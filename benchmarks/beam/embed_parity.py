"""Is a GPU-built index the same index? Compare embeddings across runtimes, and RANK, not bits.

::

    # on VPS2 (the CPU reference)
    python -m benchmarks.beam.embed_parity emit --texts sample.json --out cpu.npz

    # on the rented GPU box, after installing onnxruntime-gpu
    python -m benchmarks.beam.embed_parity emit --texts sample.json --out gpu.npz

    # anywhere
    python -m benchmarks.beam.embed_parity compare --reference cpu.npz --candidate gpu.npz

Prior work: searched with ``docs_search(source_type="memory", ...)``.
[[project-recall-splade-learned-sparse-measured-2026-08-06]] records both halves of the lesson this
is built on: a rerank-offload gate demanding exact GPU-vs-CPU ordering equality "which floating
point cannot deliver" failed on two documents tied at 9.5e-07, and a device test asserting
``cpu == cpu`` on a CPU-only box passed against buggy code. Nothing existed to check embedding
parity across runtimes.

Why this is needed at all
-------------------------
The free arm's embedder is `BAAI/bge-small-en-v1.5` through fastembed, which as installed is **CPU
ONNX**. Building its index on a GPU means `onnxruntime-gpu`: the same weights and the same graph,
but a different kernel implementation, and possibly TF32 or fp16 accumulation. The vectors will be
numerically close and **not** bit-identical.

Two failure modes, and only one of them is about numbers
--------------------------------------------------------
1. The vectors diverge enough to change retrieval. That is what would invalidate the arm, and it is
   measured here by RANKING agreement, not by a distance: a benchmark cares whether the same
   passages come back in the same order, and two vector sets can differ in the last decimal while
   ranking identically.
2. **The GPU was never used.** `onnxruntime-gpu` falls back to `CPUExecutionProvider` silently when
   CUDA is unavailable, and then this whole comparison is CPU against CPU: it passes, proves
   nothing, and certifies a run that never happened. `emit` therefore records the providers
   ONNX Runtime actually resolved, and `compare` REFUSES a candidate that ran on the same provider
   as the reference unless `--allow-same-provider` says that is intended.

Thresholds — BOTH are enforced, and they catch different things
---------------------------------------------------------------
A run passes only if it clears both. They are not redundant:

- `--min-cosine` (default 0.9999) is a numeric floor on the worst per-row cosine. It catches gross
  divergence even where the top-k happens to survive it.
- `--max-rank-disagreement` (default 0.0) gates on the top-k **SET**, not its order, with a tie
  band at the k boundary (`TIE_RTOL`).

Why the SET and not the order: `get_beam_answer_generation_prompt` re-sorts retrieved memories
chronologically before rendering them. ⚠️ It also TRUNCATES to `top_k` FIRST, in retrieval order —
so this reasoning holds only while `--cutoff >= --k`. Where the answerer's cutoff is below the
retrieval budget, retrieval order decides which memories survive and the order statistic would
need gating too. The BEAM arm runs k=45/cutoff=45, where the slice is a no-op.

Three regimes this refuses rather than scores, each found by audit after shipping:

- **k >= the document count.** The top-k set is then the whole corpus and `set_disagreement` is 0
  by construction. Measured: two INDEPENDENT random matrices at n=100 reported 0.000 with the
  shipped defaults. A gate with a silent regime where it cannot fail is worse than no gate.
- **k < 1 or queries < 1.** `k=0` compared two empty sets and passed; `queries=0` divided by zero.
- **An unrecorded execution provider.** "Could not observe what ran" is refused, not treated as
  evidence of a different runtime.

⚠️ The noise table that used to sit here was measured against the PRE-FIX code and has been
removed rather than left to describe behaviour that no longer exists. Current calibration, on
uniform random vectors at n=200/queries=64/k=45: noise <= 1e-7 passes, >= 1e-5 fails on set
disagreement. The stable sort made the set statistic MORE sensitive, not less, because tie-flip
noise no longer masks real movement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import cast

import numpy as np


#: Similarity tolerance for calling two documents tied at the top-k boundary. Chosen well ABOVE
#: fp32 jitter (~1e-7 on a cosine) and well BELOW any divergence that matters: the measured
#: CPU-vs-CUDA run moved similarities by ~1e-4, four orders of magnitude outside this band, so a
#: genuine runtime difference still fails the gate.
TIE_RTOL = 1e-9
TIE_ATOL = 1e-12


def _providers() -> list[str]:
    """The execution providers ONNX Runtime actually resolved, not the ones requested."""
    try:
        import onnxruntime  # noqa: PLC0415

        return list(onnxruntime.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime absent
        return []


def emit(
    texts_path: Path, out: Path, model: str, providers: list[str] | None = None
) -> dict[str, object]:
    """Embed `texts` here, and record the providers the SESSION actually resolved.

    `providers` is a request, not a guarantee — which is the entire reason this records the
    session's answer rather than the request. Measured on an RTX 5090 host: asking for
    `CUDAExecutionProvider` against a CUDA-13 wheel on a CUDA-12.8 box fell back to CPU with a
    `RuntimeWarning` and nothing else. A run that recorded the request would have claimed a GPU
    build that never happened.
    """
    from recall.embeddings import FastEmbedEmbedder

    texts: list[str] = json.loads(Path(texts_path).read_text(encoding="utf-8"))
    if not texts:
        raise SystemExit(f"{texts_path} holds no texts")

    embedder = FastEmbedEmbedder(model_name=model, providers=providers)
    session_providers = [
        p for p in list(getattr(embedder, "session_providers", []) or [])
        if not str(p).startswith("<")
    ]
    vectors: np.ndarray = np.asarray(embedder.embed(texts), dtype=np.float64)
    if vectors.shape[0] != len(texts):
        raise SystemExit(f"embedder returned {vectors.shape[0]} vectors for {len(texts)} texts")

    meta = {
        "model": model,
        "dim": int(vectors.shape[1]),
        "n": int(vectors.shape[0]),
        # The SESSION's providers, and — critically — WHERE that list came from. The previous
        # version silently fell back to module-level availability (`or _providers()`), which is
        # identical on a box whether or not the session touched the GPU; `compare` could not tell
        # the two apart and passed the fallback as evidence of a different runtime.
        "providers": session_providers or ["<session not reachable>"],
        "providers_source": "session" if session_providers else "unavailable",
        "available_providers": _providers(),
        "providers_requested": providers,
        # Binds this emission to its input. Without it two emissions over DIFFERENT samples of the
        # same length compare as a runtime difference.
        "texts_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vectors=vectors, meta=json.dumps(meta))
    print(json.dumps(meta, indent=2))
    return meta


def _load(path: Path) -> tuple[np.ndarray, dict]:
    # Object arrays stay disabled: these files travel between hosts, and a vector dump has no
    # business carrying executable state. Metadata rides as a JSON string for the same reason.
    data = np.load(path, allow_pickle=False)
    return data["vectors"], json.loads(str(data["meta"]))


def _unit(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit: np.ndarray = v / norms
    return unit


def rank_disagreement(
    ref: np.ndarray, cand: np.ndarray, n_queries: int, k: int
) -> dict[str, object]:
    """Fraction of sample queries whose top-k ordering differs between the two vector sets.

    A self-contained mini-retrieval: the first `n_queries` rows act as queries against the rest as
    documents. This deliberately avoids the database — the question is whether the VECTORS rank the
    same, and routing it through an HNSW index would fold that index's own approximation into the
    answer.

    Both the ordered top-k and the top-k SET are reported. A pair of documents at a near-tie can
    swap without changing which passages the answerer sees, and that is a materially milder event
    than a passage entering or leaving the context.
    """
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}")
    if k < 1:
        raise SystemExit(f"k must be >= 1, got {k}: a top-0 set is empty on both sides and would "
                         f"report perfect agreement, and a negative k silently means drop-last")
    if n_queries < 1:
        raise SystemExit(f"queries must be >= 1, got {n_queries}: it is the denominator of both "
                         f"reported rates")
    if ref.shape[0] <= n_queries:
        raise SystemExit(f"need more than {n_queries} rows to split queries from documents")

    rq, rd = _unit(ref[:n_queries]), _unit(ref[n_queries:])
    cq, cd = _unit(cand[:n_queries]), _unit(cand[n_queries:])
    # REFUSE rather than clamp. `k = min(k, n_docs)` looks harmless and is not: once k reaches the
    # document count the top-k SET is the whole corpus for every query, so `set(a) == set(b)` holds
    # by construction and `set_disagreement` — the only GATED statistic — is 0.0 for arbitrarily
    # unrelated vectors. Measured: two INDEPENDENT random matrices at n=100 with the shipped
    # defaults reported set_disagreement 0.000 (k clamped 45 -> 36); at n=110 the same pair
    # reported 0.984. A gate with a silent regime where it cannot fail is worse than no gate,
    # because it reports PARITY OK.
    if rd.shape[0] <= k:
        raise SystemExit(
            f"only {rd.shape[0]} documents for k={k}: the top-k set would be the entire corpus and "
            f"set_disagreement would be 0 by construction, not by agreement. Supply at least "
            f"{n_queries + k + 1} texts, or lower --k."
        )

    # `kind="stable"` so equal similarities break ties by index identically on BOTH sides. Under
    # the default (unstable) quicksort an EXACT tie straddling the k boundary resolves arbitrarily
    # per array, so a candidate identical to ~15 significant digits fails a zero-tolerance gate:
    # measured, 22 of 80 seed/noise configurations rejected a numerically identical build. Exact
    # ties are routine here — two identical turns in a conversation embed identically.
    ref_scores = rq @ rd.T
    cand_scores = cq @ cd.T
    ref_top = np.argsort(-ref_scores, axis=1, kind="stable")[:, :k]
    cand_top = np.argsort(-cand_scores, axis=1, kind="stable")[:, :k]

    order_diff = int(np.sum(np.any(ref_top != cand_top, axis=1)))

    # A set difference counts ONLY when the documents that moved sit meaningfully away from the
    # k-th-place similarity on both sides. A stable sort fixes ties that are exact on BOTH sides,
    # but the realistic case is a near-tie: two documents separated by ~1e-12 straddle rank k, and
    # which one lands inside is decided by noise far below the divergence being measured. Counting
    # that rejects a build identical to ~15 significant digits, which is the failure this module's
    # own history warns about — a gate demanding equality floating point cannot deliver.
    #
    # The band is on the SIMILARITY, not on the vectors, and it is tight: a real divergence moves
    # similarities by orders of magnitude more than TIE_RTOL, so it still fails.
    set_diff = 0
    for i in range(n_queries):
        ref_set, cand_set = set(ref_top[i].tolist()), set(cand_top[i].tolist())
        if ref_set == cand_set:
            continue
        ref_boundary = float(ref_scores[i, ref_top[i, -1]])
        cand_boundary = float(cand_scores[i, cand_top[i, -1]])
        moved_at_the_boundary = all(
            np.isclose(ref_scores[i, d], ref_boundary, rtol=TIE_RTOL, atol=TIE_ATOL)
            and np.isclose(cand_scores[i, d], cand_boundary, rtol=TIE_RTOL, atol=TIE_ATOL)
            for d in ref_set ^ cand_set
        )
        if not moved_at_the_boundary:
            set_diff += 1
    return {
        "queries": n_queries,
        "k": k,
        "order_disagreement": order_diff / n_queries,
        "set_disagreement": set_diff / n_queries,
        "order_differing": order_diff,
        "set_differing": set_diff,
    }


def compare(
    reference: Path,
    candidate: Path,
    *,
    min_cosine: float,
    max_rank_disagreement: float,
    n_queries: int,
    k: int,
    allow_same_provider: bool,
) -> dict[str, object]:
    ref, ref_meta = _load(reference)
    cand, cand_meta = _load(candidate)

    # Before ANY arithmetic. `cos = ...` below broadcasts, so a (200,384) vs (200,768) pair raised
    # a raw numpy ValueError from the subtraction instead of the readable message the shape guard
    # inside rank_disagreement was written to produce — and a broadcast-compatible mismatch such
    # as (200,384) vs (1,384) produced a full-length cosine array before anything checked.
    if ref.shape != cand.shape:
        raise SystemExit(
            f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}. The two emissions "
            f"must cover the same texts with the same model."
        )

    if ref_meta["model"] != cand_meta["model"]:
        raise SystemExit(
            f"different models: {ref_meta['model']} vs {cand_meta['model']}. This check answers "
            f"'same model, different runtime'; a model swap is a different arm, not a parity risk."
        )

    # POSITIVE verification, not a difference test. The original check was
    # `ref["providers"] != cand["providers"]`, which answers "are these lists unequal" and NOT "do
    # I know what the candidate ran on". An unrecorded session (`<session not reachable>`) or a
    # fallback to module-level availability compares unequal to a real CPU reference, so the
    # refusal never fired: measured, byte-identical vectors with a candidate provider of
    # `["<session not reachable>"]` returned PARITY OK. That is the module's own stated failure
    # mode — "it passes, proves nothing, and certifies a run that never happened" — reintroduced
    # by the guard against it. `_check_resume_config` in run.py states the principle this now
    # follows: "cannot check" and "checked and fine" have to land differently.
    for label, meta in (("reference", ref_meta), ("candidate", cand_meta)):
        providers = list(meta.get("providers") or [])
        if not providers or any(str(p).startswith("<") for p in providers):
            raise SystemExit(
                f"{label} did not record which providers its ONNX session resolved "
                f"({providers or 'empty'}). Parity cannot be certified from an unknown runtime: "
                f"an unrecorded session is not evidence of a different one. Re-emit with a build "
                f"whose session can be introspected."
            )
        if meta.get("providers_source") == "availability":
            raise SystemExit(
                f"{label} recorded module-level provider AVAILABILITY, not its session's "
                f"resolved providers. Availability lists CUDAExecutionProvider on a GPU box even "
                f"after a silent CPU fallback, so it cannot distinguish the two."
            )

    same_provider = ref_meta.get("providers") == cand_meta.get("providers")
    if same_provider and not allow_same_provider:
        raise SystemExit(
            "reference and candidate resolved the SAME execution providers "
            f"({cand_meta.get('providers')}). onnxruntime-gpu falls back to CPUExecutionProvider "
            "silently when CUDA is unavailable, so this comparison would be CPU against CPU: it "
            "would pass while proving nothing about the GPU build. Install onnxruntime-gpu on the "
            "candidate host, or pass --allow-same-provider if comparing two CPU runs is genuinely "
            "what you meant."
        )

    # Bind both sides to the SAME sample. Nothing else does: the documented workflow copies
    # sample.json between hosts by hand, so a regenerated or reordered sample of the same length
    # would compare as a runtime difference and be reported as one.
    ref_texts, cand_texts = ref_meta.get("texts_sha256"), cand_meta.get("texts_sha256")
    if ref_texts and cand_texts and ref_texts != cand_texts:
        raise SystemExit(
            f"the two emissions embedded DIFFERENT texts (sha256 {ref_texts[:12]}… vs "
            f"{cand_texts[:12]}…). This compares runtimes over one sample; a different sample is "
            f"a different measurement, not a parity failure."
        )

    cos = np.sum(_unit(ref) * _unit(cand), axis=1)
    diff = np.abs(ref - cand)
    ranks = rank_disagreement(ref, cand, n_queries, k)

    vectors_ok = float(cos.min()) >= min_cosine
    # The SET, not the order. `get_beam_answer_generation_prompt` re-sorts the retrieved memories
    # chronologically before rendering them, so retrieval order is discarded and never reaches the
    # model — only membership of the top-k does. Gating on order would fail a run over tie-flips
    # the benchmark is structurally incapable of noticing.
    #
    # This is not a theoretical difference. Measured on a 200-vector sample: at 1e-5 injected
    # noise the minimum cosine is 0.999999977, far above any sane cosine bar, while 3.1% of
    # queries already return a DIFFERENT top-10 order. Cosine alone is the wrong instrument;
    # order alone is too strict; membership is what the benchmark actually consumes.
    ranking_ok = float(cast(float, ranks["set_disagreement"])) <= max_rank_disagreement
    return {
        "reference": {"path": str(reference), **ref_meta},
        "candidate": {"path": str(candidate), **cand_meta},
        "vectors": {
            "n": int(ref.shape[0]),
            "min_cosine": float(cos.min()),
            "mean_cosine": float(cos.mean()),
            "max_abs_elementwise_diff": float(diff.max()),
            "threshold_min_cosine": min_cosine,
            "passed": vectors_ok,
        },
        "ranking": {
            **ranks,
            "gated_on": "set_disagreement",
            "threshold_set_disagreement": max_rank_disagreement,
            "passed": ranking_ok,
        },
        "verdict": "PARITY OK" if (vectors_ok and ranking_ok) else "PARITY FAILED",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.beam.embed_parity")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="embed a text sample with the runtime installed here")
    e.add_argument("--texts", type=Path, required=True, help="JSON array of strings")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    e.add_argument("--provider", action="append", dest="providers",
                   help="repeatable, e.g. --provider CUDAExecutionProvider. A REQUEST; the "
                        "session's actual answer is what gets recorded")

    c = sub.add_parser("compare", help="compare two emissions and report a parity verdict")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--min-cosine", type=float, default=0.9999)
    c.add_argument("--max-rank-disagreement", type=float, default=0.0,
                   help="max fraction of sample queries whose top-k SET may differ. Order is "
                        "reported but NOT gated: the BEAM answerer re-sorts memories "
                        "chronologically, so retrieval order never reaches the model")
    c.add_argument("--queries", type=int, default=64)
    c.add_argument("--k", type=int, default=45)
    c.add_argument("--allow-same-provider", action="store_true")
    c.add_argument("--out", type=Path, default=None)

    args = p.parse_args(argv)
    if args.cmd == "emit":
        emit(args.texts, args.out, args.model, args.providers)
        return 0

    report = compare(
        args.reference,
        args.candidate,
        min_cosine=args.min_cosine,
        max_rank_disagreement=args.max_rank_disagreement,
        n_queries=args.queries,
        k=args.k,
        allow_same_provider=args.allow_same_provider,
    )
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["verdict"] == "PARITY OK" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
