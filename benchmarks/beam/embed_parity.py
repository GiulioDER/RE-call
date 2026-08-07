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

- `--min-cosine` (default 0.9999) is a numeric floor on the worst per-row cosine between the two
  vector sets. It catches gross divergence even where the top-k happens to survive it.
- `--max-rank-disagreement` (default 0.0) is gated on the top-k **SET**, not its order.
  `get_beam_answer_generation_prompt` re-sorts retrieved memories chronologically before rendering
  them, so retrieval order is discarded and never reaches the model. Only membership matters, and
  gating on order would fail runs over tie-flips this benchmark cannot see.

Measured on a 200-vector sample of bge-small, injecting noise into a CPU emission:

    noise   min_cosine     set_disagree   order_disagree   verdict
    1e-7    1.000000000    0.000          0.000            OK
    1e-6    1.000000000    0.000          0.000            OK
    1e-5    0.999999977    0.000          0.031            OK
    1e-4    0.999997736    0.000          0.094            OK
    1e-3    0.999774151    0.000          0.438            FAILED  (on cosine)
    1e-2    0.978413459    0.688          1.000            FAILED  (on both)

Two things that table settles. Cosine alone is the wrong instrument: at 1e-5 the worst cosine is
0.99999998 while 3.1% of queries already reorder. And order alone is too strict: membership holds
to 1e-4 while ordering has been moving for two decades of noise. Set membership is what the
benchmark consumes, and the cosine floor is what stops a numerically broken build sneaking through
on a corpus too small to reshuffle.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np


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
    vectors = np.asarray(embedder.embed(texts), dtype=np.float64)
    if vectors.shape[0] != len(texts):
        raise SystemExit(f"embedder returned {vectors.shape[0]} vectors for {len(texts)} texts")

    meta = {
        "model": model,
        "dim": int(vectors.shape[1]),
        "n": int(vectors.shape[0]),
        # The SESSION's providers. `_providers()` (module-level availability) is kept only as a
        # diagnostic: it is identical on a box whether or not the session ever touched the GPU.
        "providers": list(getattr(embedder, "session_providers", [])) or _providers(),
        "available_providers": _providers(),
        "providers_requested": providers,
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
    return v / norms


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
    if ref.shape[0] <= n_queries:
        raise SystemExit(f"need more than {n_queries} rows to split queries from documents")

    rq, rd = _unit(ref[:n_queries]), _unit(ref[n_queries:])
    cq, cd = _unit(cand[:n_queries]), _unit(cand[n_queries:])
    k = min(k, rd.shape[0])

    ref_top = np.argsort(-(rq @ rd.T), axis=1)[:, :k]
    cand_top = np.argsort(-(cq @ cd.T), axis=1)[:, :k]

    order_diff = int(np.sum(np.any(ref_top != cand_top, axis=1)))
    set_diff = sum(
        1 for i in range(n_queries) if set(ref_top[i].tolist()) != set(cand_top[i].tolist())
    )
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

    if ref_meta["model"] != cand_meta["model"]:
        raise SystemExit(
            f"different models: {ref_meta['model']} vs {cand_meta['model']}. This check answers "
            f"'same model, different runtime'; a model swap is a different arm, not a parity risk."
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
    ranking_ok = float(ranks["set_disagreement"]) <= max_rank_disagreement
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
