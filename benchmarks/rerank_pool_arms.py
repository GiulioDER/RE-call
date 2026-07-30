"""Matched-config pool-level reranker comparison: retrieval vs local MiniLM vs Voyage rerank-2.5.

Prior work (searched before writing this, per CLAUDE.md):
`docs_search(source_type='memory', ...)` on "fine-tuning reranker cross-encoder training RE-call"
and on "RE-call training learned model closed hypothesis" ->
[[project-recall-nearmiss-signal-exhaustion-2026-07-29]] (the reranker arc, and the source of the
published numbers reproduced below), [[reference-recall-best-configuration-2026-07-28]],
[[project-recall-answerability-ladder-2026-07-29]]. The second query returned `gap_warning: true`:
**no prior work exists on reranker fine-tuning or on a matched-config local-vs-hosted pool
comparison** -- the gap this script fills. Related but NOT this: [[project_ns4_finetune_bge_2026-04-26]]
and [[archive/project_replug_lsr_2026-05-06]] fine-tuned a *bi-encoder* against a *noisy forward-return*
label; `docs/RAG_TRAINING_STUDY.md` fine-tuned a bi-encoder on toy corpora. None measured a reranker.

WHY THIS EXISTS. The published pool-level reranking result (`voyage:rerank-2.5` lifting hit@5
0.640 -> 0.870 on the ladder's 200 answerable questions) has **no local-reranker arm**. The only
MiniLM-vs-Voyage comparison ever run reordered a FIXED top-5, which measures ordering and cannot
move set membership; and the shipped reranker's own headline (hit@5 0.671 -> 0.777) comes from a
different n and a different config. So the two numbers people naturally compare were never measured
on the same footing -- a one-sided comparison. This script measures the missing arm.

It also fixes a provenance gap: the retained `voyage_pool_out.jsonl` / `ksweep_out.jsonl` artifacts
were produced by throwaway inline code with no committed generator, so their numbers could be read
but not re-derived. This script re-derives the published figures as a precondition (see INVARIANTS)
before adding anything to them.

INPUTS (all already on disk, no network, no paid API call):
  ksweep_out.jsonl       per-instance 50-chunk candidate pool, in fused retrieval order, + gold ids
  voyage_pool_out.jsonl  per-chunk `voyage:rerank-2.5` scores over the SAME 50 chunks
  locomo10.json          source of truth for question and turn text

INVARIANTS -- all hard failures, never warnings. A reranker benchmark that silently degrades to the
identity ordering, or silently drops the pool members it cannot resolve, still prints a plausible
table; that is the failure mode these guard.
  1. the two artifacts describe identical chunk sets for every instance
  2. every pool member and every question resolves to text from locomo10.json
  3. the retrieval and Voyage arms reproduce their published values exactly
  4. the MiniLM arm is not the identity ordering (i.e. the reranker actually ran)

Usage:
    python benchmarks/rerank_pool_arms.py [--batch-size 16] [--limit N] [--out results/...json]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from benchmarks.ladder.sources.locomo import load_locomo
from recall.rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION

REPO = Path(__file__).resolve().parent.parent

#: Pool-level Voyage reranking vs plain retrieval, n=200 answerable, top-50 pool, as published in
#: the memo above. Reproducing these is a precondition, not a result.
PUBLISHED = {
    "retrieval": {1: 0.375, 5: 0.640, 10: 0.730},
    "voyage": {1: 0.710, 5: 0.870, 10: 0.895},
}
#: The ladder's answerable arm. Ring 0 is the near-miss (gold-excised) arm and is out of scope here:
#: with the evidence removed there is no gold to hit, so hit@k is not the question being asked.
ANSWERABLE_RING = -2
KS = (1, 5, 10)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _hit_at(order: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(order[:k]) & gold else 0.0


def _ceiling(hit5: float) -> float:
    """The abstention ceiling as defined in the ladder work: hit@k + 0.5*(1 - hit@k)."""
    return hit5 + 0.5 * (1 - hit5)


def _summarise(per_instance: dict[int, list[float]]) -> dict:
    out = {k: sum(v) / len(v) for k, v in per_instance.items()}
    out["ceiling@5"] = _ceiling(out[5])
    return out


def _paired_bootstrap(
    a: list[float], b: list[float], n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on the per-instance difference b - a. Returns (mean, lo, hi) at 95%."""
    diffs = [x - y for x, y in zip(b, a)]
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (
        sum(diffs) / n,
        means[int(0.025 * n_resamples)],
        means[int(0.975 * n_resamples)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16, help="cross-encoder batch (memory-bound)")
    ap.add_argument("--limit", type=int, default=None, help="first N instances, for a smoke run")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rerank_pool_arms.json")
    # The 10,000 per-chunk scores are ~440 KB and reproducible from this script in ~5 min, so they
    # are opt-in rather than committed -- `finetune/README.md`'s "commit the numbers, not the
    # weights" applied to scores.
    ap.add_argument("--scores-out", type=Path, default=None, help="dump per-chunk MiniLM scores")
    ap.add_argument("--locomo", type=Path, default=REPO / "locomo10.json")
    args = ap.parse_args()

    ksweep = {r["instance_id"]: r for r in _load_jsonl(REPO / "ksweep_out.jsonl")}
    voyage = {r["instance_id"]: r for r in _load_jsonl(REPO / "voyage_pool_out.jsonl")}

    ids = sorted(i for i, r in ksweep.items() if r["ring"] == ANSWERABLE_RING)
    if args.limit:
        ids = ids[: args.limit]
    print(f"instances: {len(ids)} (ring {ANSWERABLE_RING}, answerable)")

    # --- INVARIANT 1: the artifacts describe the same pools -------------------------------------
    for i in ids:
        if i not in voyage:
            raise SystemExit(f"INVARIANT 1 FAILED: {i} missing from voyage_pool_out.jsonl")
        kset = {h["doc_id"] for h in ksweep[i]["hits"]}
        vset = {c["id"] for c in voyage[i]["per_chunk"]}
        if kset != vset:
            raise SystemExit(f"INVARIANT 1 FAILED: pool mismatch at {i}")
    print("INVARIANT 1 ok: chunk sets identical across artifacts")

    # --- INVARIANT 2: everything resolves to text ------------------------------------------------
    corpus = load_locomo(args.locomo)
    doc_text = dict(corpus.documents)
    q_text = {q.question_id: q.question for q in corpus.questions}
    for i in ids:
        qid = i.split("/", 1)[1].split("#", 1)[0]  # locomo/conv-43/qa105#original -> conv-43/qa105
        if qid not in q_text:
            raise SystemExit(f"INVARIANT 2 FAILED: no question text for {qid}")
        for h in ksweep[i]["hits"]:
            if h["doc_id"] not in doc_text:
                raise SystemExit(f"INVARIANT 2 FAILED: no text for pool member {h['doc_id']}")
    print(f"INVARIANT 2 ok: {len(doc_text)} docs / {len(q_text)} questions resolved")

    # --- orderings we already have ---------------------------------------------------------------
    retrieval_order = {i: [h["doc_id"] for h in ksweep[i]["hits"]] for i in ids}
    voyage_order = {
        i: [c["id"] for c in sorted(voyage[i]["per_chunk"], key=lambda c: c["voyage"], reverse=True)]
        for i in ids
    }

    # --- the missing arm: MiniLM over the SAME pool ----------------------------------------------
    from sentence_transformers import CrossEncoder

    print(f"loading {DEFAULT_RERANKER_MODEL} @ {DEFAULT_RERANKER_REVISION[:12]}")
    model = CrossEncoder(DEFAULT_RERANKER_MODEL, revision=DEFAULT_RERANKER_REVISION)

    pairs: list[tuple[str, str]] = []
    spans: dict[str, tuple[int, int]] = {}
    for i in ids:
        qid = i.split("/", 1)[1].split("#", 1)[0]
        q = q_text[qid]
        start = len(pairs)
        for h in ksweep[i]["hits"]:
            pairs.append((q, doc_text[h["doc_id"]]))
        spans[i] = (start, len(pairs))

    print(f"scoring {len(pairs)} (query, chunk) pairs, batch {args.batch_size}")
    t0 = time.time()
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"scored in {elapsed:.1f}s ({elapsed / max(len(pairs), 1) * 1000:.1f} ms/pair)")

    minilm_order: dict[str, list[str]] = {}
    minilm_scores: dict[str, dict[str, float]] = {}
    for i in ids:
        lo, hi = spans[i]
        member_ids = [h["doc_id"] for h in ksweep[i]["hits"]]
        s = {d: float(v) for d, v in zip(member_ids, scores[lo:hi])}
        minilm_scores[i] = s
        minilm_order[i] = sorted(member_ids, key=lambda d: s[d], reverse=True)

    # --- INVARIANT 4: the reranker actually reordered something ----------------------------------
    identical = sum(1 for i in ids if minilm_order[i] == retrieval_order[i])
    if identical == len(ids):
        raise SystemExit("INVARIANT 4 FAILED: MiniLM arm is the identity ordering (reranker inert)")
    print(f"INVARIANT 4 ok: MiniLM reordered {len(ids) - identical}/{len(ids)} pools")

    # --- score every arm --------------------------------------------------------------------------
    arms = {"retrieval": retrieval_order, "minilm": minilm_order, "voyage": voyage_order}
    per_inst: dict[str, dict[int, list[float]]] = {}
    for name, order in arms.items():
        gold_sets = {i: set(ksweep[i]["gold_doc_ids"]) for i in ids}
        per_inst[name] = {k: [_hit_at(order[i], gold_sets[i], k) for i in ids] for k in KS}

    summary = {name: _summarise(v) for name, v in per_inst.items()}

    # --- INVARIANT 3: reproduce the published arms ------------------------------------------------
    if not args.limit:
        for arm, want in PUBLISHED.items():
            for k, w in want.items():
                got = summary[arm][k]
                if abs(got - w) >= 5e-4:
                    raise SystemExit(
                        f"INVARIANT 3 FAILED: {arm} hit@{k} = {got:.4f}, published {w:.4f}"
                    )
        print("INVARIANT 3 ok: retrieval and voyage arms reproduce published values exactly")
    else:
        print("INVARIANT 3 skipped (--limit set; published values are for the full 200)")

    # --- report -----------------------------------------------------------------------------------
    print(f"\n{'arm':<12} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'ceiling@5':>11}")
    print("-" * 52)
    for name in ("retrieval", "minilm", "voyage"):
        s = summary[name]
        print(f"{name:<12} {s[1]:>8.3f} {s[5]:>8.3f} {s[10]:>8.3f} {s['ceiling@5']:>11.4f}")

    print(f"\n{'paired delta (hit@5)':<34} {'mean':>8} {'CI95 lo':>9} {'CI95 hi':>9}")
    print("-" * 62)
    comparisons = [
        ("minilm - retrieval", "retrieval", "minilm"),
        ("voyage - retrieval", "retrieval", "voyage"),
        ("voyage - minilm", "minilm", "voyage"),
    ]
    deltas = {}
    for label, a, b in comparisons:
        mean, lo, hi = _paired_bootstrap(per_inst[a][5], per_inst[b][5])
        deltas[label] = {"mean": mean, "ci95": [lo, hi]}
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"{label:<34} {mean:>+8.4f} {lo:>+9.4f} {hi:>+9.4f}{flag}")
    print("\n* = 95% CI excludes zero")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n": len(ids),
                "ring": ANSWERABLE_RING,
                "reranker_model": DEFAULT_RERANKER_MODEL,
                "reranker_revision": DEFAULT_RERANKER_REVISION,
                "pool_size": 50,
                "corpus_content_hash": corpus.content_hash,
                "summary": {k: {str(kk): vv for kk, vv in v.items()} for k, v in summary.items()},
                "paired_deltas_hit5": deltas,
                "seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
        # `write_text` defaults to newline=None, which rewrites "\n" as "\r\n" on Windows. This file
        # is committed, so that would make every Windows re-run a whole-file diff against a
        # Linux-generated one. Pin it rather than leaning on git's normalisation.
        newline="\n",
    )
    print(f"\nwrote {args.out}")
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        args.scores_out.write_text(json.dumps(minilm_scores), encoding="utf-8", newline="\n")
        print(f"wrote {args.scores_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
