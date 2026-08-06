"""Paired CIs for the SPLADE contrasts, and where the SPLADE-only gold documents land.

Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning):
  - [[reference-validation-standards]] / [[feedback_rigorous_validation]] -- bootstrap CI,
    permutation test, Holm-Bonferroni at alpha=0.05 are required before acting on a contrast.
    n_boot >= 2000, n_perm = 5000. This module implements exactly that.
  - [[project-recall-splade-learned-sparse-measured-2026-08-06]] -- the point estimates this
    quantifies.

Reads only artifacts already on disk (pools, scores, qrels). No retrieval, no GPU, no model.

Two questions:

1. Are the contrasts real? The reranked primary contrast is -0.0042, which is small enough that
   its sign is not established by a point estimate. Paired over the same 777 queries, so the
   bootstrap resamples QUERIES and the permutation flips the sign of each query's delta.

2. Where does the coverage go? Coverage rose (+0.0303 R@100) while reranked nDCG@5 did not move.
   Either the documents SPLADE adds are not gold, or they are gold and the cross-encoder buries
   them. Those have opposite implications -- keep pushing coverage, or switch to ranking -- so
   the rank distribution of SPLADE-only gold is the number that chooses the next lever.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from benchmarks.mtrag.rerank_offload import EVAL_K, rerank_order
from benchmarks.mtrag.run import DOMAINS, load_qrels

N_BOOT = 10000
N_PERM = 5000
SEED = 20260806


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def load_pool(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_header"):
                continue
            out.setdefault(str(r["qid"]), {})[str(r["doc_id"])] = float(r["score"])
    return out


def paired_stats(deltas: list[float], rng: random.Random) -> dict:
    """Bootstrap CI over QUERIES and a sign-flip permutation p, on the paired differences.

    Paired because both arms answered the same 777 queries: the query is the unit of resampling,
    which is what makes the interval about the contrast rather than about query difficulty.
    """
    n = len(deltas)
    observed = sum(deltas) / n

    means = []
    for _ in range(N_BOOT):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * N_BOOT)]
    hi = means[int(0.975 * N_BOOT) - 1]

    # Sign-flip: under the null the arm labels are exchangeable, so each query's delta is as
    # likely to have carried the opposite sign.
    extreme = 0
    for _ in range(N_PERM):
        flipped = sum(d if rng.random() < 0.5 else -d for d in deltas) / n
        if abs(flipped) >= abs(observed):
            extreme += 1
    p = (extreme + 1) / (N_PERM + 1)

    nonzero = sum(1 for d in deltas if d != 0.0)
    return {
        "n": n, "n_nonzero": nonzero, "mean_delta": observed,
        "ci_low": lo, "ci_high": hi, "p_permutation": p,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
    }


def holm(results: dict[str, dict], alpha: float = 0.05) -> None:
    """Holm-Bonferroni, in place. The primary multiple-hypothesis gate per the standards."""
    ordered = sorted(results.items(), key=lambda kv: kv[1]["p_permutation"])
    m = len(ordered)
    stopped = False
    for rank, (_name, res) in enumerate(ordered):
        res["holm_rank"] = rank
        res["holm_threshold"] = alpha / (m - rank)
        # Step-down: the first hypothesis that fails stops all weaker ones, regardless of their
        # own p. Writing it as a latch rather than a nested loop keeps that rule visible and
        # avoids re-deriving each item's position, which the first version got wrong.
        if stopped:
            res["holm_significant"] = False
            continue
        res["holm_significant"] = res["p_permutation"] < res["holm_threshold"]
        if not res["holm_significant"]:
            stopped = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--mtrag-root", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--control", default="hybrid_lexical")
    parser.add_argument("--treatment", default="hybrid_splade")
    parser.add_argument("--scores", type=Path, default=None,
                        help="scores file; defaults to <offload-dir>/scores.jsonl")
    args = parser.parse_args(argv)

    rng = random.Random(SEED)
    qrels_by_domain = load_qrels(args.mtrag_root.resolve(), args.split)
    qrels: dict[str, set[str]] = {}
    for d in DOMAINS:
        qrels.update(qrels_by_domain[d])

    control = load_pool(args.offload_dir / "pools" / f"{args.control}.jsonl")
    treatment = load_pool(args.offload_dir / "pools" / f"{args.treatment}.jsonl")
    scores = load_scores(args.scores or (args.offload_dir / "scores.jsonl"))

    shared = sorted(set(control) & set(treatment) & set(qrels))
    print(json.dumps({"event": "setup", "paired_queries": len(shared),
                      "control": args.control, "treatment": args.treatment}), flush=True)

    deltas: dict[str, list[float]] = {k: [] for k in
                                      ("raw_nDCG@5", "raw_R@100", "rr_nDCG@5", "rr_R@100")}
    # Diagnostic accumulators for question 2.
    splade_only_gold_ranks: list[int] = []
    splade_only_gold_total = 0
    queries_with_splade_only_gold = 0

    for qid in shared:
        rel = qrels[qid]
        c_raw = control[qid]["candidates"]
        t_raw = treatment[qid]["candidates"]
        c_rr = rerank_order(c_raw, scores.get(qid, {}))
        t_rr = rerank_order(t_raw, scores.get(qid, {}))

        deltas["raw_nDCG@5"].append(ndcg_at(t_raw, rel, 5) - ndcg_at(c_raw, rel, 5))
        deltas["raw_R@100"].append(recall_at(t_raw, rel, EVAL_K) - recall_at(c_raw, rel, EVAL_K))
        deltas["rr_nDCG@5"].append(ndcg_at(t_rr, rel, 5) - ndcg_at(c_rr, rel, 5))
        deltas["rr_R@100"].append(recall_at(t_rr, rel, EVAL_K) - recall_at(c_rr, rel, EVAL_K))

        # Gold the treatment's pool has and the control's does not: the coverage SPLADE adds.
        extra_gold = (set(t_raw) & rel) - set(c_raw)
        if extra_gold:
            queries_with_splade_only_gold += 1
            splade_only_gold_total += len(extra_gold)
            for doc in extra_gold:
                splade_only_gold_ranks.append(t_rr.index(doc))

    results = {name: paired_stats(vals, rng) for name, vals in deltas.items()}
    holm(results)
    for name, res in results.items():
        print(json.dumps({"event": "contrast", "metric": name, **res}), flush=True)

    splade_only_gold_ranks.sort()
    n = len(splade_only_gold_ranks)
    summary = {
        "event": "coverage_destination",
        "queries_with_splade_only_gold": queries_with_splade_only_gold,
        "splade_only_gold_documents": splade_only_gold_total,
        "post_rerank_rank": {
            "min": splade_only_gold_ranks[0] if n else None,
            "p25": splade_only_gold_ranks[n // 4] if n else None,
            "median": splade_only_gold_ranks[n // 2] if n else None,
            "p75": splade_only_gold_ranks[3 * n // 4] if n else None,
            "max": splade_only_gold_ranks[-1] if n else None,
        },
        # The decisive split: gold that reaches a depth a reader actually sees, versus gold that
        # is retrieved and then buried. The second kind is what a stronger reranker could rescue.
        "reaching_top_5": sum(1 for r in splade_only_gold_ranks if r < 5),
        "reaching_top_10": sum(1 for r in splade_only_gold_ranks if r < 10),
        "buried_below_10": sum(1 for r in splade_only_gold_ranks if r >= 10),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
