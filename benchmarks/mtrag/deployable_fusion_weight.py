"""Does down-weighting the harmful variant rescue the DEPLOYABLE query fusion?

Preregistration: `benchmarks/archive/preregistrations/PREREGISTRATION-deployable-fusion-weight.md`, frozen before this ran.

Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning):
  - [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]] -- do NOT build an LLM
    rewriter. This arm uses no gold and no LLM, so it honours that.
  - [[closed-hypothesis-recall-leg-disagreement-prf-trigger-2026-07-28]] -- per-query ADAPTIVE
    weighting is falsified. The weight here is fixed a priori and identical on every query.
  - [[reference-mtrag-dev-query-variant-overlap-2026-08-06]] -- the deciding cell.
  - The 2026-08-06 multi-query run, whose `mq_nested2_nogold` post-hoc result is the input.

⚠️ This module IMPORTS `benchmarks.mtrag.multiquery` and modifies NOTHING in it. That module is
uncommitted work belonging to a parallel session; editing it would be a lost update. The arm is
constructed here, from their frozen `MultiQueryArm` and their `fuse_arm`, so the fusion is theirs
and only the weight differs.

The question, in one line: `{last, full}` unweighted buys +0.0842 R@100 and costs -0.0447 nDCG@5,
tripping three vetoes. On the three-variant arm, `w_full = 0.5` recovered +0.0208 nDCG@5 for
-0.0106 R@100. Does the same weight make the deployable arm ranking-neutral?
"""

from __future__ import annotations

import argparse
import json
from typing import Any
import random
from pathlib import Path

from benchmarks.mtrag.analyse_contrasts import holm, ndcg_at, paired_stats, recall_at
from benchmarks.mtrag.multiquery import (
    EVAL_K,
    MultiQueryArm,
    fuse_arm,
    load_legs,
    load_queries,
)
from benchmarks.mtrag.run import DOMAINS, load_qrels

SEED = 20260806

#: The arm, frozen in the preregistration. w_full = 0.5 is the value `mq_nested3_vw` already used
#: A PRIORI in the previous run, so it carries no information from this contrast. NO OTHER WEIGHT
#: WILL BE RUN -- sweeping until one passes is how the adaptive-weighting hypothesis went wrong.
TREATMENT = MultiQueryArm(
    "mq_nested2_nogold_vw", ("last", "full"), "nested", weights=(1.0, 0.5),
    role="preregistered_deployable",
)
#: The unweighted deployable arm, re-fused here so treatment and baseline come from identical
#: legs in one process rather than being compared across runs.
UNWEIGHTED = MultiQueryArm(
    "mq_nested2_nogold", ("last", "full"), "nested", role="post_hoc_reference",
)
CONTROL = MultiQueryArm("mq_last", ("last",), "nested", role="control")

#: The ship bar and the veto family, both fixed in the preregistration.
SHIP_BAR = 0.020
DECISION_METRIC = "R@100"
VETO_METRICS = (("nDCG@5", ndcg_at, 5), ("nDCG@10", ndcg_at, 10),
                ("R@5", recall_at, 5), ("R@10", recall_at, 10))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mq-dir", type=Path, required=True)
    parser.add_argument("--mtrag-root", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.mtrag_root.resolve()
    out = args.mq_dir.resolve()
    qrels_by_domain = load_qrels(root, "dev")
    qrels: dict[str, set[str]] = {}
    for d in DOMAINS:
        qrels.update(qrels_by_domain[d])

    legs = load_legs(out, ["last", "full"])
    texts = {v: load_queries(out, v) for v in ("last", "full")}
    shared = sorted(set(legs) & set(qrels))
    rng = random.Random(SEED)

    rankings = {
        arm.name: {q: fuse_arm(arm, legs[q]) for q in shared}
        for arm in (CONTROL, UNWEIGHTED, TREATMENT)
    }

    # The deciding cell for a {last, full} fusion: queries where those two texts DIFFER. On the
    # rest both arms retrieve identically and the paired delta is 0 by construction.
    cell = [q for q in shared if texts["last"][q].strip() != texts["full"][q].strip()]
    print(json.dumps({"event": "setup", "queries": len(shared), "deciding_cell_n": len(cell),
                      "weight": TREATMENT.weights}), flush=True)

    for name, ranking in rankings.items():
        row: dict[str, Any] = {"event": "arm", "arm": name}
        for mname, fn, k in VETO_METRICS:
            row[mname] = round(sum(fn(ranking[q], qrels[q], k) for q in shared) / len(shared), 4)
        row["R@100"] = round(
            sum(recall_at(ranking[q], qrels[q], EVAL_K) for q in shared) / len(shared), 4)
        print(json.dumps(row), flush=True)

    # Contrast: treatment vs the CONTROL (not vs the unweighted arm) -- the ship decision is
    # "is the deployable fusion better than shipping nothing", and the unweighted arm is a
    # reference point, not the baseline.
    metrics = [(DECISION_METRIC, recall_at, EVAL_K), *VETO_METRICS]
    for population, ids in (("all_777", shared), ("deciding_cell", cell)):
        results: dict[str, dict] = {}
        for mname, fn, k in metrics:
            deltas = [fn(rankings[TREATMENT.name][q], qrels[q], k)
                      - fn(rankings[CONTROL.name][q], qrels[q], k) for q in ids]
            results[mname] = paired_stats(deltas, rng)
        holm(results)
        for mname, res in results.items():
            print(json.dumps({"event": "contrast", "population": population,
                              "metric": mname, **res}), flush=True)

        if population == "all_777":
            primary = results[DECISION_METRIC]
            tripped = [
                m for m, _fn, _k in VETO_METRICS
                if results[m]["mean_delta"] < 0 and results[m]["ci_excludes_zero"]
            ]
            ships = (primary["mean_delta"] >= SHIP_BAR
                     and primary["ci_excludes_zero"]
                     and primary["holm_significant"]
                     and not tripped)
            print(json.dumps({
                "event": "decision", "arm": TREATMENT.name,
                "ship_bar": SHIP_BAR, "primary_delta": primary["mean_delta"],
                "primary_ci": [primary["ci_low"], primary["ci_high"]],
                "vetoes_tripped": tripped,
                "verdict": "SHIPS" if ships else "NOT SHIP",
                # P3 was recorded as genuinely uncertain before the run. Whichever way it lands,
                # the preregistration says so.
                "P3_nDCG5_veto_cleared": "nDCG@5" not in tripped,
            }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
