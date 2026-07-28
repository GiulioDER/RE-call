"""Phase 0 diagnostic — does leg disagreement select for retrieval failures?

Design: docs/superpowers/specs/2026-07-28-weighted-fusion-prf-phase0-design.md
Predictions and kill gates were committed BEFORE this ran. Do not edit them afterwards.

Answers three questions, each with a decision rule fixed in advance:

  Q1  hit@k split on `trigger` — if the firing group is not WORSE, the trigger selects for
      successes and PRF stops here.
  Q2  firing rate — outside 5-50% the trigger needs redesigning.
  Q3  on firing misses, where the gold chunk actually was:
        a_misranked   in the fused pool, below k        -> weighted fusion's job (Phase 1)
        b_unretrieved in neither leg's pool             -> PRF's job (Phase 2); its ceiling
        c_absent      no gold labelled                  -> labelling defect, excluded
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from recall.eval.legconf import leg_confidence, more_decisive
# Reused, not reimplemented. The mapping from a hit to a LOCOMO dialog id lives in
# `_filename_to_dia_id` (stem, first underscore -> colon) and reads `metadata["file"]`. There is
# no `dia_id` key. A local copy of that rule would silently match nothing on drift, which here
# means every miss classifies as "gold was never retrieved" — inflating the PRF ceiling to 100%
# and manufacturing a green light for Phase 2.
from recall.eval.locomo import _retrieved_dia_ids
from recall.eval.metrics import wilson_ci
from recall.retriever import LegProbe

#: `hit@5` and `hit@20` published in FINDINGS §9a, backed by results/locomo/postfix_pool20.json.
EXPECTED_HIT_AT_5 = 0.671
EXPECTED_HIT_AT_20 = 0.855
#: Answerable questions in LOCOMO (categories 1-4). Exact — this is the doubled-corpus check.
EXPECTED_ANSWERABLE_N = 1536
#: Tolerance for the rate asserts. NOT zero: HNSW index builds are nondeterministic (§5b, §6),
#: so demanding equality would fail honest reruns. Wide enough to absorb build noise, far too
#: tight to absorb a structural defect — a doubled corpus moved a headline rate by far more.
HIT_RATE_TOLERANCE = 0.01


def triggered(probe: LegProbe) -> bool:
    """The lexical leg was the more decisive one on this query.

    Delegates to `more_decisive`, which scores BOTH legs at their common candidate depth.
    Comparing them at their natural depths would measure how many chunks matched the tsquery
    rather than which leg was decisive: the dense leg always returns exactly `candidate_k`
    candidates, the sparse leg only its tsquery matches, and the z-score of a sample maximum
    grows with sample size on its own. See the amendment note in the design doc.
    """
    return more_decisive(probe.sparse_ranks, [h.score for h in probe.dense])


def classify_gold(probe: LegProbe, evidence: Sequence[str], k: int) -> str:
    """Where the gold chunk sits relative to what retrieval produced.

    Note `_retrieved_dia_ids` returns DISTINCT dia ids best-rank-first, so slicing the fused
    hits to `k` before mapping (rather than mapping then slicing) is what makes "inside the
    top k" mean the same thing here as it does in `_hit_by_depth`.
    """
    if not evidence:
        return "c_absent"
    gold = set(evidence)
    if gold & set(_retrieved_dia_ids(probe.fused[:k])):
        return "hit"
    pool = set(_retrieved_dia_ids(probe.dense)) | set(_retrieved_dia_ids(probe.sparse))
    return "a_misranked" if gold & pool else "b_unretrieved"


def _mean(flags: list[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _rate(flags: list[bool]) -> dict[str, Any]:
    if not flags:
        return {"rate": 0.0, "n": 0, "ci": [None, None]}
    lo, hi = wilson_ci(flags)
    return {
        "rate": sum(1 for f in flags if f) / len(flags),
        "n": len(flags),
        "ci": [round(lo, 4), round(hi, 4)],
    }


#: Sparse-leg depth bins for the Q1 confound control.
#:
#: `more_decisive` removes the FIRST-ORDER sample-size bias but not all of it. Measured on iid
#: noise: an equal-length 5-vs-5 comparison fires 50.0% of the time, but a 5-candidate sparse leg
#: against a 20-candidate dense leg fires only 35.1%, and against a 40-candidate dense leg 33.6% —
#: because truncating a larger pool to its top m yields order statistics clustered more tightly
#: near the maximum than a fresh m-sized draw. So the trigger still correlates with how many chunks
#: matched the tsquery, and n_sparse plausibly correlates with question difficulty too.
#:
#: Q1 is therefore reported WITHIN these bins as well as overall. If the firing/not-firing gap
#: exists only across bins and vanishes inside them, that is the confound talking, not the trigger.
SPARSE_DEPTH_BINS: tuple[tuple[int, int], ...] = ((0, 4), (5, 9), (10, 19), (20, 1_000_000_000))


def _depth_bin(n: int) -> str:
    for lo, hi in SPARSE_DEPTH_BINS:
        if lo <= n <= hi:
            return f"n_sparse_{lo}+" if hi == 1_000_000_000 else f"n_sparse_{lo}-{hi}"
    return "n_sparse_other"


def _split_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """hit-rate for the firing and non-firing halves of `rows`, plus their difference."""
    f = [r["hit"] for r in rows if r["trigger"]]
    nf = [r["hit"] for r in rows if not r["trigger"]]
    fr, nfr = _rate(f), _rate(nf)
    return {
        "firing": fr,
        "not_firing": nfr,
        "delta": (fr["rate"] - nfr["rate"]) if (f and nf) else None,
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Q1/Q2/Q3 from per-question records. Pure — every figure traces to `records`."""
    scored = [r for r in records if r["bucket"] != "c_absent"]
    firing = [r for r in scored if r["trigger"]]
    not_firing = [r for r in scored if not r["trigger"]]

    q1_firing = _rate([r["hit"] for r in firing])
    q1_not = _rate([r["hit"] for r in not_firing])
    delta = (q1_firing["rate"] - q1_not["rate"]) if (firing and not_firing) else None

    buckets: dict[str, int] = {}
    for r in scored:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1

    by_category: dict[int, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in scored}):
        by_category[cat] = _rate([r["trigger"] for r in scored if r["category"] == cat])

    return {
        "n_scored": len(scored),
        "n_excluded_unlabelled": len(records) - len(scored),
        "q1_hit_at_k": {"firing": q1_firing, "not_firing": q1_not, "delta": delta},
        # The confound control. Read this BEFORE q1_hit_at_k: a Q1 effect that survives only in
        # the pooled number and disappears inside every depth bin is sparse-leg depth talking.
        "q1_stratified_by_sparse_depth": {
            label: {"n": len(rows), **_split_rates(rows)}
            for label in sorted({_depth_bin(r["n_sparse"]) for r in scored})
            if (rows := [r for r in scored if _depth_bin(r["n_sparse"]) == label])
        },
        "q2_firing_rate": _rate([r["trigger"] for r in scored]),
        "q2_firing_rate_by_category": by_category,
        "q3_buckets": buckets,
        "q3_buckets_firing_misses": {
            b: sum(1 for r in firing if r["bucket"] == b)
            for b in ("a_misranked", "b_unretrieved")
        },
    }


def check_apparatus(hit_at_5: float, hit_at_20: float, answerable_n: int) -> None:
    """Fail the run if the instrumented pipeline is not the measured one.

    A corrupted apparatus does not raise — it returns plausible numbers and a manufactured
    finding. Exit code 0 is not a measurement.
    """
    if answerable_n != EXPECTED_ANSWERABLE_N:
        raise RuntimeError(
            f"apparatus: scored {answerable_n} answerable questions, expected "
            f"{EXPECTED_ANSWERABLE_N}. The corpus or the label set is not the one §9a measured."
        )
    for name, got, want in (
        ("hit@5", hit_at_5, EXPECTED_HIT_AT_5),
        ("hit@20", hit_at_20, EXPECTED_HIT_AT_20),
    ):
        if abs(got - want) > HIT_RATE_TOLERANCE:
            raise RuntimeError(
                f"apparatus: {name} reads {got:.4f}, §9a published {want} "
                f"(tolerance {HIT_RATE_TOLERANCE}). Instrumentation changed the retrieved set; "
                f"the diagnostic below would be measuring something else."
            )
