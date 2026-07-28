"""Rank fusion, and the per-query leg decisiveness that weights it.

`leg_confidence` reports whether a retrieval leg found a clear winner or handed back a flat,
undifferentiated list: the z-score of the leg's top candidate within that leg's OWN candidates.
It is **affine-invariant**, which is what lets a cosine leg and a `ts_rank` leg be compared
without normalizing incompatible scales. That invariance is asserted as a property test.

Provenance, stated precisely because it is easy to misread: this function was built for the
Phase 0 diagnostic, and Phase 0's *trigger* — `more_decisive`, "is sparse more decisive than
dense" used to PREDICT retrieval failure — was falsified
(`results/legdiag/FINDINGS_phase0.md`). What was falsified is that use. `leg_confidence` itself
measures what it claims to, and Phase 1 uses it for a different job: deciding **which leg's
ranking should dominate this query's prefix**, not predicting whether the query will fail.
Phase 0 is in fact evidence for this use — the leg it identified as more decisive was the one
whose queries scored HIGHER (hit@5 0.708 vs 0.616).
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def leg_confidence(scores: Sequence[float]) -> float:
    """z-score of the top candidate within `scores`. 0.0 when there is no spread to measure.

    Returns 0.0 for an empty leg, a single candidate, or a perfectly flat leg — all three mean
    "this leg expressed no preference". Never negative: the maximum of a sample is always at
    least its mean.

    Not comparable across inputs of different length — the expected value of a sample maximum
    grows with sample size on its own, even under pure noise. Use `more_decisive` to compare two
    legs.
    """
    n = len(scores)
    if n < 2:
        return 0.0
    mu = sum(scores) / n
    sd = math.sqrt(sum((s - mu) ** 2 for s in scores) / n)
    if sd == 0.0:
        return 0.0
    return (max(scores) - mu) / sd


def weighted_rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Fuse best-first id rankings into one score map, weighting each ranking.

    Each id accrues ``w_L / (k + rank)`` from every ranking it appears in. `k` (default 60, the
    standard RRF damping constant — unrelated to the caller's result-count `k`) softens the top
    ranks so no single ranking dominates outright.

    `weights` defaults to uniform, which reproduces the ORDER of the unweighted formula exactly:
    a common factor scales every score and cannot reorder them. The returned dict is UNSORTED;
    callers sort by value descending.
    """
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings) if rankings else []
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(rankings)} rankings but {len(weights)} weights")
    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
    return scores
