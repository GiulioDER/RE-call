"""Per-query, per-leg decisiveness — the quantity the Phase 0 diagnostic measures.

A retrieval leg either found a clear winner or handed back a flat, undifferentiated list.
`leg_confidence` reports which, as the z-score of the leg's top candidate within that leg's
OWN candidate scores.

Why a z-score and not a normalized max: the dense leg scores in cosine (bounded, ~[0, 1]) and
the sparse leg scores in `ts_rank` (unbounded, corpus-dependent). Any statistic that survives
being compared across those two must be invariant to an affine change of units, and a z-score
is. That invariance is asserted in `tests/test_leg_confidence.py`, not assumed here.

This lives under `recall/eval/` deliberately. Nothing in the serving path consumes it yet —
Phase 1 would, if the diagnostic clears its gates. Shipping it into `recall/` before then would
be dead code in the installed package.
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


def more_decisive(sparse_scores: Sequence[float], dense_scores: Sequence[float]) -> bool:
    """True when the sparse leg was more decisive than the dense leg on this query.

    Both legs are scored over their top `min(len(sparse), len(dense))` candidates. That
    truncation is the point: `leg_confidence` is the z-score of a sample maximum, which grows
    with sample size on its own (E[conf] on iid noise runs 1.39/1.67/1.95 at n=5/10/20). The
    dense leg always returns exactly `candidate_k` candidates while the sparse leg returns only
    tsquery matches, so comparing them at their natural depths would measure how many chunks
    matched the query text rather than which leg was more decisive.

    Returns False when the common depth is below 2 — there is no spread to compare, and the
    caller treats "no information" as "do not fire".
    """
    m = min(len(sparse_scores), len(dense_scores))
    if m < 2:
        return False
    sparse_top = sorted(sparse_scores, reverse=True)[:m]
    dense_top = sorted(dense_scores, reverse=True)[:m]
    return leg_confidence(sparse_top) > leg_confidence(dense_top)
