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
    """
    n = len(scores)
    if n < 2:
        return 0.0
    mu = sum(scores) / n
    sd = math.sqrt(sum((s - mu) ** 2 for s in scores) / n)
    if sd == 0.0:
        return 0.0
    return (max(scores) - mu) / sd
