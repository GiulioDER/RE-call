"""Does a corpus statistic predict the cloud-embedder gap better than the local model's own score?

The response is `gap = hit@5(cloud) - hit@5(local)`, and it is mechanically anti-correlated with
`hit@5(local)` through the ceiling alone: a corpus where the local embedder already scores 0.9
cannot show a +0.28 gap no matter how idiosyncratic its vocabulary. So a raw correlation between
any predictor and the gap is not evidence — it is what the arithmetic guarantees.

Everything here exists to answer the one question that is not guaranteed: **does a predictor
explain variance beyond the local score?** That is a partial correlation, and with a sample of
~20 corpora it needs a permutation test rather than a parametric p-value, and Holm correction
because there is more than one candidate.

Dependency-free by design (stdlib only), matching `recall.eval.metrics`, so the analysis runs in
the offline test suite and its arithmetic can be read without trusting a library.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

Number = float | int


def _ranks(values: Sequence[Number]) -> list[float]:
    """Ranks of `values`, ties sharing their average rank.

    Average ranks rather than first-seen order: several corpora can share a hit@5 to two decimals,
    and breaking those ties arbitrarily would invent an ordering the measurement does not support.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0.0:
        # A constant column has no variation to correlate. NaN is the honest answer; 0.0 would
        # read as "measured, and there is no relation".
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x: Sequence[Number], y: Sequence[Number]) -> float:
    """Rank correlation between `x` and `y`.

    Rank rather than linear, and not as a robustness afterthought: no candidate predictor has any
    reason to be *linearly* related to the gap. `oov_rate` plausibly saturates, `crowding` is
    bounded by construction. Pearson would score a perfectly monotone but curved relation as weak
    and quietly discard a real finding.
    """
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def partial_spearman(
    x: Sequence[Number], y: Sequence[Number], control: Sequence[Number]
) -> float:
    """Rank correlation between `x` and `y` with `control` held fixed.

    This is the study's actual statistic. `x` is a candidate predictor, `y` is the gap, and
    `control` is the local model's own score — the null model. A predictor that correlates with
    the gap only because both track the local score collapses to ~0 here, which is the entire
    point of computing it.
    """
    r_xy = spearman(x, y)
    r_xz = spearman(x, control)
    r_yz = spearman(y, control)
    if any(math.isnan(r) for r in (r_xy, r_xz, r_yz)):
        return float("nan")
    den = math.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if den == 0.0:
        # The control explains one of the variables perfectly, so there is no residual variation
        # left in which a partial relation could exist.
        return float("nan")
    return (r_xy - r_xz * r_yz) / den


def permutation_p(
    x: Sequence[Number],
    y: Sequence[Number],
    control: Sequence[Number],
    *,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for `partial_spearman(x, y, control)`.

    Permutation rather than the parametric t on a partial correlation, because at ~20 corpora the
    parametric test leans on a normality assumption that a bounded, possibly saturating statistic
    computed over a hand-picked corpus set has no right to.

    `x` is permuted while `y` and `control` stay paired, which breaks any relation between the
    predictor and the gap while preserving the gap's own relationship to the local score — that is
    the null being tested: *the predictor adds nothing to what the local score already says*.

    Uses the (r+1)/(n+1) estimator, which cannot return 0. A reported p of exactly zero would
    claim more resolution than `n_perm` draws can support.
    """
    observed = partial_spearman(x, y, control)
    if math.isnan(observed):
        return float("nan")
    rng = random.Random(seed)
    shuffled = list(x)
    at_least_as_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        r = partial_spearman(shuffled, y, control)
        if not math.isnan(r) and abs(r) >= abs(observed):
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_perm + 1)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the INPUT order.

    Step-down rather than plain Bonferroni: the smallest p is multiplied by m, the next by m-1,
    and so on, which is uniformly less conservative while controlling the family-wise error rate
    just as strictly. With three predictors against ~20 corpora that difference decides whether
    anything is reportable at all.

    The running maximum is carried forward so a later test can never be adjusted below an earlier
    one — without it, correction could invert the ordering and present the weakest result as the
    strongest.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvalues[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted
