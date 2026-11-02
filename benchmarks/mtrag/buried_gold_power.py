"""Family B power precondition: can the 123-document cell detect anything?

Prior work: uses McNemar's exact paired-binomial test; this harness simulates its power for
the existing Family B design under the declared copy-mixture model.

This BLOCKS freezing Family B, and the standard it serves is specific:
`feedback-check-the-deciding-cell-has-power-2026-08-06` records a prior session that built three
guards which could not fire, could not pass, or rested on n=8. A design whose deciding cell cannot
resolve the effect it is looking for produces a null that means nothing.

The design is PAIRED: the same 123 gold documents are ranked by the control (MiniLM buries 90) and
by the treatment, so the relevant test is McNemar's on the discordant pairs. `rho` is the
tetrachoric-style association between the two rankers' bury decisions. It is NOT a free parameter
to tune until the answer is pleasant: it is estimated from the MiniLM/BGE agreement in the
2026-08-07 archive (90 and 91 of the same 123), and reported alongside the result.

Power is simulated rather than derived from a closed form, because the closed forms for McNemar
disagree at small discordant counts and the exact binomial test is what will actually be run.
"""

from __future__ import annotations

import random
from math import comb


def _binom_two_sided_p(b: int, n_discordant: int) -> float:
    """Exact two-sided binomial p at p=0.5, which is McNemar's exact test."""
    if n_discordant == 0:
        return 1.0
    k = min(b, n_discordant - b)
    tail: float = sum(comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
    return min(1.0, 2.0 * tail)


def mcnemar_power(
    n: int,
    p_control: float,
    p_treatment: float,
    rho: float,
    alpha: float = 0.05,
    trials: int = 20000,
    seed: int = 0,
) -> float:
    """Simulated power of McNemar's exact test on `n` paired binary outcomes.

    `p_control` is the control bury rate; `p_treatment` is the independent branch's bury
    probability. `rho` is the probability that treatment copies control, which models two rankers
    agreeing on easy cases; the remainder is drawn independently. The achieved treatment bury
    rate is `rho * p_control + (1 - rho) * p_treatment`, not the branch probability. This
    deliberately simple association model is unchanged by reporting its achieved rate.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    rng = random.Random(seed)
    rejections = 0
    for _ in range(trials):
        b = c = 0  # b: control buried, treatment did not. c: the reverse.
        for _ in range(n):
            control = rng.random() < p_control
            if rng.random() < rho:
                treatment = control
            else:
                treatment = rng.random() < p_treatment
            if control and not treatment:
                b += 1
            elif treatment and not control:
                c += 1
        if _binom_two_sided_p(b, b + c) < alpha:
            rejections += 1
    return rejections / trials


def minimum_detectable_shift(
    n: int,
    p_control: float,
    rho: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> float | None:
    """The largest treatment bury rate still detectable at `target_power`, or None if none is.

    Scans the independent branch probability downward from `p_control` in steps of 0.01,
    then reports its mixture marginal. `None` means no shift in range is detectable, so the
    spec's demotion rule makes Family B descriptive with no p-value.
    """
    rate = p_control
    while rate >= 0.0:
        if mcnemar_power(n, p_control, rate, rho, alpha, trials=4000, seed=7) >= target_power:
            return round(rho * p_control + (1 - rho) * rate, 4)
        rate = round(rate - 0.01, 4)
    return None
