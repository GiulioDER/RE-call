"""Paired statistics for the A/B, with the degenerate cases handled rather than hidden.

The design is paired by construction: the same task runs in both arms, so every comparison here
works on **within-pair differences** and never on two independent samples. That is worth insisting
on, because an unpaired test on this data throws away the pairing that the whole harness exists to
create, and reports a wider interval than the experiment earned.

Three tests, one per kind of endpoint:

| Endpoint | Test | Why this one |
|---|---|---|
| trap hit, success (binary) | **exact McNemar** | uses only discordant pairs, which is the whole information a paired binary comparison contains |
| tokens, wall time (continuous) | **Wilcoxon signed-rank** | no normality assumption, and these distributions are skewed by construction |
| any mean difference | **paired bootstrap** | an interval on the effect size, which is what a reader actually wants |

⚠️ **Degenerate samples are the normal case here, not the exception.** With 5 repetitions of 4
tasks, an arm that never triggers a trap gives 0/20, and `recall/eval/metrics.py:bootstrap_ci`
already records what a percentile bootstrap does with that: it resamples all-False every time and
returns `[0.00, 0.00]`, reporting CERTAINTY from twenty observations that happened to agree. Every
function here returns `None` rather than a number it cannot support, and says why in `note`.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Sequence

#: Below this many usable pairs, no test is reported. Not a convention: with fewer than six
#: discordant pairs the exact McNemar test cannot reach p < 0.05 at any split, so a "not
#: significant" result would be an artifact of the sample size rather than a finding about RE-call.
MIN_PAIRS = 6


@dataclass(frozen=True)
class PairedResult:
    """One endpoint's paired comparison. `None` means 'not supportable', never 'zero'."""

    metric: str
    n_pairs: int
    on_mean: float | None
    off_mean: float | None
    delta_mean: float | None
    #: The MEDIAN within-pair difference, reported beside the mean and never instead of it.
    #:
    #: These distributions are skewed by construction, and the two statistics can disagree about
    #: the sign. Measured on `agent-ab-additive-002` wall time: mean -33,035 ms but median
    #: +1,703 ms, because the on arm was slower in 29 of 48 pairs while a handful of baseline
    #: sessions explored for 400 seconds and dragged the mean down. Reporting the mean alone
    #: would have claimed "33 seconds faster" for a configuration that is typically slightly
    #: SLOWER and occasionally saves an enormous amount. The rank-based p-value tracks the
    #: median, so a mean-only table also looks inconsistent with its own significance test.
    delta_median: float | None
    delta_ci: tuple[float, float] | None
    p_value: float | None
    test: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delta_ci"] = list(self.delta_ci) if self.delta_ci else None
        return payload


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def paired_bootstrap(
    deltas: Sequence[float], *, n: int = 10000, confidence: float = 0.95, seed: int = 12345
) -> tuple[float, float] | None:
    """Percentile interval on the MEAN within-pair difference.

    Resamples the differences, not the two arms separately: resampling the arms independently
    would destroy the pairing and inflate the interval.

    Returns `None` when every difference is identical. Such a sample resamples to the same value
    every time and yields a zero-width interval, which reads as a precise result and is the
    opposite of one.
    """

    usable = [float(d) for d in deltas if d is not None and math.isfinite(d)]
    if len(usable) < 2 or len(set(usable)) == 1:
        return None
    rng = random.Random(seed)
    size = len(usable)
    means = sorted(
        sum(usable[rng.randrange(size)] for _ in range(size)) / size for _ in range(n)
    )
    lo_q = (1.0 - confidence) / 2.0
    lo = means[min(len(means) - 1, int(lo_q * len(means)))]
    hi = means[min(len(means) - 1, int((1.0 - lo_q) * len(means)))]
    return (lo, hi)


def mcnemar_exact(on: Sequence[bool], off: Sequence[bool]) -> tuple[float | None, int, int]:
    """Exact McNemar on paired binary outcomes; returns `(p, b, c)`.

    `b` counts pairs where the on arm is True and the off arm False, `c` the reverse. Concordant
    pairs carry no information about a difference and are excluded by the test, which is why a
    lopsided-looking table can still be inconclusive.

    Uses the exact binomial rather than the chi-square approximation, because the approximation is
    unreliable exactly where this benchmark lives: few discordant pairs.
    """

    if len(on) != len(off):
        raise ValueError("paired sequences must be the same length")
    b = sum(1 for x, y in zip(on, off) if x and not y)
    c = sum(1 for x, y in zip(on, off) if y and not x)
    if b + c == 0:
        # Every pair agreed. There is no evidence of a difference AND no evidence of sameness;
        # reporting p=1.0 would assert the second.
        return None, b, c
    from scipy.stats import binomtest

    return float(binomtest(b, b + c, 0.5).pvalue), b, c


def wilcoxon_signed_rank(deltas: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value on within-pair differences.

    Returns `None` when every difference is zero: the test is undefined there, and scipy's own
    behaviour in that case is a warning plus a nan, which is easy to serialise as a number.
    """

    usable = [float(d) for d in deltas if d is not None and math.isfinite(d)]
    nonzero = [d for d in usable if d != 0.0]
    if len(nonzero) < 1:
        return None
    from scipy.stats import wilcoxon

    return float(wilcoxon(usable, zero_method="wilcox", alternative="two-sided").pvalue)


@dataclass(frozen=True)
class TaskRate:
    """One trap's rate in each arm, across its repetitions."""

    task: str
    n_reps: int
    on_rate: float
    off_rate: float

    @property
    def delta(self) -> float:
        return self.on_rate - self.off_rate

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "delta": self.delta}


def summarize_by_task(
    pairs_by_task: dict[str, Sequence[tuple[bool, bool]]],
    *,
    n: int = 10000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, Any]:
    """The per-TASK view, which is the conservative reading of a repeated-measures design.

    Repetitions of one task are **not** independent: the same prompt, the same corpus and the same
    governing memo produce correlated outcomes, so treating 10 repetitions of 4 traps as 40
    independent pairs overstates confidence. This collapses each trap to one rate per arm, so the
    unit of evidence is the hazard rather than the session.

    ⚠️ **With few traps this view cannot reach significance at any effect size, and that is a
    property of the design rather than a result.** A sign test over 4 tasks bottoms out at p=0.125
    even when all four move the same way. So `n_improved` and the per-task deltas are reported as
    DESCRIPTIVE, and no p-value is invented for them. The cluster bootstrap resamples whole tasks,
    which is the honest interval for generalising to a new hazard, and it will be wide.
    """

    rates = [
        TaskRate(
            task=task,
            n_reps=len(pairs),
            on_rate=sum(1 for on, _ in pairs if on) / len(pairs),
            off_rate=sum(1 for _, off in pairs if off) / len(pairs),
        )
        for task, pairs in sorted(pairs_by_task.items())
        if pairs
    ]
    if not rates:
        return {"tasks": [], "n_tasks": 0, "note": "no tasks"}

    deltas = [r.delta for r in rates]
    improved = sum(1 for d in deltas if d < 0)
    worsened = sum(1 for d in deltas if d > 0)

    cluster_ci = None
    if len(rates) >= 2 and len(set(deltas)) > 1:
        rng = random.Random(seed)
        size = len(rates)
        means = sorted(
            sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(n)
        )
        lo_q = (1.0 - confidence) / 2.0
        cluster_ci = (
            means[min(len(means) - 1, int(lo_q * len(means)))],
            means[min(len(means) - 1, int((1.0 - lo_q) * len(means)))],
        )

    return {
        "tasks": [r.to_dict() for r in rates],
        "n_tasks": len(rates),
        "mean_delta": sum(deltas) / len(deltas),
        "improved": improved,
        "worsened": worsened,
        "unchanged": len(deltas) - improved - worsened,
        # Resamples TASKS, not pairs, so the interval answers "what would a new hazard do".
        "cluster_ci": list(cluster_ci) if cluster_ci else None,
        "note": (
            f"descriptive: {len(rates)} distinct traps. A sign test over this many cannot reach "
            f"p<0.05 at any effect size, so no p-value is reported for this view."
        ),
    }


def compare_binary(metric: str, pairs: Sequence[tuple[bool, bool]]) -> PairedResult:
    """Compare a paired binary endpoint, such as whether a trap was triggered."""

    on = [bool(a) for a, _ in pairs]
    off = [bool(b) for _, b in pairs]
    n = len(pairs)
    if n < MIN_PAIRS:
        return PairedResult(
            metric=metric, n_pairs=n, on_mean=_mean([float(x) for x in on]),
            off_mean=_mean([float(x) for x in off]), delta_mean=None, delta_median=None,
            delta_ci=None,
            p_value=None, test="mcnemar_exact",
            note=f"only {n} pairs; below {MIN_PAIRS} no split of the discordant pairs can reach "
                 f"p < 0.05, so a p-value here would describe the sample size, not the effect",
        )
    p, b, c = mcnemar_exact(on, off)
    deltas = [float(x) - float(y) for x, y in zip(on, off)]
    note = f"discordant pairs: on-only={b}, off-only={c}"
    if p is None:
        note += "; every pair agreed, so the test is undefined"
    return PairedResult(
        metric=metric, n_pairs=n, on_mean=_mean([float(x) for x in on]),
        off_mean=_mean([float(x) for x in off]), delta_mean=_mean(deltas),
        delta_median=statistics.median(deltas) if deltas else None,
        delta_ci=paired_bootstrap(deltas), p_value=p, test="mcnemar_exact", note=note,
    )


def compare_continuous(metric: str, pairs: Sequence[tuple[float, float]]) -> PairedResult:
    """Compare a paired continuous endpoint, such as input tokens or wall time."""

    usable = [
        (float(a), float(b))
        for a, b in pairs
        # A missing measurement stays missing. Substituting zero would report an unmeasured
        # session as a free one, which is the schema's standing rule and matters most here,
        # where the whole endpoint is a cost.
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    n = len(usable)
    dropped = len(pairs) - n
    note = f"{dropped} pair(s) dropped for a missing measurement" if dropped else ""
    if n < MIN_PAIRS:
        return PairedResult(
            metric=metric, n_pairs=n, on_mean=_mean([a for a, _ in usable]),
            off_mean=_mean([b for _, b in usable]), delta_mean=None, delta_median=None,
            delta_ci=None,
            p_value=None, test="wilcoxon_signed_rank",
            note=(note + "; " if note else "") + f"only {n} usable pairs, below {MIN_PAIRS}",
        )
    deltas = [a - b for a, b in usable]
    return PairedResult(
        metric=metric, n_pairs=n, on_mean=_mean([a for a, _ in usable]),
        off_mean=_mean([b for _, b in usable]), delta_mean=_mean(deltas),
        delta_median=statistics.median(deltas) if deltas else None,
        delta_ci=paired_bootstrap(deltas), p_value=wilcoxon_signed_rank(deltas),
        test="wilcoxon_signed_rank", note=note,
    )
