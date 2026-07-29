"""The 2x2 per rung, the lambda-priced curve, and the H1 gate.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Stdlib only, matching `recall/eval/metrics.py` and `recall/eval/gap_study.py`: the arithmetic that
decides whether this benchmark lives should be readable without trusting a library.

`answered_answerable` is NOT `correct_answer`. v1 has no judge, so answering an answerable question
is scored as success without the content being verified. Every accuracy computed from this module
is an UPPER BOUND, and the field name is where that is enforced rather than remembered.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmarks.ladder.manifest import LABEL_ANSWERABLE, LABEL_UNANSWERABLE, Instance


@dataclass(frozen=True)
class Cell:
    correct_abstain: int
    false_answer: int
    false_abstain: int
    answered_answerable: int


def confusion_by_ring(
    instances: Sequence[Instance], abstained: Mapping[str, bool]
) -> dict[int, Cell]:
    """Ring -> 2x2.

    Instances with no recorded response are SKIPPED, not counted as abstentions: a missing row is
    missing data, and scoring it as an abstention would flatter a crashed run into looking
    well-calibrated.
    """
    counts: dict[int, list[int]] = {}
    for inst in instances:
        if inst.instance_id not in abstained:
            continue
        cell = counts.setdefault(inst.ring, [0, 0, 0, 0])
        did_abstain = abstained[inst.instance_id]
        if inst.label == LABEL_UNANSWERABLE:
            cell[0 if did_abstain else 1] += 1
        elif inst.label == LABEL_ANSWERABLE:
            cell[2 if did_abstain else 3] += 1
    return {ring: Cell(*vals) for ring, vals in counts.items()}


def correct_abstain_rate(cell: Cell) -> float:
    """Over unanswerable instances only. Returns 0.0 when there are none."""
    n = cell.correct_abstain + cell.false_answer
    return cell.correct_abstain / n if n else 0.0


def lambda_cost(cell: Cell, lam: float) -> float:
    """Expected cost with cost(false answer) = lam x cost(false abstention).

    lam=1 reproduces BEAM's implicit weighting. Fixed in PREREGISTRATION-ladder.md before results.
    """
    return lam * cell.false_answer + cell.false_abstain


def _paired_flags(
    instances: Sequence[Instance], abstained: Mapping[str, bool], low: int, high: int
) -> tuple[list[int], list[int]]:
    """Per pair_id, the abstention flag at each of two rungs — pairs present at BOTH only."""
    at: dict[int, dict[str, int]] = {low: {}, high: {}}
    for inst in instances:
        if inst.ring in at and inst.instance_id in abstained:
            at[inst.ring][inst.pair_id] = int(abstained[inst.instance_id])
    shared = sorted(set(at[low]) & set(at[high]))
    return [at[low][p] for p in shared], [at[high][p] for p in shared]


def paired_difference_ci(
    instances: Sequence[Instance],
    abstained: Mapping[str, bool],
    low: int,
    high: int,
    *,
    seed: int = 0,
    iterations: int = 10_000,
) -> tuple[float, float, float]:
    """Mean paired difference in correct-abstain (high rung minus low), with a bootstrap 95 % CI.

    Paired because the same question appears at both rungs; an unpaired test would attribute
    question difficulty to the rung.
    """
    lo_flags, hi_flags = _paired_flags(instances, abstained, low, high)
    n = len(lo_flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    deltas = [h - lo for lo, h in zip(lo_flags, hi_flags)]
    observed = sum(deltas) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(iterations))
    return observed, means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def h1_verdict(
    difference: float, ci_low: float, ci_high: float, *, threshold: float = 0.15
) -> str:
    """P1 from the pre-registration: difference > 0.15 AND the bootstrap CI excludes zero.

    FAIL is the kill condition, not a disappointing result: a flat curve means the axis is a
    fiction, and no paid comparative arm runs.
    """
    excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    return "PASS" if difference > threshold and excludes_zero else "FAIL"
