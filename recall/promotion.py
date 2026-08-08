"""Conservative, reproducible promotion gates for retrieval and generation experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import isfinite
from statistics import mean


@dataclass(frozen=True)
class QuestionOutcome:
    question_id: str
    corpus: str
    baseline_hit5: float
    candidate_hit5: float
    baseline_mrr: float
    candidate_mrr: float


@dataclass(frozen=True)
class SafetyMetrics:
    false_confidence: float
    false_abstention: float
    #: ``None`` means NOT MEASURED, and like a PENDING latency it is a FAILURE rather than an
    #: exemption. An arm scored under a degraded trust policy has every verdict overwritten with
    #: ``unverified`` (`recall/trust.py`), which destroys the distinction this rate is made of: a
    #: superseded hit and a clean one are then the same observation. Encoding that as ``0.0``
    #: would SATISFY the zero-tolerance check below by never having measured it, which is the
    #: failure mode this whole gate exists to refuse.
    superseded_trust_rate: float | None


@dataclass(frozen=True)
class RetrievalGateInput:
    outcomes: tuple[QuestionOutcome, ...]
    baseline_safety: SafetyMetrics
    candidate_safety: SafetyMetrics
    security_green: bool
    #: ``None`` means PENDING: the p95 has not been measured on a host that can carry the claim.
    #: PENDING is a FAILURE, not an exemption. The alternative encodings were both worse — a
    #: default of 0.0 makes an unmeasured latency the fastest possible one, and omitting the check
    #: makes a missing measurement indistinguishable from a passing one. This program has no idle
    #: reference host (see docs/ENTERPRISE_PROGRAM_STATUS.md's standing blockers), so PENDING is
    #: the state every real decision is in today, and it must block rather than pass silently.
    latency_p95_ms: float | None
    latency_budget_ms: float


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    macro_hit5_delta: float
    bootstrap_interval: tuple[float, float]
    corpus_hit5_delta: dict[str, float]
    holm_p_values: dict[str, float]
    failures: tuple[str, ...]


def _groups(outcomes: tuple[QuestionOutcome, ...]) -> dict[str, list[QuestionOutcome]]:
    grouped: dict[str, list[QuestionOutcome]] = {}
    ids: set[str] = set()
    for outcome in outcomes:
        identity = f"{outcome.corpus}\x00{outcome.question_id}"
        if identity in ids:
            raise ValueError(f"duplicate question outcome: {outcome.corpus}/{outcome.question_id}")
        ids.add(identity)
        grouped.setdefault(outcome.corpus, []).append(outcome)
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("at least one outcome per corpus is required")
    return grouped


def stratified_bootstrap_hit5(
    outcomes: tuple[QuestionOutcome, ...], *, samples: int = 10_000, seed: int = 20260803
) -> tuple[float, tuple[float, float]]:
    """Macro paired delta and percentile interval, resampled within each corpus."""
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    grouped = _groups(outcomes)
    corpus_deltas = [
        mean(item.candidate_hit5 - item.baseline_hit5 for item in values)
        for values in grouped.values()
    ]
    point = mean(corpus_deltas)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        deltas: list[float] = []
        for values in grouped.values():
            resampled = [rng.choice(values) for _ in values]
            deltas.append(
                mean(item.candidate_hit5 - item.baseline_hit5 for item in resampled)
            )
        draws.append(mean(deltas))
    draws.sort()
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]
    return point, (low, high)


def _paired_sign_p(differences: list[float], seed: int) -> float:
    nonzero = [value for value in differences if value]
    if not nonzero:
        return 1.0
    observed = mean(nonzero)
    if observed <= 0:
        return 1.0
    exact = len(nonzero) <= 18
    permutations = (1 << len(nonzero)) if exact else 20_000
    rng = random.Random(seed)
    at_least = 0
    for index in range(permutations):
        if exact:
            simulated = mean(
                value if index & (1 << position) else -value
                for position, value in enumerate(nonzero)
            )
        else:
            simulated = mean(value if rng.random() < 0.5 else -value for value in nonzero)
        if simulated >= observed:
            at_least += 1
    return (at_least + (0 if exact else 1)) / (permutations + (0 if exact else 1))


def holm_correct(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    corrected: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        corrected[name] = running
    return corrected


def evaluate_retrieval_promotion(
    gate: RetrievalGateInput, *, bootstrap_samples: int = 10_000
) -> PromotionDecision:
    grouped = _groups(gate.outcomes)
    point, interval = stratified_bootstrap_hit5(
        gate.outcomes, samples=bootstrap_samples
    )
    corpus_delta = {
        corpus: mean(item.candidate_hit5 - item.baseline_hit5 for item in values)
        for corpus, values in grouped.items()
    }
    raw_p = {
        corpus: _paired_sign_p(
            [item.candidate_hit5 - item.baseline_hit5 for item in values],
            seed=20260803 + index,
        )
        for index, (corpus, values) in enumerate(sorted(grouped.items()))
    }
    corrected = holm_correct(raw_p)
    failures: list[str] = []
    if interval[0] <= 0:
        failures.append("macro paired hit@5 bootstrap interval does not clear zero")
    for corpus, delta in corpus_delta.items():
        if delta < -0.02:
            failures.append(f"{corpus} hit@5 regresses by more than two percentage points")
    if not any(value < 0.05 and corpus_delta[name] > 0 for name, value in corrected.items()):
        failures.append("no improving corpus has Holm corrected paired significance below 0.05")
    if gate.candidate_safety.false_confidence - gate.baseline_safety.false_confidence > 0.02:
        failures.append("false confidence regresses by more than two percentage points")
    if gate.candidate_safety.false_abstention - gate.baseline_safety.false_abstention > 0.02:
        failures.append("false abstention regresses by more than two percentage points")
    if gate.candidate_safety.superseded_trust_rate is None:
        failures.append(
            "superseded trust rate was NOT MEASURED — the candidate arm ran under a degraded "
            "trust policy, so every verdict is 'unverified' and a stale hit is indistinguishable "
            "from a clean one"
        )
    elif gate.candidate_safety.superseded_trust_rate != 0.0:
        failures.append("superseded trust rate is not zero")
    if not gate.security_green:
        failures.append("security verification is not green")
    if gate.latency_p95_ms is None:
        failures.append(
            "retrieval p95 latency is PENDING — no measurement from a reference host, so the "
            "latency budget could not be evaluated"
        )
    elif not (isfinite(gate.latency_p95_ms) and gate.latency_p95_ms > 0.0):
        # NaN is the one that matters, and it is reachable without malice: a percentile over an
        # empty sample is NaN, and `nan > budget` is False, so an unmeasured latency would pass
        # the budget check while being reported as MEASURED. That is the same shape as the None
        # case above and must fail the same way, not slip through as the fastest possible p95.
        failures.append(
            f"retrieval p95 latency {gate.latency_p95_ms!r} is not a positive finite "
            f"measurement, so the latency budget could not be evaluated"
        )
    elif gate.latency_p95_ms > gate.latency_budget_ms:
        failures.append("retrieval p95 exceeds its profile budget")
    return PromotionDecision(
        promoted=not failures,
        macro_hit5_delta=point,
        bootstrap_interval=interval,
        corpus_hit5_delta=corpus_delta,
        holm_p_values=corrected,
        failures=tuple(failures),
    )
