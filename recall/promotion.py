"""Conservative, reproducible promotion gates for retrieval and generation experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from recall.reasoning_proposals import InferenceProposal

AcceptedProposalState = Literal["proposed", "reviewed", "accepted", "rejected", "promoted"]


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
    #: reference host (see docs/archive/ENTERPRISE_PROGRAM_STATUS.md's standing blockers), so PENDING is
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


@dataclass(frozen=True)
class ReviewedProposal:
    proposal: InferenceProposal
    state: AcceptedProposalState
    reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    audit_note: str | None = None


@dataclass(frozen=True)
class PromotedFact:
    """Reviewed proposal stored separately from raw inference proposals."""

    fact_id: str
    proposal_id: str
    relation: str
    subject_id: str
    object_id: str
    reviewer_id: str
    source_generation_id: str
    source_provider_id: str
    source_model_id: str
    source_model_revision: str
    proposal_evidence_ids: tuple[str, ...]
    promoted_at: datetime
    audit_note: str
    state: AcceptedProposalState = "promoted"


def review_proposal(
    proposal: InferenceProposal,
    *,
    reviewer_id: str,
    reviewed_at: datetime | None,
    audit_note: str,
) -> ReviewedProposal:
    """Move a raw proposal into review after recording reviewer, time, and audit evidence."""

    _require_review_fields(reviewer_id, reviewed_at, audit_note)
    return ReviewedProposal(
        proposal=proposal,
        state="reviewed",
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        audit_note=audit_note,
    )


def accept_reviewed_proposal(review: ReviewedProposal) -> ReviewedProposal:
    """Accept a reviewed proposal without creating trusted metadata yet."""

    if review.state != "reviewed":
        raise ValueError("proposal must be reviewed before it can be accepted")
    _require_review_fields(review.reviewer_id, review.reviewed_at, review.audit_note)
    return ReviewedProposal(
        proposal=review.proposal,
        state="accepted",
        reviewer_id=review.reviewer_id,
        reviewed_at=review.reviewed_at,
        audit_note=review.audit_note,
    )


def reject_reviewed_proposal(review: ReviewedProposal) -> ReviewedProposal:
    """Reject a reviewed proposal while preserving its review identity and audit note."""

    if review.state != "reviewed":
        raise ValueError("proposal must be reviewed before it can be rejected")
    _require_review_fields(review.reviewer_id, review.reviewed_at, review.audit_note)
    return ReviewedProposal(
        proposal=review.proposal,
        state="rejected",
        reviewer_id=review.reviewer_id,
        reviewed_at=review.reviewed_at,
        audit_note=review.audit_note,
    )


def promote_accepted_proposal(
    review: ReviewedProposal,
    *,
    promoted_at: datetime,
    audit_note: str | None = None,
) -> PromotedFact:
    """Create a trusted promoted fact from an accepted proposal.

    Promotion requires reviewer identity, review timestamp, audit note, source evidence ids, and
    source provider generation identity. The raw proposal remains separate from the promoted fact.
    """

    if review.state != "accepted":
        raise ValueError("proposal must be accepted before promotion")
    _require_review_fields(review.reviewer_id, review.reviewed_at, review.audit_note)
    note = audit_note or review.audit_note
    if note is None or not note.strip():
        raise ValueError("promotion audit note is required")
    proposal = review.proposal
    if not proposal.source_evidence_ids:
        raise ValueError("promotion requires proposal evidence ids")
    if not proposal.provider_id or not proposal.model_id or not proposal.provider_revision:
        raise ValueError("promotion requires source generation identity")
    return PromotedFact(
        fact_id=f"promoted:{proposal.id}",
        proposal_id=proposal.id,
        relation=proposal.proposed_relation,
        subject_id=proposal.subject_id,
        object_id=proposal.object_id,
        reviewer_id=str(review.reviewer_id),
        source_generation_id=proposal.generation_id,
        source_provider_id=proposal.provider_id,
        source_model_id=proposal.model_id,
        source_model_revision=proposal.provider_revision,
        proposal_evidence_ids=proposal.source_evidence_ids,
        promoted_at=promoted_at,
        audit_note=note,
    )


def reviewed_promotion_is_trusted_metadata(value: object) -> bool:
    """Trust retrieval metadata only when it is a fully reviewed promoted fact."""

    return (
        isinstance(value, PromotedFact)
        and value.state == "promoted"
        and bool(value.reviewer_id)
        and bool(value.audit_note.strip())
        and bool(value.proposal_evidence_ids)
        and bool(value.source_generation_id)
        and bool(value.source_provider_id)
        and bool(value.source_model_id)
        and bool(value.source_model_revision)
    )


def _require_review_fields(
    reviewer_id: str | None,
    reviewed_at: datetime | None,
    audit_note: str | None,
) -> None:
    if reviewer_id is None or not reviewer_id.strip():
        raise ValueError("reviewer identity is required")
    if reviewed_at is None:
        raise ValueError("review timestamp is required")
    if audit_note is None or not audit_note.strip():
        raise ValueError("audit note is required")


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
