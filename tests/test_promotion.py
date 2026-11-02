from datetime import datetime, timezone
from dataclasses import replace

import pytest

from recall.promotion import (
    QuestionOutcome,
    RetrievalGateInput,
    SafetyMetrics,
    accept_reviewed_proposal,
    evaluate_retrieval_promotion,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_proposals import InferenceProposal
from recall.trust import metadata_is_trusted


def test_promotion_gate_rejects_a_corpus_regression() -> None:
    outcomes = tuple(
        [QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(20)]
        + [QuestionOutcome(str(i), "b", 1.0, 0.0, 1.0, 0.0) for i in range(20)]
    )
    safe = SafetyMetrics(0.1, 0.1, 0.0)
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, safe, safe, True, 100, 250),
        bootstrap_samples=500,
    )
    assert not decision.promoted
    assert any("b hit@5 regresses" in failure for failure in decision.failures)


def _proposal() -> InferenceProposal:
    return InferenceProposal(
        id="p1",
        source_evidence_ids=("node-a", "node-b"),
        proposed_relation="supersedes",
        subject_id="old.md",
        object_id="new.md",
        explanation="review candidate",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="recall.deterministic",
        provider_revision="rev-a",
        confidence=0.8,
        uncertainty=(),
        generation_id="gen-a",
    )


def test_reviewed_proposal_state_machine_promotes_into_separate_fact() -> None:
    when = datetime(2026, 8, 10, tzinfo=timezone.utc)

    reviewed = review_proposal(
        _proposal(),
        reviewer_id="reviewer@example.com",
        reviewed_at=when,
        audit_note="Evidence ids match the corpus change.",
    )
    accepted = accept_reviewed_proposal(reviewed)
    promoted = promote_accepted_proposal(accepted, promoted_at=when)

    assert reviewed.state == "reviewed"
    assert accepted.state == "accepted"
    assert promoted.state == "promoted"
    assert promoted.proposal_id == "p1"
    assert promoted.proposal_evidence_ids == ("node-a", "node-b")
    assert metadata_is_trusted(promoted) is True
    assert metadata_is_trusted(reviewed.proposal) is False


def test_promotion_requires_review_identity_evidence_timestamp_and_audit_note() -> None:
    when = datetime(2026, 8, 10, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="reviewer"):
        review_proposal(_proposal(), reviewer_id="", reviewed_at=when, audit_note="checked")
    with pytest.raises(ValueError, match="timestamp"):
        review_proposal(_proposal(), reviewer_id="reviewer", reviewed_at=None, audit_note="checked")
    with pytest.raises(ValueError, match="audit note"):
        review_proposal(_proposal(), reviewer_id="reviewer", reviewed_at=when, audit_note="")

    reviewed = review_proposal(
        replace(_proposal(), source_evidence_ids=()),
        reviewer_id="reviewer",
        reviewed_at=when,
        audit_note="checked",
    )
    with pytest.raises(ValueError, match="accepted"):
        promote_accepted_proposal(reviewed, promoted_at=when)

    accepted = accept_reviewed_proposal(reviewed)
    with pytest.raises(ValueError, match="evidence ids"):
        promote_accepted_proposal(accepted, promoted_at=when)


@pytest.mark.parametrize("metric", ["false_confidence", "false_abstention"])
@pytest.mark.parametrize("errors", range(50))
def test_two_percentage_point_safety_boundary_is_inclusive(metric, errors) -> None:
    """NUM-001: one additional error out of 50 is always within the policy limit.

    Red proof: 5302950e4 rejects several exact boundaries, including 6/50 to 7/50.
    Target: evaluate_retrieval_promotion's safety comparisons. This exhausts the
    integer domain from the numeric property while keeping other gates passing.
    """
    outcomes = tuple(QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(5))
    safe = SafetyMetrics(0.0, 0.0, 0.0)
    baseline = replace(safe, **{metric: errors / 50})
    candidate = replace(safe, **{metric: (errors + 1) / 50})
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, baseline, candidate, True, 100, 250),
        bootstrap_samples=100,
    )
    assert decision.promoted, decision.failures


@pytest.mark.parametrize("metric,label", [
    ("false_confidence", "false confidence"), ("false_abstention", "false abstention"),
])
@pytest.mark.parametrize("increase", [0.020000000001, 0.03])
def test_safety_regression_above_the_boundary_still_refuses(metric, label, increase) -> None:
    """NUM-001 control: tolerating representation error must not widen the policy."""
    outcomes = tuple(QuestionOutcome(str(i), "a", 0.0, 1.0, 0.0, 1.0) for i in range(5))
    baseline = SafetyMetrics(0.12, 0.12, 0.0)
    candidate = replace(baseline, **{metric: 0.12 + increase})
    decision = evaluate_retrieval_promotion(
        RetrievalGateInput(outcomes, baseline, candidate, True, 100, 250),
        bootstrap_samples=100,
    )
    assert decision.failures == (f"{label} regresses by more than two percentage points",)
