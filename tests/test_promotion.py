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
