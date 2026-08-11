"""Contract for `proposal_precision_recall`, the scorer every proposal arm is judged by.

Properties, one test per property:

1. An empty predicted set scores precision NaN, never 0.0. Declining to answer is not being
   wrong, and the rule based baseline proposes zero edges.
2. An empty expectation set scores recall NaN, never 1.0. A perfect score from no data is
   publishable nonsense.
3. NaN never leaks into the counts. `true_positive` / `false_positive` / `false_negative` stay
   integers on both empty cases.
4. The populated session 3 case is unchanged by this contract: precision 1.0, recall 1.0,
   tp 2, fp 0, fn 0.
5. `requires_review` is not asserted. It is excluded from precision and returned as a separate
   referral rate, so relabelling cannot buy precision.
6. The referral rate itself is a rate with no data on an empty relation, so NaN.
7. The status policy is an explicit parameter, and counted and referred statuses must be
   disjoint. Overlap is rejected rather than silently folded.
8. Every relation is scorable separately, not just `supersedes`.
"""

from __future__ import annotations

import math

import pytest

from recall.reasoning_proposals import proposal_precision_recall
from recall.reasoning_proposals.types import InferenceProposal, ProposalStatus, ProposedRelation

SEARCH_PAIR = ("search_policy_v1.md", "search_policy_v2.md")
CACHE_PAIR = ("cache_policy_v1.md", "cache_policy_v2.md")


def _proposal(
    subject: str,
    obj: str,
    *,
    relation: ProposedRelation = "supersedes",
    status: ProposalStatus = "candidate",
    rule_id: str = "test.rule",
) -> InferenceProposal:
    return InferenceProposal(
        id=f"{relation}:{subject}:{obj}:{rule_id}:{status}",
        source_evidence_ids=("evidence-1",),
        proposed_relation=relation,
        subject_id=subject,
        object_id=obj,
        explanation="fixture",
        model_id="rules",
        pipeline_id="pipeline-1",
        provider_id="recall.deterministic",
        provider_revision="test-v1",
        confidence=None,
        uncertainty=(),
        generation_id="gen-1",
        status=status,
        rule_id=rule_id,
    )


def _session3_proposals() -> tuple[InferenceProposal, ...]:
    """The exact relation/status mix pinned in results/reasoning_session3_proposals.json."""

    return (
        _proposal(*SEARCH_PAIR, status="requires_review", rule_id="repeated_decision_subject"),
        _proposal(*CACHE_PAIR, status="requires_review", rule_id="temporal_ordering"),
        _proposal(
            *CACHE_PAIR,
            relation="contradicts",
            status="requires_review",
            rule_id="contradictory_validity_windows",
        ),
        _proposal(*CACHE_PAIR, status="requires_review", rule_id="repeated_decision_subject"),
        _proposal(*CACHE_PAIR, status="candidate", rule_id="explicit_version_naming"),
        _proposal(*SEARCH_PAIR, status="candidate", rule_id="direct_textual_reference"),
        _proposal(*SEARCH_PAIR, status="candidate", rule_id="explicit_version_naming"),
        _proposal(*SEARCH_PAIR, status="requires_review", rule_id="temporal_ordering"),
    )


def test_empty_predicted_set_scores_precision_nan_not_zero() -> None:
    metrics = proposal_precision_recall([], {SEARCH_PAIR})

    assert math.isnan(metrics["precision"])


def test_empty_expectation_set_scores_recall_nan_not_one() -> None:
    metrics = proposal_precision_recall([_proposal(*SEARCH_PAIR)], set())

    assert math.isnan(metrics["recall"])


def test_empty_cases_keep_integer_counts() -> None:
    nothing_predicted = proposal_precision_recall([], {SEARCH_PAIR})
    nothing_expected = proposal_precision_recall([_proposal(*SEARCH_PAIR)], set())

    assert nothing_predicted["true_positive"] == 0
    assert nothing_predicted["false_positive"] == 0
    assert nothing_predicted["false_negative"] == 1
    assert nothing_expected["true_positive"] == 0
    assert nothing_expected["false_positive"] == 1
    assert nothing_expected["false_negative"] == 0


def test_populated_session3_case_is_unchanged() -> None:
    metrics = proposal_precision_recall(_session3_proposals(), {SEARCH_PAIR, CACHE_PAIR})

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_requires_review_is_not_asserted_against_precision() -> None:
    proposals = [
        _proposal(*SEARCH_PAIR, status="candidate"),
        _proposal("wrong_a.md", "wrong_b.md", status="requires_review", rule_id="referred"),
    ]

    metrics = proposal_precision_recall(proposals, {SEARCH_PAIR})

    assert metrics["precision"] == 1.0
    assert metrics["false_positive"] == 0
    assert metrics["asserted"] == 1
    assert metrics["referred"] == 1
    assert metrics["referral_rate"] == 0.5


def test_relabelling_everything_requires_review_cannot_buy_precision() -> None:
    proposals = [
        _proposal(*SEARCH_PAIR, status="requires_review"),
        _proposal("wrong_a.md", "wrong_b.md", status="requires_review", rule_id="referred"),
    ]

    metrics = proposal_precision_recall(proposals, {SEARCH_PAIR})

    assert math.isnan(metrics["precision"])
    assert metrics["asserted"] == 0
    assert metrics["referral_rate"] == 1.0


def test_referral_rate_is_nan_when_the_relation_has_no_proposals() -> None:
    metrics = proposal_precision_recall([], {SEARCH_PAIR})

    assert math.isnan(metrics["referral_rate"])
    assert metrics["referred"] == 0
    assert metrics["asserted"] == 0


def test_rejected_proposals_are_neither_asserted_nor_referred() -> None:
    proposals = [_proposal("wrong_a.md", "wrong_b.md", status="rejected")]

    metrics = proposal_precision_recall(proposals, {SEARCH_PAIR})

    assert metrics["asserted"] == 0
    assert metrics["referred"] == 0
    assert metrics["false_positive"] == 0
    assert math.isnan(metrics["referral_rate"])


def test_status_policy_is_an_explicit_parameter() -> None:
    proposals = [_proposal(*SEARCH_PAIR, status="requires_review")]

    metrics = proposal_precision_recall(
        proposals,
        {SEARCH_PAIR},
        counted_statuses=("candidate", "requires_review"),
        referred_statuses=(),
    )

    assert metrics["precision"] == 1.0
    assert metrics["asserted"] == 1
    assert metrics["referred"] == 0


def test_counted_and_referred_statuses_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        proposal_precision_recall(
            [],
            {SEARCH_PAIR},
            counted_statuses=("candidate", "requires_review"),
            referred_statuses=("requires_review",),
        )


def test_each_relation_is_scored_separately() -> None:
    proposals = [
        _proposal(*SEARCH_PAIR, relation="supersedes", status="candidate"),
        _proposal(*CACHE_PAIR, relation="contradicts", status="candidate"),
        _proposal(*SEARCH_PAIR, relation="same_entity", status="candidate"),
        _proposal(*CACHE_PAIR, relation="references", status="requires_review"),
    ]

    supersedes = proposal_precision_recall(proposals, {SEARCH_PAIR}, relation="supersedes")
    contradicts = proposal_precision_recall(proposals, {CACHE_PAIR}, relation="contradicts")
    same_entity = proposal_precision_recall(proposals, {CACHE_PAIR}, relation="same_entity")
    references = proposal_precision_recall(proposals, {CACHE_PAIR}, relation="references")

    assert supersedes["precision"] == 1.0
    assert supersedes["recall"] == 1.0
    assert contradicts["precision"] == 1.0
    assert contradicts["recall"] == 1.0
    assert same_entity["precision"] == 0.0
    assert same_entity["recall"] == 0.0
    assert math.isnan(references["precision"])
    assert references["referral_rate"] == 1.0


def test_relation_defaults_to_supersedes() -> None:
    proposals = [_proposal(*CACHE_PAIR, relation="contradicts", status="candidate")]

    metrics = proposal_precision_recall(proposals, {CACHE_PAIR})

    assert math.isnan(metrics["precision"])
    assert metrics["asserted"] == 0
