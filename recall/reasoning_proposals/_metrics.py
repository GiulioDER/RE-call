"""Evaluation metrics for inference proposal sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from recall.reasoning_proposals.types import InferenceProposal, ProposalStatus, ProposedRelation

#: Statuses that count as the provider ASSERTING a relation, and so are scored.
ASSERTED_STATUSES: tuple[ProposalStatus, ...] = ("candidate",)
#: Statuses that count as the provider REFERRING a relation to a human instead of asserting it.
#: These are reported as a referral rate and never folded into precision.
REFERRED_STATUSES: tuple[ProposalStatus, ...] = ("requires_review",)

_NAN = float("nan")


def proposal_precision_recall(
    proposals: Sequence[InferenceProposal],
    expected_pairs: set[tuple[str, str]],
    *,
    relation: ProposedRelation = "supersedes",
    counted_statuses: Sequence[ProposalStatus] = ASSERTED_STATUSES,
    referred_statuses: Sequence[ProposalStatus] = REFERRED_STATUSES,
) -> Mapping[str, float | int]:
    """Score one relation kind, keeping asserted proposals and referrals strictly apart.

    Both rates are NaN on no data, matching `recall.eval.metrics.fraction_true`: a rate with no
    data is NOT a score. Precision 0.0 from an empty predicted set would read as "the provider was
    wrong" when the truth is "the provider declined to answer" (the rule based baseline proposes
    zero edges and would otherwise publish a 0.0). Recall 1.0 from an empty expectation set is a
    perfect score derived from nothing.

    `requires_review` is a referral, not an assertion. Counting it as predicted would let a
    provider buy precision by relabelling every shaky proposal, so it is excluded from the scored
    set and surfaced as `referral_rate` instead. The policy is a parameter rather than a constant
    so an arm may declare a different one, but the two sets must stay disjoint.
    """

    counted = tuple(counted_statuses)
    referred_only = tuple(referred_statuses)
    overlap = set(counted) & set(referred_only)
    if overlap:
        raise ValueError(
            "counted_statuses and referred_statuses must be disjoint; both contain: "
            + ", ".join(sorted(overlap))
        )

    of_relation = [proposal for proposal in proposals if proposal.proposed_relation == relation]
    asserted = [proposal for proposal in of_relation if proposal.status in counted]
    referrals = [proposal for proposal in of_relation if proposal.status in referred_only]

    predicted = {(proposal.subject_id, proposal.object_id) for proposal in asserted}
    true_positive = len(predicted & expected_pairs)
    false_positive = len(predicted - expected_pairs)
    false_negative = len(expected_pairs - predicted)

    considered = len(asserted) + len(referrals)
    return MappingProxyType(
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": true_positive / len(predicted) if predicted else _NAN,
            "recall": true_positive / len(expected_pairs) if expected_pairs else _NAN,
            "asserted": len(asserted),
            "referred": len(referrals),
            "referral_rate": len(referrals) / considered if considered else _NAN,
        }
    )


__all__ = ["ASSERTED_STATUSES", "REFERRED_STATUSES", "proposal_precision_recall"]
