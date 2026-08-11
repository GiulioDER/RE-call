"""Evaluation metrics for inference proposal sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import get_args

from recall.reasoning_proposals.types import InferenceProposal, ProposalStatus, ProposedRelation

_VALID_STATUSES: frozenset[str] = frozenset(get_args(ProposalStatus))

#: Statuses that count as the provider ASSERTING a relation, and so are scored.
ASSERTED_STATUSES: tuple[ProposalStatus, ...] = ("candidate",)
#: Statuses that count as the provider REFERRING a relation to a human instead of asserting it.
#: These are reported as a referral rate and never folded into precision.
REFERRED_STATUSES: tuple[ProposalStatus, ...] = ("requires_review",)

_NAN = float("nan")


def _validated_statuses(
    statuses: Sequence[ProposalStatus], name: str, *, allow_empty: bool
) -> tuple[ProposalStatus, ...]:
    """Reject a policy that would score zero in silence rather than raise.

    A bare `"candidate"` is the dangerous one: `tuple()` shreds a string into single characters,
    so the policy matches nothing and the arm publishes precision NaN beside recall 0.0, which
    reads as two contradictory findings rather than as the misconfiguration it is.
    """

    if isinstance(statuses, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of statuses, not a bare string")
    values = tuple(statuses)
    if not values and not allow_empty:
        raise ValueError(f"{name} must name at least one status")
    unknown = sorted(set(values) - _VALID_STATUSES)
    if unknown:
        raise ValueError(
            f"{name} contains unknown proposal statuses: "
            + ", ".join(unknown)
            + "; valid statuses are "
            + ", ".join(sorted(_VALID_STATUSES))
        )
    return values


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

    Mind the two denominators. Precision divides by the DEDUPLICATED (subject, object) pair set;
    `asserted_proposals` and `referred_proposals` count raw proposals, because several rules may
    fire on the same pair and the referral split is about how the provider labelled its own
    output. Session 3 publishes three asserted proposals over two pairs, so the keys name their
    unit and `true_positive / asserted_proposals` is not the published precision.
    """

    counted = _validated_statuses(counted_statuses, "counted_statuses", allow_empty=False)
    referred_only = _validated_statuses(referred_statuses, "referred_statuses", allow_empty=True)
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
            "asserted_proposals": len(asserted),
            "referred_proposals": len(referrals),
            "referral_rate": len(referrals) / considered if considered else _NAN,
        }
    )


__all__ = ["ASSERTED_STATUSES", "REFERRED_STATUSES", "proposal_precision_recall"]
