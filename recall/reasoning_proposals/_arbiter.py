"""Turn the supersession arbiter's verdicts into inference proposals.

Inside `recall.reasoning_proposals` for the reason `_extracted.py` gives: proposal ids are the
library's to compute, through the private `_make_proposal`, and a provider built elsewhere would
have to duplicate that construction or export it.

It calls nothing. `recall.supersession_arbiter` ran the model and applied the gate; this replays
the pairs that passed into the proposal protocol, so `propose` is pure with respect to the graph,
and a failed call can never surface as a malformed provider here.

Every proposal is `requires_review`, with the stated probability as `confidence` and the two
quotes in `metadata`. The probability is NOT calibrated across stores, and the uncertainty says
so on every proposal rather than once in a docstring nobody reviewing reads.

Direction: `subject_id` is the SUPERSEDED memo (the older) and `object_id` the superseding one,
as in `_extracted._shape_of` and `_deterministic._direct_reference_proposals`. `recall/rewrite.py`
writes `supersedes:` onto `object_id`, naming `subject_id`.
"""

from __future__ import annotations

from recall.reasoning_graph import ReasoningGraphProjection
from recall.reasoning_proposals._deterministic import _make_proposal, _source_nodes
from recall.reasoning_proposals.types import InferenceProposal, ProposalContext
from recall.supersession_arbiter import ArbiterRun, ArbiterVerdict

ARBITER_RULE_ID = "supersession_arbiter.stated_probability"


class ArbiterProposalProvider:
    """A `ModelBackedProposalProvider` over verdicts the arbiter's gate already passed."""

    def __init__(
        self, run: ArbiterRun, *, provider_id: str, model_id: str, provider_revision: str
    ) -> None:
        self._verdicts = run.verdicts
        self.provider_id = provider_id
        self.model_id = model_id
        self.provider_revision = provider_revision
        #: One proposal per verdict at most, so anything above this is a bug in the adapter.
        self.max_proposals = max(1, len(run.verdicts))

    def propose(
        self, graph: ReasoningGraphProjection, context: ProposalContext
    ) -> tuple[InferenceProposal, ...]:
        node_id_by_file = {
            node.file: node.id for node in _source_nodes(graph) if node.file is not None
        }
        by_id: dict[str, InferenceProposal] = {}
        for verdict in self._verdicts:
            older_id = node_id_by_file.get(verdict.older)
            newer_id = node_id_by_file.get(verdict.newer)
            if older_id is None or newer_id is None:
                # Not in this graph generation; `_coerce_provider_proposal` would refuse the
                # unknown evidence id and fail the whole batch.
                continue
            proposal = _proposal(graph, context, verdict, older_id, newer_id)
            by_id[proposal.id] = proposal
        return tuple(by_id[key] for key in sorted(by_id))


def _proposal(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    verdict: ArbiterVerdict,
    older_id: str,
    newer_id: str,
) -> InferenceProposal:
    subject = f" about {verdict.subject}" if verdict.subject else ""
    return _make_proposal(
        graph=graph,
        context=context,
        source_evidence_ids=(older_id, newer_id),
        proposed_relation="supersedes",
        # subject is the SUPERSEDED memo; object is the one doing the superseding.
        subject_id=verdict.older,
        object_id=verdict.newer,
        explanation=(
            f"{context.model_id} judged these a supersession pair{subject} with stated "
            f"probability {verdict.probability:.2f}; {verdict.newer} is the newer by "
            f"{verdict.direction_source}."
        ),
        confidence=verdict.probability,
        uncertainty=(
            "the probability is the model's own statement, and its threshold is not "
            "calibrated for this corpus",
            f"direction comes from {verdict.direction_source}, not from the text",
        ),
        status="requires_review",
        rule_id=ARBITER_RULE_ID,
        metadata={
            "subject": verdict.subject,
            "quote_older": verdict.quote_older,
            "quote_newer": verdict.quote_newer,
            "direction_source": verdict.direction_source,
        },
    )


__all__ = ["ARBITER_RULE_ID", "ArbiterProposalProvider"]
