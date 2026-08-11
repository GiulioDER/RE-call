"""Turn validated extracted claims into inference proposals.

This adapter lives INSIDE `recall.reasoning_proposals` because it needs `_make_proposal` and
`_proposal_id`, which are private. Building proposals anywhere else would mean either
duplicating the id construction — the exact drift `_providers._coerce_provider_proposal`
exists to catch — or exporting a private so that a caller can compute an identity the library
is supposed to own. The model supplies semantics; the library computes ids.

Three postures, all of them refusals:

- **Every proposal is `requires_review`.** The measured prior on this problem is that four
  candidates survived the mechanical rules and all four were wrong (`recall/fix.py`).
  Swapping rules for a model is not evidence that changed. Nothing reaches corpus metadata
  without a named human either way (`recall/promotion.py`), and this keeps the queue honest
  about what it is: a reviewing aid, not an automation.
- **`confidence` is None.** There is no calibrated score behind an extracted claim, and a
  number invented to fill the field would be read as one.
- **One provider identity per adapter.** Extractions from two engines cannot be attributed to
  one provider, because the audit record would then be wrong about what produced a claim.

Direction, the hazard worth stating twice: a supersession claim read from file F naming target
T means F REPLACES T, so `subject_id` is T (the superseded document) and `object_id` is F.
That matches `_deterministic._direct_reference_proposals`. Reversing it declares the live memo
stale and demotes it beneath the one it replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recall.reasoning_graph import ReasoningGraphProjection
from recall.reasoning_proposals._deterministic import _make_proposal, _source_nodes
from recall.reasoning_proposals.types import (
    InferenceProposal,
    ProposalContext,
    ProposedRelation,
)
from recall.truth_extraction.types import (
    MAX_CLAIMS_PER_FILE,
    ExtractedClaim,
    FileExtraction,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
)


class ExtractedClaimProposalProvider:
    """A `ModelBackedProposalProvider` over claims that already passed the validation ladder.

    It calls nothing: the extraction happened on the ingest path, and this replays its result
    into the proposal protocol. `propose` is therefore pure with respect to the graph.
    """

    def __init__(
        self,
        extractions: Sequence[FileExtraction],
        *,
        max_proposals: int | None = None,
    ) -> None:
        self._extractions = tuple(extractions)
        identities = {
            (item.engine_id, item.model_id, item.revision, item.prompt_revision)
            for item in self._extractions
        }
        if len(identities) > 1:
            raise ValueError(
                "extractions must share one engine identity; got "
                f"{len(identities)} distinct identities"
            )
        engine_id, model_id, revision, prompt_revision = (
            identities.pop() if identities else ("recall.truth_extraction", "none", "none", "none")
        )
        self.provider_id = engine_id
        self.model_id = model_id
        #: Engine revision and prompt revision together: the same engine under a reworded
        #: prompt is a different extractor and must not share an audit identity.
        self.provider_revision = f"{revision}+{prompt_revision}"
        #: Structural bound. Claims already passed the per file ceiling, so anything above
        #: this is a bug in the adapter rather than a runaway provider.
        self.max_proposals = (
            max_proposals
            if max_proposals is not None
            else max(1, MAX_CLAIMS_PER_FILE * len(self._extractions))
        )

    def propose(
        self, graph: ReasoningGraphProjection, context: ProposalContext
    ) -> tuple[InferenceProposal, ...]:
        node_id_by_file = {
            node.file: node.id for node in _source_nodes(graph) if node.file is not None
        }
        by_id: dict[str, InferenceProposal] = {}
        for extraction in self._extractions:
            evidence_id = node_id_by_file.get(extraction.file)
            if evidence_id is None:
                # The file is not in this graph generation. Citing nothing is not an option:
                # `_coerce_provider_proposal` rejects a proposal with no known evidence.
                continue
            for claim in extraction.claims:
                proposal = self._proposal(
                    graph=graph,
                    context=context,
                    claim=claim,
                    file=extraction.file,
                    evidence_id=evidence_id,
                    node_id_by_file=node_id_by_file,
                )
                if proposal is not None:
                    # Identical claims hash to identical ids, and `_reject_duplicate_ids`
                    # treats a repeat as a malformed batch. Collapse them here instead.
                    by_id[proposal.id] = proposal
        return tuple(by_id[key] for key in sorted(by_id))

    def _proposal(
        self,
        *,
        graph: ReasoningGraphProjection,
        context: ProposalContext,
        claim: ExtractedClaim,
        file: str,
        evidence_id: str,
        node_id_by_file: Mapping[str, str],
    ) -> InferenceProposal | None:
        shape = _shape_of(
            claim, file=file, evidence_id=evidence_id, node_id_by_file=node_id_by_file
        )
        if shape is None:
            return None
        relation, subject_id, object_id, explanation, evidence = shape
        return _make_proposal(
            graph=graph,
            context=context,
            source_evidence_ids=evidence,
            proposed_relation=relation,
            subject_id=subject_id,
            object_id=object_id,
            explanation=explanation,
            confidence=None,
            uncertainty=(
                "the claim is read from prose, not from authored frontmatter",
                "a quote proves the text exists, not that the author meant it as a declaration",
            ),
            status="requires_review",
            rule_id=f"truth_extraction.{claim.kind}",
            metadata={"file": file, "claim_kind": claim.kind, "quote": claim.quote},
        )


def _shape_of(
    claim: ExtractedClaim,
    *,
    file: str,
    evidence_id: str,
    node_id_by_file: Mapping[str, str],
) -> tuple[ProposedRelation, str, str, str, tuple[str, ...]] | None:
    if isinstance(claim, SupersessionClaim):
        target_evidence = node_id_by_file.get(claim.superseded)
        if target_evidence is None:
            # The ladder resolves a target against `corpus_names`, which can name a file this
            # graph generation has no source node for. Emitting the proposal anyway would give
            # `proposal_to_graph_edge` an empty `from_node_id`: a candidate supersession edge
            # pointing at nothing. The claim stays in the extraction record for review.
            return None
        return (
            "supersedes",
            # subject is the SUPERSEDED document; object is the one doing the superseding.
            claim.superseded,
            file,
            f"{file} states in prose that it supersedes {claim.superseded}.",
            (target_evidence, evidence_id),
        )
    if isinstance(claim, ValidityClaim):
        return (
            "declares_validity",
            file,
            f"{claim.key}:{claim.date}",
            f"{file} states {claim.key} {claim.date} in prose without declaring it.",
            (evidence_id,),
        )
    if isinstance(claim, StatusClaim):
        return (
            "declares_status",
            file,
            f"status:{claim.value}",
            f"{file} states status {claim.value} in prose without declaring it.",
            (evidence_id,),
        )
    return (
        "same_entity",
        claim.entity,
        claim.alias,
        f"{file} treats {claim.alias!r} as another name for {claim.entity!r}.",
        (evidence_id,),
    )


__all__ = ["ExtractedClaimProposalProvider"]
