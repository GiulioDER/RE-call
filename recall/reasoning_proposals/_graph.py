"""Graph conversion helpers for inference proposals."""

from __future__ import annotations

from recall.lineage import canonical_sha256
from recall.reasoning_graph import ReasoningGraphEdge, ReasoningGraphProjection
from recall.reasoning_proposals._deterministic import _source_nodes
from recall.reasoning_proposals.types import PROPOSAL_SCHEMA_VERSION, InferenceProposal

def proposal_to_graph_edge(
    graph: ReasoningGraphProjection, proposal: InferenceProposal
) -> ReasoningGraphEdge:
    """Represent a supersession proposal as a graph candidate edge.

    Only `supersedes` proposals can become candidate edges. The returned edge remains separate from
    authored edges and is not a trust input unless a caller explicitly passes it as proposal data.
    """
    if proposal.proposed_relation != "supersedes":
        raise ValueError("only supersedes proposals can be represented as candidate graph edges")
    node_by_file = {node.file: node for node in _source_nodes(graph)}
    from_node = node_by_file.get(proposal.subject_id)
    to_node = node_by_file.get(proposal.object_id)
    payload = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "tenant_id": graph.tenant_id,
        "generation_id": graph.generation_id,
        "proposal_id": proposal.id,
        "from_file": proposal.subject_id,
        "to_file": proposal.object_id,
    }
    return ReasoningGraphEdge(
        id=f"rg_edge_candidate_{canonical_sha256(payload)[:24]}",
        kind="inferred_candidate_supersedes",
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        from_node_id=from_node.id if from_node is not None else "",
        to_node_id=to_node.id if to_node is not None else None,
        from_file=proposal.subject_id,
        to_file=proposal.object_id,
        authored_reference=None,
        asserted_at=None,
        provenance={
            "proposal_id": proposal.id,
            "provider_id": proposal.provider_id,
            "model_id": proposal.model_id,
            "provider_revision": proposal.provider_revision,
            "source_evidence_ids": proposal.source_evidence_ids,
        },
        metadata={
            "status": proposal.status,
            "confidence": proposal.confidence,
            "uncertainty": proposal.uncertainty,
            "explanation": proposal.explanation,
        },
    )
