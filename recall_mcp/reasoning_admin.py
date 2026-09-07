"""Deterministic reasoning proposals and rewrite planning for MCP."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from recall.reasoning_graph import project_store_graph
from recall.reasoning_proposals import deterministic_inference_proposals
from recall_mcp.models import ReasoningProposalItem, ReasoningProposalResult, RewritePlanResult

if TYPE_CHECKING:
    from recall.store import PgVectorStore


def apply_command_for(claim: str) -> str:
    """Return the exact CLI command that declares ``claim``."""
    return (
        f"recall rewrite apply <corpus> --claim {claim} "
        f"--reviewer <your-id> --note <why> --apply"
    )


def rewrite_plan(
    store: PgVectorStore,
    *,
    proposal_id: str,
    _project_store_graph_fn: Callable[..., Any] = project_store_graph,
    _deterministic_inference_proposals_fn: Callable[..., Any] = deterministic_inference_proposals,
    _apply_command_for_fn: Callable[[str], str] = apply_command_for,
) -> RewritePlanResult:
    """Describe what declaring ``proposal_id`` would write, without writing anything."""
    from recall.rewrite import claim_key, destination, route_relation

    graph = _project_store_graph_fn(store, include_text=True)
    proposals = _deterministic_inference_proposals_fn(
        graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
    )
    found = next((p for p in proposals if p.id == proposal_id), None)
    if found is None:
        raise ValueError(f"no proposal {proposal_id!r} in this generation")
    routed = route_relation(found.proposed_relation, found.subject_id, found.object_id)
    claim = claim_key(found.proposed_relation, found.subject_id, found.object_id)
    return RewritePlanResult(
        proposal_id=found.id,
        claim=claim,
        relation=found.proposed_relation,
        key=routed.key,
        value=routed.value,
        edit_file=routed.edit_file,
        block=destination(routed.key),
        apply_command=_apply_command_for_fn(claim),
        rejection_checked=False,
    )


def stored_extracted_proposals(graph: object) -> tuple[object, ...]:
    """Refuse when no persisted extraction record is available on the query path."""
    raise ValueError(
        "no extraction record exists for this generation. Run `recall extract run <path>` on "
        "the ingest side first; extraction never runs on the query path."
    )


def reasoning_proposals(
    store: PgVectorStore,
    *,
    limit: int = 100,
    include_extracted: bool = False,
    _project_store_graph_fn: Callable[..., Any] = project_store_graph,
    _deterministic_inference_proposals_fn: Callable[..., Any] = deterministic_inference_proposals,
    _stored_extracted_proposals_fn: Callable[[object], tuple[object, ...]] = stored_extracted_proposals,
) -> ReasoningProposalResult:
    if limit < 1:
        raise ValueError("proposal limit must be positive")
    graph = _project_store_graph_fn(store, include_text=True)
    proposals = _deterministic_inference_proposals_fn(
        graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
    )
    if include_extracted:
        proposals = proposals + _stored_extracted_proposals_fn(graph)  # type: ignore[operator]
    returned = proposals[:limit]
    return ReasoningProposalResult(
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        proposal_count=len(proposals),
        review_count=sum(1 for proposal in proposals if proposal.status == "requires_review"),
        returned_count=len(returned),
        truncated=len(proposals) > len(returned),
        proposals=[
            ReasoningProposalItem(
                id=proposal.id,
                status=proposal.status,
                relation=proposal.proposed_relation,
                subject_id=proposal.subject_id,
                object_id=proposal.object_id,
                confidence=proposal.confidence,
                rule_id=proposal.rule_id,
                generation_id=proposal.generation_id,
                pipeline_id=proposal.pipeline_id,
                provider_id=proposal.provider_id,
                model_id=proposal.model_id,
                provider_revision=proposal.provider_revision,
                source_evidence_ids=list(proposal.source_evidence_ids),
                uncertainty=list(proposal.uncertainty),
            )
            for proposal in returned
        ],
    )
