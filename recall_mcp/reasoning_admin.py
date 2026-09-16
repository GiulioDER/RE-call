"""Deterministic reasoning proposals and rewrite planning for MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from recall_mcp.models import ReasoningProposalItem, ReasoningProposalResult, RewritePlanResult

if TYPE_CHECKING:
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall.store import PgVectorStore


def apply_command_for(claim: str) -> str:
    """Return the exact CLI command that declares ``claim``."""
    return (
        f"recall rewrite apply <corpus> --claim {claim} "
        f"--reviewer <your-id> --note <why> --apply"
    )


def _stored_extracted_proposals(graph: object) -> tuple[object, ...]:
    """Refuse when no persisted extraction record is available on the query path."""
    raise ValueError(
        "no extraction record exists for this generation. Run `recall extract run <path>` on "
        "the ingest side first; extraction never runs on the query path."
    )


def rewrite_plan(
    store: PgVectorStore,
    *,
    proposal_id: str,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> RewritePlanResult:
    """Describe what declaring ``proposal_id`` would write, without writing anything."""
    from recall.rewrite import claim_key, destination, route_relation
    from recall_mcp import service

    graph = service._authorized_graph(
        store,
        service._store_graph(
            store,
            include_text=True,
            policy_fingerprint=service._combined_graph_policy_fingerprint(
                security_policy=security_policy
            ),
        ),
        security_policy,
        access_context,
    )
    proposals = service._cached_deterministic_proposals(
        graph,
        pipeline_id=graph.pipeline_fingerprint or "legacy",
        policy_scope=service._proposal_policy_scope(security_policy, access_context),
    )
    found = next((proposal for proposal in proposals if proposal.id == proposal_id), None)
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
        apply_command=apply_command_for(claim),
        rejection_checked=False,
    )


def reasoning_proposals(
    store: PgVectorStore,
    *,
    limit: int = 100,
    include_extracted: bool = False,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> ReasoningProposalResult:
    """Return bounded deterministic proposals for the authorized graph view."""
    from recall_mcp import service

    if limit < 1:
        raise ValueError("proposal limit must be positive")
    graph = service._authorized_graph(
        store,
        service._store_graph(
            store,
            include_text=True,
            policy_fingerprint=service._combined_graph_policy_fingerprint(
                security_policy=security_policy
            ),
        ),
        security_policy,
        access_context,
    )
    proposals = service._cached_deterministic_proposals(
        graph,
        pipeline_id=graph.pipeline_fingerprint or "legacy",
        policy_scope=service._proposal_policy_scope(security_policy, access_context),
    )
    if include_extracted:
        proposals = proposals + _stored_extracted_proposals(graph)  # type: ignore[operator]
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
