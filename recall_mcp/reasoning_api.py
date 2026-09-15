"""Public MCP reasoning response operations.

This module owns the response boundary for reasoning queries and audits.  The bounded execution
engine remains in :mod:`recall_mcp.service` for now, while this smaller boundary owns request
validation, budget construction, and the compatibility contract used by the server.

The service import is lazy on purpose.  ``recall_mcp.service`` exposes these functions at their
historical import path, so importing this module must not create a cycle while that compatibility
alias is installed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from recall.answer_provider import OllamaAnswerProvider
from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.query_class import resolve_graph_expansion, route_query
from recall.reasoning import ReasoningPolicy, ReasoningResponse
from recall.reasoning_planner import ReasoningBudget
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.trust_policy import TrustPolicy
from recall_mcp.models import ReasoningAuditResult

if TYPE_CHECKING:
    from recall.store import PgVectorStore


def _reasoning_policy(mode: str, graph_expansion: str = "off") -> ReasoningPolicy:
    """Build the bounded policy for a public reasoning request."""
    if mode == "retrieval_only":
        return ReasoningPolicy(name="retrieval_only", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "review_required":
        return ReasoningPolicy(name="review_required", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "proposal_assisted":
        return ReasoningPolicy(name="proposal_assisted", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "evidence_assembly":
        return ReasoningPolicy(name="evidence_assembly", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    raise ValueError(f"unknown reasoning mode: {mode!r}")


def reasoning_query(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    *,
    source: str | None = None,
    k: int = 5,
    mode: str = "proposal_assisted",
    max_steps: int | None = None,
    max_graph_nodes: int | None = None,
    max_graph_entities: int | None = None,
    max_evidence_tokens: int = 2048,
    expand_retrieval: bool = False,
    graph_expansion: str = "auto",
    answer_provider: OllamaAnswerProvider | None = None,
    as_of: datetime | None = None,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> ReasoningResponse:
    """Run a bounded reasoning query with optional graph expansion."""
    if graph_expansion not in {"auto", "off", "one_hop"}:
        raise ValueError("graph_expansion must be auto, off, or one_hop")
    graph_expansion = resolve_graph_expansion(query, graph_expansion)  # type: ignore[arg-type]
    route = route_query(query)
    graph_budget = route.graph_budget
    budget = ReasoningBudget(
        max_steps=graph_budget.max_steps if max_steps is None else max_steps,
        max_graph_nodes=(
            graph_budget.max_graph_nodes if max_graph_nodes is None else max_graph_nodes
        ),
        max_evidence_tokens=max_evidence_tokens,
        max_graph_hops=1 if graph_expansion == "one_hop" else 0,
        max_graph_entities=(
            graph_budget.max_graph_entities if max_graph_entities is None else max_graph_entities
        ),
    )
    reasoning_policy = _reasoning_policy(mode, graph_expansion)
    reasoning_policy = replace(
        reasoning_policy,
        allow_retrieval_expansion=expand_retrieval,
    )
    if policy is not None and not policy.strict:
        reasoning_policy = replace(reasoning_policy, require_certified_evidence=False)

    def execute() -> ReasoningResponse:
        # Kept lazy so the public boundary does not import the service during module loading.
        from recall_mcp import service

        return service._execute_reasoning_query(
            store,
            embedder,
            query,
            source,
            k,
            calibration,
            policy,
            security_policy,
            access_context,
            graph_expansion,
            expand_retrieval,
            answer_provider,
            reasoning_policy,
            budget,
            as_of,
        )

    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        with snapshot():
            return execute()
    return execute()


def reasoning_audit(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    query: str = "reasoning audit sentinel",
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> ReasoningAuditResult:
    """Run the bounded reasoning integration audit without exposing corpus content."""
    from recall_mcp import service

    projection = service.reasoning_projection(
        store,
        include_text=False,
        security_policy=security_policy,
        access_context=access_context,
    )
    proposals = service.reasoning_proposals(
        store,
        security_policy=security_policy,
        access_context=access_context,
    )
    response = reasoning_query(
        store,
        embedder,
        query,
        mode="proposal_assisted",
        max_steps=4,
        policy=policy,
        calibration=calibration,
        security_policy=security_policy,
        access_context=access_context,
    )
    refusal_reasons = sorted(
        {
            reason
            for reason in [
                response.refusal_reason,
                response.trusted_evidence.failure_code,
            ]
            if reason
        }
    )
    return ReasoningAuditResult(
        tenant_id=projection.tenant_id,
        generation_id=projection.generation_id,
        trust_state=response.trust_state,
        proposal_count=proposals.proposal_count,
        review_count=proposals.review_count,
        diagnostic_count=projection.diagnostic_count,
        refusal_reasons=refusal_reasons,
        checks={
            "tenant_scoped": response.tenant_id == store.tenant,
            "generation_identity_present": bool(response.generation_id),
            "trust_metadata_present": bool(response.trust_state and response.calibration_status),
            "trace_metadata_present": response.reasoning_trace is not None,
            "development_mode_explicit": (
                (policy is not None and not policy.strict) or response.trust_state == "trusted"
            ),
        },
    )


__all__ = ["reasoning_audit", "reasoning_query"]
