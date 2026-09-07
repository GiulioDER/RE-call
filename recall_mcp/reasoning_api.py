"""Reasoning response APIs for the MCP boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from recall.answer_provider import OllamaAnswerProvider
from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.evidence import EvidenceBundle
from recall.observability import METRICS
from recall.reasoning import (
    REASONING_API_VERSION,
    GenerationSelection,
    ReasoningDiagnostics,
    ReasoningGraphProvider,
    ReasoningProposalProvider,
    ReasoningProviderPorts,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningRetriever,
    SemanticGraphExpansionResult,
    reason,
)
from recall.reasoning_expansion import (
    ExpansionProposal,
    ReasoningExpansionRetriever,
    resolve_expansion_provider,
)
from recall.reasoning_graph import ReasoningGraphProjection, project_store_graph
from recall.reasoning_planner import ReasoningBudget
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
    deterministic_inference_proposals,
)
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall.types import TrustedResult
from recall_mcp.models import ReasoningAuditResult
from recall_mcp.graph_expansion import _retrieval_graph
from recall_mcp.reasoning_common import _reasoning_generation, _reasoning_policy
from recall_mcp.retrieval import _retrieve_trusted

if TYPE_CHECKING:
    from recall.store import PgVectorStore


def _strict_reasoning_refusal(
    refusal: TrustRefusal,
    *,
    tenant_id: str,
    generation: GenerationSelection,
    budget: ReasoningBudget,
) -> ReasoningResponse:
    bundle = EvidenceBundle(
        query="",
        decision="abstain",
        reason_code=refusal.code.value,
        decision_state="no_supporting_evidence",
        calibrated=False,
        stale=False,
        embedding_profile="legacy",
        retrieval_profile="legacy",
        index_generation=refusal.generation_id or generation.generation_id or "legacy",
        items=(),
        trust_state="refused",
        failure_code=refusal.code.value,
    )
    response = ReasoningResponse(
        schema_version=REASONING_API_VERSION,
        outcome="abstained",
        answer=None,
        clarification_request=None,
        trusted_evidence=bundle,
        inference_proposals=(),
        provider_failures=(),
        reasoning_trace=None,
        contradictions=(),
        unsupported_gaps=(),
        citations=(),
        calibration_id=refusal.calibration_id,
        calibration_status=refusal.calibration_status,
        tenant_id=refusal.tenant_id or tenant_id,
        generation_id=refusal.generation_id or generation.generation_id,
        pipeline_fingerprint=refusal.pipeline_fingerprint or generation.pipeline_fingerprint,
        corpus_fingerprint=refusal.corpus_fingerprint or generation.corpus_fingerprint,
        query_set_digest=refusal.query_set_digest,
        trust_state="refused",
        refusal_reason=refusal.code.value,
        diagnostics=ReasoningDiagnostics(
            latency_ms=0,
            budget=budget,
            budget_used=None,
            retrieval_stage_ms={},
            generator_invoked=False,
            citations_normalized=False,
        ),
    )
    METRICS.increment(
        "recall_reasoning_outcome_total",
        outcome=response.outcome,
        trust_state=response.trust_state,
        refusal_reason=refusal.code.value,
    )
    return response


def reasoning_query(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    *,
    source: str | None = None,
    k: int = 5,
    mode: str = "proposal_assisted",
    max_steps: int = 12,
    max_graph_nodes: int = 32,
    max_evidence_tokens: int = 2048,
    expand_retrieval: bool = False,
    graph_expansion: str = "off",
    answer_provider: OllamaAnswerProvider | None = None,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    _reason_fn: Callable[..., ReasoningResponse] = reason,
    _resolve_expansion_provider_fn: Callable[..., Any] = resolve_expansion_provider,
    _retrieve_trusted_fn: Callable[..., Any] = _retrieve_trusted,
    _retrieval_graph_fn: Callable[..., ReasoningGraphProjection] = _retrieval_graph,
    _project_store_graph_fn: Callable[..., ReasoningGraphProjection] = project_store_graph,
    _deterministic_inference_proposals_fn: Callable[..., Any] = deterministic_inference_proposals,
    _expand_semantic_graph_fn: Callable[..., SemanticGraphExpansionResult] | None = None,
    _strict_reasoning_refusal_fn: Callable[..., ReasoningResponse] = _strict_reasoning_refusal,
) -> ReasoningResponse:
    budget = ReasoningBudget(
        max_steps=max_steps,
        max_graph_nodes=max_graph_nodes,
        max_evidence_tokens=max_evidence_tokens,
        max_graph_hops=1 if graph_expansion == "one_hop" else 0,
    )
    if graph_expansion not in {"off", "one_hop"}:
        raise ValueError("graph_expansion must be 'off' or 'one_hop'")
    reasoning_policy = _reasoning_policy(mode, graph_expansion)
    reasoning_policy = replace(
        reasoning_policy,
        allow_retrieval_expansion=expand_retrieval,
    )
    if policy is not None and not policy.strict:
        reasoning_policy = replace(reasoning_policy, require_certified_evidence=False)

    expand_fn = _expand_semantic_graph_fn
    if expand_fn is None:
        from recall_mcp.graph_expansion import _expand_semantic_graph

        expand_fn = _expand_semantic_graph

    def execute() -> ReasoningResponse:
        generation = _reasoning_generation(store)
        retrieval_cache: dict[str, TrustedResult] = {}

        def retrieve(request: ReasoningRequest) -> TrustedResult:
            del request
            if "result" not in retrieval_cache:
                result = _retrieve_trusted_fn(
                    store, embedder, query, source, k, calibration, policy
                ).result
                generation_id = result.generation_id or str(
                    getattr(store, "generation_id", "legacy")
                )
                retrieval_cache["result"] = replace(
                    result,
                    tenant_id=result.tenant_id or store.tenant,
                    generation_id=generation_id,
                )
            return retrieval_cache["result"]

        def graph_provider(
            request: ReasoningRequest, retrieval: TrustedResult
        ) -> ReasoningGraphProjection:
            del request
            if source is not None:
                return _retrieval_graph_fn(retrieval, include_text=True)
            return _project_store_graph_fn(store, include_text=True)

        def proposal_provider(
            request: ReasoningRequest,
            graph: ReasoningGraphProjection,
            retrieval: TrustedResult,
        ) -> Sequence[InferenceProposal] | ProposalProtocolReport:
            del request, retrieval
            return _deterministic_inference_proposals_fn(
                graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
            )

        def graph_expansion_provider(
            request: ReasoningRequest, retrieval: TrustedResult
        ) -> SemanticGraphExpansionResult:
            return expand_fn(store, request, retrieval, calibration, embedder)

        expansion_provider = (
            _resolve_expansion_provider_fn() if expand_retrieval else None
        )

        def expansion_retriever(
            request: ReasoningRequest,
            proposal: ExpansionProposal,
            initial: TrustedResult,
        ) -> TrustedResult:
            del request, initial
            expanded = _retrieve_trusted_fn(
                store, embedder, proposal.query, source, k, calibration, policy
            ).result
            return replace(
                expanded,
                tenant_id=expanded.tenant_id or store.tenant,
                generation_id=expanded.generation_id or generation.generation_id,
            )

        retriever_port: ReasoningRetriever = retrieve
        graph_port: ReasoningGraphProvider = graph_provider
        proposal_port: ReasoningProposalProvider = proposal_provider

        request = ReasoningRequest(
            query=query,
            tenant_id=store.tenant,
            generation=generation,
            providers=ReasoningProviderPorts(
                retriever=retriever_port,
                graph_provider=graph_port,
                proposal_provider=proposal_port,
                expansion_provider=expansion_provider,
                expansion_retriever=cast(ReasoningExpansionRetriever, expansion_retriever),
                graph_expansion_provider=graph_expansion_provider,
                answer_provider=answer_provider,
            ),
            policy=reasoning_policy,
            budget=budget,
        )
        try:
            return _reason_fn(request)
        except TrustRefusal as exc:
            return _strict_reasoning_refusal_fn(
                exc,
                tenant_id=store.tenant,
                generation=generation,
                budget=budget,
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
    _reasoning_projection_fn: Callable[..., Any],
    _reasoning_proposals_fn: Callable[..., Any],
    _reasoning_query_fn: Callable[..., ReasoningResponse],
) -> ReasoningAuditResult:
    projection = _reasoning_projection_fn(store, include_text=False)
    proposals = _reasoning_proposals_fn(store)
    response = _reasoning_query_fn(
        store,
        embedder,
        query,
        mode="proposal_assisted",
        max_steps=4,
        policy=policy,
        calibration=calibration,
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
