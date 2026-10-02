"""Execution of one `recall_reasoning_query`: generation binding, trusted retrieval, graph
expansion, proposals and the strict refusal.

Moved out of `recall_mcp.service`, which re-exports every name defined here. Names that tests
monkeypatch on `recall_mcp.service` are reached through `_svc()` at call time, so those patches
keep applying to this code.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import replace
from datetime import datetime

from recall.answer_provider import OllamaAnswerProvider
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueLineageError,
    AtomicRescueSelectionError,
)
from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.evidence import (
    EvidenceBundle,
    EvidencePolicy,
)
from recall.observability import (
    get_logger as _get_logger,
    METRICS,
    performance_trace_scope,
    PerformanceTrace,
)
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,
    ReasoningDiagnostics,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    ReasoningResponse,
    SemanticGraphExpansionResult,
)
from recall.reasoning_expansion import (
    ExpansionProposal,
    ReasoningExpansionRetriever,
)
from recall.reasoning_graph import ReasoningGraphProjection
from recall.reasoning_planner import ReasoningBudget
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
)
from recall.security_policy import (
    AccessContext,
    SourceSecurityPolicy,
)
from recall.source_conditioning import SourceConditioningArtifactError
from recall.store import PgVectorStore
from recall.trust_policy import (
    TrustPolicy,
    TrustRefusal,
)
from recall.types import (
    RetrievalResult,
    TrustedResult,
)

from recall_mcp.graph_projection import (
    _authorized_graph,
    _combined_graph_policy_fingerprint,
)
from recall_mcp.reasoning_common import _reasoning_generation
from recall_mcp.reasoning_diagnostics import (
    _atomic_rescue_shadow_payload,
    _atomic_rescue_shadow_sampled,
    _document_expansion_benchmark_audit_enabled,
    _document_expansion_benchmark_audit_payload,
    _retrieval_leg_benchmark_audit_enabled,
    _source_admission_benchmark_audit_enabled,
    _source_conditioning_reuse_benchmark_audit_enabled,
    _source_conditioning_reused_audits,
    _source_conditioning_shadow_payload,
    _source_conditioning_shadow_sampled,
)
from recall_mcp.semantic_graph_cache import (
    _assemble_graph_first_context,
    _cached_deterministic_proposals,
    _expand_semantic_graph,
    _graph_precision_policy_fingerprint,
    _graph_tail_replacement_margin,
    _proposal_policy_scope,
    _provisional_graph_seed_result,
    _resolve_graph_calibration,
    _retrieval_graph,
    GRAPH_FIRST_CONTEXT_K,
    GRAPH_FIRST_RETRIEVAL_K,
)
from recall_mcp.settings import runtime_environment
from typing import (
    Any,
    cast,
)
import copy
import time


_log = _get_logger("mcp.service")


def _svc() -> Any:
    """`recall_mcp.service`, looked up at call time so its monkeypatch seams keep applying."""
    from recall_mcp import service

    return service


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


def _execute_reasoning_query(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
    graph_expansion: str,
    expand_retrieval: bool,
    answer_provider: OllamaAnswerProvider | None,
    reasoning_policy: ReasoningPolicy,
    budget: ReasoningBudget,
    as_of: datetime | None,
    *,
    answer_profile: str = "dated",
) -> ReasoningResponse:
    """Execute one generation bound reasoning request inside the store snapshot."""
    generation = _reasoning_generation(store)
    retrieval_cache: dict[str, TrustedResult] = {}
    retrieval_context: dict[str, list[float] | None] = {}
    graph_first_expansion: dict[str, SemanticGraphExpansionResult] = {}

    def capture_retrieval_query_vector(vector: list[float]) -> None:
        retrieval_context["query_vector"] = vector
        request._context.query_vector = vector

    def graph_first_transform(raw: RetrievalResult) -> RetrievalResult:
        provisional = _provisional_graph_seed_result(raw, store, request)
        expansion = _expand_semantic_graph(
            store,
            request,
            provisional,
            calibration,
            embedder,
            security_policy=security_policy,
            access_context=access_context,
            defer_trust_evaluation=True,
            excluded_chunk_ids=frozenset(hit.chunk.id for hit in raw.hits),
        )
        graph_first_expansion["result"] = expansion
        active_calibration = _resolve_graph_calibration(store, request, calibration)
        return _assemble_graph_first_context(
            raw,
            expansion.scored_candidates,
            calibration=active_calibration,
            tail_replacement_margin=_graph_tail_replacement_margin(),
        )

    def retrieve(request: ReasoningRequest) -> TrustedResult:
        if "result" not in retrieval_cache:
            performance = request._context.performance
            shadow_values = dict(runtime_environment()) if performance is not None else {}
            shadow_mode = (
                shadow_values.get("RECALL_SOURCE_CONDITIONING_MODE", "off").strip().lower()
            )
            shadow_sampled = False
            shadow_configuration_error = False
            if performance is not None and shadow_mode != "off":
                try:
                    shadow_sampled = _source_conditioning_shadow_sampled(query, shadow_values)
                except SourceConditioningArtifactError:
                    shadow_configuration_error = True
            atomic_mode = shadow_values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower()
            atomic_sampled = False
            atomic_configuration_error = False
            if performance is not None and atomic_mode == "shadow":
                try:
                    atomic_sampled = _atomic_rescue_shadow_sampled(query, shadow_values)
                except AtomicRescueArtifactError:
                    atomic_configuration_error = True
            capture_source_trace = (
                performance is not None
                and shadow_mode != "off"
                and not shadow_configuration_error
                and shadow_sampled
                and security_policy is None
                and access_context is None
            )
            capture_atomic_trace = (
                performance is not None
                and atomic_mode == "shadow"
                and not atomic_configuration_error
                and atomic_sampled
                and source is None
                and security_policy is None
                and access_context is None
            )
            capture_shadow_trace = capture_source_trace or capture_atomic_trace
            leg_audit: dict[str, object] | None = None
            source_admission_audit: dict[str, object] | None = None
            if performance is None:
                executed = _svc()._retrieve_trusted(
                    store,
                    embedder,
                    query,
                    source,
                    k,
                    calibration,
                    policy,
                    security_policy=security_policy,
                    access_context=access_context,
                    pool_k=GRAPH_FIRST_RETRIEVAL_K if graph_expansion == "one_hop" else None,
                    pre_trust_transform=(
                        graph_first_transform if graph_expansion == "one_hop" else None
                    ),
                    query_vector_callback=(
                        capture_retrieval_query_vector if graph_expansion == "one_hop" else None
                    ),
                    capture_candidate_trace=capture_shadow_trace,
                )
            else:
                with performance.span("baseline_retrieval_ms"):
                    executed = _svc()._retrieve_trusted(
                        store,
                        embedder,
                        query,
                        source,
                        k,
                        calibration,
                        policy,
                        security_policy=security_policy,
                        access_context=access_context,
                        pool_k=GRAPH_FIRST_RETRIEVAL_K if graph_expansion == "one_hop" else None,
                        pre_trust_transform=(
                            graph_first_transform if graph_expansion == "one_hop" else None
                        ),
                        query_vector_callback=(
                            capture_retrieval_query_vector if graph_expansion == "one_hop" else None
                        ),
                        capture_candidate_trace=capture_shadow_trace,
                    )
            if performance is not None and _retrieval_leg_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "retrieval leg benchmark audit is unavailable with source security policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "retrieval leg benchmark audit did not capture a query vector"
                    )
                with performance.span("retrieval_leg_benchmark_audit_ms"):
                    leg_audit = _svc()._retrieval_leg_benchmark_audit_payload(
                        store,
                        query,
                        query_vector,
                        source,
                    )
                performance.set("retrieval_leg_benchmark_audit", leg_audit)
            if performance is not None and _document_expansion_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "document expansion benchmark audit is unavailable with source security "
                        "policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "document expansion benchmark audit did not capture a query vector"
                    )
                with performance.span("document_expansion_benchmark_audit_ms"):
                    document_audit = _document_expansion_benchmark_audit_payload(
                        store,
                        embedder,
                        query,
                        query_vector,
                        source,
                        k,
                        calibration,
                        policy,
                        executed.profile,
                    )
                performance.set("document_expansion_benchmark_audit", document_audit)
            if performance is not None and _source_admission_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "source admission benchmark audit is unavailable with source security "
                        "policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "source admission benchmark audit did not capture a query vector"
                    )
                with performance.span("source_admission_benchmark_audit_ms"):
                    source_admission_audit = _svc()._source_admission_benchmark_audit_payload(
                        store,
                        embedder,
                        query,
                        query_vector,
                        source,
                        calibration,
                        policy,
                        executed.profile,
                    )
                performance.set("source_admission_benchmark_audit", source_admission_audit)
            if performance is not None and shadow_mode != "off":
                if shadow_configuration_error:
                    performance.set(
                        "source_conditioning_shadow",
                        {"status": "error", "error_code": "configuration_error"},
                    )
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total",
                        status="configuration_error",
                    )
                elif not shadow_sampled:
                    performance.set("source_conditioning_shadow", {"status": "not_sampled"})
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total", status="not_sampled"
                    )
                elif security_policy is not None or access_context is not None:
                    performance.set(
                        "source_conditioning_shadow",
                        {"status": "skipped", "reason_code": "source_security"},
                    )
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total",
                        status="security_skipped",
                    )
                else:
                    shadow_started = time.perf_counter()
                    trace_trust_ms = 0.0
                    shadow_payload: dict[str, object] | None = None
                    artifact_path = shadow_values.get(
                        "RECALL_SOURCE_CONDITIONING_ARTIFACT", ""
                    ).strip()
                    try:
                        if executed.candidate_trace is None:
                            raise RuntimeError("sampled shadow has no candidate trace")
                        trace_trusted = executed.candidate_trace[1]
                        trace_trust_ms = float(
                            trace_trusted.diagnostics.stage_ms.get(
                                "source_conditioning_trace_trust", 0.0
                            )
                        )
                        reused_legs, reused_pool = _source_conditioning_reused_audits(
                            executed.candidate_trace, executed.profile
                        )
                        if not artifact_path:
                            raise SourceConditioningArtifactError(
                                "source conditioning shadow artifact path is required"
                            )
                        shadow_payload = _source_conditioning_shadow_payload(
                            artifact_path=artifact_path,
                            leg_audit=reused_legs,
                            pool_audit=reused_pool,
                            baseline=executed.result,
                            embedder=embedder,
                            profile=executed.profile,
                            policy=shadow_values.get(
                                "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY", "alpha008"
                            )
                            .strip()
                            .lower(),
                        )
                    except (OSError, SourceConditioningArtifactError):
                        performance.set(
                            "source_conditioning_shadow",
                            {"status": "error", "error_code": "artifact_error"},
                        )
                        METRICS.increment(
                            "recall_source_conditioning_shadow_total",
                            status="artifact_error",
                        )
                    except Exception:  # BROAD-CATCH: fail-open
                        _log.exception("source conditioning shadow computation failed")
                        performance.set(
                            "source_conditioning_shadow",
                            {"status": "error", "error_code": "computation_error"},
                        )
                        METRICS.increment(
                            "recall_source_conditioning_shadow_total",
                            status="computation_error",
                        )
                    reuse_ms = trace_trust_ms + (time.perf_counter() - shadow_started) * 1000.0
                    performance.set_span("source_conditioning_shadow_ms", reuse_ms)
                    if shadow_payload is not None:
                        performance.set("source_conditioning_shadow", shadow_payload)
                        METRICS.increment("recall_source_conditioning_shadow_total", status="ok")
                        added_count = shadow_payload.get("added_count")
                        if isinstance(added_count, int) and not isinstance(added_count, bool):
                            METRICS.increment(
                                "recall_source_conditioning_shadow_added_items_total",
                                value=added_count,
                            )
                        lane_counts_payload = shadow_payload.get("lane_counts")
                        if isinstance(lane_counts_payload, Mapping):
                            for lane, count in lane_counts_payload.items():
                                if isinstance(count, int) and not isinstance(count, bool):
                                    METRICS.increment(
                                        "recall_source_conditioning_shadow_lane_total",
                                        value=count,
                                        lane=str(lane),
                                    )
                        if _source_conditioning_reuse_benchmark_audit_enabled():
                            duplicate_started = time.perf_counter()
                            try:
                                query_vector = executed.query_vector
                                if query_vector is None:
                                    raise RuntimeError(
                                        "reuse benchmark did not capture a query vector"
                                    )
                                duplicate_legs = _svc()._retrieval_leg_benchmark_audit_payload(
                                    store, query, query_vector, source
                                )
                                duplicate_pool = _svc()._source_admission_benchmark_audit_payload(
                                    store,
                                    embedder,
                                    query,
                                    query_vector,
                                    source,
                                    calibration,
                                    policy,
                                    executed.profile,
                                )
                                duplicate_payload = _source_conditioning_shadow_payload(
                                    artifact_path=artifact_path,
                                    leg_audit=duplicate_legs,
                                    pool_audit=duplicate_pool,
                                    baseline=executed.result,
                                    embedder=embedder,
                                    profile=executed.profile,
                                )
                            except Exception:  # BROAD-CATCH: fail-open
                                _log.exception("source conditioning reuse benchmark audit failed")
                                performance.set(
                                    "source_conditioning_trace_reuse_audit",
                                    {"status": "error"},
                                )
                            else:
                                duplicate_ms = (time.perf_counter() - duplicate_started) * 1000.0
                                performance.set_span(
                                    "source_conditioning_duplicate_audit_ms", duplicate_ms
                                )
                                performance.set(
                                    "source_conditioning_trace_reuse_audit",
                                    {
                                        "status": "ok",
                                        "selected_hash_parity": (
                                            shadow_payload["selected_chunk_hashes"]
                                            == duplicate_payload["selected_chunk_hashes"]
                                        ),
                                        "reused_selected_chunk_hashes": shadow_payload[
                                            "selected_chunk_hashes"
                                        ],
                                        "duplicate_selected_chunk_hashes": duplicate_payload[
                                            "selected_chunk_hashes"
                                        ],
                                        "reuse_ms": round(reuse_ms, 3),
                                        "duplicate_ms": round(duplicate_ms, 3),
                                    },
                                )
            if performance is not None and atomic_mode == "shadow":
                atomic_payload: dict[str, object] | None = None
                atomic_started = time.perf_counter()
                benchmark_result = (
                    copy.deepcopy(executed.result)
                    if shadow_values.get("RECALL_BENCHMARK_PIN", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                    else None
                )
                if atomic_configuration_error:
                    atomic_payload = {"status": "error", "error_code": "configuration_error"}
                elif not atomic_sampled:
                    atomic_payload = {"status": "not_sampled"}
                elif source is not None:
                    atomic_payload = {"status": "skipped", "reason_code": "source_scope"}
                elif security_policy is not None or access_context is not None:
                    atomic_payload = {"status": "skipped", "reason_code": "security_scope"}
                else:
                    artifact_path = shadow_values.get("RECALL_ATOMIC_RESCUE_ARTIFACT", "").strip()
                    try:
                        if not artifact_path:
                            raise AtomicRescueArtifactError(
                                "atomic rescue shadow artifact path is required"
                            )
                        if executed.query_vector is None:
                            raise AtomicRescueSelectionError(
                                "atomic rescue shadow has no query vector"
                            )
                        if executed.candidate_trace is None:
                            raise AtomicRescueSelectionError(
                                "atomic rescue shadow has no candidate trace"
                            )
                        atomic_payload = _atomic_rescue_shadow_payload(
                            artifact_path=artifact_path,
                            query=query,
                            query_vector=executed.query_vector,
                            candidate_trace=executed.candidate_trace,
                            baseline=executed.result,
                            embedder=embedder,
                            expected_path=(
                                shadow_values.get(
                                    "RECALL_BENCHMARK_ATOMIC_RESCUE_EXPECTED", ""
                                ).strip()
                                or None
                                if shadow_values.get("RECALL_BENCHMARK_PIN", "")
                                .strip()
                                .lower()
                                in {"1", "true", "yes", "on"}
                                else None
                            ),
                        )
                    except AtomicRescueLineageError:
                        atomic_payload = {"status": "error", "error_code": "lineage_error"}
                    except (OSError, AtomicRescueArtifactError):
                        atomic_payload = {"status": "error", "error_code": "artifact_error"}
                    except AtomicRescueSelectionError:
                        atomic_payload = {"status": "error", "error_code": "computation_error"}
                    except Exception:  # BROAD-CATCH: fail-open
                        _log.exception("atomic rescue shadow computation failed")
                        atomic_payload = {"status": "error", "error_code": "computation_error"}
                if benchmark_result is not None:
                    atomic_payload["benchmark_public_result_unchanged"] = (
                        executed.result == benchmark_result
                    )
                atomic_ms = (time.perf_counter() - atomic_started) * 1000.0
                performance.set_span("atomic_rescue_shadow_ms", atomic_ms)
                performance.set("atomic_rescue_shadow", atomic_payload)
                status = str(atomic_payload.get("status", "error"))
                reason = atomic_payload.get("reason_code") or atomic_payload.get("error_code")
                METRICS.increment(
                    "recall_atomic_rescue_shadow_total",
                    status=str(reason) if reason is not None else status,
                )
                selector_ms = atomic_payload.get("selector_ms")
                if isinstance(selector_ms, (int, float)) and not isinstance(selector_ms, bool):
                    METRICS.observe("recall_atomic_rescue_selector_ms", float(selector_ms))
            result = executed.result
            generation_id = result.generation_id or str(getattr(store, "generation_id", "legacy"))
            retrieval_cache["result"] = replace(
                result,
                tenant_id=result.tenant_id or store.tenant,
                generation_id=generation_id,
            )
            retrieval_context["query_vector"] = getattr(executed, "query_vector", None)
        request._context.query_vector = retrieval_context.get("query_vector")
        return retrieval_cache["result"]

    def graph_provider(
        request: ReasoningRequest, retrieval: TrustedResult
    ) -> ReasoningGraphProjection:
        if source is not None:
            graph = _retrieval_graph(retrieval, include_text=True)
        else:
            graph = _svc()._store_graph(
                store,
                include_text=True,
                policy_fingerprint=_combined_graph_policy_fingerprint(
                    security_policy=security_policy,
                    graph_policy_fingerprint=(
                        _graph_precision_policy_fingerprint()
                        if graph_expansion == "one_hop"
                        else None
                    ),
                ),
            )
        return _authorized_graph(store, graph, security_policy, access_context)

    def proposal_provider(
        request: ReasoningRequest,
        graph: ReasoningGraphProjection,
        retrieval: TrustedResult,
    ) -> Sequence[InferenceProposal] | ProposalProtocolReport:
        del retrieval
        return _cached_deterministic_proposals(
            graph,
            pipeline_id=graph.pipeline_fingerprint or "legacy",
            policy_scope=request.policy_scope
            or _proposal_policy_scope(security_policy, access_context),
        )

    def graph_expansion_provider(
        request: ReasoningRequest, retrieval: TrustedResult
    ) -> SemanticGraphExpansionResult:
        prepared = graph_first_expansion.get("result")
        if prepared is not None:
            return replace(prepared, retrieval=retrieval)
        return _expand_semantic_graph(
            store,
            request,
            retrieval,
            calibration,
            embedder,
            security_policy=security_policy,
            access_context=access_context,
        )

    expansion_provider = _svc().resolve_expansion_provider() if expand_retrieval else None

    def expansion_retriever(
        request: ReasoningRequest,
        proposal: ExpansionProposal,
        initial: TrustedResult,
    ) -> TrustedResult:
        del request, initial
        expanded: TrustedResult = _svc()._retrieve_trusted(
            store,
            embedder,
            proposal.query,
            source,
            k,
            calibration,
            policy,
            security_policy=security_policy,
            access_context=access_context,
        ).result
        return replace(
            expanded,
            tenant_id=expanded.tenant_id or store.tenant,
            generation_id=expanded.generation_id or generation.generation_id,
        )

    request = ReasoningRequest(
        query=query,
        tenant_id=store.tenant,
        generation=generation,
        providers=ReasoningProviderPorts(
            retriever=retrieve,
            graph_provider=graph_provider,
            proposal_provider=proposal_provider,
            expansion_provider=expansion_provider,
            expansion_retriever=cast(ReasoningExpansionRetriever, expansion_retriever),
            graph_expansion_provider=graph_expansion_provider,
            answer_provider=answer_provider,
        ),
        policy=reasoning_policy,
        budget=budget,
        evidence_policy=EvidencePolicy(
            max_items=GRAPH_FIRST_CONTEXT_K if graph_expansion == "one_hop" else max(1, k)
        ),
        as_of=as_of,
        policy_scope=_proposal_policy_scope(security_policy, access_context),
        answer_profile=answer_profile,
    )
    request._context.performance = PerformanceTrace()
    try:
        with performance_trace_scope(request._context.performance):
            return cast(ReasoningResponse, _svc().reason(request))
    except TrustRefusal as exc:
        return _strict_reasoning_refusal(
            exc,
            tenant_id=store.tenant,
            generation=generation,
            budget=budget,
        )
