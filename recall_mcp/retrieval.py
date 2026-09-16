"""Retrieval application boundary for MCP and in process clients.

Profile startup is owned here. Search and evidence remain forwarding façades until their
dependencies are moved in a later retrieval slice. The legacy service import path is preserved
for callers during the migration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Mapping
from collections.abc import Callable
from typing import TYPE_CHECKING

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.profiles import FAST_PROFILE, RetrievalProfile, resolve_retrieval_profile
from recall.profiles import QUALITY_PROFILE, RetrievalAdmission, RetrievalOverloaded
from recall.query_class import route_query, routing_mode
from recall.rerank import COREB_CODE_RERANKER_MODEL
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import EvidenceCard, RetrievalResult, TrustedHit, TrustedResult
from recall.evidence import EvidenceItem
from recall_mcp.models import EvidenceCardModel, EvidenceItemModel, SearchHit
from recall.observability import METRICS, get_logger
from recall.retriever import RetrievalCandidateTrace
from recall.entailment import EntailmentJudge
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall_mcp.factories import (
    _admission,
    _build_reranker,
    _positive_env,
    _require_remote_model_code_enabled,
    _validate_quality_reranker_config,
    resolve_reranker,
)
from recall_mcp.settings import runtime_environment

_log = get_logger("mcp.service")

MAX_SEARCH_K = 50
MAX_QUERY_CHARS = 4096


@dataclass(frozen=True)
class _Retrieval:
    """One executed retrieval, with everything the two cost surfaces are computed from."""

    result: TrustedResult
    timed: TimedEmbedder
    profile: RetrievalProfile
    request_started: float
    admission_wait_ms: float
    #: `k` AFTER both clamps (MAX_SEARCH_K, then the profile's `returned_k`). Returned because a
    #: caller that needs to bound anything by `k` must bound it by the effective one: the raw
    #: argument is what the client asked for, not what the process allowed.
    effective_k: int
    #: The baseline query vector, retained only for providers inside this request.
    query_vector: list[float] | None = None
    #: Private full candidate trace, present only for a sampled source conditioning shadow.
    candidate_trace: tuple[RetrievalCandidateTrace, TrustedResult, Calibration] | None = None


def _retrieve_trusted(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
    pool_k: int | None = None,
    pre_trust_transform: Callable[[RetrievalResult], RetrievalResult] | None = None,
    query_vector_callback: Callable[[list[float]], None] | None = None,
    capture_candidate_trace: bool = False,
    *,
    reranker_builder: Callable[..., object] = _build_reranker,
    admission_factory: Callable[[RetrievalProfile], RetrievalAdmission] = _admission,
    trusted_search_fn: Callable[..., TrustedResult] = trusted_search,
) -> _Retrieval:
    """The guarded, instrumented retrieval shared by search and evidence assembly.

    The optional factories preserve the legacy service test seams while this module owns the
    retrieval execution boundary. The guards and observations stay in one implementation so
    search, evidence, graph-first, and reasoning paths cannot silently diverge.
    """
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"query is {len(query)} characters, over the {MAX_QUERY_CHARS}-character limit. "
            "Search cost scales with query length while the rate budget does not, so an "
            "unbounded query is a shared-database denial of service. Ask a shorter question."
        )
    values = dict(runtime_environment() if env is None else env)
    profile = resolve_retrieval_profile(values)
    selected_mode = routing_mode(values.get("RECALL_ROUTING_MODE", "shadow"))
    if selected_mode == "active" and profile.name == "legacy":
        decision = route_query(query)
        profile = FAST_PROFILE if decision.profile == "fast" else QUALITY_PROFILE
    requested_k = k if pool_k is None else pool_k
    k = max(1, min(requested_k, MAX_SEARCH_K))
    if profile.name != "legacy" and pool_k is None:
        k = min(k, profile.returned_k)
    timed = TimedEmbedder(embedder)
    generation = str(getattr(store, "generation_id", "legacy"))
    request_started = time.perf_counter()
    admission_wait_ms = 0.0
    candidate_traces: list[tuple[RetrievalCandidateTrace, TrustedResult, Calibration]] = []

    def capture_trace(
        raw: RetrievalCandidateTrace,
        trusted: TrustedResult,
        active_calibration: Calibration,
    ) -> None:
        candidate_traces.append((raw, trusted, active_calibration))

    try:
        from recall.decision_ledger import DecisionLedger

        ledger = DecisionLedger.from_env(store, env=values, actor="mcp-service")
        with admission_factory(profile):
            admission_wait_ms = (time.perf_counter() - request_started) * 1000.0
            effective_pre_trust_transform = pre_trust_transform
            if pre_trust_transform is not None and query_vector_callback is not None:

                def capture_query_vector(value: RetrievalResult) -> RetrievalResult:
                    query_vector = timed.last_query_vector
                    if query_vector is not None:
                        query_vector_callback(query_vector)
                    return pre_trust_transform(value)

                effective_pre_trust_transform = capture_query_vector
            result = trusted_search_fn(
                store,
                timed,
                query,
                k=k,
                source=source,
                calibration=calibration,
                reranker=reranker_builder(profile, env=values),
                candidate_k=profile.candidate_k,
                retrieval_profile=profile.name,
                index_generation=generation,
                policy=policy,
                entailment=entailment,
                security_policy=security_policy,
                access_context=access_context,
                ledger=ledger,
                env=values,
                pre_trust_transform=effective_pre_trust_transform,
                candidate_trace_callback=capture_trace if capture_candidate_trace else None,
            )
    except RetrievalOverloaded as exc:
        METRICS.increment(
            "recall_retrieval_rejected_total", profile=profile.name, reason=exc.reason
        )
        raise
    except BaseException:
        METRICS.observe(
            "recall_retrieval_total_ms",
            round((time.perf_counter() - request_started) * 1000.0, 3),
            profile=profile.name,
        )
        METRICS.increment("recall_retrieval_failed_total", profile=profile.name)
        raise
    if capture_candidate_trace and len(candidate_traces) != 1:
        raise RuntimeError("sampled shadow did not retain exactly one candidate trace")
    return _Retrieval(
        result,
        timed,
        profile,
        request_started,
        admission_wait_ms,
        k,
        query_vector=timed.last_query_vector,
        candidate_trace=candidate_traces[0] if candidate_traces else None,
    )


def _cost_surface(
    retrieval: _Retrieval, assembly_started: float
) -> tuple[dict[str, float], float, bool]:
    """Return retrieval stage timings, total latency, and the budget verdict."""
    profile = retrieval.profile
    stage_ms = dict(retrieval.result.diagnostics.stage_ms)
    stage_ms["admission_wait"] = round(retrieval.admission_wait_ms, 3)
    stage_ms["evidence_assembly"] = round((time.perf_counter() - assembly_started) * 1000.0, 3)
    elapsed_ms = (time.perf_counter() - retrieval.request_started) * 1000.0
    total_ms = round(elapsed_ms, 3)
    served_ms = elapsed_ms - retrieval.admission_wait_ms
    budget = profile.enforced_budget_ms
    budget_exceeded = budget is not None and served_ms > budget
    for stage, value in stage_ms.items():
        METRICS.observe("recall_retrieval_stage_ms", value, profile=profile.name, stage=stage)
    METRICS.observe("recall_retrieval_total_ms", total_ms, profile=profile.name)
    if budget_exceeded:
        METRICS.increment("recall_retrieval_budget_exceeded_total", profile=profile.name)
        _log.warning(
            "retrieval served in %.1f ms against the %d ms budget of profile %r "
            "(%.1f ms queued, %.1f ms total)",
            served_ms,
            budget,
            profile.name,
            retrieval.admission_wait_ms,
            total_ms,
        )
    return stage_ms, total_ms, budget_exceeded


def _search_hit_model(hit: TrustedHit, *, include_scores: bool) -> SearchHit:
    """Project a trusted hit into the search response shape."""
    return SearchHit(
        chunk_id=hit.chunk.id,
        source=hit.provenance.file or hit.chunk.source,
        score=round(hit.cosine, 4) if include_scores else None,
        confidence=round(hit.confidence, 4) if include_scores else None,
        verdict=hit.verdict,
        superseded_by=hit.validity.superseded_by,
        valid_until=hit.validity.valid_until.isoformat() if hit.validity.valid_until else None,
        valid_from=hit.validity.valid_from.isoformat() if hit.validity.valid_from else None,
        ordinal=hit.provenance.ord,
        indexed_at=hit.provenance.indexed_at.isoformat() if hit.provenance.indexed_at else None,
        text=hit.chunk.text,
    )


def _trusted_evidence_item_model(item: TrustedHit) -> EvidenceItemModel:
    """Project an independently trusted related hit into the evidence response shape."""
    return EvidenceItemModel(
        chunk_id=item.chunk.id,
        text=item.chunk.text,
        source=item.provenance.file or item.chunk.source,
        ordinal=item.provenance.ord,
        indexed_at=item.provenance.indexed_at.isoformat() if item.provenance.indexed_at else None,
        valid_from=item.validity.valid_from.isoformat() if item.validity.valid_from else None,
        valid_until=item.validity.valid_until.isoformat() if item.validity.valid_until else None,
        cosine=round(item.cosine, 4),
        confidence=round(item.confidence, 4),
        verdict=item.verdict,
    )


def _evidence_item_model(item: EvidenceItem, related_ids: set[str]) -> EvidenceItemModel:
    """Project a bundle item while preserving the score distinction for related evidence."""
    is_related = item.chunk_id in related_ids
    return EvidenceItemModel(
        chunk_id=item.chunk_id,
        text=item.text,
        source=item.source,
        ordinal=item.ordinal,
        indexed_at=item.indexed_at.isoformat() if item.indexed_at else None,
        valid_from=item.valid_from.isoformat() if item.valid_from else None,
        valid_until=item.valid_until.isoformat() if item.valid_until else None,
        cosine=None if is_related else round(item.cosine, 4),
        confidence=None if is_related else round(item.confidence, 4),
        verdict=item.verdict,
        authority=item.authority,
    )


def _evidence_card_model(card: EvidenceCard) -> EvidenceCardModel:
    """Project a provenance card into the public evidence response shape."""
    return EvidenceCardModel(
        card_id=card.card_id,
        chunk_id=card.chunk_id,
        source=card.source,
        source_digest=card.source_digest,
        valid_from=card.valid_from.isoformat() if card.valid_from else None,
        valid_until=card.valid_until.isoformat() if card.valid_until else None,
        first_indexed_at=card.first_indexed_at.isoformat() if card.first_indexed_at else None,
        indexed_at=card.indexed_at.isoformat() if card.indexed_at else None,
        tenant_id=card.tenant_id,
        generation_id=card.generation_id,
        pipeline_fingerprint=card.pipeline_fingerprint,
        corpus_fingerprint=card.corpus_fingerprint,
        calibration_id=card.calibration_id,
        calibration_status=card.calibration_status,
        trust_state=card.trust_state,
        verdict=card.verdict,
        confidence=card.confidence,
        rank=card.rank,
        supersession_links=list(card.supersession_links),
        contradiction_links=list(card.contradiction_links),
        support_refs=list(card.support_refs),
        structured_facts=[fact.to_payload() for fact in card.structured_facts],
        schema_version=card.schema_version,
    )

if TYPE_CHECKING:
    from recall.calibration import Calibration
    from recall.embeddings import Embedder
    from recall.entailment import EntailmentJudge
    from recall.store import PgVectorStore
    from recall.trust_policy import TrustPolicy
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall_mcp.service import EvidenceResult, SearchResult


def search_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    reasoning_available: bool = False,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> SearchResult:
    """Run retrieval through the legacy service implementation during extraction."""
    from recall_mcp import service

    if entailment is None and security_policy is None and access_context is None:
        if env is None:
            return service.search_memory(
                store,
                embedder,
                query,
                source,
                k,
                calibration,
                policy,
                explain,
                include_related,
                related_relation,
                related_max_items,
                reasoning_available,
            )
        return service.search_memory(
            store,
            embedder,
            query,
            source,
            k,
            calibration,
            policy,
            explain,
            include_related,
            related_relation,
            related_max_items,
            reasoning_available,
            env=env,
        )
    if env is None:
        return service.search_memory(
            store,
            embedder,
            query,
            source,
            k,
            calibration,
            policy,
            explain,
            include_related,
            related_relation,
            related_max_items,
            reasoning_available,
            entailment=entailment,
            security_policy=security_policy,
            access_context=access_context,
        )
    return service.search_memory(
        store,
        embedder,
        query,
        source,
        k,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
        reasoning_available,
        entailment=entailment,
        security_policy=security_policy,
        access_context=access_context,
        env=env,
    )


def evidence_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    max_items: int | None = None,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> EvidenceResult:
    """Build generator neutral evidence through the legacy service implementation."""
    from recall_mcp import service

    if entailment is None and security_policy is None and access_context is None:
        if env is None:
            return service.evidence_memory(
                store,
                embedder,
                query,
                source,
                k,
                max_items,
                calibration,
                policy,
                explain,
                include_related,
                related_relation,
                related_max_items,
            )
        return service.evidence_memory(
            store,
            embedder,
            query,
            source,
            k,
            max_items,
            calibration,
            policy,
            explain,
            include_related,
            related_relation,
            related_max_items,
            env=env,
        )
    if env is None:
        return service.evidence_memory(
            store,
            embedder,
            query,
            source,
            k,
            max_items,
            calibration,
            policy,
            explain,
            include_related,
            related_relation,
            related_max_items,
            entailment=entailment,
            security_policy=security_policy,
            access_context=access_context,
        )
    return service.evidence_memory(
        store,
        embedder,
        query,
        source,
        k,
        max_items,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
        entailment=entailment,
        security_policy=security_policy,
        access_context=access_context,
        env=env,
    )


def startup_retrieval_profile(env: dict[str, str] | None = None) -> RetrievalProfile:
    """Resolve and fully validate the process profile. Called once, at server startup.

    Resolution alone used to happen on the first search, which meant a contradictory
    ``RECALL_RETRIEVAL_PROFILE`` / ``RECALL_RERANK`` pair, or a quality profile with no pinned
    reranker artifact, produced a server that started clean and failed on its first client
    request. Startup validation keeps that failure at startup.

    This deliberately does not import torch or load the model. It runs before the store is opened,
    and a configuration error should be reported in milliseconds. The artifact itself is verified
    when the reranker is built.
    """
    values = dict(runtime_environment()) if env is None else env
    selected_routing_mode = routing_mode(values.get("RECALL_ROUTING_MODE", "shadow"))
    profile = resolve_retrieval_profile(values)
    if selected_routing_mode == "active" and profile.name == "legacy":
        # Active routing may select QUALITY_PROFILE on temporal and status queries even when no
        # process profile was configured. Validate that artifact at startup and size the worker
        # pool for FAST_PROFILE, the larger of the two active admission pools.
        _validate_quality_reranker_config(values)
        return FAST_PROFILE
    if profile.name == "quality":
        _validate_quality_reranker_config(values)
    elif profile.name == "code":
        rerank_values = dict(values)
        rerank_values.setdefault("RECALL_RERANK", "1")
        rerank_values.setdefault("RECALL_RERANK_MODEL", "coreb-code")
        spec = resolve_reranker(rerank_values)
        assert spec is not None
        if spec[0] != COREB_CODE_RERANKER_MODEL:
            raise ValueError("the code retrieval profile requires RECALL_RERANK_MODEL=coreb-code")
        _require_remote_model_code_enabled(values, "coreb-code")
        _positive_env(values, "RECALL_RERANK_BATCH_SIZE", 4)
    return profile


__all__ = ["evidence_memory", "search_memory", "startup_retrieval_profile"]
