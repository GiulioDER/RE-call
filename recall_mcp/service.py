from __future__ import annotations

# This module is a compatibility facade. The imports below intentionally preserve the public
# service-owned names and monkeypatch seams while implementations move to focused boundaries.
# ruff: noqa: F401

import copy
import hashlib
import json
import os
import random
from contextlib import AbstractContextManager, nullcontext, suppress
import mimetypes
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import psycopg
from pydantic import BaseModel, Field

from recall_mcp.models import (
    EvidenceCardModel,
    EvidenceItemModel,
    EvidenceResult,
    ForgetResult,
    CurrentStateRecordModel,  # noqa: F401  # legacy public import
    CurrentStateResult,  # noqa: F401  # legacy public import
    InventoryEntry,  # noqa: F401  # legacy public import
    InventoryResult,  # noqa: F401  # legacy public import
    IndexResult,  # noqa: F401  # legacy public import
    MemoryStatsResult,  # noqa: F401  # legacy public import
    ReasoningAuditResult,  # noqa: F401  # legacy public import
    ReasoningProjectionResult,
    ReasoningProposalItem,
    ReasoningProposalResult,
    RewritePlanResult,
    SearchHit,
    SearchResult,
)

from recall.calibration import Calibration
from recall.calibration_v2 import CalibrationRepository
from recall.answer_provider import OllamaAnswerProvider
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueLineageError,
    AtomicRescueSelectionError,
    atomic_rescue_expectation_parity,
    atomic_rescue_reference_parity,
    load_atomic_rescue_artifact,
    select_atomic_rescue,
)
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall.embeddings import (
    Embedder,
    HashingEmbedder,
    REMOTE_MODEL_CODE_OPT_IN,
    embedder_artifact_digest,
    embed_query,
    embedding_profile_id,
    resolve_registered_embedder,
    resolve_embedder,
)
from recall.errors import RecallError
from recall.control_plane import ControlPlane
from recall.frontmatter import supersedes_key
from recall.index import Chunker, candidate_files, chunk_text  # noqa: F401  # legacy public import
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import ExtractingLocalObjectReader
from recall.generations import (
    GenerationManager,
    InvalidGenerationTransition,
    NoActiveGeneration,
    UnsafePromotion,
)
from recall.observability import (
    METRICS,
    PerformanceTrace,
    current_performance_trace,
    get_logger,
    performance_trace_scope,
)
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.source_conditioning import (
    SOURCE_CONDITIONING_SHADOW_POLICIES,
    SourceConditioningArtifactError,
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from recall.profiles import (
    FAST_PROFILE,
    QUALITY_PROFILE,
    RetrievalAdmission,
    RetrievalOverloaded,
    RetrievalProfile,
    resolve_retrieval_profile,
)
from recall.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidencePolicy,
    build_evidence_bundle,
    render_evidence_prompt,
)
from recall.explanations import RetrievalExplanation
from recall.graph_first import (
    GraphFirstCandidate,
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
    build_graph_first_candidates,
)
from recall.query_class import route_query, routing_mode
from recall.query_construction import (
    MAX_QUERY_CANDIDATES,
    MAX_QUERY_CHARS as MAX_QUERY_CONSTRUCTION_QUERY_CHARS,
    MAX_QUERY_CONSTRUCTION_ROUNDS,
    QueryConstructionArm,
    QueryConstructionRequest,
    QueryFrame,
    QueryProposal,
    QueryValidation,
    RetrievalSignal,
    build_control_proposals,
    build_original_model_challenge,
    parse_query_frame,
    should_request_original_model_refinement,
    validate_query_proposals,
)
from recall.retriever import RetrievalCandidateTrace
from recall.related import RelatedEvidenceResult, trusted_related
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,
    ReasoningDiagnostics,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    ReasoningResponse,
    SemanticGraphExpansionResult,
    reason,
)
from recall.reasoning_expansion import (
    ExpansionProposal,
    ReasoningExpansionRetriever,
    merge_trusted_results,
    resolve_expansion_provider,
)
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    build_reasoning_graph,
    project_store_graph,  # noqa: F401  # legacy patch seam
)
from recall.reasoning_planner import ReasoningBudget, _reset_planner_index_cache
from recall.semantic_graph import (
    RELATION_KINDS,
    SemanticGraphProjection,
    relation_coverage,
)
from recall.query_entity_resolution import resolve_query_entities
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
    deterministic_inference_proposals,
)
from recall.rerank import (
    COREB_CODE_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    KNOWN_RERANKER_REVISIONS,
    RERANKER_MODEL_ALIASES,
    Reranker,
)
from recall.retriever import DocumentExpansionPolicy, StructuralExpansionPolicy
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder
from recall.trust import decision_state_for, evaluate, is_trusted, resolve_successor, trusted_search
from recall.types import (
    AtomicFact,
    EvidenceCard,
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import factories as _factories
from recall_mcp import reasoning_api as _reasoning_api
from recall_mcp import retrieval as _retrieval
from recall_mcp import graph_first_api as _graph_first_api
from recall_mcp import query_construction_api as _query_construction_api
from recall_mcp.compat import serving_json  # noqa: F401  # legacy public import
from recall_mcp import graph_expansion as _graph_expansion
from recall_mcp import graph_projection as _graph_projection
from recall_mcp.settings import runtime_environment
from recall_mcp.status import (
    JobLedger,  # noqa: F401  # legacy public import
    calibration_status,  # noqa: F401  # legacy public import
    job_status,  # noqa: F401  # legacy public import
)
from recall_mcp.lifecycle import (
    MAX_FORGET_SOURCES,  # noqa: F401  # legacy public import
    current_state_memory,  # noqa: F401  # legacy public import
    forget_memory,  # noqa: F401  # legacy public import
    memory_inventory,  # noqa: F401  # legacy public import
    memory_stats,  # noqa: F401  # legacy public import
    withdraw_memory,  # noqa: F401  # on the service hub beside forget_memory
)
from recall_mcp.indexing import (
    DEFAULT_MAX_INDEX_BYTES,  # noqa: F401  # legacy public import
    DEFAULT_MAX_INDEX_FILES,  # noqa: F401  # legacy public import
    REDACTED_PATH,  # noqa: F401  # legacy public import
    IndexPreflightError,  # noqa: F401  # legacy public import
    _scrub_paths,  # noqa: F401  # legacy public import
    _set_candidate_files_provider,
    index_memory,  # noqa: F401  # legacy public import
)
from recall_mcp.provenance import (
    FACT_WRITE_DSN_ENV,  # noqa: F401  # legacy public import
    _fact_write_dsn,  # noqa: F401  # legacy public import
    apply_fact_memory,  # noqa: F401  # legacy public import
    current_facts_memory,  # noqa: F401  # legacy public import
    register_evidence_cards,  # noqa: F401  # legacy public import
)
from recall_mcp.graph_projection import (
    _authorization_scope,
    _authorized_graph,  # noqa: F401  # legacy public import
    _combined_graph_policy_fingerprint,  # noqa: F401  # legacy public import
    _store_graph,  # noqa: F401  # legacy public import
    _store_graph_with_readiness,  # noqa: F401  # legacy public import
    reasoning_projection,  # noqa: F401  # legacy public import
)
from recall_mcp.reasoning_common import (
    _query_construction_anchors,
    _query_construction_evidence,
    _query_construction_generation,
    _query_construction_hit,
    _query_construction_retrieval,
    _reasoning_generation,
    _same_generation,
)
from recall_mcp.reasoning_admin import (
    _stored_extracted_proposals,  # noqa: F401  # legacy public import
    apply_command_for,  # noqa: F401  # legacy public import
    reasoning_proposals,  # noqa: F401  # legacy public import
    rewrite_plan,  # noqa: F401  # legacy public import
)
from recall_mcp.retrieval import (
    MAX_QUERY_CHARS,  # noqa: F401  # legacy public import
    MAX_SEARCH_K,  # noqa: F401  # legacy public import
    _Retrieval,  # noqa: F401  # legacy public import
    startup_retrieval_profile,  # noqa: F401  # legacy public import
)

# Implementations moved out of this module in S5 (2026-10). Every name stays importable here.
from recall_mcp.semantic_graph_cache import (  # noqa: E402,F401  # re-exported
    GRAPH_COSINE_MARGIN,
    GRAPH_DIAGNOSTIC_ONLY_RELATIONS,
    GRAPH_DIRECTIONAL_RELATIONS,
    GRAPH_FILL_POLICY,
    GRAPH_FILL_SLOT_COUNT,
    GRAPH_FIRST_CONTEXT_K,
    GRAPH_FIRST_RETRIEVAL_K,
    GRAPH_FIRST_SEED_K,
    GRAPH_HUB_DEGREE_THRESHOLD,
    GRAPH_PRECISION_POLICY_VERSION,
    GRAPH_PRECISION_VARIANTS,
    GRAPH_RELATION_CONTROLS,
    GRAPH_RERANK_CORROBORATION_CAP,
    GRAPH_RERANK_WEIGHTS,
    GRAPH_TAIL_REPLACEMENT_MARGIN,
    GRAPH_TAIL_REPLACEMENT_MARGINS,
    MAX_GRAPH_RESCORING_CANDIDATES,
    _DETERMINISTIC_PROPOSAL_CACHE,
    _DETERMINISTIC_PROPOSAL_CACHE_LOCK,
    _DETERMINISTIC_PROPOSAL_CACHE_MAX,
    _DETERMINISTIC_PROPOSAL_INFLIGHT,
    _GRAPH_PROJECTION_LOCK,
    _GraphCandidate,
    _ProposalFlight,
    _SEMANTIC_GRAPH_CACHE,
    _SEMANTIC_GRAPH_CACHE_MAX,
    _SEMANTIC_GRAPH_INDEXES,
    _SEMANTIC_GRAPH_INDEX_CACHE_MAX,
    _SEMANTIC_GRAPH_INFLIGHT,
    _SemanticGraphFlight,
    _SemanticGraphIndexes,
    _assemble_graph_first_context,
    _cached_deterministic_proposals,
    _cached_semantic_graph,
    _expand_semantic_graph,
    _generation_scope,
    _graph_candidate_rerank_score,
    _graph_expansion_dependencies,
    _graph_precision_feature_flags,
    _graph_precision_policy_fingerprint,
    _graph_precision_settings,
    _graph_tail_replacement_margin,
    _merge_graph_hits,
    _proposal_policy_scope,
    _provisional_graph_seed_result,
    _reset_graph_projection_cache,
    _resolve_graph_calibration,
    _retrieval_graph,
    _semantic_graph_indexes,
    _shuffle_graph_relation_endpoints,
    _validate_security_context,
)
from recall_mcp.related_api import (  # noqa: E402,F401  # re-exported
    RelatedResult,
    records_memory,
    related_memory,
)
from recall_mcp.stale_reports_api import report_stale  # noqa: E402,F401  # re-exported
from recall_mcp.use_reports_api import report_use  # noqa: E402,F401  # re-exported
from recall_mcp.reasoning_diagnostics import (  # noqa: E402,F401  # re-exported
    BENCHMARK_DOCUMENT_EXPANSION_CHUNKS,
    BENCHMARK_DOCUMENT_EXPANSION_SOURCES,
    BENCHMARK_RETRIEVAL_LEG_DEPTH,
    _PinnedBenchmarkQueryEmbedder,
    _atomic_rescue_shadow_payload,
    _atomic_rescue_shadow_sampled,
    _benchmark_bundle_payload,
    _benchmark_candidate_identity,
    _benchmark_candidate_rows,
    _benchmark_trusted_pool_payload,
    _document_expansion_benchmark_audit_enabled,
    _document_expansion_benchmark_audit_payload,
    _retrieval_leg_benchmark_audit_enabled,
    _retrieval_leg_benchmark_audit_payload,
    _source_admission_benchmark_audit_enabled,
    _source_admission_benchmark_audit_payload,
    _source_conditioning_reuse_benchmark_audit_enabled,
    _source_conditioning_reused_audits,
    _source_conditioning_shadow_payload,
    _source_conditioning_shadow_sampled,
)
from recall_mcp.reasoning_engine import (  # noqa: E402,F401  # re-exported
    _execute_reasoning_query,
    _strict_reasoning_refusal,
)
from recall_mcp.desktop_ingest import (  # noqa: E402,F401  # re-exported
    _DESKTOP_CORPUS_PREFIX,
    _carry_forward,
    _certify_upload,
    _digest_of,
    _generated_calibration_queries,
    _local_path,
    _query_set_for,
    _reclaim_failed,
    _release_superseded,
    _restamped_note,
    _roots_of,
    _vanished_note,
    generation_ingest,
    publish_calibration,
    run_calibration,
)

# Compatibility aliases for diagnostics and tests that inspected the former service-owned cache.
_GRAPH_PROJECTIONS = _graph_projection._GRAPH_PROJECTIONS
_GRAPH_PROJECTION_INFLIGHT = _graph_projection._GRAPH_PROJECTION_INFLIGHT
_GRAPH_PROJECTION_CACHE_MAX = _graph_projection._GRAPH_PROJECTION_CACHE_MAX

_log = get_logger("mcp.service")

HASHING_DIM = 64  # offline HashingEmbedder width; matches the eval/test default
# Query construction is a two-phase, client-callable protocol. Keep its prompt and graph budgets
# below the broader search limits because every continuation can trigger bounded retrieval work.
MAX_QUERY_CONSTRUCTION_PROMPT_CHARS = _query_construction_api.MAX_QUERY_CONSTRUCTION_PROMPT_CHARS
MAX_QUERY_CONSTRUCTION_GRAPH_NODES = _query_construction_api.MAX_QUERY_CONSTRUCTION_GRAPH_NODES

_set_candidate_files_provider(lambda: candidate_files)


# One implementation of each embedder factory, in `recall_mcp.factories`. This module carried a
# second, drifted copy of both until 2026-10 (different resolver, exception type and environment
# source); the names stay importable from here for compatibility.
from recall_mcp.factories import make_embedder, make_profile_embedder  # noqa: E402,F401


#: Cross-encoder reranking, opt-in via `RECALL_RERANK`.
#:
#: Measured on LOCOMO at n=1,536 (FINDINGS §11): hit@5 **0.671 -> 0.777**, intervals disjoint from
#: the baseline through k=10 — the largest single retrieval gain in this project, and roughly twice
#: the best embedder effect. It closes 57% of the distance to the candidate pool's own ceiling.
#:
#: OFF by default because it costs ~1,050 ms per query on CPU. A memory server that silently
#: quadrupled every query's latency to improve a benchmark would be choosing for the operator.
#: Worth enabling when a human is waiting on the answer; leave it off for high-volume automated
#: retrieval or constrained hardware.
def resolve_reranker(env: dict[str, str] | None = None) -> tuple[str, str | None] | None:
    """Resolve through the production owner, using this service's settings snapshot."""
    return _factories.resolve_reranker(env if env is not None else dict(runtime_environment()))


# Compatibility exports share the factory policy used by the actual reranker builder.
from recall_mcp.factories import (  # noqa: E402
    _positive_env,
    _remote_model_code_enabled,
    _require_remote_model_code_enabled,
    _validate_quality_reranker_config,
)


def _new_reranker(
    env: dict[str, str] | None = None,
    profile: RetrievalProfile | None = None,
) -> "Reranker | None":  # pragma: no cover
    """Instantiate the configured reranker, or None. Imports torch only when actually enabled."""
    if profile is None:
        return _factories._new_reranker(env)
    return _factories._new_reranker(env, profile=profile)


def _reset_reranker_cache() -> None:
    """Drop the per-process reranker. For tests, a server should never need this."""
    _factories._reset_reranker_cache()


def _build_reranker(
    profile: RetrievalProfile | None = None, env: dict[str, str] | None = None
) -> "Reranker | None":
    """Compatibility wrapper for the single reranker owner in ``recall_mcp.factories``."""
    return _factories._build_reranker(profile=profile, env=env, builder=_new_reranker)


def _admission(profile: RetrievalProfile) -> RetrievalAdmission:
    """Compatibility wrapper for the single admission owner in ``recall_mcp.factories``."""
    return _factories._admission(profile)


def _retrieve_trusted(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
    pool_k: int | None = None,
    pre_trust_transform: Callable[[RetrievalResult], RetrievalResult] | None = None,
    query_vector_callback: Callable[[list[float]], None] | None = None,
    capture_candidate_trace: bool = False,
    *,
    paged_depth: bool = False,
) -> _Retrieval:
    """Compatibility adapter for the retrieval execution owner."""
    return _retrieval._retrieve_trusted(
        store,
        embedder,
        query,
        source,
        k,
        calibration,
        policy,
        security_policy,
        access_context,
        env,
        pool_k,
        pre_trust_transform,
        query_vector_callback,
        capture_candidate_trace,
        reranker_builder=_build_reranker,
        admission_factory=_admission,
        trusted_search_fn=trusted_search,
        paged_depth=paged_depth,
    )


# Compatibility aliases for response assembly helpers now owned by retrieval.
_cost_surface = _retrieval._cost_surface
_search_hit_model = _retrieval._search_hit_model
_trusted_evidence_item_model = _retrieval._trusted_evidence_item_model
_evidence_item_model = _retrieval._evidence_item_model
_evidence_card_model = _retrieval._evidence_card_model


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
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> SearchResult:
    """Compatibility wrapper for search response assembly owned by retrieval."""
    return _retrieval.search_memory(
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
        security_policy,
        access_context,
        env,
        _retrieve_trusted_fn=_retrieve_trusted,
        _search_hit_model_fn=_search_hit_model,
        _search_advice_fn=_search_advice,
        _cost_surface_fn=_cost_surface,
        trusted_related_fn=trusted_related,
    )

# Compatibility aliases for advice assembly now owned by retrieval.
UNCALIBRATED_NOTE = _retrieval.UNCALIBRATED_NOTE
STALE_INDEX_NOTE = _retrieval.STALE_INDEX_NOTE
REASONING_BLOCKED_NOTE = _retrieval.REASONING_BLOCKED_NOTE
REASONING_SUPERSEDED_NOTE = _retrieval.REASONING_SUPERSEDED_NOTE
_search_advice = _retrieval._search_advice
_advice_suffixes = _retrieval._advice_suffixes
_evidence_advice = _retrieval._evidence_advice


def evidence_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int | None = None,
    max_items: int | None = None,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> EvidenceResult:
    """Compatibility wrapper for evidence assembly owned by retrieval."""
    return _retrieval.evidence_memory(
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
        security_policy,
        access_context,
        env,
        _retrieve_trusted_fn=_retrieve_trusted,
        _evidence_item_model_fn=_evidence_item_model,
        _evidence_advice_fn=_evidence_advice,
        _cost_surface_fn=_cost_surface,
        _trusted_evidence_item_model_fn=_trusted_evidence_item_model,
        _evidence_card_model_fn=_evidence_card_model,
        trusted_related_fn=trusted_related,
        register_evidence_cards_fn=register_evidence_cards,
    )

def _query_construction_graph(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    retrieval: TrustedResult,
    generation: GenerationSelection,
    calibration: Calibration | None,
    graph_expansion: str,
    max_graph_nodes: int,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> tuple[TrustedResult, dict[str, object]]:
    """Compatibility wrapper for query graph orchestration owned by query_construction_api."""
    return _query_construction_api._query_construction_graph(
        store,
        embedder,
        query,
        retrieval,
        generation,
        calibration,
        graph_expansion,
        max_graph_nodes,
        security_policy=security_policy,
        access_context=access_context,
        _expand_semantic_graph_fn=_expand_semantic_graph,
    )


def graph_first_retrieval(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    *,
    mode: GraphFirstMode = "hybrid",
    source: str | None = None,
    k: int = 5,
    max_candidates: int = MAX_GRAPH_FIRST_CANDIDATES,
    expected_generation_id: str | None = None,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for graph-first retrieval owned by graph_first_api."""
    return _graph_first_api.graph_first_retrieval(
        store,
        embedder,
        query,
        mode=mode,
        source=source,
        k=k,
        max_candidates=max_candidates,
        expected_generation_id=expected_generation_id,
        policy=policy,
        calibration=calibration,
        security_policy=security_policy,
        access_context=access_context,
        _retrieve_trusted_fn=_retrieve_trusted,
        _store_graph_fn=_store_graph,
        _cached_semantic_graph_fn=_cached_semantic_graph,
        _combined_graph_policy_fingerprint_fn=_combined_graph_policy_fingerprint,
        _validate_security_context_fn=_validate_security_context,
    )


def query_construction_challenge(
    store: PgVectorStore,
    embedder: Embedder,
    original_prompt: str,
    query: str,
    *,
    arm: QueryConstructionArm = "original_loop",
    source: str | None = None,
    k: int = 5,
    round_index: int = 0,
    frame: Mapping[str, object] | None = None,
    expected_generation_id: str | None = None,
    graph_expansion: str = "off",
    max_graph_nodes: int = 32,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for query construction owned by query_construction_api."""
    return _query_construction_api.query_construction_challenge(
        store,
        embedder,
        original_prompt,
        query,
        arm=arm,
        source=source,
        k=k,
        round_index=round_index,
        frame=frame,
        expected_generation_id=expected_generation_id,
        graph_expansion=graph_expansion,
        max_graph_nodes=max_graph_nodes,
        policy=policy,
        calibration=calibration,
        security_policy=security_policy,
        access_context=access_context,
        _retrieve_trusted_fn=_retrieve_trusted,
        _query_construction_graph_fn=_query_construction_graph,
    )


reasoning_audit = _reasoning_api.reasoning_audit
reasoning_query = _reasoning_api.reasoning_query


def tenant_scopes(store: PgVectorStore, tenants: Sequence[str]) -> dict[str, object]:
    """Keep tenant metadata shaping behind the authenticated store boundary."""
    return {"tenants": sorted({str(store.tenant), *(str(value) for value in tenants)})}

