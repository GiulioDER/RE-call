from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from recall_mcp.models import (
    EvidenceCardModel,  # noqa: F401  # legacy public import
    EvidenceResult,
    IndexResult,
    ReasoningAuditResult,
    ReasoningProjectionResult,
    ReasoningProposalItem,
    ReasoningProposalResult,
    RewritePlanResult,
    SearchHit,  # noqa: F401  # legacy public import
    SearchResult,
)

from recall.calibration import Calibration
from recall.answer_provider import OllamaAnswerProvider
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall.embeddings import (
    Embedder,
    embed_query,
)
from recall.control_plane import ControlPlane
from recall.index import Chunker, candidate_files, chunk_text
from recall.observability import METRICS
from recall.profiles import (
    FAST_PROFILE,  # noqa: F401  # legacy public import
    QUALITY_PROFILE,  # noqa: F401  # legacy public import
    RetrievalAdmission,
    RetrievalOverloaded,  # noqa: F401  # legacy public import
    RetrievalProfile,
    resolve_retrieval_profile,  # noqa: F401  # legacy public import
)
from recall.evidence import (
    EvidenceBundle,
    EvidencePolicy,  # noqa: F401  # legacy public import
    build_evidence_bundle,  # noqa: F401  # legacy public import
    render_evidence_prompt,  # noqa: F401  # legacy public import
)
from recall.provenance_controller import EvidenceCardStore  # noqa: F401  # legacy public import
from recall.explanations import RetrievalExplanation  # noqa: F401  # legacy public import
from recall.graph_first import (
    GraphFirstCandidate,
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
    build_graph_first_candidates,
)
from recall.query_class import route_query, routing_mode  # noqa: F401  # legacy public import
from recall.query_construction import (
    MAX_QUERY_CANDIDATES,
    MAX_QUERY_CHARS as MAX_QUERY_CONSTRUCTION_QUERY_CHARS,
    MAX_QUERY_CONSTRUCTION_ROUNDS,
    QueryConstructionArm,
    QueryConstructionRequest,
    QueryProposal,
    RetrievalSignal,
    build_control_proposals,
    build_original_model_challenge,
    parse_query_frame,
    should_request_original_model_refinement,
    validate_query_proposals,
)
from recall.related import trusted_related  # noqa: F401  # legacy public import
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,
    ReasoningDiagnostics,
    ReasoningGraphProvider,
    ReasoningPolicy,
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
    merge_trusted_results,
    resolve_expansion_provider,
)
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    build_reasoning_graph,
    project_store_graph,
)
from recall.reasoning_planner import ReasoningBudget
from recall.semantic_graph import SemanticGraphProjection
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
    deterministic_inference_proposals,
)
from recall.rerank import COREB_CODE_RERANKER_MODEL, Reranker  # noqa: F401
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder  # noqa: F401  # legacy public import
from recall.trust import decision_state_for, evaluate, is_trusted, trusted_search
from recall.types import (
    Chunk,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,  # noqa: F401  # legacy public import
)
from recall_mcp import factories as _factories, generation_admin as _generation_admin, retrieval as _retrieval
from recall_mcp.indexing import (
    DEFAULT_MAX_INDEX_BYTES,  # noqa: F401  # legacy public import
    DEFAULT_MAX_INDEX_FILES,  # noqa: F401  # legacy public import
    REDACTED_PATH,  # noqa: F401  # legacy public import
    _scrub_paths,
    index_memory as _index_memory,
)
from recall_mcp.lifecycle import (
    MAX_CURRENT_STATE_RECORDS,  # noqa: F401  # legacy public import
    MAX_FORGET_SOURCES,  # noqa: F401  # legacy public import
    current_state_memory,  # noqa: F401  # legacy public import
    forget_memory,  # noqa: F401  # legacy public import
    memory_inventory,  # noqa: F401  # legacy public import
    memory_stats,  # noqa: F401  # legacy public import
)
from recall_mcp.status import (
    JobLedger,  # noqa: F401  # legacy public import
    calibration_status,  # noqa: F401  # legacy public import
    job_status,  # noqa: F401  # legacy public import
)
from recall_mcp.provenance import (
    FACT_WRITE_DSN_ENV,  # noqa: F401  # legacy public import
    _fact_write_dsn,  # noqa: F401  # legacy public import
    apply_fact_memory,  # noqa: F401  # legacy public import
    current_facts_memory,  # noqa: F401  # legacy public import
)
from recall_mcp.generation_admin import (
    _DESKTOP_CORPUS_PREFIX,  # noqa: F401  # legacy public import
    _carry_forward,  # noqa: F401  # legacy public import
    _certify_upload,  # noqa: F401  # legacy public import
    _digest_of,  # noqa: F401  # legacy public import
    _local_path,  # noqa: F401  # legacy public import
    _query_set_for,  # noqa: F401  # legacy public import
    _reclaim_failed,  # noqa: F401  # legacy public import
    _release_superseded,  # noqa: F401  # legacy public import
    _restamped_note,  # noqa: F401  # legacy public import
    _roots_of,  # noqa: F401  # legacy public import
    _vanished_note,  # noqa: F401  # legacy public import
    generation_ingest,  # noqa: F401  # legacy public import
)
from recall_mcp.factories import (
    _positive_env,  # noqa: F401  # legacy public import
    _require_remote_model_code_enabled,  # noqa: F401  # legacy public import
    _validate_quality_reranker_config,  # noqa: F401  # legacy public import
    make_embedder,  # noqa: F401  # legacy public import
    make_profile_embedder,  # noqa: F401  # legacy public import
    resolve_reranker,  # noqa: F401  # legacy public import
)
from recall_mcp.compat import serving_json  # noqa: F401  # legacy public import
from recall_mcp.retrieval import (
    MAX_QUERY_CHARS,  # noqa: F401  # legacy public import
    MAX_SEARCH_K,  # noqa: F401  # legacy public import
    _Retrieval,
    REASONING_BLOCKED_NOTE,  # noqa: F401  # legacy public import
    REASONING_SUPERSEDED_NOTE,  # noqa: F401  # legacy public import
    STALE_INDEX_NOTE,  # noqa: F401  # legacy public import
    UNCALIBRATED_NOTE,  # noqa: F401  # legacy public import
    _advice_suffixes,  # noqa: F401  # legacy public import
    _cost_surface,
    _evidence_advice,  # noqa: F401  # legacy public import
    related_memory,  # noqa: F401  # legacy public import
    register_evidence_cards,
    startup_retrieval_profile,  # noqa: F401  # legacy public import
)

#: Upper bound on a search query, in characters. `k` bounds the RESULT set; this bounds the
#: WORK, which is a different quantity and the one an attacker controls. `query_sparse` builds a
#: disjunctive tsquery from every distinct lexeme of the query, so server cost scales with the
#: text sent while `RateLimiter` debits exactly one read token regardless of its size. At the
#: defaults (read 120/min, POOL_SIZE 8, statement_timeout 15s) that asymmetry lets one tenant
#: hold every pooled connection on 15-second scans, against the single Postgres every tenant
#: shares — so the blast radius is not confined to the tenant that caused it.
#:
#: 4096 characters is ~1000 words: orders of magnitude above any natural-language question
#: (this project's own 150-question eval set averages 15.9 content terms), so the bound refuses
#: only input that was never a question. Deliberately NOT configurable — an operator who can
#: raise a DoS bound under deadline will, and the ceiling protects co-tenants who had no say.
# Query construction is a two-phase, client-callable protocol. Keep its prompt and graph budgets
# below the broader search limits because every continuation can trigger bounded retrieval work.
MAX_QUERY_CONSTRUCTION_PROMPT_CHARS = 4_000
MAX_QUERY_CONSTRUCTION_GRAPH_NODES = 128
# Cosine reranking may inspect a bounded oversample of structural candidates so a lower-confidence
# relation can still win on query relevance without turning graph expansion into an unbounded query.
MAX_GRAPH_RESCORING_CANDIDATES = 512








































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
_RERANK_TRUE = frozenset({"1", "true", "yes", "on"})
_RERANK_FALSE = frozenset({"", "0", "false", "no", "off"})












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
        reranker_builder=_build_reranker,
        admission_factory=_admission,
        trusted_search_fn=trusted_search,
    )


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
) -> SearchResult:
    """Compatibility wrapper for the retrieval response owner."""
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
        _retrieve_trusted_fn=_retrieve_trusted,
        _trusted_related_fn=trusted_related,
        _cost_surface_fn=_cost_surface,
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
) -> EvidenceResult:
    """Compatibility wrapper for the retrieval evidence owner."""
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
        _retrieve_trusted_fn=_retrieve_trusted,
        _trusted_related_fn=trusted_related,
        _register_evidence_cards_fn=register_evidence_cards,
        _cost_surface_fn=_cost_surface,
    )


def _reasoning_policy(mode: str, graph_expansion: str = "off") -> ReasoningPolicy:
    if mode == "retrieval_only":
        return ReasoningPolicy(name="retrieval_only", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "review_required":
        return ReasoningPolicy(name="review_required", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "proposal_assisted":
        return ReasoningPolicy(name="proposal_assisted", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    if mode == "evidence_assembly":
        return ReasoningPolicy(name="evidence_assembly", graph_expansion=graph_expansion)  # type: ignore[arg-type]
    raise ValueError("unknown reasoning mode")


def _reasoning_generation(store: PgVectorStore) -> GenerationSelection:
    binding = getattr(store, "generation_binding", None)
    if callable(binding):
        payload = binding()
        return GenerationSelection(
            generation_id=str(payload["generation_id"]),
            pipeline_fingerprint=str(payload["pipeline_fingerprint"]),
            corpus_fingerprint=str(payload["corpus_fingerprint"]),
        )
    generation_id = str(getattr(store, "generation_id", "legacy"))
    return GenerationSelection(generation_id=generation_id if generation_id != "legacy" else None)


def _query_construction_generation(generation: GenerationSelection) -> dict[str, object]:
    return {
        "generation_id": generation.generation_id,
        "pipeline_fingerprint": generation.pipeline_fingerprint,
        "corpus_fingerprint": generation.corpus_fingerprint,
    }


def _query_construction_hit(trusted_hit: TrustedHit) -> dict[str, object]:
    chunk = trusted_hit.chunk
    return {
        "chunk_id": chunk.id,
        "source": chunk.source,
        "text": chunk.text[:2_000],
        "score": trusted_hit.cosine,
        "confidence": trusted_hit.confidence,
        "verdict": trusted_hit.verdict,
        "ordinal": trusted_hit.provenance.ord,
    }


def _query_construction_retrieval(result: TrustedResult) -> dict[str, object]:
    return {
        "query": result.query,
        "decision_state": result.decision_state or decision_state_for(
            result.hits, gap_warning=result.gap_warning
        ),
        "abstained": result.abstained,
        "reason": result.reason,
        "gap_warning": result.gap_warning,
        "trust_state": result.trust_state,
        "calibration_status": result.calibration_status,
        "tenant_id": result.tenant_id,
        "generation_id": result.generation_id,
        "pipeline_fingerprint": result.pipeline_fingerprint,
        "corpus_fingerprint": result.corpus_fingerprint,
        "hits": [_query_construction_hit(hit) for hit in result.hits],
    }


def _query_construction_evidence(result: TrustedResult) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "chunk_id": hit.chunk.id,
            "source": hit.chunk.source,
            "text": hit.chunk.text,
            "verdict": hit.verdict,
        }
        for hit in result.hits[:5]
        if is_trusted(hit)
    )


def _query_construction_anchors(result: TrustedResult) -> tuple[str, ...]:
    """Expose only bounded corpus identifiers as graph anchors, never generated text."""

    anchors: list[str] = []
    for hit in result.hits:
        if hit.verdict != "ok":
            continue
        for value in (hit.chunk.source, hit.chunk.id):
            if value and value not in anchors:
                anchors.append(value)
            if len(anchors) >= 8:
                return tuple(anchors)
    return tuple(anchors)


def _same_generation(expected: GenerationSelection, result: TrustedResult) -> None:
    checks = (
        ("generation_id", expected.generation_id, result.generation_id),
        ("pipeline_fingerprint", expected.pipeline_fingerprint, result.pipeline_fingerprint),
        ("corpus_fingerprint", expected.corpus_fingerprint, result.corpus_fingerprint),
    )
    for name, requested, actual in checks:
        if requested is not None and actual != requested:
            raise ValueError(f"retrieval {name} does not match the construction generation")


def _query_construction_graph(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    retrieval: TrustedResult,
    generation: GenerationSelection,
    calibration: Calibration | None,
    graph_expansion: str,
    max_graph_nodes: int,
) -> tuple[TrustedResult, dict[str, object]]:
    if graph_expansion == "off":
        return retrieval, {
            "readiness": "not_requested",
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    graph_request = ReasoningRequest(
        query=query,
        tenant_id=store.tenant,
        generation=generation,
        providers=ReasoningProviderPorts(retriever=lambda _request: retrieval),
        policy=ReasoningPolicy(name="retrieval_only", graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=max_graph_nodes, max_graph_hops=1),
    )
    try:
        expanded = _expand_semantic_graph(
            store, graph_request, retrieval, calibration, embedder
        )
    except Exception as exc:  # BROAD-CATCH: fail-open
        return retrieval, {
            "readiness": "GRAPH_PROVIDER_ERROR",
            "error": type(exc).__name__,
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    return expanded.retrieval, {
        "readiness": expanded.readiness,
        "entities_inspected": expanded.entities_inspected,
        "relations_inspected": expanded.relations_inspected,
        "candidates_discovered": expanded.candidates_discovered,
        "candidates_rejected": expanded.candidates_rejected,
        "diagnostics_encountered": expanded.diagnostics_encountered,
        "latency_ms": expanded.latency_ms,
    }


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
) -> dict[str, object]:
    """Probe bounded graph-derived query seeds before ordinary trusted retrieval."""
    if mode not in {"entity", "relation", "hybrid"}:
        raise ValueError("mode must be 'entity', 'relation', or 'hybrid'")
    if not 1 <= max_candidates <= MAX_GRAPH_FIRST_CANDIDATES:
        raise ValueError(
            f"max_candidates must be between 1 and {MAX_GRAPH_FIRST_CANDIDATES}"
        )
    if not query.strip():
        raise ValueError("query must be non-empty")

    generation = _reasoning_generation(store)
    if expected_generation_id is not None and expected_generation_id != generation.generation_id:
        return {
            "status": "refused",
            "mode": mode,
            "refusal_reason": "generation_mismatch",
            "generation": _query_construction_generation(generation),
            "diagnostics": {"retrieval_calls": 0, "graph": {"readiness": "not_checked"}},
        }

    graph_started = time.perf_counter()
    semantic: SemanticGraphProjection | None = None
    graph_reason: str | None = None
    readiness_reader = getattr(store, "graph_readiness", None)
    loader = getattr(store, "load_semantic_graph", None)
    try:
        readiness = readiness_reader() if callable(readiness_reader) else None
        if callable(loader) and generation.generation_id is not None:
            semantic = cast(SemanticGraphProjection | None, loader(generation.generation_id))
        else:
            semantic = project_store_graph(store, include_text=False).semantic_graph
        if readiness is not None and not readiness.ready:
            graph_reason = "graph_not_ready"
        elif semantic is None:
            graph_reason = "graph_not_ready"
        elif semantic.tenant_id != store.tenant:
            graph_reason = "tenant_mismatch"
        elif generation.generation_id and semantic.generation_id != generation.generation_id:
            graph_reason = "generation_mismatch"
        elif (
            generation.pipeline_fingerprint
            and semantic.pipeline_fingerprint != generation.pipeline_fingerprint
        ):
            graph_reason = "pipeline_mismatch"
        elif (
            generation.corpus_fingerprint
            and semantic.corpus_fingerprint != generation.corpus_fingerprint
        ):
            graph_reason = "corpus_mismatch"
    except Exception as exc:  # BROAD-CATCH: fail-open
        graph_reason = type(exc).__name__
        semantic = None

    graph_candidates: tuple[GraphFirstCandidate, ...] = ()
    if semantic is not None and graph_reason is None:
        graph_candidates = build_graph_first_candidates(
            semantic, query, mode=mode, max_candidates=max_candidates
        )

    baseline = _retrieve_trusted(store, embedder, query, source, k, calibration, policy).result
    baseline = replace(
        baseline,
        tenant_id=baseline.tenant_id or store.tenant,
        generation_id=baseline.generation_id or generation.generation_id,
    )
    _same_generation(generation, baseline)

    candidate_results: list[TrustedResult] = []
    failures: list[str] = []
    for candidate in graph_candidates:
        try:
            result = _retrieve_trusted(
                store, embedder, candidate.query, source, k, calibration, policy
            ).result
            result = replace(
                result,
                tenant_id=result.tenant_id or store.tenant,
                generation_id=result.generation_id or generation.generation_id,
            )
            _same_generation(generation, result)
            candidate_results.append(result)
        except Exception as exc:  # BROAD-CATCH: fail-open
            failures.append(type(exc).__name__)

    merged = merge_trusted_results(baseline, candidate_results, original_query=query)
    merged = replace(
        merged,
        tenant_id=merged.tenant_id or store.tenant,
        generation_id=merged.generation_id or generation.generation_id,
    )
    baseline_ids = {hit.chunk.id for hit in baseline.hits if is_trusted(hit)}
    merged_ids = {hit.chunk.id for hit in merged.hits if is_trusted(hit)}
    return {
        "status": "complete",
        "mode": mode,
        "generation": _query_construction_generation(generation),
        "baseline_retrieval": _query_construction_retrieval(baseline),
        "candidate_queries": [candidate.to_dict() for candidate in graph_candidates],
        "candidate_retrievals": [
            _query_construction_retrieval(result) for result in candidate_results
        ],
        "retrieval": _query_construction_retrieval(merged),
        "new_trusted_chunk_ids": sorted(merged_ids - baseline_ids),
        "diagnostics": {
            "retrieval_calls": 1 + len(candidate_results),
            "model_calls": 0,
            "token_cost": 0,
            "graph": {
                "readiness": "ready" if semantic is not None and graph_reason is None else "not_ready",
                "reason": graph_reason,
                "entities_inspected": len(semantic.entities) if semantic is not None else 0,
                "mentions_inspected": len(semantic.mentions) if semantic is not None else 0,
                "relations_inspected": len(semantic.relations) if semantic is not None else 0,
                "diagnostics_encountered": len(semantic.diagnostics) if semantic is not None else 0,
                "candidates_discovered": len(graph_candidates),
                "candidates_accepted": len(graph_candidates),
                "candidates_rejected": 0,
                "candidate_retrieval_failures": len(failures),
                "latency_ms": round((time.perf_counter() - graph_started) * 1000.0, 3),
            },
            "new_trusted_items": len(merged_ids - baseline_ids),
            "provider_failures": failures,
        },
    }


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
) -> dict[str, object]:
    """Run one stateless phase of original model query construction.

    With no frame, this retrieves the original query and returns a challenge prompt. With a frame,
    it validates the model output, executes the selected bounded controller, and returns either a
    final retrieval result or the next challenge. The original model is always outside this
    service, which keeps the MCP tool deterministic and makes the benchmark replayable.
    """

    if arm not in {"original_loop", "pyramid"}:
        raise ValueError("arm must be 'original_loop' or 'pyramid'")
    if graph_expansion not in {"off", "one_hop"}:
        raise ValueError("graph_expansion must be 'off' or 'one_hop'")
    if not 0 <= round_index < MAX_QUERY_CONSTRUCTION_ROUNDS:
        raise ValueError("round_index must be 0 or 1")
    if not original_prompt.strip():
        raise ValueError("original_prompt must be non-empty")
    if len(original_prompt) > MAX_QUERY_CONSTRUCTION_PROMPT_CHARS:
        raise ValueError("original_prompt is too long")
    if not query.strip():
        raise ValueError("query must be non-empty")
    if len(query) > MAX_QUERY_CONSTRUCTION_QUERY_CHARS:
        raise ValueError("query is too long")
    if not 1 <= max_graph_nodes <= MAX_QUERY_CONSTRUCTION_GRAPH_NODES:
        raise ValueError(
            f"max_graph_nodes must be between 1 and {MAX_QUERY_CONSTRUCTION_GRAPH_NODES}"
        )

    generation = _reasoning_generation(store)
    if expected_generation_id is not None and expected_generation_id != generation.generation_id:
        return {
            "status": "refused",
            "arm": arm,
            "round_index": round_index,
            "refusal_reason": "generation_mismatch",
            "generation": _query_construction_generation(generation),
            "diagnostics": {"retrieval_calls": 0, "challenge_issued": False},
        }

    baseline = _retrieve_trusted(
        store, embedder, query, source, k, calibration, policy
    ).result
    baseline = replace(
        baseline,
        tenant_id=baseline.tenant_id or store.tenant,
        generation_id=baseline.generation_id or generation.generation_id,
    )
    _same_generation(generation, baseline)
    baseline_evidence = _query_construction_evidence(baseline)
    request = QueryConstructionRequest(
        original_prompt=original_prompt,
        original_query=query,
        trusted_evidence=baseline_evidence,
        graph_anchors=_query_construction_anchors(baseline),
        gap_reason=baseline.reason or "retrieval_gap",
        round_index=round_index,
    )

    if frame is None:
        challenge = build_original_model_challenge(request)
        return {
            "status": "challenge",
            "arm": arm,
            "round_index": round_index,
            "challenge_prompt": challenge.prompt,
            "frame_schema": [
                "task_object",
                "intended_action",
                "failure_or_risk",
                "memory_need",
                "artifacts",
                "query",
                "need_more",
            ],
            "generation": _query_construction_generation(generation),
            "retrieval": _query_construction_retrieval(baseline),
            "diagnostics": {
                "retrieval_calls": 1,
                "challenge_issued": True,
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "rejected_candidate_count": 0,
                "original_model_calls": 1,
                "graph": {"readiness": "deferred_until_trusted_seed"},
            },
        }

    try:
        parsed_frame = parse_query_frame(frame)
    except (TypeError, ValueError) as exc:
        return {
            "status": "fallback",
            "arm": arm,
            "round_index": round_index,
            "refusal_reason": "invalid_frame",
            "error": str(exc),
            "generation": _query_construction_generation(generation),
            "retrieval": _query_construction_retrieval(baseline),
            "diagnostics": {
                "retrieval_calls": 1,
                "challenge_issued": False,
                "original_model_calls": 1,
            },
        }

    proposals: tuple[QueryProposal, ...]
    if arm == "original_loop":
        proposals = (
            QueryProposal(
                parsed_frame.query,
                "literal",
                "original model refinement",
                tuple(
                    str(item["chunk_id"])
                    for item in baseline_evidence
                    if item.get("verdict") == "ok"
                ),
            ),
        )
    else:
        proposals = build_control_proposals(
            parsed_frame,
            original_query=query,
            trusted_evidence=baseline_evidence,
        )
    validation = validate_query_proposals(
        QueryConstructionRequest(
            original_prompt=original_prompt,
            original_query=query,
            trusted_evidence=baseline_evidence,
            graph_anchors=_query_construction_anchors(baseline),
            gap_reason=baseline.reason or "retrieval_gap",
            round_index=round_index,
            max_candidates=MAX_QUERY_CANDIDATES,
        ),
        proposals,
    )

    expanded_results: list[TrustedResult] = []
    failures: list[str] = []
    for proposal in validation.accepted:
        try:
            candidate = _retrieve_trusted(
                store, embedder, proposal.query, source, k, calibration, policy
            ).result
            candidate = replace(
                candidate,
                tenant_id=candidate.tenant_id or store.tenant,
                generation_id=candidate.generation_id or generation.generation_id,
            )
            _same_generation(generation, candidate)
            expanded_results.append(candidate)
        except Exception as exc:  # BROAD-CATCH: fail-open
            failures.append(type(exc).__name__)

    merged = merge_trusted_results(baseline, expanded_results, original_query=query)
    merged = replace(
        merged,
        tenant_id=merged.tenant_id or store.tenant,
        generation_id=merged.generation_id or generation.generation_id,
    )
    baseline_ids = {hit.chunk.id for hit in baseline.hits if is_trusted(hit)}
    merged_ids = {hit.chunk.id for hit in merged.hits if is_trusted(hit)}
    new_ids = tuple(sorted(merged_ids - baseline_ids))
    if new_ids:
        graph_result, graph_diagnostics = _query_construction_graph(
            store,
            embedder,
            parsed_frame.query,
            merged,
            generation,
            calibration,
            graph_expansion,
            max_graph_nodes,
        )
    else:
        graph_result = merged
        graph_diagnostics = {
            "readiness": "deferred_until_trusted_seed",
            "entities_inspected": 0,
            "relations_inspected": 0,
            "candidates_discovered": 0,
            "candidates_rejected": 0,
            "diagnostics_encountered": 0,
            "latency_ms": 0.0,
        }
    signal = RetrievalSignal(
        trusted_items=len([hit for hit in graph_result.hits if is_trusted(hit)]),
        new_trusted_items=len(new_ids),
        gap_warning=graph_result.gap_warning or graph_result.abstained,
        agent_says_need_more=parsed_frame.need_more,
    )
    needs_followup = should_request_original_model_refinement(
        signal, round_index=round_index
    )
    response: dict[str, object] = {
        "status": "challenge" if needs_followup else "complete",
        "arm": arm,
        "round_index": round_index,
        "frame": {
            "task_object": parsed_frame.task_object,
            "intended_action": parsed_frame.intended_action,
            "failure_or_risk": parsed_frame.failure_or_risk,
            "memory_need": parsed_frame.memory_need,
            "artifacts": list(parsed_frame.artifacts),
            "query": parsed_frame.query,
            "need_more": parsed_frame.need_more,
        },
        "generation": _query_construction_generation(generation),
        "retrieval": _query_construction_retrieval(graph_result),
        "new_trusted_chunk_ids": list(new_ids),
        "accepted_candidates": [
            {
                "query": proposal.query,
                "kind": proposal.kind,
                "rationale": proposal.rationale,
                "parent_chunk_ids": list(proposal.parent_chunk_ids),
            }
            for proposal in validation.accepted
        ],
        "rejected_candidates": [
            {"query": proposal.query, "kind": proposal.kind, "reason": reason}
            for proposal, reason in validation.rejected
        ],
        "diagnostics": {
            "retrieval_calls": 1 + len(expanded_results),
            "challenge_issued": needs_followup,
            "candidate_count": len(proposals),
            "accepted_candidate_count": len(validation.accepted),
            "rejected_candidate_count": len(validation.rejected),
            "new_trusted_items": len(new_ids),
            "original_model_calls": 1 + (1 if needs_followup else 0),
            "provider_failures": failures,
            "graph": graph_diagnostics,
        },
    }
    if needs_followup:
        followup_request = QueryConstructionRequest(
            original_prompt=original_prompt,
            original_query=parsed_frame.query,
            trusted_evidence=_query_construction_evidence(graph_result),
            graph_anchors=_query_construction_anchors(graph_result),
            gap_reason=graph_result.reason or "retrieval_gap",
            round_index=round_index + 1,
        )
        response["next_challenge_prompt"] = build_original_model_challenge(
            followup_request
        ).prompt
        response["next_round_index"] = round_index + 1
    return response


_GRAPH_PROJECTION_LOCK = threading.Lock()
_GRAPH_PROJECTIONS: dict[tuple[str, str, bool, str | None], ReasoningGraphProjection] = {}
_GRAPH_PROJECTION_CACHE_MAX = 4


def _reset_graph_projection_cache() -> None:
    with _GRAPH_PROJECTION_LOCK:
        _GRAPH_PROJECTIONS.clear()


def _store_graph_with_readiness(
    store: PgVectorStore, *, include_text: bool
) -> tuple[ReasoningGraphProjection, Any]:
    """Project immutable generations once while leaving mutable legacy stores uncached."""
    snapshot = getattr(store, "snapshot", None)
    lookup = getattr(store, "active_generation_id", None)
    if not callable(snapshot) and not callable(lookup):
        return project_store_graph(store, include_text=include_text), None
    scope: AbstractContextManager[Any] = (
        snapshot() if callable(snapshot) else nullcontext(None)
    )
    with scope as pinned:
        if pinned is not None:
            generation_id = str(pinned)
        elif not callable(lookup):
            return project_store_graph(store, include_text=include_text), None
        else:
            generation_id = str(lookup())
        readiness_reader = getattr(store, "graph_readiness", None)
        readiness = readiness_reader() if callable(readiness_reader) else None
        fingerprint = getattr(readiness, "graph_fingerprint", None) if readiness else None
        key = (store.tenant, generation_id, include_text, fingerprint)
        with _GRAPH_PROJECTION_LOCK:
            cached = _GRAPH_PROJECTIONS.get(key)
        if cached is not None:
            return cached, readiness
        graph = project_store_graph(store, include_text=include_text)
        if graph.generation_id != generation_id:
            return graph, readiness
        with _GRAPH_PROJECTION_LOCK:
            if key not in _GRAPH_PROJECTIONS:
                while len(_GRAPH_PROJECTIONS) >= _GRAPH_PROJECTION_CACHE_MAX:
                    _GRAPH_PROJECTIONS.pop(next(iter(_GRAPH_PROJECTIONS)))
            _GRAPH_PROJECTIONS[key] = graph
        return graph, readiness


def _store_graph(store: PgVectorStore, *, include_text: bool) -> ReasoningGraphProjection:
    return _store_graph_with_readiness(store, include_text=include_text)[0]


def reasoning_projection(
    store: PgVectorStore, *, include_text: bool = False
) -> ReasoningProjectionResult:
    graph, readiness = _store_graph_with_readiness(store, include_text=include_text)
    semantic = graph.semantic_graph
    return ReasoningProjectionResult(
        schema_version=graph.schema_version,
        graph_id=graph.graph_id,
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        node_count=len(graph.nodes),
        authored_edge_count=len(graph.authored_edges),
        inferred_candidate_edge_count=len(graph.inferred_candidate_edges),
        diagnostic_count=len(graph.diagnostics),
        trust_state="trusted" if graph.generation_id != "legacy" else "degraded",
        semantic_graph_ready=bool(readiness.ready) if readiness is not None else semantic is not None,
        semantic_graph_reason=getattr(readiness, "reason", None) if readiness is not None else None,
        semantic_entity_count=len(semantic.entities) if semantic is not None else 0,
        semantic_mention_count=len(semantic.mentions) if semantic is not None else 0,
        semantic_relation_count=len(semantic.relations) if semantic is not None else 0,
        semantic_diagnostic_count=len(semantic.diagnostics) if semantic is not None else 0,
    )








def apply_command_for(claim: str) -> str:
    """The exact CLI command that declares `claim`.

    A function rather than an inline f-string so a test can assert on the VALUE. Asserting on
    this module's SOURCE does not work: the surrounding comment explains why a proposal id
    cannot be handed off, and that explanation contains the very flag name being ruled out.
    """
    return (
        f"recall rewrite apply <corpus> --claim {claim} "
        f"--reviewer <your-id> --note <why> --apply"
    )


def rewrite_plan(store: PgVectorStore, *, proposal_id: str) -> RewritePlanResult:
    """Describe what declaring `proposal_id` would write, without writing anything.

    Read only by construction: it routes the relation and reports the result. It never
    constructs a `PromotedFact`, never touches a file, and imports nothing that writes.
    """
    from recall.rewrite import claim_key, destination, route_relation

    graph = project_store_graph(store, include_text=True)
    proposals = deterministic_inference_proposals(
        graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
    )
    found = next((p for p in proposals if p.id == proposal_id), None)
    if found is None:
        # The id is echoed because the caller supplied it; nothing about the corpus leaks.
        raise ValueError(f"no proposal {proposal_id!r} in this generation")
    routed = route_relation(found.proposed_relation, found.subject_id, found.object_id)
    # The CLAIM key, not the proposal id, is what crosses to the CLI. This tool's proposals come
    # from the deterministic rules over the STORE graph; `recall rewrite apply --proposal`
    # resolves ids against the filesystem extractor. Provider, tenant, generation and pipeline
    # are hashed into an id, so the two id spaces are disjoint by construction and every id this
    # tool emitted was one the CLI exits 2 on. Claim keys are generation independent and match.
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


def _stored_extracted_proposals(graph: object) -> tuple[object, ...]:
    """Replay extractions recorded at ingest into the proposal protocol.

    Refuses, because there is nothing to replay. `FileExtraction` is persisted nowhere the query
    path can read: `recall/truth_extraction/_cache.py` defines `ExtractionCache` as a Protocol
    with no shipped database implementation, and no module outside `recall.truth_extraction` and
    `recall.reasoning_proposals._extracted` references the type at all.

    An empty tuple would be the obvious stub and the wrong one. `--include-extracted` would then
    report "0 proposals", which a caller reads as *the extractor ran and found nothing* when the
    truth is *nothing was ever recorded*, and those two call for opposite responses from whoever
    asked. Refusing says which one it is.

    This never builds an engine. Extraction runs on the INGEST path, and constructing one here
    would put a model backed component on the query path, where `max_model_calls` is 0.
    """
    raise ValueError(
        "no extraction record exists for this generation. Run `recall extract run <path>` on "
        "the ingest side first; extraction never runs on the query path."
    )


def reasoning_proposals(
    store: PgVectorStore, *, limit: int = 100, include_extracted: bool = False
) -> ReasoningProposalResult:
    if limit < 1:
        raise ValueError("proposal limit must be positive")
    graph = project_store_graph(store, include_text=True)
    proposals = deterministic_inference_proposals(
        graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
    )
    if include_extracted:
        # Mirrors `include_text`: defaulting to False keeps existing behaviour byte identical,
        # so no caller that did not ask for this sees any change.
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


def _retrieval_graph(
    retrieval: TrustedResult, *, include_text: bool = True
) -> ReasoningGraphProjection:
    chunks = [hit.chunk for hit in retrieval.hits if hit.verdict == "ok"]
    return build_reasoning_graph(
        chunks,
        tenant_id=retrieval.tenant_id or "default",
        generation_id=retrieval.generation_id or "legacy",
        pipeline_fingerprint=retrieval.pipeline_fingerprint,
        corpus_fingerprint=retrieval.corpus_fingerprint,
        include_text=include_text,
    )


def _expand_semantic_graph(
    store: PgVectorStore,
    request: ReasoningRequest,
    retrieval: TrustedResult,
    calibration: Calibration | None,
    embedder: Embedder,
) -> SemanticGraphExpansionResult:
    """Expand trusted seeds through one persisted semantic hop and re-run trust evaluation."""
    started = time.perf_counter()
    readiness_reader = getattr(store, "graph_readiness", None)
    readiness = readiness_reader() if callable(readiness_reader) else None
    graph = project_store_graph(store, include_text=True)
    semantic = graph.semantic_graph
    if semantic is None or (readiness is not None and not readiness.ready):
        return SemanticGraphExpansionResult(
            retrieval=retrieval,
            readiness="GRAPH_NOT_READY",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    trusted_seed_ids = {hit.chunk.id for hit in retrieval.hits if is_trusted(hit)}
    mentions_by_chunk: dict[str, set[str]] = {}
    chunks_by_entity: dict[str, set[str]] = {}
    for mention in semantic.mentions:
        mentions_by_chunk.setdefault(mention.chunk_id, set()).add(mention.entity_id)
        chunks_by_entity.setdefault(mention.entity_id, set()).add(mention.chunk_id)
    ambiguous_entities = {
        entity_id
        for diagnostic in semantic.diagnostics
        if diagnostic.kind == "ambiguous_entity"
        for entity_id in diagnostic.entity_ids
    }
    seed_entities = {
        entity_id
        for chunk_id in trusted_seed_ids
        for entity_id in mentions_by_chunk.get(chunk_id, ())
        if entity_id not in ambiguous_entities
    }

    relation_rank: dict[str, tuple[float, int, str]] = {}
    relation_count = 0
    for relation in semantic.relations:
        if relation.status != "authored":
            continue
        if relation.subject_id in ambiguous_entities or relation.object_id in ambiguous_entities:
            continue
        # A mention of an entity is not enough to activate every relation attached to it. The
        # relation itself must be evidenced by one of the trusted seed chunks. Otherwise a common
        # entity acts as a hub and leaks unrelated documents into the answer bundle.
        if not set(relation.evidence_chunk_ids).intersection(trusted_seed_ids):
            continue
        if relation.subject_id not in seed_entities and relation.object_id not in seed_entities:
            continue
        relation_count += 1
        neighbor = (
            relation.object_id
            if relation.subject_id in seed_entities
            else relation.subject_id
        )
        support_ids = chunks_by_entity.get(neighbor, set())
        for chunk_id in support_ids:
            if chunk_id in trusted_seed_ids:
                continue
            rank = (float(relation.confidence), len(support_ids), chunk_id)
            if rank > relation_rank.get(chunk_id, (-1.0, -1, "")):
                relation_rank[chunk_id] = rank

    graph_candidate_ids = tuple(
        sorted(
            relation_rank,
            key=lambda chunk_id: (
                -relation_rank[chunk_id][0],
                -relation_rank[chunk_id][1],
                chunk_id,
            ),
        )
    )
    query_vector = embed_query(embedder, request.query)
    max_candidates = max(0, request.budget.max_graph_nodes - len(trusted_seed_ids))
    # Score every structural candidate before applying the node budget.  The relation ordering is
    # only a tie-breaker; truncating it before cosine scoring can discard a lower-confidence relation
    # whose evidence is more relevant to the query than the first structural candidates.
    score_limit = min(
        len(graph_candidate_ids),
        min(MAX_GRAPH_RESCORING_CANDIDATES, max_candidates * 4),
    )
    query_scores = store.cosines_for(graph_candidate_ids[:score_limit], query_vector)
    ordered_candidate_ids = tuple(
        sorted(
            (chunk_id for chunk_id in graph_candidate_ids if chunk_id in query_scores),
            key=lambda chunk_id: (
                -query_scores[chunk_id],
                -relation_rank[chunk_id][0],
                -relation_rank[chunk_id][1],
                chunk_id,
            ),
        )
    )
    bounded_ids = ordered_candidate_ids[:max_candidates]
    node_by_chunk = {
        node.chunk_id: node
        for node in graph.nodes
        if node.kind == "chunk" and node.chunk_id is not None
    }
    scored: list[ScoredChunk] = []
    for chunk_id in bounded_ids:
        node = node_by_chunk.get(chunk_id)
        text = node.metadata.get("_recall_evidence_text") if node is not None else None
        if node is None or not isinstance(text, str):
            continue
        metadata = dict(node.metadata)
        metadata.pop("_recall_evidence_text", None)
        scored.append(
            ScoredChunk(
                chunk=Chunk(chunk_id, node.source, text, metadata),
                # Trust calibration is fitted on query dense cosine. Relation confidence is
                # structural metadata and must never stand in for query relevance here.
                score=query_scores[chunk_id],
            )
        )

    active_calibration = calibration
    if active_calibration is None:
        resolver = getattr(store, "resolve_calibration", None)
        if callable(resolver):
            resolution = resolver()
            artifact = getattr(resolution, "artifact", None)
            if artifact is not None:
                active_calibration = artifact.runtime
    supersession: dict[str, str] = {}
    unresolved: frozenset[str] = frozenset()
    if scored:
        supersession, unresolved = store.supersession()
    candidate_result = RetrievalResult(
        query=retrieval.query,
        hits=scored,
        gap_warning=False,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
    )
    generation_binding: dict[str, str] = {
        "tenant_id": retrieval.tenant_id or store.tenant,
        "generation_id": retrieval.generation_id or graph.generation_id or "",
        "pipeline_fingerprint": retrieval.pipeline_fingerprint or graph.pipeline_fingerprint or "",
        "corpus_fingerprint": retrieval.corpus_fingerprint or graph.corpus_fingerprint or "",
    }
    evaluated = evaluate(
        candidate_result,
        supersession,
        active_calibration,
        datetime.now(UTC),
        unresolved,
        calibration_id=retrieval.calibration_id,
        calibration_status=retrieval.calibration_status,
        generation_binding=generation_binding,
        query_set_digest=retrieval.query_set_digest,
    )
    accepted = [hit for hit in evaluated.hits if is_trusted(hit)]
    accepted_ids = {hit.chunk.id for hit in accepted}
    merged = list(retrieval.hits)
    merged.extend(hit for hit in accepted if hit.chunk.id not in {item.chunk.id for item in merged})
    expanded = replace(
        retrieval,
        hits=merged,
        abstained=not any(is_trusted(hit) for hit in merged),
        reason="" if any(is_trusted(hit) for hit in merged) else evaluated.reason,
    )
    rejected = len(ordered_candidate_ids) - len(accepted_ids)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    METRICS.increment("recall_graph_query_total")
    METRICS.increment("recall_graph_expansion_total")
    METRICS.increment("recall_graph_candidates_total", value=len(ordered_candidate_ids))
    METRICS.increment("recall_graph_rejected_candidates_total", value=max(0, rejected))
    METRICS.increment("recall_graph_diagnostics_total", value=len(semantic.diagnostics))
    METRICS.observe("recall_graph_latency_ms", latency_ms)
    return SemanticGraphExpansionResult(
        retrieval=expanded,
        readiness="ready",
        entities_inspected=len(seed_entities),
        relations_inspected=relation_count,
        candidates_discovered=len(ordered_candidate_ids),
        candidates_rejected=max(0, rejected),
        diagnostics_encountered=len(semantic.diagnostics),
        latency_ms=latency_ms,
    )


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

    def execute() -> ReasoningResponse:
        generation = _reasoning_generation(store)
        retrieval_cache: dict[str, TrustedResult] = {}

        def retrieve(request: ReasoningRequest) -> TrustedResult:
            del request
            if "result" not in retrieval_cache:
                result = _retrieve_trusted(
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
                return _retrieval_graph(retrieval, include_text=True)
            return project_store_graph(store, include_text=True)

        def proposal_provider(
            request: ReasoningRequest,
            graph: ReasoningGraphProjection,
            retrieval: TrustedResult,
        ) -> Sequence[InferenceProposal] | ProposalProtocolReport:
            del request, retrieval
            return deterministic_inference_proposals(
                graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
            )

        def graph_expansion_provider(
            request: ReasoningRequest, retrieval: TrustedResult
        ) -> SemanticGraphExpansionResult:
            return _expand_semantic_graph(store, request, retrieval, calibration, embedder)

        expansion_provider = resolve_expansion_provider() if expand_retrieval else None

        def expansion_retriever(
            request: ReasoningRequest,
            proposal: ExpansionProposal,
            initial: TrustedResult,
        ) -> TrustedResult:
            del request, initial
            expanded = _retrieve_trusted(
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
            return reason(request)
        except TrustRefusal as exc:
            return _strict_reasoning_refusal(
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
) -> ReasoningAuditResult:
    projection = reasoning_projection(store, include_text=False)
    proposals = reasoning_proposals(store)
    response = reasoning_query(
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
            # `policy is not None` was a proxy for "a relaxed policy was supplied", and it held only
            # while callers passed a policy exclusively to relax the gate. Once the server resolves
            # one from the environment and always passes it, that proxy is constant-True and the
            # field stops meaning anything. Ask the policy what it is instead of inferring it from
            # whether it exists.
            "development_mode_explicit": (
                (policy is not None and not policy.strict) or response.trust_state == "trusted"
            ),
        },
    )


def index_memory(
    store: PgVectorStore,
    embedder: Embedder,
    path: str,
    on_measured: Callable[[int, int], None] | None = None,
    shadow_store: PgVectorStore | None = None,
    shadow_embedder: Embedder | None = None,
    control_plane: ControlPlane | None = None,
    glob: str | None = None,
    chunker: Chunker = chunk_text,
) -> IndexResult:
    """Compatibility wrapper for the local filesystem indexing owner."""
    return _index_memory(
        store,
        embedder,
        path,
        on_measured,
        shadow_store,
        shadow_embedder,
        control_plane,
        glob,
        chunker,
        _candidate_files_fn=candidate_files,
        _scrub_paths_fn=_scrub_paths,
    )








def tenant_scopes(store: PgVectorStore, tenants: Sequence[str]) -> dict[str, object]:
    """Keep tenant metadata shaping behind the authenticated store boundary."""
    return {"tenants": sorted({str(store.tenant), *(str(value) for value in tenants)})}




def _generated_calibration_queries(store: PgVectorStore, generation_id: str) -> list[dict[str, object]]:
    """Compatibility wrapper for generation administration's query generator."""
    return _generation_admin._generated_calibration_queries(store, generation_id)


def run_calibration(
    store: PgVectorStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for the generation calibration owner."""
    return _generation_admin.run_calibration(
        store,
        embedder,
        generation_id,
        queries,
        _generated_calibration_queries_fn=_generated_calibration_queries,
    )


def publish_calibration(store: PgVectorStore, calibration_id: str) -> dict[str, object]:
    """Compatibility wrapper for the generation calibration owner."""
    return _generation_admin.publish_calibration(store, calibration_id)
