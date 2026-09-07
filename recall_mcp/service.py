from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from recall_mcp.models import (
    EvidenceCardModel,  # noqa: F401  # legacy public import
    EvidenceItemModel,  # noqa: F401  # legacy public import
    EvidenceResult,
    IndexResult,
    ReasoningAuditResult,
    ReasoningProjectionResult,
    ReasoningProposalItem,  # noqa: F401  # legacy public import
    ReasoningProposalResult,
    RewritePlanResult,
    SearchHit,  # noqa: F401  # legacy public import
    SearchResult,
)

from recall.calibration import Calibration
from recall.answer_provider import OllamaAnswerProvider
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall.embeddings import Embedder
from recall.control_plane import ControlPlane
from recall.index import Chunker, candidate_files, chunk_text
from recall.profiles import (
    FAST_PROFILE,  # noqa: F401  # legacy public import
    QUALITY_PROFILE,  # noqa: F401  # legacy public import
    RetrievalAdmission,
    RetrievalOverloaded,  # noqa: F401  # legacy public import
    RetrievalProfile,
    resolve_retrieval_profile,  # noqa: F401  # legacy public import
)
from recall.evidence import (
    EvidenceBundle,  # noqa: F401  # legacy public import
    EvidencePolicy,  # noqa: F401  # legacy public import
    build_evidence_bundle,  # noqa: F401  # legacy public import
    render_evidence_prompt,  # noqa: F401  # legacy public import
)
from recall.provenance_controller import EvidenceCardStore  # noqa: F401  # legacy public import
from recall.explanations import RetrievalExplanation  # noqa: F401  # legacy public import
from recall.graph_first import (
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
)
from recall.query_class import route_query, routing_mode  # noqa: F401  # legacy public import
from recall.related import trusted_related  # noqa: F401  # legacy public import
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,  # noqa: F401  # legacy public import
    ReasoningDiagnostics,  # noqa: F401  # legacy public import
    ReasoningGraphProvider,  # noqa: F401  # legacy public import
    ReasoningPolicy,  # noqa: F401  # legacy public import
    ReasoningProposalProvider,  # noqa: F401  # legacy public import
    ReasoningProviderPorts,  # noqa: F401  # legacy public import
    ReasoningRequest,  # noqa: F401  # legacy public import
    ReasoningResponse,
    ReasoningRetriever,  # noqa: F401  # legacy public import
    SemanticGraphExpansionResult,  # noqa: F401  # legacy public import
    reason,
)
from recall.reasoning_expansion import (
    ExpansionProposal,  # noqa: F401  # legacy public import
    ReasoningExpansionRetriever,  # noqa: F401  # legacy public import
    resolve_expansion_provider,
)
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    project_store_graph,
)
from recall.reasoning_planner import ReasoningBudget
from recall.reasoning_proposals import (
    InferenceProposal,  # noqa: F401  # legacy public import
    ProposalProtocolReport,  # noqa: F401  # legacy public import
    deterministic_inference_proposals,
)
from recall.rerank import COREB_CODE_RERANKER_MODEL, Reranker  # noqa: F401
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder  # noqa: F401  # legacy public import
from recall.trust import trusted_search
from recall.types import TrustedResult  # noqa: F401  # legacy public import
from recall_mcp import (
    factories as _factories,
    generation_admin as _generation_admin,
    graph_projection as _graph_projection,
    query_construction_api as _query_construction,
    reasoning_api as _reasoning_api,
    reasoning_admin as _reasoning_admin,
    retrieval as _retrieval,
)
from recall_mcp.graph_first_api import graph_first_retrieval as _graph_first_retrieval
from recall_mcp.graph_expansion import (
    MAX_GRAPH_RESCORING_CANDIDATES,  # noqa: F401  # legacy public import
    _expand_semantic_graph,
    _retrieval_graph,
)
from recall_mcp.query_construction_api import (
    MAX_QUERY_CANDIDATES,  # noqa: F401  # legacy public import
    MAX_QUERY_CONSTRUCTION_GRAPH_NODES,  # noqa: F401  # legacy public import
    MAX_QUERY_CONSTRUCTION_PROMPT_CHARS,  # noqa: F401  # legacy public import
    MAX_QUERY_CONSTRUCTION_QUERY_CHARS,  # noqa: F401  # legacy public import
    MAX_QUERY_CONSTRUCTION_ROUNDS,  # noqa: F401  # legacy public import
    QueryConstructionArm,  # noqa: F401  # legacy public import
    QueryConstructionRequest,  # noqa: F401  # legacy public import
    QueryProposal,  # noqa: F401  # legacy public import
    RetrievalSignal,  # noqa: F401  # legacy public import
    build_control_proposals,  # noqa: F401  # legacy public import
    build_original_model_challenge,  # noqa: F401  # legacy public import
    parse_query_frame,  # noqa: F401  # legacy public import
    should_request_original_model_refinement,  # noqa: F401  # legacy public import
    validate_query_proposals,  # noqa: F401  # legacy public import
)
from recall_mcp.reasoning_common import (
    _query_construction_anchors,  # noqa: F401  # legacy public import
    _query_construction_evidence,  # noqa: F401  # legacy public import
    _query_construction_generation,  # noqa: F401  # legacy public import
    _query_construction_retrieval,  # noqa: F401  # legacy public import
    _reasoning_generation,  # noqa: F401  # legacy public import
    _reasoning_policy,  # noqa: F401  # legacy public import
    _same_generation,  # noqa: F401  # legacy public import
)
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
    return _query_construction._query_construction_graph(
        store,
        embedder,
        query,
        retrieval,
        generation,
        calibration,
        graph_expansion,
        max_graph_nodes,
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
) -> dict[str, object]:
    """Compatibility wrapper for graph-first retrieval orchestration."""
    return _graph_first_retrieval(
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
        _retrieve_trusted_fn=_retrieve_trusted,
        _project_store_graph_fn=project_store_graph,
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
) -> dict[str, object]:
    """Compatibility wrapper for the stateless query construction owner."""
    return _query_construction.query_construction_challenge(
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
        _retrieve_trusted_fn=_retrieve_trusted,
        _query_construction_graph_fn=_query_construction_graph,
    )


_GRAPH_PROJECTION_LOCK = _graph_projection._GRAPH_PROJECTION_LOCK
_GRAPH_PROJECTIONS = _graph_projection._GRAPH_PROJECTIONS
_GRAPH_PROJECTION_CACHE_MAX = _graph_projection._GRAPH_PROJECTION_CACHE_MAX


def _reset_graph_projection_cache() -> None:
    _graph_projection._reset_graph_projection_cache()


def _store_graph_with_readiness(
    store: PgVectorStore, *, include_text: bool
) -> tuple[ReasoningGraphProjection, Any]:
    return _graph_projection._store_graph_with_readiness(
        store,
        include_text=include_text,
        _project_store_graph_fn=project_store_graph,
    )


def _store_graph(store: PgVectorStore, *, include_text: bool) -> ReasoningGraphProjection:
    return _graph_projection._store_graph(
        store,
        include_text=include_text,
        _project_store_graph_fn=project_store_graph,
    )


def reasoning_projection(
    store: PgVectorStore, *, include_text: bool = False
) -> ReasoningProjectionResult:
    return _graph_projection.reasoning_projection(
        store,
        include_text=include_text,
        _project_store_graph_fn=project_store_graph,
    )

def apply_command_for(claim: str) -> str:
    """Compatibility wrapper for the deterministic rewrite command owner."""
    return _reasoning_admin.apply_command_for(claim)


def rewrite_plan(store: PgVectorStore, *, proposal_id: str) -> RewritePlanResult:
    """Compatibility wrapper for the read only rewrite plan owner."""
    return _reasoning_admin.rewrite_plan(
        store,
        proposal_id=proposal_id,
        _project_store_graph_fn=project_store_graph,
        _deterministic_inference_proposals_fn=deterministic_inference_proposals,
        _apply_command_for_fn=apply_command_for,
    )


def _stored_extracted_proposals(graph: object) -> tuple[object, ...]:
    return _reasoning_admin.stored_extracted_proposals(graph)


def reasoning_proposals(
    store: PgVectorStore, *, limit: int = 100, include_extracted: bool = False
) -> ReasoningProposalResult:
    return _reasoning_admin.reasoning_proposals(
        store,
        limit=limit,
        include_extracted=include_extracted,
        _project_store_graph_fn=project_store_graph,
        _deterministic_inference_proposals_fn=deterministic_inference_proposals,
        _stored_extracted_proposals_fn=_stored_extracted_proposals,
    )




def _strict_reasoning_refusal(
    refusal: TrustRefusal,
    *,
    tenant_id: str,
    generation: GenerationSelection,
    budget: ReasoningBudget,
) -> ReasoningResponse:
    return _reasoning_api._strict_reasoning_refusal(
        refusal,
        tenant_id=tenant_id,
        generation=generation,
        budget=budget,
    )


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
    return _reasoning_api.reasoning_query(
        store,
        embedder,
        query,
        source=source,
        k=k,
        mode=mode,
        max_steps=max_steps,
        max_graph_nodes=max_graph_nodes,
        max_evidence_tokens=max_evidence_tokens,
        expand_retrieval=expand_retrieval,
        graph_expansion=graph_expansion,
        answer_provider=answer_provider,
        policy=policy,
        calibration=calibration,
        _reason_fn=reason,
        _resolve_expansion_provider_fn=resolve_expansion_provider,
        _retrieve_trusted_fn=_retrieve_trusted,
        _retrieval_graph_fn=_retrieval_graph,
        _project_store_graph_fn=project_store_graph,
        _deterministic_inference_proposals_fn=deterministic_inference_proposals,
        _expand_semantic_graph_fn=_expand_semantic_graph,
        _strict_reasoning_refusal_fn=_strict_reasoning_refusal,
    )


def reasoning_audit(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    query: str = "reasoning audit sentinel",
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
) -> ReasoningAuditResult:
    return _reasoning_api.reasoning_audit(
        store,
        embedder,
        query=query,
        policy=policy,
        calibration=calibration,
        _reasoning_projection_fn=reasoning_projection,
        _reasoning_proposals_fn=reasoning_proposals,
        _reasoning_query_fn=reasoning_query,
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
