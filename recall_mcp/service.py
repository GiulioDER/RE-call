from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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
from recall.embeddings import Embedder
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
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
)
from recall.query_class import route_query, routing_mode  # noqa: F401  # legacy public import
from recall.related import trusted_related  # noqa: F401  # legacy public import
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,
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
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    project_store_graph,
)
from recall.reasoning_planner import ReasoningBudget
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
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
    _reasoning_generation,
    _reasoning_policy,
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
