"""Semantic graph serving for MCP: graph-first candidates, precision policy, the bounded
semantic graph and proposal caches, and graph expansion.

Moved out of `recall_mcp.service`, which re-exports every name defined here. Names that tests
monkeypatch on `recall_mcp.service` are reached through `_svc()` at call time, so those patches
keep applying to this code.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import (
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractContextManager,
    nullcontext,
)
from dataclasses import (
    dataclass,
    replace,
)
from datetime import (
    datetime,
    UTC,
)

from recall.calibration import Calibration
from recall.embeddings import (
    embed_query,
    Embedder,
)
from recall.frontmatter import supersedes_key
from recall.observability import (
    current_performance_trace,
    get_logger as _get_logger,
    METRICS,
)
from recall.query_entity_resolution import resolve_query_entities
from recall.reasoning import (
    ReasoningRequest,
    SemanticGraphExpansionResult,
)
from recall.reasoning_graph import (
    build_reasoning_graph,
    ReasoningGraphProjection,
)
from recall.reasoning_planner import _reset_planner_index_cache
from recall.reasoning_proposals import InferenceProposal
from recall.security_policy import (
    AccessContext,
    SourceSecurityPolicy,
)
from recall.semantic_graph import (
    RELATION_KINDS,
    SemanticGraphProjection,
)
from recall.store import PgVectorStore
from recall.trust import resolve_successor
from recall.types import (
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
)

from recall_mcp import (
    graph_expansion as _graph_expansion,
    graph_projection as _graph_projection,
)
from recall_mcp.graph_projection import (
    _authorization_scope,
    _combined_graph_policy_fingerprint,
)
from typing import (
    Any,
    cast,
)
import hashlib
import os
import random
import threading


_log = _get_logger("mcp.service")


def _svc() -> Any:
    """`recall_mcp.service`, looked up at call time so its monkeypatch seams keep applying."""
    from recall_mcp import service

    return service


# Cosine reranking may inspect a bounded oversample of structural candidates so a lower-confidence
# relation can still win on query relevance without turning graph expansion into an unbounded query.
MAX_GRAPH_RESCORING_CANDIDATES = 512

# Graph first intentionally follows the successful LoCoMo context shape. The retrieval pool is
# wider than the final context, eight direct items are protected, and graph candidates can fill
# the remaining two slots after all bounded candidates have been rescored.
GRAPH_FIRST_RETRIEVAL_K = 20
GRAPH_FIRST_SEED_K = 8
GRAPH_FIRST_CONTEXT_K = 10

GRAPH_PRECISION_POLICY_VERSION = "semantic_graph_precision_v2"
GRAPH_DIRECTIONAL_RELATIONS = frozenset(
    {"supports", "references", "depends_on", "caused", "supersedes"}
)
GRAPH_DIAGNOSTIC_ONLY_RELATIONS = frozenset({"contradicts", "same_entity"})
GRAPH_HUB_DEGREE_THRESHOLD = 32
# Kept as a compatibility setting for old diagnostic runners. It no longer rejects candidates.
GRAPH_COSINE_MARGIN = 0.10
GRAPH_RERANK_WEIGHTS = (0.60, 0.20, 0.10, 0.10)
GRAPH_RERANK_CORROBORATION_CAP = 2
GRAPH_FILL_POLICY = "direct_first_fill_missing"
GRAPH_FILL_SLOT_COUNT = 5
GRAPH_TAIL_REPLACEMENT_MARGIN = 0.05
GRAPH_TAIL_REPLACEMENT_MARGINS = frozenset({0.05, 0.10, 0.15, 0.20})
GRAPH_PRECISION_VARIANTS = frozenset(
    {
        "baseline",
        "directional",
        "corroboration",
        "hub",
        "cosine",
        "selective",
        "combined",
        "combined_no_selective",
    }
)
GRAPH_RELATION_CONTROLS = frozenset({"none", "shuffled", "removed"})


class _SemanticGraphFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: SemanticGraphProjection | None = None
        self.error: BaseException | None = None


_GRAPH_PROJECTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class _SemanticGraphIndexes:
    """Immutable adjacency indexes derived from one persisted graph generation."""

    mentions_by_chunk: Mapping[str, frozenset[str]]
    chunks_by_entity: Mapping[str, frozenset[str]]
    relation_indexes_by_entity: Mapping[str, frozenset[int]]
    ambiguous_entities: frozenset[str]
    validity_by_chunk: Mapping[str, tuple[datetime | None, datetime | None]]


_GraphCandidate = _graph_expansion.GraphCandidate


def _resolve_graph_calibration(
    store: PgVectorStore,
    request: ReasoningRequest,
    calibration: Calibration | None,
) -> Calibration | None:
    """Resolve the generation calibration used by graph selection when none was supplied."""
    if calibration is not None:
        return calibration
    resolver = getattr(store, "resolve_calibration", None)
    if not callable(resolver):
        return None
    with _generation_scope(store, request.generation.generation_id):
        resolution = resolver()
    artifact = getattr(resolution, "artifact", None)
    return artifact.runtime if artifact is not None else None


def _graph_candidate_rerank_score(
    candidate: _GraphCandidate,
    cosine: float,
    calibration: Calibration | None,
) -> float:
    return _graph_expansion.graph_candidate_rerank_score(candidate, cosine, calibration)


def _merge_graph_hits(
    retrieval: TrustedResult,
    accepted: Sequence[TrustedHit],
    candidate_scores: Mapping[str, float],
    calibration: Calibration | None,
    max_items: int = GRAPH_FILL_SLOT_COUNT,
    tail_replacement_margin: float | None = None,
) -> list[TrustedHit]:
    return _graph_expansion.merge_graph_hits(
        retrieval,
        accepted,
        candidate_scores,
        calibration,
        max_items=max_items,
        tail_replacement_margin=tail_replacement_margin,
    )


def _assemble_graph_first_context(
    retrieval: RetrievalResult,
    graph_candidates: Sequence[ScoredChunk],
    *,
    seed_k: int = GRAPH_FIRST_SEED_K,
    context_k: int = GRAPH_FIRST_CONTEXT_K,
    calibration: Calibration | None = None,
    tail_replacement_margin: float | None = None,
) -> RetrievalResult:
    return _graph_expansion.assemble_graph_first_context(
        retrieval,
        graph_candidates,
        seed_k=seed_k,
        context_k=context_k,
        calibration=calibration,
        tail_replacement_margin=tail_replacement_margin,
    )


def _provisional_graph_seed_result(
    retrieval: RetrievalResult,
    store: PgVectorStore,
    request: ReasoningRequest,
) -> TrustedResult:
    """Adapt raw top seeds to the graph provider's seed interface without claiming trust."""
    seeds: list[TrustedHit] = []
    for scored in retrieval.hits[:GRAPH_FIRST_SEED_K]:
        metadata = scored.chunk.metadata
        ord_value = metadata.get("ord")
        seeds.append(
            TrustedHit(
                chunk=scored.chunk,
                cosine=float(scored.score),
                confidence=0.0,
                verdict="ok",
                provenance=Provenance(
                    source=scored.chunk.source,
                    file=(
                        str(metadata["file"])
                        if metadata.get("file") is not None
                        else scored.chunk.source
                    ),
                    ord=ord_value if isinstance(ord_value, int) else None,
                    indexed_at=scored.indexed_at,
                    first_indexed_at=scored.first_indexed_at,
                ),
                validity=Validity(None, None, None),
            )
        )
    return TrustedResult(
        query=retrieval.query,
        hits=seeds,
        abstained=not seeds,
        reason="" if seeds else "no supporting retrieval",
        gap_warning=retrieval.gap_warning,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
        tenant_id=getattr(store, "tenant", None),
        generation_id=request.generation.generation_id or getattr(store, "generation_id", None),
        pipeline_fingerprint=request.generation.pipeline_fingerprint,
        corpus_fingerprint=request.generation.corpus_fingerprint,
    )


_SEMANTIC_GRAPH_INDEXES: OrderedDict[
    tuple[str, str, str, tuple[tuple[str, str, str], ...]], _SemanticGraphIndexes
] = OrderedDict()
_SEMANTIC_GRAPH_INDEX_CACHE_MAX = 4
_SEMANTIC_GRAPH_CACHE: OrderedDict[
    tuple[str, str, str | None, str | None, str | None], SemanticGraphProjection | None
] = OrderedDict()
_SEMANTIC_GRAPH_INFLIGHT: dict[
    tuple[str, str, str | None, str | None, str | None], _SemanticGraphFlight
] = {}
_SEMANTIC_GRAPH_CACHE_MAX = 4


class _ProposalFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: tuple[InferenceProposal, ...] | None = None
        self.error: BaseException | None = None


_DETERMINISTIC_PROPOSAL_CACHE: OrderedDict[
    tuple[str, str, str, str, str], tuple[InferenceProposal, ...]
] = OrderedDict()
_DETERMINISTIC_PROPOSAL_INFLIGHT: dict[tuple[str, str, str, str, str], _ProposalFlight] = {}
_DETERMINISTIC_PROPOSAL_CACHE_LOCK = threading.Lock()
_DETERMINISTIC_PROPOSAL_CACHE_MAX = 16


def _reset_graph_projection_cache() -> None:
    _graph_projection._reset_graph_projection_cache()
    with _GRAPH_PROJECTION_LOCK:
        _SEMANTIC_GRAPH_INDEXES.clear()
        _SEMANTIC_GRAPH_CACHE.clear()
        _SEMANTIC_GRAPH_INFLIGHT.clear()
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        _DETERMINISTIC_PROPOSAL_CACHE.clear()
        _DETERMINISTIC_PROPOSAL_INFLIGHT.clear()
    _reset_planner_index_cache()


#: The partition for the authorization view used to make proposals: the same scope the
#: filtered graph cache keys on, so a proposal is never reused across views.
_proposal_policy_scope = _authorization_scope


def _cached_deterministic_proposals(
    graph: ReasoningGraphProjection,
    *,
    pipeline_id: str,
    policy_scope: str,
) -> tuple[InferenceProposal, ...]:
    """Cache deterministic proposal output for one immutable graph serving identity."""
    key = (
        graph.tenant_id,
        graph.generation_id,
        graph.fingerprint,
        pipeline_id,
        policy_scope,
    )
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        cached = _DETERMINISTIC_PROPOSAL_CACHE.get(key)
        if cached is not None:
            _DETERMINISTIC_PROPOSAL_CACHE.move_to_end(key)
            return cached
        flight = _DETERMINISTIC_PROPOSAL_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _ProposalFlight()
            _DETERMINISTIC_PROPOSAL_INFLIGHT[key] = flight
    assert flight is not None
    if not owner:
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        assert flight.result is not None
        return flight.result
    try:
        proposals = tuple(_svc().deterministic_inference_proposals(graph, pipeline_id=pipeline_id))
    except BaseException as exc:
        with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
            flight.error = exc
            _DETERMINISTIC_PROPOSAL_INFLIGHT.pop(key, None)
            flight.done.set()
        raise
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        while len(_DETERMINISTIC_PROPOSAL_CACHE) >= _DETERMINISTIC_PROPOSAL_CACHE_MAX:
            _DETERMINISTIC_PROPOSAL_CACHE.popitem(last=False)
        _DETERMINISTIC_PROPOSAL_CACHE[key] = proposals
        flight.result = proposals
        _DETERMINISTIC_PROPOSAL_INFLIGHT.pop(key, None)
        flight.done.set()
    return proposals


def _semantic_graph_indexes(semantic: SemanticGraphProjection) -> _SemanticGraphIndexes:
    """Build graph adjacency once per immutable tenant, generation, and graph identity."""
    performance = current_performance_trace()
    relation_identity = tuple(
        (relation.id, relation.subject_id, relation.object_id) for relation in semantic.relations
    )
    key = (semantic.tenant_id, semantic.generation_id, semantic.graph_id, relation_identity)
    with _GRAPH_PROJECTION_LOCK:
        cached = _SEMANTIC_GRAPH_INDEXES.get(key)
        if cached is not None:
            _SEMANTIC_GRAPH_INDEXES.move_to_end(key)
            if performance is not None:
                performance.add("adjacency_cache_hits")
            return cached

    span: AbstractContextManager[Any]
    if performance is not None:
        performance.add("adjacency_cache_misses")
        span = performance.span("adjacency_construction_ms")
    else:
        span = nullcontext()
    with span:
        mentions_by_chunk: dict[str, set[str]] = {}
        chunks_by_entity: dict[str, set[str]] = {}
        relation_indexes_by_entity: dict[str, set[int]] = {}
        validity_by_chunk: dict[str, tuple[datetime | None, datetime | None]] = {}
        for mention in semantic.mentions:
            mentions_by_chunk.setdefault(mention.chunk_id, set()).add(mention.entity_id)
            chunks_by_entity.setdefault(mention.entity_id, set()).add(mention.chunk_id)
            if mention.chunk_id not in validity_by_chunk:
                try:
                    valid_from = mention.metadata.get("valid_from")
                    valid_until = mention.metadata.get("valid_until")
                    start = (
                        datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
                        if valid_from
                        else None
                    )
                    end = (
                        datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
                        if valid_until
                        else None
                    )
                    if start is not None and start.tzinfo is None:
                        start = start.replace(tzinfo=UTC)
                    if end is not None and end.tzinfo is None:
                        end = end.replace(tzinfo=UTC)
                    validity_by_chunk[mention.chunk_id] = (start, end)
                except (TypeError, ValueError):
                    validity_by_chunk[mention.chunk_id] = (None, None)
        for relation_index, relation in enumerate(semantic.relations):
            relation_indexes_by_entity.setdefault(relation.subject_id, set()).add(relation_index)
            relation_indexes_by_entity.setdefault(relation.object_id, set()).add(relation_index)
        indexes = _SemanticGraphIndexes(
            mentions_by_chunk={key: frozenset(value) for key, value in mentions_by_chunk.items()},
            chunks_by_entity={key: frozenset(value) for key, value in chunks_by_entity.items()},
            relation_indexes_by_entity={
                key: frozenset(value) for key, value in relation_indexes_by_entity.items()
            },
            ambiguous_entities=frozenset(
                entity_id
                for diagnostic in semantic.diagnostics
                if diagnostic.kind == "ambiguous_entity"
                for entity_id in diagnostic.entity_ids
            ),
            validity_by_chunk=validity_by_chunk,
        )
    with _GRAPH_PROJECTION_LOCK:
        existing = _SEMANTIC_GRAPH_INDEXES.get(key)
        if existing is not None:
            _SEMANTIC_GRAPH_INDEXES.move_to_end(key)
            return existing
        while len(_SEMANTIC_GRAPH_INDEXES) >= _SEMANTIC_GRAPH_INDEX_CACHE_MAX:
            _SEMANTIC_GRAPH_INDEXES.popitem(last=False)
        _SEMANTIC_GRAPH_INDEXES[key] = indexes
    return indexes


def _validate_security_context(
    store: PgVectorStore,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
) -> None:
    if security_policy is None:
        return
    if access_context is None:
        raise ValueError("access_context is required when security_policy is configured")
    store_tenant = getattr(store, "tenant", None)
    if isinstance(store_tenant, str) and store_tenant != access_context.tenant:
        raise PermissionError("access context tenant does not match the serving store")


def _generation_scope(
    store: PgVectorStore, generation_id: str | None
) -> AbstractContextManager[Any]:
    """Pin generation-scoped store calls when a request names an immutable generation."""
    pin_generation = getattr(store, "pin_generation", None)
    if generation_id and callable(pin_generation):
        return cast(AbstractContextManager[Any], pin_generation(generation_id))
    return nullcontext()


def _cached_semantic_graph(
    store: PgVectorStore,
    generation_id: str,
    readiness: Any,
    policy_fingerprint: str | None,
) -> SemanticGraphProjection | None:
    """Load the lazy semantic graph through a bounded generation and rebuild cache."""
    graph_fingerprint = getattr(readiness, "graph_fingerprint", None) if readiness else None
    key = (
        store.tenant,
        generation_id,
        _graph_projection._corpus_fingerprint(store, generation_id),
        graph_fingerprint,
        policy_fingerprint,
    )
    with _GRAPH_PROJECTION_LOCK:
        if key in _SEMANTIC_GRAPH_CACHE:
            cached = _SEMANTIC_GRAPH_CACHE[key]
            _SEMANTIC_GRAPH_CACHE.move_to_end(key)
            performance = current_performance_trace()
            if performance is not None:
                performance.add("projection_cache_hits")
            return cached

        flight = _SEMANTIC_GRAPH_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _SemanticGraphFlight()
            _SEMANTIC_GRAPH_INFLIGHT[key] = flight
    assert flight is not None
    performance = current_performance_trace()
    if performance is not None:
        performance.add("projection_cache_misses")
        performance.add(
            "projection_single_flight_owners" if owner else "projection_single_flight_waiters"
        )
    if not owner:
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        return flight.result

    try:
        loader = getattr(store, "load_semantic_graph", None)
        if callable(loader):
            with _generation_scope(store, generation_id):
                semantic = cast(SemanticGraphProjection | None, loader(generation_id))
        else:
            with _generation_scope(store, generation_id):
                semantic = _svc()._store_graph(
                    store,
                    include_text=False,
                    policy_fingerprint=policy_fingerprint,
                ).semantic_graph
        if semantic is not None and readiness is not None and getattr(readiness, "ready", True):
            actual = semantic.readiness()
            if (
                actual.graph_fingerprint != getattr(readiness, "graph_fingerprint", None)
                or (getattr(readiness, "tenant_id", actual.tenant_id) != actual.tenant_id)
                or (
                    getattr(readiness, "generation_id", actual.generation_id)
                    != actual.generation_id
                )
                or getattr(readiness, "graph_id", actual.graph_id) != actual.graph_id
            ):
                _log.warning(
                    "semantic graph does not match its generation readiness marker for %s",
                    generation_id,
                )
                semantic = None
    except BaseException as exc:
        with _GRAPH_PROJECTION_LOCK:
            flight.error = exc
            _SEMANTIC_GRAPH_INFLIGHT.pop(key, None)
            flight.done.set()
        raise
    with _GRAPH_PROJECTION_LOCK:
        while len(_SEMANTIC_GRAPH_CACHE) >= _SEMANTIC_GRAPH_CACHE_MAX:
            _SEMANTIC_GRAPH_CACHE.popitem(last=False)
        _SEMANTIC_GRAPH_CACHE[key] = semantic
        flight.result = semantic
        _SEMANTIC_GRAPH_INFLIGHT.pop(key, None)
        flight.done.set()
    return semantic


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


def _graph_precision_settings() -> tuple[str, str, int, int, float]:
    variant = os.environ.get("RECALL_GRAPH_PRECISION_VARIANT", "combined").strip().lower()
    if variant not in GRAPH_PRECISION_VARIANTS:
        variant = "combined"
    relation_control = os.environ.get("RECALL_GRAPH_RELATION_CONTROL", "none").strip().lower()
    if relation_control not in GRAPH_RELATION_CONTROLS:
        relation_control = "none"
    try:
        relation_control_seed = int(
            os.environ.get("RECALL_GRAPH_RELATION_CONTROL_SEED", "20260825")
        )
    except ValueError:
        relation_control_seed = 20260825
    try:
        hub_threshold = int(
            os.environ.get("RECALL_GRAPH_HUB_DEGREE_THRESHOLD", str(GRAPH_HUB_DEGREE_THRESHOLD))
        )
    except ValueError:
        hub_threshold = GRAPH_HUB_DEGREE_THRESHOLD
    if hub_threshold not in {16, 32, 64}:
        hub_threshold = GRAPH_HUB_DEGREE_THRESHOLD
    try:
        cosine_margin = float(
            os.environ.get("RECALL_GRAPH_COSINE_MARGIN", str(GRAPH_COSINE_MARGIN))
        )
    except ValueError:
        cosine_margin = GRAPH_COSINE_MARGIN
    if cosine_margin not in {0.05, 0.10, 0.15, 0.20}:
        cosine_margin = GRAPH_COSINE_MARGIN
    return variant, relation_control, relation_control_seed, hub_threshold, cosine_margin


def _graph_tail_replacement_margin() -> float | None:
    """Read the opt-in calibrated margin for replacing one direct tail item."""
    raw = os.environ.get("RECALL_GRAPH_TAIL_REPLACEMENT_MARGIN", "off").strip().lower()
    if raw in {"", "off", "none"}:
        return None
    if raw in {"on", "true"}:
        return GRAPH_TAIL_REPLACEMENT_MARGIN
    try:
        margin = float(raw)
    except ValueError:
        return None
    return margin if margin in GRAPH_TAIL_REPLACEMENT_MARGINS else None


def _graph_precision_policy_fingerprint(
    settings: tuple[str, str, int, int, float] | None = None,
) -> str:
    variant, relation_control, relation_control_seed, hub_threshold, _legacy_cosine_margin = (
        settings if settings is not None else _graph_precision_settings()
    )
    return hashlib.sha256(
        "|".join(
            (
                GRAPH_PRECISION_POLICY_VERSION,
                variant,
                relation_control,
                str(relation_control_seed),
                str(hub_threshold),
                ",".join(sorted(GRAPH_DIRECTIONAL_RELATIONS)),
                ",".join(sorted(GRAPH_DIAGNOSTIC_ONLY_RELATIONS)),
                "rerank=" + ",".join(f"{weight:.2f}" for weight in GRAPH_RERANK_WEIGHTS),
                f"corroboration_cap={GRAPH_RERANK_CORROBORATION_CAP}",
                f"fill_policy={GRAPH_FILL_POLICY}",
                f"fill_slots={GRAPH_FILL_SLOT_COUNT}",
                "tail_replacement_margin="
                + (
                    "off"
                    if _graph_tail_replacement_margin() is None
                    else f"{_graph_tail_replacement_margin():.2f}"
                ),
            )
        ).encode("utf-8")
    ).hexdigest()


def _graph_precision_feature_flags(variant: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return the graph precision features enabled by a diagnostic variant.

    The fourth flag is now calibrated reranking. The former cosine admission flag is retained in
    this tuple for diagnostic compatibility, but no variant performs hard cosine rejection.
    """
    return (
        variant in {"directional", "combined", "combined_no_selective"},
        variant in {"corroboration", "combined", "combined_no_selective"},
        variant in {"hub", "combined", "combined_no_selective"},
        variant in {"cosine", "combined", "combined_no_selective"},
        variant in {"selective", "combined"},
    )


def _shuffle_graph_relation_endpoints(
    relations: Sequence[Any],
    seed: int,
) -> tuple[Any, ...]:
    """Rewire object endpoints while preserving the directed degree sequences."""
    if len(relations) < 2:
        return tuple(relations)
    subjects = [relation.subject_id for relation in relations]
    objects = [relation.object_id for relation in relations]
    original_objects = tuple(objects)
    random.Random(seed).shuffle(objects)
    if tuple(objects) == original_objects and len(set(objects)) > 1:
        objects = objects[1:] + objects[:1]
    return tuple(
        replace(relation, subject_id=subject_id, object_id=object_id)
        for relation, subject_id, object_id in zip(relations, subjects, objects)
    )


def _graph_expansion_dependencies() -> _graph_expansion.GraphExpansionDependencies:
    """Bind graph orchestration to current service seams at call time."""
    return _graph_expansion.GraphExpansionDependencies(
        cached_semantic_graph=_cached_semantic_graph,
        combined_graph_policy_fingerprint=_combined_graph_policy_fingerprint,
        generation_scope=_generation_scope,
        graph_precision_feature_flags=_graph_precision_feature_flags,
        graph_precision_policy_fingerprint=_graph_precision_policy_fingerprint,
        graph_precision_settings=_graph_precision_settings,
        graph_tail_replacement_margin=_graph_tail_replacement_margin,
        resolve_graph_calibration=_resolve_graph_calibration,
        semantic_graph_indexes=_semantic_graph_indexes,
        shuffle_graph_relation_endpoints=_shuffle_graph_relation_endpoints,
        store_graph=_svc()._store_graph,
        validate_security_context=_validate_security_context,
        embed_query=embed_query,
        evaluate=_svc().evaluate,
        resolve_query_entities=resolve_query_entities,
        resolve_successor=resolve_successor,
        supersedes_key=supersedes_key,
        metrics=METRICS,
        graph_directional_relations=GRAPH_DIRECTIONAL_RELATIONS,
        graph_diagnostic_only_relations=GRAPH_DIAGNOSTIC_ONLY_RELATIONS,
        max_graph_rescoring_candidates=MAX_GRAPH_RESCORING_CANDIDATES,
        relation_kinds=frozenset(RELATION_KINDS),
    )


def _expand_semantic_graph(
    store: PgVectorStore,
    request: ReasoningRequest,
    retrieval: TrustedResult,
    calibration: Calibration | None,
    embedder: Embedder,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    defer_trust_evaluation: bool = False,
    excluded_chunk_ids: frozenset[str] = frozenset(),
) -> SemanticGraphExpansionResult:
    return _graph_expansion.expand_semantic_graph(
        store,
        request,
        retrieval,
        calibration,
        embedder,
        dependencies=_graph_expansion_dependencies(),
        security_policy=security_policy,
        access_context=access_context,
        defer_trust_evaluation=defer_trust_evaluation,
        excluded_chunk_ids=excluded_chunk_ids,
    )
