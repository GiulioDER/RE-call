"""Pure graph candidate scoring and evidence allocation helpers.

This module deliberately has no dependency on ``recall_mcp.service``.  The service layer owns
database access, readiness, security, and trust orchestration; this module owns only the deterministic
candidate state and the final graph versus direct evidence allocation rules.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence
from typing import cast

from recall.calibration import Calibration
from recall.reasoning import ReasoningRequest, SemanticGraphExpansionResult
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall.store import EdgeCandidates
from recall.trust import is_trusted
from recall.types import Chunk, RetrievalResult, ScoredChunk, TrustedHit, TrustedResult

GRAPH_RERANK_WEIGHTS = (0.60, 0.20, 0.10, 0.10)
GRAPH_RERANK_CORROBORATION_CAP = 2
GRAPH_FILL_SLOT_COUNT = 5
GRAPH_FIRST_SEED_K = 8
GRAPH_FIRST_CONTEXT_K = 10


@dataclass(frozen=True)
class GraphExpansionDependencies:
    """Service-owned operations required by the graph expansion orchestration.

    The orchestration is deliberately kept independent from ``recall_mcp.service``. The service
    builds this object at call time, which preserves monkeypatchable compatibility seams and keeps
    database, cache, policy, and observability ownership in the service layer.
    """

    cached_semantic_graph: Callable[..., Any]
    combined_graph_policy_fingerprint: Callable[..., Any]
    generation_scope: Callable[..., Any]
    graph_precision_feature_flags: Callable[..., Any]
    graph_precision_policy_fingerprint: Callable[..., Any]
    graph_precision_settings: Callable[..., Any]
    graph_tail_replacement_margin: Callable[..., Any]
    resolve_graph_calibration: Callable[..., Any]
    semantic_graph_indexes: Callable[..., Any]
    shuffle_graph_relation_endpoints: Callable[..., Any]
    store_graph: Callable[..., Any]
    validate_security_context: Callable[..., Any]
    embed_query: Callable[..., Any]
    evaluate: Callable[..., Any]
    resolve_query_entities: Callable[..., Any]
    resolve_successor: Callable[..., Any]
    supersedes_key: Callable[..., Any]
    metrics: Any
    graph_directional_relations: frozenset[str]
    graph_diagnostic_only_relations: frozenset[str]
    max_graph_rescoring_candidates: int
    relation_kinds: frozenset[str]


@dataclass
class GraphCandidate:
    """Mutable admission state for one graph neighbor chunk."""

    neighbor_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    trusted_seed_chunk_ids: set[str] = field(default_factory=set)
    relation_evidence_chunk_ids: set[str] = field(default_factory=set)
    best_confidence: float = 0.0
    neighbor_chunk_count: int = 0
    relation_types: set[str] = field(default_factory=set)
    path_length: int = 1


def calibrated_graph_relevance(cosine: float, calibration: Calibration | None) -> float:
    """Map a query cosine into the bounded relevance feature used by graph reranking."""
    if calibration is not None:
        return calibration.confidence(float(cosine))
    return max(0.0, min(1.0, (float(cosine) + 1.0) / 2.0))


def graph_corroboration(candidate: GraphCandidate) -> float:
    """Return a bounded signal for distinct seed and relation support."""
    seed_support = min(len(candidate.trusted_seed_chunk_ids) / GRAPH_RERANK_CORROBORATION_CAP, 1.0)
    relation_support = min(len(candidate.relation_ids) / GRAPH_RERANK_CORROBORATION_CAP, 1.0)
    return (seed_support + relation_support) / 2.0


def graph_candidate_rerank_score(
    candidate: GraphCandidate,
    cosine: float,
    calibration: Calibration | None,
) -> float:
    """Combine calibrated relevance with structural evidence without changing trust inputs."""
    cosine_signal = calibrated_graph_relevance(cosine, calibration)
    relation_signal = max(0.0, min(1.0, float(candidate.best_confidence)))
    path_signal = 1.0 / max(1, int(candidate.path_length))
    corroboration_signal = graph_corroboration(candidate)
    cosine_weight, relation_weight, path_weight, corroboration_weight = GRAPH_RERANK_WEIGHTS
    return (
        cosine_weight * cosine_signal
        + relation_weight * relation_signal
        + path_weight * path_signal
        + corroboration_weight * corroboration_signal
    )


def merge_graph_hits(
    retrieval: TrustedResult,
    accepted: Sequence[TrustedHit],
    candidate_scores: Mapping[str, float],
    calibration: Calibration | None,
    max_items: int = GRAPH_FILL_SLOT_COUNT,
    tail_replacement_margin: float | None = None,
) -> list[TrustedHit]:
    """Keep direct evidence first, with an opt-in calibrated one-item tail replacement."""
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise ValueError("max_items must be a positive int")
    if tail_replacement_margin is not None and (
        isinstance(tail_replacement_margin, bool)
        or not isinstance(tail_replacement_margin, (int, float))
        or tail_replacement_margin < 0
        or tail_replacement_margin > 1
    ):
        raise ValueError("tail_replacement_margin must be between 0 and 1")
    if not accepted:
        return list(retrieval.hits)
    existing_ids = {hit.chunk.id for hit in retrieval.hits}
    graph_hits = [hit for hit in accepted if hit.chunk.id not in existing_ids]
    if not graph_hits:
        return list(retrieval.hits)

    direct = [hit for hit in retrieval.hits if is_trusted(hit)]
    fill_slots = max_items - len(direct)
    ranked_graph = sorted(
        enumerate(graph_hits),
        key=lambda item: (-float(candidate_scores[item[1].chunk.id]), item[0], item[1].chunk.id),
    )
    if fill_slots <= 0:
        if tail_replacement_margin is not None and direct:
            direct_tail = direct[max_items - 1] if len(direct) >= max_items else direct[-1]
            tail_signal = calibrated_graph_relevance(direct_tail.cosine, calibration)
            for _order, candidate in ranked_graph:
                candidate_signal = calibrated_graph_relevance(candidate.cosine, calibration)
                if candidate_signal > tail_signal + float(tail_replacement_margin):
                    demoted = [hit for hit in retrieval.hits if not is_trusted(hit)]
                    return direct[: max_items - 1] + [candidate] + demoted
        return list(retrieval.hits)
    graph_fill = [hit for _order, hit in ranked_graph[:fill_slots]]
    demoted = [hit for hit in retrieval.hits if not is_trusted(hit)]
    return direct + graph_fill + demoted


def assemble_graph_first_context(
    retrieval: RetrievalResult,
    graph_candidates: Sequence[ScoredChunk],
    *,
    seed_k: int = GRAPH_FIRST_SEED_K,
    context_k: int = GRAPH_FIRST_CONTEXT_K,
    calibration: Calibration | None = None,
    tail_replacement_margin: float | None = None,
    max_graph_items: int | None = None,
    compare_weakest_tail: bool = True,
    drop_replaced_tail: bool = False,
    calibrate_margin: bool = True,
) -> RetrievalResult:
    """Protect the direct prefix, then let bounded graph candidates compete for the tail."""
    if tail_replacement_margin is not None and (
        isinstance(tail_replacement_margin, bool)
        or not isinstance(tail_replacement_margin, (int, float))
        or tail_replacement_margin < 0
        or tail_replacement_margin > 1
    ):
        raise ValueError("tail_replacement_margin must be between 0 and 1")
    direct_prefix = list(retrieval.hits[:seed_k])
    remaining = max(0, context_k - len(direct_prefix))
    graph_limit = remaining if max_graph_items is None else min(remaining, max_graph_items)
    selected_ids = {hit.chunk.id for hit in direct_prefix}
    graph_fill: list[ScoredChunk] = []
    direct_tail = list(retrieval.hits[seed_k:context_k])
    def relevance(score: float) -> float:
        return (
            calibrated_graph_relevance(score, calibration)
            if calibrate_margin
            else float(score)
        )
    tail = (
        min(direct_tail, key=lambda hit: relevance(hit.score))
        if compare_weakest_tail and direct_tail
        else (direct_tail[0] if direct_tail else None)
    )
    if graph_limit:
        for candidate in graph_candidates:
            if candidate.chunk.id in selected_ids:
                continue
            if tail_replacement_margin is not None:
                if tail is None:
                    break
                candidate_signal = relevance(candidate.score)
                tail_signal = relevance(tail.score)
                if candidate_signal < tail_signal + float(tail_replacement_margin):
                    continue
            graph_fill.append(candidate)
            selected_ids.add(candidate.chunk.id)
            if len(graph_fill) >= graph_limit:
                break
    fallback = [hit for hit in retrieval.hits[seed_k:] if hit.chunk.id not in selected_ids]
    if drop_replaced_tail and tail_replacement_margin is not None and graph_fill:
        fallback = fallback[len(graph_fill) :]
    return replace(
        retrieval,
        hits=(direct_prefix + graph_fill + fallback)[:context_k],
    )


@dataclass
class ExpansionStats:
    """Mutable counters shared by graph admission phases and final result emission."""

    rejections: dict[str, int] = field(default_factory=dict)
    refusals: dict[str, int] = field(default_factory=dict)
    relation_seed_activations: dict[str, int] = field(default_factory=dict)
    relation_candidates_accepted: dict[str, int] = field(default_factory=dict)
    relation_new_trusted_evidence: dict[str, int] = field(default_factory=dict)
    semantic_diagnostic_count: int = 0

    def reject(self, reason: str, count: int = 1) -> None:
        if count > 0:
            self.rejections[reason] = self.rejections.get(reason, 0) + count

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1


def _finish_expansion(
    *,
    started: float,
    stats: ExpansionStats,
    performance: Any,
    metrics: Any,
    policy_fingerprint: str,
    result: TrustedResult,
    readiness: str,
    entities: int = 0,
    relations: int = 0,
    candidates: int = 0,
    gate_reason: str | None = None,
    scored_candidates: Sequence[ScoredChunk] = (),
    candidate_relation_types: Mapping[str, tuple[str, ...]] | None = None,
) -> SemanticGraphExpansionResult:
    """Record expansion telemetry and construct the stable expansion result."""
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if performance is not None:
        performance.set("graph_readiness", readiness)
        performance.add("candidate_count", candidates)
    rejection_items = tuple(sorted(stats.rejections.items()))
    refusal_items = tuple(sorted(stats.refusals.items()))
    metrics.increment("recall_graph_query_total")
    metrics.increment("recall_graph_expansion_total")
    metrics.increment("recall_graph_candidates_total", value=candidates)
    metrics.increment(
        "recall_graph_rejected_candidates_total",
        value=sum(stats.rejections.values()),
    )
    metrics.increment("recall_graph_diagnostics_total", value=stats.semantic_diagnostic_count)
    metrics.increment("recall_graph_policy_total", policy=policy_fingerprint[:16])
    if gate_reason is not None:
        metrics.increment("recall_graph_gate_refused_total", reason=gate_reason)
    for refusal_reason, count in refusal_items:
        metrics.increment(
            "recall_graph_expansion_refused_total", value=count, reason=refusal_reason
        )
    for rejection_reason, count in rejection_items:
        metric = (
            "recall_graph_relations_rejected_total"
            if rejection_reason.startswith("relation_")
            or rejection_reason in {"ambiguous_entity", "hub_entity"}
            else "recall_graph_candidates_rejected_total"
        )
        metrics.increment(metric, value=count, reason=rejection_reason)
    metrics.observe("recall_graph_latency_ms", latency_ms)
    return SemanticGraphExpansionResult(
        retrieval=result,
        readiness=readiness,
        entities_inspected=entities,
        relations_inspected=relations,
        candidates_discovered=candidates,
        candidates_rejected=sum(stats.rejections.values()),
        relation_seed_activations=dict(stats.relation_seed_activations),
        relation_candidates_accepted=dict(stats.relation_candidates_accepted),
        relation_new_trusted_evidence=dict(stats.relation_new_trusted_evidence),
        diagnostics_encountered=stats.semantic_diagnostic_count,
        latency_ms=latency_ms,
        admission_rejections=rejection_items,
        expansion_refusals=refusal_items,
        gate_reason=gate_reason,
        policy_fingerprint=policy_fingerprint,
        scored_candidates=tuple(scored_candidates),
        candidate_relation_types=dict(candidate_relation_types or {}),
    )


@dataclass(frozen=True)
class RelationAdmission:
    """Relation-derived candidate state before text is fetched."""

    candidate_relations_by_chunk: Mapping[str, tuple[Any, ...]]
    supersession: Mapping[str, str]
    unresolved: frozenset[str]
    edge_candidates: EdgeCandidates
    relation_count: int


def _admit_relations(
    *,
    store: PgVectorStore,
    semantic: Any,
    indexes: Any,
    seed_entities: set[str],
    resolved_query_entities: set[str],
    trusted_seed_ids: set[str],
    as_of: datetime,
    entity_budget: int,
    use_directional: bool,
    graph_directional_relations: frozenset[str],
    graph_diagnostic_only_relations: frozenset[str],
    use_hub_suppression: bool,
    hub_threshold: int,
    stats: ExpansionStats,
) -> RelationAdmission:
    """Admit authored relation neighbors while applying graph safety and budget rules."""
    candidate_relations_by_chunk: dict[str, list[Any]] = {}
    neighboring_entities: set[str] = set()
    relation_count = 0
    supersession: dict[str, str] = {}
    unresolved: frozenset[str] = frozenset()
    edge_candidates: EdgeCandidates = {}
    supersession_reader = getattr(store, "supersession_all", None)
    if callable(supersession_reader):
        supersession_result = supersession_reader()
        if isinstance(supersession_result, tuple) and len(supersession_result) >= 2:
            supersession = dict(supersession_result[0])
            unresolved = frozenset(supersession_result[1])
            if len(supersession_result) >= 3 and isinstance(supersession_result[2], Mapping):
                edge_candidates = cast(EdgeCandidates, supersession_result[2])
    else:
        supersession_reader = getattr(store, "supersession", None)
        if callable(supersession_reader):
            supersession, unresolved = supersession_reader()

    ambiguous_entities = indexes.ambiguous_entities
    chunks_by_entity = indexes.chunks_by_entity
    relation_indexes = sorted(
        {
            relation_index
            for entity_id in seed_entities
            for relation_index in indexes.relation_indexes_by_entity.get(entity_id, ())
        }
    )
    for relation_index in relation_indexes:
        relation = semantic.relations[relation_index]
        if relation.status != "authored":
            stats.reject("relation_non_authored")
            continue
        if relation.effective_at is not None and relation.effective_at > as_of:
            stats.reject("relation_not_yet_effective")
            continue
        if relation.valid_from is not None and as_of < relation.valid_from:
            stats.reject("relation_not_yet_valid")
            continue
        if relation.valid_until is not None and as_of > relation.valid_until:
            stats.reject("relation_expired")
            continue
        if relation.subject_id in ambiguous_entities or relation.object_id in ambiguous_entities:
            stats.reject("ambiguous_entity")
            continue
        if use_directional and relation.relation in graph_diagnostic_only_relations:
            stats.reject("relation_type")
            continue
        if use_directional and relation.relation not in graph_directional_relations:
            stats.reject("relation_type")
            continue
        if relation.relation == "supersedes" and relation.subject_id not in seed_entities:
            stats.reject("relation_direction")
            continue
        if not set(relation.evidence_chunk_ids).intersection(trusted_seed_ids):
            stats.reject("relation_evidence_not_trusted")
            continue
        seed_entity_for_relation: str | None = None
        if use_directional:
            if relation.subject_id not in seed_entities:
                if relation.object_id in seed_entities:
                    stats.reject("relation_direction")
                else:
                    stats.reject("relation_not_seeded")
                continue
            seed_entity_for_relation = relation.subject_id
        elif relation.subject_id not in seed_entities and relation.object_id not in seed_entities:
            stats.reject("relation_not_seeded")
            continue
        else:
            seed_entity_for_relation = (
                relation.subject_id if relation.subject_id in seed_entities else relation.object_id
            )
        if (
            use_hub_suppression
            and seed_entity_for_relation is not None
            and len(chunks_by_entity.get(seed_entity_for_relation, set())) > hub_threshold
            and seed_entity_for_relation not in resolved_query_entities
        ):
            stats.reject("hub_entity")
            continue
        relation_count += 1
        stats.relation_seed_activations[relation.relation] += 1
        neighbor = (
            relation.object_id if relation.subject_id in seed_entities else relation.subject_id
        )
        if neighbor not in neighboring_entities:
            if len(neighboring_entities) >= entity_budget:
                stats.reject("entity_budget")
                continue
            neighboring_entities.add(neighbor)
        for chunk_id in sorted(chunks_by_entity.get(neighbor, frozenset())):
            if chunk_id not in trusted_seed_ids:
                candidate_relations_by_chunk.setdefault(chunk_id, []).append(relation)

    return RelationAdmission(
        candidate_relations_by_chunk={
            chunk_id: tuple(relations)
            for chunk_id, relations in candidate_relations_by_chunk.items()
        },
        supersession=supersession,
        unresolved=unresolved,
        edge_candidates=edge_candidates,
        relation_count=relation_count,
    )


@dataclass(frozen=True)
class CandidateMaterialization:
    """Bounded graph candidates and their authorized text payload."""

    candidates_by_chunk: Mapping[str, GraphCandidate]
    chunks_by_id: Mapping[str, Chunk]
    candidate_count: int
    scorable_ids: tuple[str, ...]
    metadata_only: bool = False
    prefetched_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ScoredGraphCandidates:
    """Query-scored graph candidates ready for trust evaluation or context assembly."""

    candidate_scores: Mapping[str, float]
    scored: tuple[ScoredChunk, ...]


def _score_candidates(
    *,
    store: PgVectorStore,
    request: ReasoningRequest,
    embedder: Embedder,
    scorable_ids: Sequence[str],
    candidates_by_chunk: Mapping[str, GraphCandidate],
    chunks_by_id: Mapping[str, Chunk],
    active_calibration: Calibration | None,
    use_calibrated_rerank: bool,
    use_corroboration: bool,
    performance: Any,
    generation_scope: Callable[..., Any],
    embed_query: Callable[..., list[float]],
    graph_candidate_rerank_score: Callable[..., float],
    stats: ExpansionStats,
    precomputed_query_scores: Mapping[str, float] | None = None,
) -> ScoredGraphCandidates:
    """Embed and rank the bounded graph candidate set without changing admission semantics."""
    supplied_scores = precomputed_query_scores or {}
    query_scores = {
        chunk_id: float(supplied_scores[chunk_id])
        for chunk_id in scorable_ids
        if chunk_id in supplied_scores
    }
    score_ids = tuple(chunk_id for chunk_id in scorable_ids if chunk_id not in query_scores)
    if score_ids:
        query_vector = request._context.query_vector
        if query_vector is None:
            if performance is None:
                query_vector = embed_query(embedder, request.query)
            else:
                with performance.span("query_embedding_ms"):
                    query_vector = embed_query(embedder, request.query)
        elif performance is not None:
            performance.set("query_embedding_reused", True)
        if performance is None:
            with generation_scope(store, request.generation.generation_id):
                query_scores.update(store.cosines_for(score_ids, query_vector))
        else:
            with performance.span("cosine_rescoring_ms"):
                with generation_scope(store, request.generation.generation_id):
                    measured_scores = store.cosines_for(score_ids, query_vector)
            query_scores.update(measured_scores)
            performance.add("candidate_scored_count", len(measured_scores))
    if performance is not None and supplied_scores:
        performance.add("candidate_score_reused_count", len(scorable_ids) - len(score_ids))

    candidate_scores: dict[str, float] = {}
    admitted_ids: list[str] = []
    for chunk_id, candidate in candidates_by_chunk.items():
        if chunk_id not in query_scores:
            stats.reject("missing_query_score")
            continue
        candidate_scores[chunk_id] = (
            graph_candidate_rerank_score(
                candidate,
                float(query_scores[chunk_id]),
                active_calibration,
            )
            if use_calibrated_rerank
            else float(query_scores[chunk_id])
        )
        admitted_ids.append(chunk_id)

    if use_calibrated_rerank:
        admitted_ids.sort(key=lambda chunk_id: (-candidate_scores[chunk_id], chunk_id))
    else:
        admitted_ids.sort(
            key=lambda chunk_id: (
                -float(query_scores[chunk_id]),
                -(
                    len(candidates_by_chunk[chunk_id].trusted_seed_chunk_ids)
                    if use_corroboration
                    else 0
                ),
                -(len(candidates_by_chunk[chunk_id].relation_ids) if use_corroboration else 0),
                -candidates_by_chunk[chunk_id].best_confidence,
                -(
                    candidates_by_chunk[chunk_id].neighbor_chunk_count
                    if not use_corroboration
                    else 0
                ),
                chunk_id,
            )
        )
    for chunk_id in admitted_ids:
        for relation_type in candidates_by_chunk[chunk_id].relation_types:
            stats.relation_candidates_accepted[relation_type] += 1

    scored: list[ScoredChunk] = []
    for chunk_id in admitted_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            stats.reject("missing_chunk")
            continue
        scored.append(
            ScoredChunk(
                chunk=chunk,
                # Trust calibration is fitted on query dense cosine. Relation confidence is
                # structural metadata and must never stand in for query relevance here.
                score=query_scores[chunk_id],
            )
        )
    return ScoredGraphCandidates(
        candidate_scores=candidate_scores,
        scored=tuple(scored),
    )


def _assemble_trusted_expansion(
    *,
    retrieval: TrustedResult,
    request: ReasoningRequest,
    scored_candidates: ScoredGraphCandidates,
    candidates_by_chunk: Mapping[str, GraphCandidate],
    supersession: Mapping[str, str],
    active_calibration: Calibration | None,
    as_of: datetime,
    unresolved: frozenset[str],
    generation_binding: Mapping[str, str],
    performance: Any,
    evaluate: Callable[..., TrustedResult],
    merge_graph_hits: Callable[..., list[TrustedHit]],
    graph_tail_replacement_margin: Callable[..., float | None],
    stats: ExpansionStats,
) -> TrustedResult:
    """Run the one trust pass and merge accepted graph evidence into the retrieval result."""
    scored = list(scored_candidates.scored)
    candidate_result = RetrievalResult(
        query=retrieval.query,
        hits=scored,
        gap_warning=False,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
    )
    if performance is None:
        evaluated = evaluate(
            candidate_result,
            supersession,
            active_calibration,
            as_of,
            unresolved,
            known_as_of=request.known_as_of,
            calibration_id=retrieval.calibration_id,
            calibration_status=retrieval.calibration_status,
            generation_binding=generation_binding,
            query_set_digest=retrieval.query_set_digest,
        )
    else:
        with performance.span("trust_reevaluation_ms"):
            evaluated = evaluate(
                candidate_result,
                supersession,
                active_calibration,
                as_of,
                unresolved,
                known_as_of=request.known_as_of,
                calibration_id=retrieval.calibration_id,
                calibration_status=retrieval.calibration_status,
                generation_binding=generation_binding,
                query_set_digest=retrieval.query_set_digest,
            )
        performance.add("trust_reevaluated_count", len(scored))
    accepted = [hit for hit in evaluated.hits if is_trusted(hit)]
    accepted_ids = {hit.chunk.id for hit in accepted}
    for chunk_id in accepted_ids:
        accepted_candidate = candidates_by_chunk.get(chunk_id)
        if accepted_candidate is None:
            continue
        for relation_type in accepted_candidate.relation_types:
            stats.relation_new_trusted_evidence[relation_type] += 1
    merged = merge_graph_hits(
        retrieval,
        accepted,
        scored_candidates.candidate_scores,
        active_calibration,
        max_items=request.evidence_policy.max_items,
        tail_replacement_margin=graph_tail_replacement_margin(),
    )
    has_trusted_hit = any(is_trusted(hit) for hit in merged)
    expanded = replace(
        retrieval,
        hits=merged,
        abstained=not has_trusted_hit,
        reason="" if has_trusted_hit else evaluated.reason,
    )
    stats.reject("trust", len(scored) - len(accepted_ids))
    return expanded


def _fetch_candidate_chunks(
    *,
    store: PgVectorStore,
    request: ReasoningRequest,
    candidate_ids: Sequence[str],
    performance: Any,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
    stats: ExpansionStats,
    generation_scope: Callable[..., Any],
) -> dict[str, Chunk]:
    """Fetch one bounded payload batch and apply source authorization and redaction."""
    batch_loader = getattr(store, "chunks_by_ids", None)
    fetch_scope: AbstractContextManager[Any] = (
        performance.span("candidate_fetch_ms") if performance is not None else nullcontext()
    )
    with fetch_scope:
        if callable(batch_loader):
            with generation_scope(store, request.generation.generation_id):
                fetched = batch_loader(tuple(candidate_ids))
            if isinstance(fetched, Mapping):
                chunks_by_id = {
                    str(chunk_id): chunk
                    for chunk_id, chunk in fetched.items()
                    if isinstance(chunk, Chunk)
                }
            else:
                chunks_by_id = {chunk.id: chunk for chunk in fetched if isinstance(chunk, Chunk)}
        else:
            iterator = getattr(store, "iter_chunks", None)
            candidate_id_set = set(candidate_ids)
            chunks_by_id = (
                {
                    chunk.id: chunk
                    for chunk in iterator()
                    if isinstance(chunk, Chunk) and chunk.id in candidate_id_set
                }
                if callable(iterator)
                else {}
            )
    if performance is not None:
        performance.add("candidate_fetched_count", len(chunks_by_id))
        performance.add(
            "candidate_payload_bytes",
            sum(
                len(chunk.text.encode("utf-8")) + len(chunk.source.encode("utf-8"))
                for chunk in chunks_by_id.values()
            ),
        )

    if security_policy is not None:
        assert access_context is not None
        authorized_chunks: dict[str, Chunk] = {}
        for chunk_id in candidate_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                stats.reject("missing_chunk")
                continue
            decision = security_policy.decide(chunk.source, access_context)
            if not decision.allowed:
                stats.reject("security_policy")
                continue
            redacted_text, _ = security_policy.redact_with_decision(chunk.text, decision)
            authorized_chunks[chunk_id] = (
                chunk if redacted_text == chunk.text else replace(chunk, text=redacted_text)
            )
        chunks_by_id = authorized_chunks
    else:
        for chunk_id in candidate_ids:
            if chunk_id not in chunks_by_id:
                stats.reject("missing_chunk")
    return chunks_by_id


def _retain_loaded_candidates(
    *,
    request: ReasoningRequest,
    candidates_by_chunk: dict[str, GraphCandidate],
    chunks_by_id: Mapping[str, Chunk],
    candidate_relations_by_chunk: Mapping[str, Sequence[Any]],
    indexes: Any,
    seed_entities: set[str],
    trusted_seed_ids: set[str],
    as_of: datetime,
    supersession: Mapping[str, str],
    edge_candidates: EdgeCandidates,
    candidate_budget: int,
    supersedes_key: Callable[..., Any],
    resolve_successor: Callable[..., Any],
    stats: ExpansionStats,
) -> CandidateMaterialization:
    """Recheck loaded metadata, then build structural state for admitted chunks."""
    retained_chunks = dict(chunks_by_id)
    for chunk_id in tuple(candidates_by_chunk):
        chunk = retained_chunks.get(chunk_id)
        if chunk is None:
            del candidates_by_chunk[chunk_id]
            continue
        window = indexes.validity_by_chunk.get(chunk_id)
        if window is None or window == (None, None):
            try:
                valid_from = chunk.metadata.get("valid_from")
                valid_until = chunk.metadata.get("valid_until")
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
                window = (start, end)
            except (TypeError, ValueError):
                stats.reject("invalid_temporal_metadata")
                del candidates_by_chunk[chunk_id]
                continue
        valid_from, valid_until = window
        if valid_from is not None and as_of < valid_from:
            stats.reject("temporal_not_yet_valid")
            del candidates_by_chunk[chunk_id]
            continue
        if valid_until is not None and as_of > valid_until:
            stats.reject("temporal_expired")
            del candidates_by_chunk[chunk_id]
            continue
        file_value = chunk.metadata.get("file") or chunk.source
        if isinstance(file_value, str):
            successor = resolve_successor(
                supersedes_key(file_value),
                supersession,
                edge_candidates,
                request.known_as_of,
            )
            if successor is not None:
                stats.reject("superseded")
                del candidates_by_chunk[chunk_id]

    admitted_candidates: dict[str, GraphCandidate] = {}
    for chunk_id in sorted(candidates_by_chunk):
        if len(admitted_candidates) >= candidate_budget:
            stats.reject("budget")
            continue
        candidate = GraphCandidate()
        for relation in candidate_relations_by_chunk[chunk_id]:
            neighbor = (
                relation.object_id if relation.subject_id in seed_entities else relation.subject_id
            )
            candidate.neighbor_ids.add(neighbor)
            candidate.relation_ids.add(relation.id)
            candidate.trusted_seed_chunk_ids.update(
                set(relation.evidence_chunk_ids).intersection(trusted_seed_ids)
            )
            candidate.relation_evidence_chunk_ids.update(relation.evidence_chunk_ids)
            candidate.relation_types.add(relation.relation)
            candidate.best_confidence = max(candidate.best_confidence, relation.confidence)
            candidate.neighbor_chunk_count = max(
                candidate.neighbor_chunk_count,
                len(indexes.chunks_by_entity.get(neighbor, ())),
            )
        admitted_candidates[chunk_id] = candidate
    scorable_ids = tuple(
        chunk_id for chunk_id in admitted_candidates if chunk_id in retained_chunks
    )
    return CandidateMaterialization(
        admitted_candidates,
        retained_chunks,
        len(admitted_candidates),
        scorable_ids,
    )


def _materialize_candidates(
    *,
    store: PgVectorStore,
    request: ReasoningRequest,
    candidate_relations_by_chunk: Mapping[str, Sequence[Any]],
    indexes: Any,
    seed_entities: set[str],
    trusted_seed_ids: set[str],
    as_of: datetime,
    supersession: Mapping[str, str],
    edge_candidates: EdgeCandidates,
    candidate_budget: int,
    defer_trust_evaluation: bool,
    max_graph_rescoring_candidates: int,
    performance: Any,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
    stats: ExpansionStats,
    generation_scope: Callable[..., Any],
    supersedes_key: Callable[..., Any],
    resolve_successor: Callable[..., Any],
    excluded_chunk_ids: frozenset[str],
    prefetched_chunks: Mapping[str, Chunk] | None = None,
) -> CandidateMaterialization:
    """Filter graph metadata, fetch only bounded text, authorize it, and build candidate state."""
    supplied_chunks = prefetched_chunks or {}
    candidate_ids: list[str] = []
    for chunk_id in sorted(
        candidate_relations_by_chunk,
        key=lambda value: (0 if value in supplied_chunks else 1, value),
    ):
        window = indexes.validity_by_chunk.get(chunk_id)
        if window is not None and window != (None, None):
            valid_from, valid_until = window
            if valid_from is not None and as_of < valid_from:
                stats.reject("temporal_not_yet_valid")
                continue
            if valid_until is not None and as_of > valid_until:
                stats.reject("temporal_expired")
                continue
        successor = resolve_successor(
            supersedes_key(chunk_id),
            supersession,
            edge_candidates,
            request.known_as_of,
        )
        if successor is not None:
            stats.reject("superseded")
            continue
        candidate_ids.append(chunk_id)

    fetch_budget = (
        min(candidate_budget, max_graph_rescoring_candidates)
        if defer_trust_evaluation
        else min(
            candidate_budget,
            max(0, request.evidence_policy.max_items - len(trusted_seed_ids)),
        )
    )
    if len(candidate_ids) > fetch_budget:
        stats.reject("budget", len(candidate_ids) - fetch_budget)
        candidate_ids = candidate_ids[:fetch_budget]
    if not candidate_ids:
        return CandidateMaterialization({}, {}, 0, ())

    candidates_by_chunk: dict[str, GraphCandidate] = {
        chunk_id: GraphCandidate() for chunk_id in candidate_ids
    }
    reusable_chunks = (
        {
            chunk_id: supplied_chunks[chunk_id]
            for chunk_id in candidate_ids
            if chunk_id in supplied_chunks
        }
        if security_policy is None
        else {}
    )
    fetch_ids = tuple(chunk_id for chunk_id in candidate_ids if chunk_id not in reusable_chunks)
    metadata_loader = getattr(store, "chunk_metadata_by_ids", None)
    batch_loader = getattr(store, "chunks_by_ids", None)
    metadata_only = (
        bool(fetch_ids)
        and defer_trust_evaluation
        and security_policy is None
        and callable(metadata_loader)
        and callable(batch_loader)
    )
    if metadata_only:
        assert callable(metadata_loader)
        with generation_scope(store, request.generation.generation_id):
            fetched = metadata_loader(fetch_ids)
        if isinstance(fetched, Mapping):
            loaded_chunks = {
                str(chunk_id): chunk
                for chunk_id, chunk in fetched.items()
                if isinstance(chunk, Chunk)
            }
        else:
            loaded_chunks = {chunk.id: chunk for chunk in fetched if isinstance(chunk, Chunk)}
        chunks_by_id = {**reusable_chunks, **loaded_chunks}
        for chunk_id in fetch_ids:
            if chunk_id not in loaded_chunks:
                stats.reject("missing_chunk")
    elif not fetch_ids:
        chunks_by_id = dict(reusable_chunks)
    else:
        loaded_chunks = _fetch_candidate_chunks(
            store=store,
            request=request,
            candidate_ids=fetch_ids,
            performance=performance,
            security_policy=security_policy,
            access_context=access_context,
            stats=stats,
            generation_scope=generation_scope,
        )
        chunks_by_id = {**reusable_chunks, **loaded_chunks}

    retained = _retain_loaded_candidates(
        request=request,
        candidates_by_chunk=candidates_by_chunk,
        chunks_by_id=chunks_by_id,
        candidate_relations_by_chunk=candidate_relations_by_chunk,
        indexes=indexes,
        seed_entities=seed_entities,
        trusted_seed_ids=trusted_seed_ids,
        as_of=as_of,
        supersession=supersession,
        edge_candidates=edge_candidates,
        candidate_budget=candidate_budget,
        supersedes_key=supersedes_key,
        resolve_successor=resolve_successor,
        stats=stats,
    )
    return replace(
        retained,
        metadata_only=metadata_only,
        prefetched_ids=frozenset(reusable_chunks),
    )


def expand_semantic_graph(
    store: PgVectorStore,
    request: ReasoningRequest,
    retrieval: TrustedResult,
    calibration: Calibration | None,
    embedder: Embedder,
    dependencies: GraphExpansionDependencies,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    defer_trust_evaluation: bool = False,
    excluded_chunk_ids: frozenset[str] = frozenset(),
    candidate_chunk_ids: frozenset[str] | None = None,
    prefetched_candidates: Mapping[str, ScoredChunk] | None = None,
) -> SemanticGraphExpansionResult:
    """Expand seeds through one precise persisted semantic hop.

    The normal path keeps its historical trusted seed and direct first fill behavior. The graph
    first path uses ``defer_trust_evaluation`` to return scored graph candidates to the retrieval
    boundary, where the direct prefix and final context are assembled before one trust pass.
    """
    _cached_semantic_graph = dependencies.cached_semantic_graph
    _combined_graph_policy_fingerprint = dependencies.combined_graph_policy_fingerprint
    _generation_scope = dependencies.generation_scope
    _graph_precision_feature_flags = dependencies.graph_precision_feature_flags
    _graph_precision_policy_fingerprint = dependencies.graph_precision_policy_fingerprint
    _graph_precision_settings = dependencies.graph_precision_settings
    _graph_tail_replacement_margin = dependencies.graph_tail_replacement_margin
    _resolve_graph_calibration = dependencies.resolve_graph_calibration
    _semantic_graph_indexes = dependencies.semantic_graph_indexes
    _shuffle_graph_relation_endpoints = dependencies.shuffle_graph_relation_endpoints
    _store_graph = dependencies.store_graph
    _validate_security_context = dependencies.validate_security_context
    embed_query = dependencies.embed_query
    evaluate = dependencies.evaluate
    resolve_query_entities = dependencies.resolve_query_entities
    resolve_successor = dependencies.resolve_successor
    supersedes_key = dependencies.supersedes_key
    METRICS = dependencies.metrics
    GRAPH_DIRECTIONAL_RELATIONS = dependencies.graph_directional_relations
    GRAPH_DIAGNOSTIC_ONLY_RELATIONS = dependencies.graph_diagnostic_only_relations
    MAX_GRAPH_RESCORING_CANDIDATES = dependencies.max_graph_rescoring_candidates
    RELATION_KINDS = dependencies.relation_kinds
    started = time.perf_counter()
    variant, relation_control, relation_control_seed, hub_threshold, _cosine_margin = (
        _graph_precision_settings()
    )
    (
        use_directional,
        use_corroboration,
        use_hub_suppression,
        use_calibrated_rerank,
        use_selective_gate,
    ) = _graph_precision_feature_flags(variant)
    policy_fingerprint = _combined_graph_policy_fingerprint(
        security_policy=security_policy,
        graph_policy_fingerprint=_graph_precision_policy_fingerprint(
            (variant, relation_control, relation_control_seed, hub_threshold, _cosine_margin)
        ),
    )
    assert policy_fingerprint is not None
    _validate_security_context(store, security_policy, access_context)
    stats = ExpansionStats(
        relation_seed_activations={relation: 0 for relation in RELATION_KINDS},
        relation_candidates_accepted={relation: 0 for relation in RELATION_KINDS},
        relation_new_trusted_evidence={relation: 0 for relation in RELATION_KINDS},
    )
    performance = request._context.performance

    def reject(reason: str, count: int = 1) -> None:
        stats.reject(reason, count)

    def refuse(reason: str) -> None:
        stats.refuse(reason)

    def finish(
        *,
        result: TrustedResult,
        readiness: str,
        entities: int = 0,
        relations: int = 0,
        candidates: int = 0,
        gate_reason: str | None = None,
        scored_candidates: Sequence[ScoredChunk] = (),
        candidate_relation_types: Mapping[str, tuple[str, ...]] | None = None,
    ) -> SemanticGraphExpansionResult:
        return _finish_expansion(
            started=started,
            stats=stats,
            performance=performance,
            metrics=METRICS,
            policy_fingerprint=policy_fingerprint,
            result=result,
            readiness=readiness,
            entities=entities,
            relations=relations,
            candidates=candidates,
            gate_reason=gate_reason,
            scored_candidates=scored_candidates,
            candidate_relation_types=candidate_relation_types,
        )

    readiness_reader = getattr(store, "graph_readiness", None)
    if performance is None:
        readiness = readiness_reader() if callable(readiness_reader) else None
    else:
        with performance.span("graph_readiness_check_ms"):
            readiness = readiness_reader() if callable(readiness_reader) else None
    if readiness is not None and not readiness.ready:
        refuse("graph_not_ready")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="graph_not_ready",
        )

    trusted_seed_ids = {hit.chunk.id for hit in retrieval.hits if is_trusted(hit)}
    if not trusted_seed_ids:
        refuse("no_trusted_seed")
        return finish(
            result=retrieval,
            readiness="ready",
            gate_reason="no_trusted_seed",
        )
    if (
        not defer_trust_evaluation
        and use_selective_gate
        and len(trusted_seed_ids) >= 2
        and not retrieval.gap_warning
    ):
        refuse("selective_gate")
        return finish(
            result=retrieval,
            readiness="ready",
            gate_reason="graph_gate_not_met",
        )

    # The persisted semantic graph is already the query side's graph metadata. Projecting the
    # store here would stream every chunk, including its text, before this path knows which
    # bounded candidates it needs.
    if request.generation.generation_id:
        if performance is None:
            semantic = _cached_semantic_graph(
                store,
                request.generation.generation_id,
                readiness,
                policy_fingerprint,
            )
        else:
            with performance.span("projection_load_ms"):
                semantic = _cached_semantic_graph(
                    store,
                    request.generation.generation_id,
                    readiness,
                    policy_fingerprint,
                )
    else:
        if performance is None:
            semantic = _store_graph(
                store,
                include_text=False,
                policy_fingerprint=policy_fingerprint,
            ).semantic_graph
        else:
            with performance.span("projection_load_ms"):
                semantic = _store_graph(
                    store,
                    include_text=False,
                    policy_fingerprint=policy_fingerprint,
                ).semantic_graph
    if semantic is None:
        refuse("graph_not_ready")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="graph_not_ready",
        )
    stats.semantic_diagnostic_count = len(semantic.diagnostics)
    if semantic.tenant_id != request.tenant_id:
        refuse("tenant_mismatch")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="tenant_mismatch",
        )
    if (
        retrieval.generation_id
        and semantic.generation_id
        and retrieval.generation_id != semantic.generation_id
    ):
        refuse("generation_mismatch")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="generation_mismatch",
        )
    if (
        request.generation.pipeline_fingerprint
        and semantic.pipeline_fingerprint != request.generation.pipeline_fingerprint
    ):
        refuse("pipeline_mismatch")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="pipeline_mismatch",
        )
    if (
        request.generation.corpus_fingerprint
        and semantic.corpus_fingerprint != request.generation.corpus_fingerprint
    ):
        refuse("corpus_mismatch")
        return finish(
            result=retrieval,
            readiness="GRAPH_NOT_READY",
            gate_reason="corpus_mismatch",
        )
    if relation_control == "removed":
        semantic = replace(semantic, relations=())
    elif relation_control == "shuffled" and semantic.relations:
        semantic = replace(
            semantic,
            relations=_shuffle_graph_relation_endpoints(
                semantic.relations,
                relation_control_seed,
            ),
        )
    indexes = _semantic_graph_indexes(semantic)
    mentions_by_chunk = indexes.mentions_by_chunk
    ambiguous_entities = indexes.ambiguous_entities
    as_of = request.as_of or datetime.now(UTC)
    query_resolution = resolve_query_entities(
        semantic,
        request.query,
        reference_time=as_of,
    )
    resolved_query_entities = set(query_resolution.entity_ids)
    if performance is not None:
        performance.add("query_entities_resolved", len(resolved_query_entities))
        performance.add("query_dates_resolved", len(query_resolution.dates))
        performance.add("query_clauses", len(query_resolution.clauses))
    seed_entities = {
        entity_id
        for chunk_id in trusted_seed_ids
        for entity_id in mentions_by_chunk.get(chunk_id, ())
        if entity_id not in ambiguous_entities
    }
    # Query resolution can activate a graph endpoint that is represented by a unique file or an
    # alias on another chunk. It does not create evidence: relation evidence must still intersect
    # a trusted seed below, and every admitted neighbor still passes ordinary trust evaluation.
    seed_entities.update(resolved_query_entities)

    candidate_budget = max(0, request.budget.max_graph_nodes - len(trusted_seed_ids))
    entity_budget = request.budget.max_graph_entities
    candidates_by_chunk: dict[str, GraphCandidate] = {}
    admission = _admit_relations(
        store=store,
        semantic=semantic,
        indexes=indexes,
        seed_entities=seed_entities,
        resolved_query_entities=resolved_query_entities,
        trusted_seed_ids=trusted_seed_ids,
        as_of=as_of,
        entity_budget=entity_budget,
        use_directional=use_directional,
        graph_directional_relations=GRAPH_DIRECTIONAL_RELATIONS,
        graph_diagnostic_only_relations=GRAPH_DIAGNOSTIC_ONLY_RELATIONS,
        use_hub_suppression=use_hub_suppression,
        hub_threshold=hub_threshold,
        stats=stats,
    )
    candidate_relations_by_chunk = admission.candidate_relations_by_chunk
    if candidate_chunk_ids is not None:
        outside_scope = len(candidate_relations_by_chunk) - sum(
            chunk_id in candidate_chunk_ids for chunk_id in candidate_relations_by_chunk
        )
        if outside_scope:
            reject("outside_candidate_scope", outside_scope)
        candidate_relations_by_chunk = {
            chunk_id: relations
            for chunk_id, relations in candidate_relations_by_chunk.items()
            if chunk_id in candidate_chunk_ids
        }
    relation_count = admission.relation_count
    supersession = dict(admission.supersession)
    unresolved = admission.unresolved
    edge_candidates = admission.edge_candidates

    materialized = _materialize_candidates(
        store=store,
        request=request,
        candidate_relations_by_chunk=candidate_relations_by_chunk,
        indexes=indexes,
        seed_entities=seed_entities,
        trusted_seed_ids=trusted_seed_ids,
        as_of=as_of,
        supersession=supersession,
        edge_candidates=edge_candidates,
        candidate_budget=candidate_budget,
        defer_trust_evaluation=defer_trust_evaluation,
        max_graph_rescoring_candidates=MAX_GRAPH_RESCORING_CANDIDATES,
        performance=performance,
        security_policy=security_policy,
        access_context=access_context,
        stats=stats,
        generation_scope=_generation_scope,
        supersedes_key=supersedes_key,
        resolve_successor=resolve_successor,
        excluded_chunk_ids=excluded_chunk_ids,
        prefetched_chunks={
            chunk_id: candidate.chunk
            for chunk_id, candidate in (prefetched_candidates or {}).items()
        },
    )
    candidates_by_chunk = dict(materialized.candidates_by_chunk)
    chunks_by_id = dict(materialized.chunks_by_id)
    candidate_count = materialized.candidate_count
    scorable_ids = materialized.scorable_ids
    if not candidates_by_chunk:
        reject("no_eligible_relation")
        return finish(
            result=retrieval,
            readiness="ready",
            entities=len(seed_entities),
            relations=relation_count,
            candidates=0,
            gate_reason="graph_gate_not_met",
        )
    if not scorable_ids:
        return finish(
            result=retrieval,
            readiness="ready",
            entities=len(seed_entities),
            relations=relation_count,
            candidates=candidate_count,
            gate_reason="security_policy" if security_policy is not None else "missing_chunk",
        )
    active_calibration = _resolve_graph_calibration(store, request, calibration)
    scored_candidates = _score_candidates(
        store=store,
        request=request,
        embedder=embedder,
        scorable_ids=scorable_ids,
        candidates_by_chunk=candidates_by_chunk,
        chunks_by_id=chunks_by_id,
        active_calibration=active_calibration,
        use_calibrated_rerank=use_calibrated_rerank,
        use_corroboration=use_corroboration,
        performance=performance,
        generation_scope=_generation_scope,
        embed_query=embed_query,
        graph_candidate_rerank_score=graph_candidate_rerank_score,
        stats=stats,
        precomputed_query_scores={
            chunk_id: candidate.score
            for chunk_id, candidate in (prefetched_candidates or {}).items()
        },
    )
    if materialized.metadata_only:
        scored_ids = tuple(
            hit.chunk.id
            for hit in scored_candidates.scored
            if hit.chunk.id not in excluded_chunk_ids
            and hit.chunk.id not in materialized.prefetched_ids
        )[:GRAPH_FIRST_CONTEXT_K]
        text_chunks = (
            _fetch_candidate_chunks(
                store=store,
                request=request,
                candidate_ids=scored_ids,
                performance=performance,
                security_policy=None,
                access_context=None,
                stats=stats,
                generation_scope=_generation_scope,
            )
            if scored_ids
            else {}
        )
        scored_candidates = ScoredGraphCandidates(
            candidate_scores=scored_candidates.candidate_scores,
            scored=tuple(
                replace(
                    hit,
                    chunk=(
                        materialized.chunks_by_id[hit.chunk.id]
                        if hit.chunk.id in materialized.prefetched_ids
                        else text_chunks[hit.chunk.id]
                    ),
                )
                for hit in scored_candidates.scored
                if (hit.chunk.id in text_chunks or hit.chunk.id in materialized.prefetched_ids)
                and hit.chunk.id not in excluded_chunk_ids
            ),
        )
    if defer_trust_evaluation:
        return finish(
            result=retrieval,
            readiness="ready",
            entities=len(seed_entities),
            relations=relation_count,
            candidates=candidate_count,
            scored_candidates=scored_candidates.scored,
            candidate_relation_types={
                hit.chunk.id: tuple(sorted(candidates_by_chunk[hit.chunk.id].relation_types))
                for hit in scored_candidates.scored
                if hit.chunk.id in candidates_by_chunk
            },
        )
    generation_binding: dict[str, str] = {
        "tenant_id": retrieval.tenant_id or store.tenant,
        "generation_id": retrieval.generation_id or semantic.generation_id or "",
        "pipeline_fingerprint": retrieval.pipeline_fingerprint
        or semantic.pipeline_fingerprint
        or "",
        "corpus_fingerprint": retrieval.corpus_fingerprint or semantic.corpus_fingerprint or "",
    }
    expanded = _assemble_trusted_expansion(
        retrieval=retrieval,
        request=request,
        scored_candidates=scored_candidates,
        candidates_by_chunk=candidates_by_chunk,
        supersession=supersession,
        active_calibration=active_calibration,
        as_of=as_of,
        unresolved=unresolved,
        generation_binding=generation_binding,
        performance=performance,
        evaluate=evaluate,
        merge_graph_hits=merge_graph_hits,
        graph_tail_replacement_margin=_graph_tail_replacement_margin,
        stats=stats,
    )
    return finish(
        result=expanded,
        readiness="ready",
        entities=len(seed_entities),
        relations=relation_count,
        candidates=candidate_count,
    )
