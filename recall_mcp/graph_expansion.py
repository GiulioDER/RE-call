"""Bounded semantic graph expansion for MCP reasoning."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from dataclasses import replace
from typing import TYPE_CHECKING

from recall.embeddings import Embedder, embed_query
from recall.reasoning import ReasoningRequest, SemanticGraphExpansionResult
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    build_reasoning_graph,
    project_store_graph,
)
from recall.trust import evaluate, is_trusted
from recall.types import Chunk, RetrievalResult, ScoredChunk, TrustedResult
from recall.observability import METRICS

if TYPE_CHECKING:
    from recall.calibration import Calibration
    from recall.store import PgVectorStore

MAX_GRAPH_RESCORING_CANDIDATES = 512


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
            relation.object_id if relation.subject_id in seed_entities else relation.subject_id
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
