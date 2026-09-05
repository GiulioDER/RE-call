"""Deterministic graph-first candidate query construction.

The graph is allowed to nominate query terms before retrieval, but it is never an evidence
source. Every resulting query must still pass the ordinary retrieval and trust layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from recall.reasoning_graph import ReasoningGraphProjection
from recall.semantic_graph import (
    SemanticEntity,
    SemanticGraphProjection,
    normalize_entity_name,
)

GraphFirstMode = Literal["entity", "relation", "hybrid"]
GRAPH_FIRST_RELATIONS = frozenset({"supports", "references", "depends_on", "caused"})
MAX_GRAPH_FIRST_CANDIDATES = 3
MAX_GRAPH_FIRST_QUERY_CHARS = 2_000


@dataclass(frozen=True)
class GraphFirstCandidate:
    """One graph-derived query proposal, with only graph identifiers as provenance."""

    query: str
    kind: Literal["entity", "relation"]
    entity_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "kind": self.kind,
            "entity_ids": list(self.entity_ids),
            "relation_ids": list(self.relation_ids),
        }


def _query_contains_label(query: str, label: str) -> bool:
    normalized_query = normalize_entity_name(query)
    normalized_label = normalize_entity_name(label)
    return bool(normalized_label) and f" {normalized_label} " in f" {normalized_query} "


def _entity_labels(entity: SemanticEntity) -> tuple[str, ...]:
    return tuple(dict.fromkeys((entity.canonical_name, *entity.aliases)))


def _append_candidate(
    candidates: list[GraphFirstCandidate], candidate: GraphFirstCandidate, original_query: str
) -> None:
    if not candidate.query.strip() or len(candidate.query) > MAX_GRAPH_FIRST_QUERY_CHARS:
        return
    if normalize_entity_name(candidate.query) == normalize_entity_name(original_query):
        return
    if any(existing.query == candidate.query for existing in candidates):
        return
    candidates.append(candidate)


def build_graph_first_candidates(
    graph: ReasoningGraphProjection | SemanticGraphProjection,
    query: str,
    *,
    mode: GraphFirstMode = "hybrid",
    max_candidates: int = MAX_GRAPH_FIRST_CANDIDATES,
) -> tuple[GraphFirstCandidate, ...]:
    """Build bounded graph-derived query variants from exact entity and alias matches.

    Only authored directional relations are eligible. Ambiguous entities and model candidate
    relations are excluded. The result is deterministic and does not contain evidence text.
    """

    if mode not in {"entity", "relation", "hybrid"}:
        raise ValueError("mode must be 'entity', 'relation', or 'hybrid'")
    if not 1 <= max_candidates <= MAX_GRAPH_FIRST_CANDIDATES:
        raise ValueError(
            f"max_candidates must be between 1 and {MAX_GRAPH_FIRST_CANDIDATES}"
        )
    if not query.strip() or len(query) > MAX_GRAPH_FIRST_QUERY_CHARS:
        raise ValueError("query must be non-empty and at most 2000 characters")
    semantic = (
        graph.semantic_graph
        if isinstance(graph, ReasoningGraphProjection)
        else graph
    )
    if semantic is None:
        return ()
    ambiguous = {
        entity_id
        for diagnostic in semantic.diagnostics
        if diagnostic.kind == "ambiguous_entity"
        for entity_id in diagnostic.entity_ids
    }
    entities = {
        entity.id: entity
        for entity in semantic.entities
        if entity.id not in ambiguous
    }
    matched = tuple(
        entity
        for entity in sorted(entities.values(), key=lambda item: item.id)
        if any(_query_contains_label(query, label) for label in _entity_labels(entity))
    )
    matched_ids = {entity.id for entity in matched}
    candidates: list[GraphFirstCandidate] = []
    if mode in {"entity", "hybrid"}:
        for entity in matched:
            _append_candidate(
                candidates,
                GraphFirstCandidate(
                    query=f"{query} {entity.canonical_name}",
                    kind="entity",
                    entity_ids=(entity.id,),
                ),
                query,
            )
    if mode in {"relation", "hybrid"}:
        for relation in sorted(semantic.relations, key=lambda item: item.id):
            if relation.status != "authored" or relation.relation not in GRAPH_FIRST_RELATIONS:
                continue
            if relation.subject_id not in matched_ids and relation.object_id not in matched_ids:
                continue
            neighbor_id = (
                relation.object_id
                if relation.subject_id in matched_ids
                else relation.subject_id
            )
            neighbor = entities.get(neighbor_id)
            if neighbor is None:
                continue
            _append_candidate(
                candidates,
                GraphFirstCandidate(
                    query=f"{query} {neighbor.canonical_name} {relation.relation.replace('_', ' ')}",
                    kind="relation",
                    entity_ids=tuple(sorted({relation.subject_id, relation.object_id})),
                    relation_ids=(relation.id,),
                ),
                query,
            )
    return tuple(candidates[:max_candidates])


__all__ = [
    "GRAPH_FIRST_RELATIONS",
    "GraphFirstCandidate",
    "GraphFirstMode",
    "MAX_GRAPH_FIRST_CANDIDATES",
    "MAX_GRAPH_FIRST_QUERY_CHARS",
    "build_graph_first_candidates",
]
