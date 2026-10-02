"""Deterministic, evidence backed semantic graph projection.

The semantic graph is deliberately conservative.  It is derived from chunk metadata and
explicit relation declarations, and every relation points back to one or more source chunks.
It never changes retrieval verdicts or corpus metadata.
"""

from __future__ import annotations

from recall._frozen import freeze_value as _freeze  # noqa: F401  # imported from here by tests
from recall.semantic_graph._build import (
    build_semantic_graph,
)
from recall.semantic_graph._model import (
    ENTITY_KINDS,
    RELATION_KINDS,
    RELATION_STATUSES,
    SEMANTIC_GRAPH_SCHEMA_VERSION as SEMANTIC_GRAPH_SCHEMA_VERSION,
    EntityKind as EntityKind,
    ExtractionMethod as ExtractionMethod,
    GraphReadiness,
    RelationKind as RelationKind,
    RelationStatus as RelationStatus,
    SemanticEntity,
    SemanticGraphDiagnostic,
    SemanticGraphProjection,
    SemanticGraphStore as SemanticGraphStore,
    SemanticMention,
    SemanticRelation,
    normalize_entity_name,
    relation_coverage,
)
from recall.semantic_graph._store import (
    delete_semantic_graph,
    load_semantic_graph,
    read_graph_readiness as read_graph_readiness,
    write_semantic_graph,
)

__all__ = [
    "ENTITY_KINDS",
    "RELATION_KINDS",
    "RELATION_STATUSES",
    "GraphReadiness",
    "SemanticEntity",
    "SemanticGraphDiagnostic",
    "SemanticGraphProjection",
    "SemanticMention",
    "SemanticRelation",
    "build_semantic_graph",
    "delete_semantic_graph",
    "load_semantic_graph",
    "normalize_entity_name",
    "relation_coverage",
    "write_semantic_graph",
]
