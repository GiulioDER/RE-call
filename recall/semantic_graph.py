"""Deterministic, evidence backed semantic graph projection.

The semantic graph is deliberately conservative.  It is derived from chunk metadata and
explicit relation declarations, and every relation points back to one or more source chunks.
It never changes retrieval verdicts or corpus metadata.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import re
from types import MappingProxyType
import unicodedata
from typing import Any, Literal, Protocol

from psycopg.types.json import Jsonb

from recall.lineage import canonical_sha256
from recall.types import Chunk

SEMANTIC_GRAPH_SCHEMA_VERSION = 1

EntityKind = Literal[
    "person",
    "project",
    "service",
    "file",
    "decision",
    "event",
    "concept",
    "unknown",
]
RelationKind = Literal[
    "supports",
    "contradicts",
    "references",
    "depends_on",
    "caused",
    "same_entity",
]
RelationStatus = Literal["authored", "candidate"]
ExtractionMethod = Literal["metadata", "filename", "heading", "explicit_relation"]

ENTITY_KINDS: tuple[EntityKind, ...] = (
    "person",
    "project",
    "service",
    "file",
    "decision",
    "event",
    "concept",
    "unknown",
)
RELATION_KINDS: tuple[RelationKind, ...] = (
    "supports",
    "contradicts",
    "references",
    "depends_on",
    "caused",
    "same_entity",
)


def normalize_entity_name(value: str) -> str:
    """Return the stable exact matching key for one entity label."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class SemanticEntity:
    id: str
    tenant_id: str
    generation_id: str
    canonical_name: str
    normalized_name: str
    kind: EntityKind
    aliases: tuple[str, ...] = ()
    extraction_method: ExtractionMethod = "metadata"
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class SemanticMention:
    id: str
    tenant_id: str
    generation_id: str
    entity_id: str
    chunk_id: str
    mention_text: str
    extraction_method: ExtractionMethod
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class SemanticRelation:
    id: str
    tenant_id: str
    generation_id: str
    subject_id: str
    object_id: str
    relation: RelationKind
    evidence_chunk_ids: tuple[str, ...]
    extraction_method: ExtractionMethod
    confidence: float
    status: RelationStatus = "authored"
    uncertainty: tuple[str, ...] = ()
    pipeline_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_chunk_ids", tuple(sorted(set(self.evidence_chunk_ids))))
        if not self.evidence_chunk_ids:
            raise ValueError("semantic relations require at least one evidence chunk")
        object.__setattr__(self, "uncertainty", tuple(self.uncertainty))
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class SemanticGraphDiagnostic:
    id: str
    tenant_id: str
    generation_id: str
    kind: Literal["ambiguous_entity", "invalid_relation", "missing_evidence"]
    reference: str | None
    message: str
    entity_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphReadiness:
    ready: bool
    tenant_id: str
    generation_id: str
    graph_id: str | None
    graph_fingerprint: str | None
    entity_count: int
    mention_count: int
    relation_count: int
    diagnostic_count: int
    reason: str | None = None


@dataclass(frozen=True)
class SemanticGraphProjection:
    schema_version: int
    graph_id: str
    tenant_id: str
    generation_id: str
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    entities: tuple[SemanticEntity, ...]
    mentions: tuple[SemanticMention, ...]
    relations: tuple[SemanticRelation, ...]
    diagnostics: tuple[SemanticGraphDiagnostic, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "graph_id": self.graph_id,
                "entities": [entity.id for entity in self.entities],
                "mentions": [mention.id for mention in self.mentions],
                "relations": [relation.id for relation in self.relations],
                "diagnostics": [diagnostic.id for diagnostic in self.diagnostics],
            }
        )

    def readiness(self) -> GraphReadiness:
        return GraphReadiness(
            ready=True,
            tenant_id=self.tenant_id,
            generation_id=self.generation_id,
            graph_id=self.graph_id,
            graph_fingerprint=self.fingerprint,
            entity_count=len(self.entities),
            mention_count=len(self.mentions),
            relation_count=len(self.relations),
            diagnostic_count=len(self.diagnostics),
        )


class SemanticGraphStore(Protocol):
    """Persistence contract for an immutable, tenant and generation bound graph."""

    def write_generation_graph(self, graph: SemanticGraphProjection) -> None:
        """Atomically replace the graph rows for ``graph.generation_id``."""

    def load_generation_graph(
        self, tenant_id: str, generation_id: str
    ) -> SemanticGraphProjection | None:
        """Load one generation graph, or ``None`` when it has not been built."""

    def delete_generation_graph(self, tenant_id: str, generation_id: str) -> int:
        """Delete all derived graph rows and return the number of root rows removed."""

    def graph_readiness(self, tenant_id: str, generation_id: str) -> GraphReadiness:
        """Report whether the persisted graph matches the generation marker."""

def write_semantic_graph(conn: Any, graph: SemanticGraphProjection) -> None:
    """Replace one generation's graph rows inside the caller's transaction."""
    conn.execute(
        "DELETE FROM recall_graph_entities_v1 WHERE tenant_id = %s AND generation_id = %s",
        (graph.tenant_id, graph.generation_id),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO recall_graph_entities_v1 "
            "(tenant_id, generation_id, entity_id, canonical_name, normalized_name, entity_kind, "
            "aliases, extraction_method, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    entity.tenant_id,
                    entity.generation_id,
                    entity.id,
                    entity.canonical_name,
                    entity.normalized_name,
                    entity.kind,
                    Jsonb(list(entity.aliases)),
                    entity.extraction_method,
                    entity.confidence,
                    Jsonb(_thaw(entity.metadata)),
                )
                for entity in graph.entities
            ],
        )

        cur.executemany(
            "INSERT INTO recall_graph_mentions_v1 "
            "(tenant_id, generation_id, mention_id, entity_id, chunk_id, mention_text, "
            "extraction_method, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    mention.tenant_id,
                    mention.generation_id,
                    mention.id,
                    mention.entity_id,
                    mention.chunk_id,
                    mention.mention_text,
                    mention.extraction_method,
                    mention.confidence,
                    Jsonb(_thaw(mention.metadata)),
                )
                for mention in graph.mentions
            ],
        )
        cur.executemany(
            "INSERT INTO recall_graph_relations_v1 "
            "(tenant_id, generation_id, relation_id, subject_id, object_id, relation, "
            "extraction_method, confidence, status, uncertainty, pipeline_fingerprint, "
            "corpus_fingerprint, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    relation.tenant_id,
                    relation.generation_id,
                    relation.id,
                    relation.subject_id,
                    relation.object_id,
                    relation.relation,
                    relation.extraction_method,
                    relation.confidence,
                    relation.status,
                    Jsonb(list(relation.uncertainty)),
                    relation.pipeline_fingerprint,
                    relation.corpus_fingerprint,
                    Jsonb(_thaw(relation.metadata)),
                )
                for relation in graph.relations
            ],
        )
        cur.executemany(
            "INSERT INTO recall_graph_relation_evidence_v1 "
            "(tenant_id, generation_id, relation_id, chunk_id) VALUES (%s, %s, %s, %s)",
            [
                (graph.tenant_id, graph.generation_id, relation.id, chunk_id)
                for relation in graph.relations
                for chunk_id in relation.evidence_chunk_ids
            ],
        )


def delete_semantic_graph(conn: Any, tenant_id: str, generation_id: str) -> int:
    """Delete all graph rows for one tenant and generation inside the caller's transaction."""
    result = conn.execute(
        "DELETE FROM recall_graph_entities_v1 WHERE tenant_id = %s AND generation_id = %s",
        (tenant_id, generation_id),
    )
    return int(result.rowcount)


def load_semantic_graph(conn: Any, tenant_id: str, generation_id: str) -> SemanticGraphProjection | None:
    """Load a generation graph, returning ``None`` when no graph has been built."""
    generation_row = conn.execute(
        "SELECT pipeline_fingerprint, corpus_fingerprint, validation_summary "
        "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
        (tenant_id, generation_id),
    ).fetchone()
    entity_rows = conn.execute(
        "SELECT entity_id, canonical_name, normalized_name, entity_kind, aliases, "
        "extraction_method, confidence, metadata FROM recall_graph_entities_v1 "
        "WHERE tenant_id = %s AND generation_id = %s ORDER BY entity_id",
        (tenant_id, generation_id),
    ).fetchall()
    mention_rows = conn.execute(
        "SELECT mention_id, entity_id, chunk_id, mention_text, extraction_method, "
        "confidence, metadata FROM recall_graph_mentions_v1 "
        "WHERE tenant_id = %s AND generation_id = %s ORDER BY mention_id",
        (tenant_id, generation_id),
    ).fetchall()
    relation_rows = conn.execute(
        "SELECT r.relation_id, r.subject_id, r.object_id, r.relation, r.extraction_method, "
        "r.confidence, r.status, r.uncertainty, r.pipeline_fingerprint, r.corpus_fingerprint, "
        "r.metadata, array_agg(e.chunk_id ORDER BY e.chunk_id) "
        "FROM recall_graph_relations_v1 r LEFT JOIN recall_graph_relation_evidence_v1 e "
        "ON e.tenant_id = r.tenant_id AND e.generation_id = r.generation_id "
        "AND e.relation_id = r.relation_id WHERE r.tenant_id = %s AND r.generation_id = %s "
        "GROUP BY r.relation_id, r.subject_id, r.object_id, r.relation, "
        "r.extraction_method, r.confidence, r.status, r.uncertainty, "
        "r.pipeline_fingerprint, r.corpus_fingerprint, r.metadata "
        "ORDER BY r.relation_id",
        (tenant_id, generation_id),
    ).fetchall()
    marker = generation_row[2].get("semantic_graph") if generation_row and isinstance(generation_row[2], dict) else None
    if not entity_rows and not mention_rows and not relation_rows and not isinstance(marker, dict):
        return None
    entities = tuple(
        SemanticEntity(
            id=str(row[0]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            canonical_name=str(row[1]),
            normalized_name=str(row[2]),
            kind=row[3],
            aliases=tuple(row[4] or ()),
            extraction_method=row[5],
            confidence=float(row[6]),
            metadata=row[7] or {},
        )
        for row in entity_rows
    )
    mentions = tuple(
        SemanticMention(
            id=str(row[0]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            entity_id=str(row[1]),
            chunk_id=str(row[2]),
            mention_text=str(row[3]),
            extraction_method=row[4],
            confidence=float(row[5]),
            metadata=row[6] or {},
        )
        for row in mention_rows
    )
    relations = tuple(
        SemanticRelation(
            id=str(row[0]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            subject_id=str(row[1]),
            object_id=str(row[2]),
            relation=row[3],
            evidence_chunk_ids=tuple(str(item) for item in (row[11] or ()) if item is not None),
            extraction_method=row[4],
            confidence=float(row[5]),
            status=row[6],
            uncertainty=tuple(row[7] or ()),
            pipeline_fingerprint=str(row[8]) if row[8] else None,
            corpus_fingerprint=str(row[9]) if row[9] else None,
            metadata=row[10] or {},
        )
        for row in relation_rows
    )
    diagnostics = tuple(
        SemanticGraphDiagnostic(
            id=str(item["id"]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind=item["kind"],
            reference=str(item["reference"]) if item.get("reference") is not None else None,
            message=str(item["message"]),
            entity_ids=tuple(str(value) for value in item.get("entity_ids", ())),
            relation_ids=tuple(str(value) for value in item.get("relation_ids", ())),
        )
        for item in (marker.get("diagnostics", ()) if isinstance(marker, dict) else ())
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("kind"), str)
        and isinstance(item.get("message"), str)
    )
    graph_id = (
        str(marker["graph_id"])
        if isinstance(marker, dict) and marker.get("graph_id")
        else _identity(
            "graph",
            {
                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "generation_id": generation_id,
                "pipeline_fingerprint": generation_row[0] if generation_row else None,
                "corpus_fingerprint": generation_row[1] if generation_row else None,
                "entities": [entity.id for entity in entities],
                "mentions": [mention.id for mention in mentions],
                "relations": [relation.id for relation in relations],
                "diagnostics": (),
            },
        )
    )
    return SemanticGraphProjection(
        schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
        graph_id=graph_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=(
            str(generation_row[0])
            if generation_row and generation_row[0]
            else (relations[0].pipeline_fingerprint if relations else None)
        ),
        corpus_fingerprint=(
            str(generation_row[1])
            if generation_row and generation_row[1]
            else (relations[0].corpus_fingerprint if relations else None)
        ),
        entities=entities,
        mentions=mentions,
        relations=relations,
        diagnostics=tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.id)),
    )


def _identity(kind: str, payload: Mapping[str, Any]) -> str:
    return f"sg_{kind}_{canonical_sha256(dict(payload))[:24]}"


def _as_labels(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _entity_specs(chunk: Chunk) -> list[tuple[str, EntityKind, ExtractionMethod]]:
    specs: list[tuple[str, EntityKind, ExtractionMethod]] = []
    file_name = chunk.metadata.get("file")
    if isinstance(file_name, str) and file_name.strip():
        specs.append((file_name.strip(), "file", "filename"))
    for key in ENTITY_KINDS:
        for label in _as_labels(chunk.metadata.get(key)):
            specs.append((label, key, "metadata"))
    for metadata_key in ("entities",):
        for label in _as_labels(chunk.metadata.get(metadata_key)):
            specs.append((label, "unknown", "metadata"))
    for line in chunk.text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if heading:
            specs.append((heading.group(1).strip(), "concept", "heading"))
    return specs


def _alias_specs(value: Any) -> list[tuple[str, str]]:
    """Read the canonical-name -> aliases form of explicit alias metadata."""
    if not isinstance(value, Mapping):
        return []
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in value.items():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        for alias in _as_labels(aliases):
            pairs.append((canonical.strip(), alias))
    return pairs


def _relation_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def build_semantic_graph(
    chunks: Sequence[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
) -> SemanticGraphProjection:
    """Build a deterministic graph from explicit metadata and conservative headings."""
    ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.id))
    entity_by_key: dict[tuple[str, EntityKind], SemanticEntity] = {}
    labels_by_key: dict[str, set[EntityKind]] = defaultdict(set)
    mentions: list[SemanticMention] = []
    diagnostics: list[SemanticGraphDiagnostic] = []
    declared_aliases = {
        normalize_entity_name(alias)
        for chunk in ordered_chunks
        for _canonical, alias in _alias_specs(chunk.metadata.get("entity_aliases"))
    }

    def get_entity(label: str, kind: EntityKind, method: ExtractionMethod) -> SemanticEntity:
        normalized = normalize_entity_name(label)
        key = (normalized, kind)
        labels_by_key[normalized].add(kind)
        entity = entity_by_key.get(key)
        if entity is None:
            entity = SemanticEntity(
                id=_identity(
                    "entity",
                    {
                        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                        "tenant_id": tenant_id,
                        "generation_id": generation_id,
                        "normalized_name": normalized,
                        "kind": kind,
                    },
                ),
                tenant_id=tenant_id,
                generation_id=generation_id,
                canonical_name=label,
                normalized_name=normalized,
                kind=kind,
                aliases=(label,),
                extraction_method=method,
            )
            entity_by_key[key] = entity
        elif label not in entity.aliases:
            entity = replace(entity, aliases=(*entity.aliases, label))
            entity_by_key[key] = entity
        return entity

    entity_ids_by_chunk: dict[str, dict[str, SemanticEntity]] = {}
    mention_ids: set[str] = set()
    for chunk in ordered_chunks:
        by_normalized: dict[str, SemanticEntity] = {}
        for label, kind, method in _entity_specs(chunk):
            normalized = normalize_entity_name(label)
            entity: SemanticEntity | None
            if not normalized:
                continue
            if normalized in declared_aliases:
                continue
            entity = get_entity(label, kind, method)
            by_normalized.setdefault(normalized, entity)
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": entity.id,
                    "chunk_id": chunk.id,
                    "mention_text": label,
                },
            )
            by_normalized.setdefault(normalized, entity)
        entity_ids_by_chunk[chunk.id] = by_normalized

    alias_candidates: dict[str, set[str]] = defaultdict(set)
    alias_text_by_chunk: dict[str, list[tuple[str, SemanticEntity]]] = defaultdict(list)
    for chunk in ordered_chunks:
        for canonical, alias in _alias_specs(chunk.metadata.get("entity_aliases")):
            canonical_key = normalize_entity_name(canonical)
            candidates = [
                entity
                for (normalized, _kind), entity in entity_by_key.items()
                if normalized == canonical_key
            ]
            if not candidates:
                candidates = [get_entity(canonical, "unknown", "metadata")]
            if len(candidates) != 1:
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": "ambiguous_entity",
                                "reference": normalize_entity_name(alias),
                                "canonical": canonical_key,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="ambiguous_entity",
                        reference=normalize_entity_name(alias),
                        message="entity alias resolves to multiple canonical entities",
                        entity_ids=tuple(sorted(entity.id for entity in candidates)),
                    )
                )
                continue
            entity = candidates[0]
            alias_key = normalize_entity_name(alias)
            alias_candidates[alias_key].add(entity.id)
            alias_text_by_chunk[chunk.id].append((alias, entity))
            entity = replace(entity, aliases=tuple(sorted(set((*entity.aliases, alias)))) )
            entity_by_key[(normalize_entity_name(entity.canonical_name), entity.kind)] = entity

    ambiguous_names = {
        normalized for normalized, entity_ids in alias_candidates.items() if len(entity_ids) > 1
    }
    for normalized in sorted(ambiguous_names):
        diagnostics.append(
            SemanticGraphDiagnostic(
                id=_identity(
                    "diagnostic",
                    {
                        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                        "tenant_id": tenant_id,
                        "generation_id": generation_id,
                        "kind": "ambiguous_entity",
                        "reference": normalized,
                    },
                ),
                tenant_id=tenant_id,
                generation_id=generation_id,
                kind="ambiguous_entity",
                reference=normalized,
                message="entity alias resolves to multiple canonical entities",
                entity_ids=tuple(sorted(alias_candidates[normalized])),
            )
        )

    for chunk in ordered_chunks:
        by_normalized = entity_ids_by_chunk[chunk.id]
        for label, kind, method in _entity_specs(chunk):
            normalized = normalize_entity_name(label)
            if normalized in alias_candidates and normalized not in ambiguous_names:
                entity = next(
                    entity for entity in entity_by_key.values() if entity.id == next(iter(alias_candidates[normalized]))
                )
                by_normalized[normalized] = entity
            else:
                entity = by_normalized.get(normalized)
            if entity is None:
                continue
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": entity.id,
                    "chunk_id": chunk.id,
                    "mention_text": label,
                },
            )
            if mention_id in mention_ids:
                continue
            mention_ids.add(mention_id)
            mentions.append(
                SemanticMention(
                    id=mention_id,
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    entity_id=entity.id,
                    chunk_id=chunk.id,
                    mention_text=label,
                    extraction_method=method,
                )
            )
        for alias, entity in alias_text_by_chunk.get(chunk.id, ()):
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": entity.id,
                    "chunk_id": chunk.id,
                    "mention_text": alias,
                },
            )
            if mention_id in mention_ids:
                continue
            mention_ids.add(mention_id)
            mentions.append(
                SemanticMention(
                    id=mention_id,
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    entity_id=entity.id,
                    chunk_id=chunk.id,
                    mention_text=alias,
                    extraction_method="metadata",
                )
            )

    for normalized, kinds in sorted(labels_by_key.items()):
        if len(kinds) > 1 and "unknown" not in kinds:
            entity_ids = tuple(
                entity_by_key[(normalized, kind)].id for kind in sorted(kinds)
            )
            diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "ambiguous_entity",
                            "reference": normalized,
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="ambiguous_entity",
                    reference=normalized,
                    message=f"entity label {normalized!r} has multiple explicit kinds",
                    entity_ids=entity_ids,
                )
            )

    relations: dict[str, SemanticRelation] = {}
    for chunk in ordered_chunks:
        local = entity_ids_by_chunk.get(chunk.id, {})
        for raw in _relation_specs(chunk.metadata.get("relations")):
            relation = raw.get("relation")
            subject = raw.get("subject")
            object_value = raw.get("object")
            if relation not in RELATION_KINDS or not isinstance(subject, str) or not isinstance(object_value, str):
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": "invalid_relation",
                                "reference": chunk.id,
                                "value": raw,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="invalid_relation",
                        reference=chunk.id,
                        message="relation metadata must name a supported relation and two strings",
                    )
                )
                continue
            subject_key = normalize_entity_name(subject)
            object_key = normalize_entity_name(object_value)
            subject_entity = local.get(subject_key)
            object_entity = local.get(object_key)
            if subject_key in ambiguous_names or object_key in ambiguous_names:
                subject_entity = None
                object_entity = None
            if subject_entity is None or object_entity is None:
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": "missing_evidence",
                                "reference": chunk.id,
                                "subject": subject,
                                "object": object_value,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="missing_evidence",
                        reference=chunk.id,
                        message="relation endpoints must be mentioned by the supporting chunk",
                    )
                )
                continue
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": subject_entity.id,
                    "object_id": object_entity.id,
                    "relation": relation,
                    "evidence_chunk_ids": [chunk.id],
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=subject_entity.id,
                object_id=object_entity.id,
                relation=relation,
                evidence_chunk_ids=(chunk.id,),
                extraction_method="explicit_relation",
                confidence=float(raw.get("confidence", 1.0)),
                status="authored",
                uncertainty=tuple(item for item in raw.get("uncertainty", ()) if isinstance(item, str)),
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={"source": chunk.source},
            )

    entities = tuple(sorted(entity_by_key.values(), key=lambda entity: entity.id))
    ordered_mentions = tuple(sorted(mentions, key=lambda mention: mention.id))
    ordered_relations = tuple(sorted(relations.values(), key=lambda relation: relation.id))
    ordered_diagnostics = tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.id))
    graph_id = _identity(
        "graph",
        {
            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "entities": [entity.id for entity in entities],
            "mentions": [mention.id for mention in ordered_mentions],
            "relations": [relation.id for relation in ordered_relations],
            "diagnostics": [diagnostic.id for diagnostic in ordered_diagnostics],
        },
    )
    return SemanticGraphProjection(
        schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
        graph_id=graph_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        entities=entities,
        mentions=ordered_mentions,
        relations=ordered_relations,
        diagnostics=ordered_diagnostics,
    )


__all__ = [
    "ENTITY_KINDS",
    "RELATION_KINDS",
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
    "write_semantic_graph",
]
