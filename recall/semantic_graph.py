"""Deterministic, evidence backed semantic graph projection.

The semantic graph is deliberately conservative.  It is derived from chunk metadata and
explicit relation declarations, and every relation points back to one or more source chunks.
It never changes retrieval verdicts or corpus metadata.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
import posixpath
import re
import unicodedata
from datetime import UTC, datetime, time
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from psycopg.types.json import Jsonb

from recall._frozen import freeze_value as _freeze
from recall.frontmatter import dependencies_from_metadata, supersedes_key
from recall.lineage import canonical_sha256
from recall.types import Chunk

SEMANTIC_GRAPH_SCHEMA_VERSION = 2

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
    "supersedes",
]
RelationStatus = Literal["authored", "candidate"]
ExtractionMethod = Literal[
    "metadata", "filename", "heading", "explicit_relation", "explicit_reference"
]

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
    "supersedes",
)
RELATION_STATUSES: tuple[RelationStatus, ...] = ("authored", "candidate")


def normalize_entity_name(value: str) -> str:
    """Return the stable exact matching key for one entity label."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


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
    effective_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_chunk_ids", tuple(sorted(set(self.evidence_chunk_ids))))
        if not self.evidence_chunk_ids:
            raise ValueError("semantic relations require at least one evidence chunk")
        object.__setattr__(self, "uncertainty", tuple(self.uncertainty))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        for name in ("effective_at", "valid_from", "valid_until"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            elif value is not None:
                value = value.astimezone(UTC)
            object.__setattr__(self, name, value)
        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_from >= self.valid_until:
                raise ValueError("semantic relation valid_until must be after valid_from")


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


def read_graph_readiness(conn: Any, tenant_id: str, generation_id: str) -> GraphReadiness:
    """Read the compact generation marker without loading graph member rows.

    The marker is written in the same transaction as the immutable graph. The serving path
    still validates the loaded projection against this fingerprint before caching it, so this
    fast check removes the repeated payload transfer without making a partial graph acceptable.
    """
    row = conn.execute(
        "SELECT validation_summary FROM recall_generations "
        "WHERE tenant_id = %s AND generation_id = %s",
        (tenant_id, generation_id),
    ).fetchone()
    summary = row[0] if row and isinstance(row[0], Mapping) else None
    marker = summary.get("semantic_graph") if isinstance(summary, Mapping) else None
    required = ("graph_id", "graph_fingerprint")
    counts = ("entity_count", "mention_count", "relation_count", "diagnostic_count")
    if (
        not isinstance(marker, Mapping)
        or marker.get("ready") is not True
        or any(not isinstance(marker.get(field), str) or not marker.get(field) for field in required)
        or any(
            (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for field in counts
            for value in (marker.get(field),)
        )
    ):
        return GraphReadiness(
            ready=False,
            tenant_id=tenant_id,
            generation_id=generation_id,
            graph_id=None,
            graph_fingerprint=None,
            entity_count=0,
            mention_count=0,
            relation_count=0,
            diagnostic_count=0,
            reason="GRAPH_NOT_READY",
        )
    return GraphReadiness(
        ready=True,
        tenant_id=tenant_id,
        generation_id=generation_id,
        graph_id=str(marker["graph_id"]),
        graph_fingerprint=str(marker["graph_fingerprint"]),
        entity_count=int(marker["entity_count"]),
        mention_count=int(marker["mention_count"]),
        relation_count=int(marker["relation_count"]),
        diagnostic_count=int(marker["diagnostic_count"]),
    )


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
                "relations": [
                    {
                        "id": relation.id,
                        "effective_at": relation.effective_at,
                        "valid_from": relation.valid_from,
                        "valid_until": relation.valid_until,
                    }
                    for relation in self.relations
                ],
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


def relation_coverage(
    graph: SemanticGraphProjection,
) -> dict[str, dict[str, int]]:
    """Return complete relation and status counts, including kinds with no rows.

    Coverage reports are deliberately zero filled. A SQL ``GROUP BY`` cannot distinguish a
    missing relation kind from a kind that was forgotten by the query, which made the live graph
    census look more complete than it was.
    """
    coverage: dict[str, dict[str, int]] = {
        relation: {status: 0 for status in RELATION_STATUSES}
        for relation in RELATION_KINDS
    }
    for item in graph.relations:
        if item.relation in coverage and item.status in RELATION_STATUSES:
            coverage[item.relation][item.status] += 1
    return coverage


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
    # The delete below is scoped by the graph's own tenant and generation, so every member
    # must carry that same identity: a foreign member would otherwise be written into a
    # scope the delete never clears, and survive the next rebuild.
    for member_kind, members in (
        ("entity", graph.entities),
        ("mention", graph.mentions),
        ("relation", graph.relations),
    ):
        for member in members:
            if member.tenant_id != graph.tenant_id:
                raise ValueError(f"{member_kind} {member.id} tenant_id does not match graph")
            if member.generation_id != graph.generation_id:
                raise ValueError(f"{member_kind} {member.id} generation_id does not match graph")
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
                    graph.tenant_id,
                    graph.generation_id,
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
                    graph.tenant_id,
                    graph.generation_id,
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
                    graph.tenant_id,
                    graph.generation_id,
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
    # One transaction, so the four reads observe one snapshot: a concurrent forget() commits
    # between its statements, and four autocommit SELECTs would each see a different state.
    with conn.transaction():
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
    loaded_relations: list[SemanticRelation] = []
    load_diagnostics: list[SemanticGraphDiagnostic] = []
    for row in relation_rows:
        evidence_chunk_ids = tuple(str(item) for item in (row[11] or ()) if item is not None)
        if not evidence_chunk_ids:
            # A relation whose evidence rows are gone cannot satisfy the dataclass
            # invariant (every relation points back to at least one chunk). Skipping it
            # with a diagnostic keeps the rest of the generation loadable; raising here
            # would abort every load of the generation over one orphaned row.
            relation_id = str(row[0])
            load_diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "missing_evidence",
                            "reference": relation_id,
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="missing_evidence",
                    reference=relation_id,
                    message=f"relation {relation_id} has no surviving evidence rows",
                    relation_ids=(relation_id,),
                )
            )
            continue
        relation_metadata = row[10] or {}
        try:
            effective_at, valid_from, valid_until = _relation_temporal_fields(relation_metadata)
        except ValueError as exc:
            relation_id = str(row[0])
            load_diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "invalid_relation",
                            "reference": relation_id,
                            "value": repr(relation_metadata),
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="invalid_relation",
                    reference=relation_id,
                    message=str(exc),
                )
            )
            continue
        loaded_relations.append(
            SemanticRelation(
                id=str(row[0]),
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=str(row[1]),
                object_id=str(row[2]),
                relation=row[3],
                evidence_chunk_ids=evidence_chunk_ids,
                extraction_method=row[4],
                confidence=float(row[5]),
                status=row[6],
                uncertainty=tuple(row[7] or ()),
                pipeline_fingerprint=str(row[8]) if row[8] else None,
                corpus_fingerprint=str(row[9]) if row[9] else None,
                metadata=relation_metadata,
                effective_at=effective_at,
                valid_from=valid_from,
                valid_until=valid_until,
            )
        )
    relations = tuple(loaded_relations)
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
        diagnostics=tuple(
            sorted((*diagnostics, *load_diagnostics), key=lambda diagnostic: diagnostic.id)
        ),
    )


def _identity(kind: str, payload: Mapping[str, Any]) -> str:
    return f"sg_{kind}_{canonical_sha256(dict(payload))[:24]}"


def _as_labels(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _graph_metadata(chunk: Chunk) -> Mapping[str, Any]:
    value = chunk.metadata.get("recall_graph")
    return value if isinstance(value, Mapping) else {}


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
    graph_entities = _graph_metadata(chunk).get("entities")
    if isinstance(graph_entities, Sequence) and not isinstance(
        graph_entities, (str, bytes, bytearray)
    ):
        for item in graph_entities:
            if not isinstance(item, Mapping):
                continue
            graph_label = item.get("name")
            kind_value = item.get("kind", "unknown")
            if (
                isinstance(graph_label, str)
                and isinstance(kind_value, str)
                and kind_value in ENTITY_KINDS
            ):
                specs.append((graph_label, kind_value, "metadata"))
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


def _chunk_alias_specs(chunk: Chunk) -> list[tuple[str, str]]:
    pairs = _alias_specs(chunk.metadata.get("entity_aliases"))
    graph = _graph_metadata(chunk)
    pairs.extend(_alias_specs(graph.get("aliases")))
    return pairs


def _relation_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _chunk_relation_specs(chunk: Chunk) -> list[dict[str, Any]]:
    relations = _relation_specs(chunk.metadata.get("relations"))
    graph = _graph_metadata(chunk)
    if "__parse_error__" in graph:
        relations.append({"relation": "__invalid_graph_annotation__"})
    relations.extend(_relation_specs(graph.get("relations")))
    return relations


def _relation_metadata(raw: Mapping[str, Any], source: str, target: str | None = None) -> dict[str, Any]:
    """Preserve the bounded provenance labels used by deterministic benchmark relations."""
    metadata: dict[str, Any] = {"source": source}
    for key in ("structural_type", "structural_key"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            metadata[key] = value.strip()
    raw_metadata = raw.get("metadata")
    if isinstance(raw_metadata, Mapping):
        for key in ("structural_type", "structural_key"):
            value = raw_metadata.get(key)
            if isinstance(value, str) and value.strip() and len(value) <= 512:
                metadata[key] = value.strip()
    if target is not None:
        metadata["target"] = target
    return metadata


def _parse_temporal_value(value: Any, field_name: str, *, end_of_day: bool = False) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(f"bad {field_name} date {value!r}") from exc
            end_of_day = True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    if end_of_day and parsed.time() == time.min:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def _relation_temporal_fields(
    raw: Mapping[str, Any],
) -> tuple[datetime | None, datetime | None, datetime | None]:
    nested = raw.get("metadata")
    values: dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
    values.update(raw)
    effective_at = values.get("effective_at", values.get("effective_date"))
    effective = _parse_temporal_value(effective_at, "effective_at")
    valid_from = _parse_temporal_value(values.get("valid_from"), "valid_from")
    valid_until = _parse_temporal_value(
        values.get("valid_until"), "valid_until", end_of_day=True
    )
    if valid_from is not None and valid_until is not None and valid_from >= valid_until:
        raise ValueError("relation valid_until must be after valid_from")
    return effective, valid_from, valid_until


def _temporal_metadata(
    effective_at: datetime | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
) -> dict[str, str]:
    return {
        key: value.isoformat()
        for key, value in (
            ("effective_at", effective_at),
            ("valid_from", valid_from),
            ("valid_until", valid_until),
        )
        if value is not None
    }


def _chunk_temporal_metadata(chunk: Chunk) -> dict[str, str]:
    try:
        valid_from = _parse_temporal_value(chunk.metadata.get("valid_from"), "valid_from")
        valid_until = _parse_temporal_value(
            chunk.metadata.get("valid_until"), "valid_until", end_of_day=True
        )
    except ValueError:
        return {}
    return _temporal_metadata(None, valid_from, valid_until)


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")


def _reference_target(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme or parsed.netloc:
        return ""
    target = value.strip().split("#", 1)[0].split("?", 1)[0]
    target = target.replace("\\", "/")
    return posixpath.basename(target).strip()


def _explicit_reference_targets(text: str) -> tuple[str, ...]:
    targets = [match.group(1) for match in _WIKILINK_RE.finditer(text)]
    targets.extend(match.group(1) for match in _MARKDOWN_LINK_RE.finditer(text))
    return tuple(target for target in (_reference_target(item) for item in targets) if target)


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
    # Secondary indexes over entity_by_key, maintained at entity creation so both alias
    # lookups below stay O(1). Keys are appended in creation order, which is exactly the
    # insertion order a linear scan of entity_by_key would have observed.
    entity_keys_by_name: dict[str, list[tuple[str, EntityKind]]] = defaultdict(list)
    entity_key_by_id: dict[str, tuple[str, EntityKind]] = {}
    labels_by_key: dict[str, set[EntityKind]] = defaultdict(set)
    mentions: list[SemanticMention] = []
    diagnostics: list[SemanticGraphDiagnostic] = []
    declared_aliases = {
        normalize_entity_name(alias)
        for chunk in ordered_chunks
        for _canonical, alias in _chunk_alias_specs(chunk)
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
            entity_keys_by_name[normalized].append(key)
            entity_key_by_id[entity.id] = key
        elif label not in entity.aliases:
            entity = replace(entity, aliases=(*entity.aliases, label))
            entity_by_key[key] = entity
        return entity

    # Each chunk's entity specs are needed by two passes below; splitting text and matching
    # headings per line is expensive enough that it should happen once per chunk, not twice.
    specs_per_chunk = [_entity_specs(chunk) for chunk in ordered_chunks]
    entity_ids_by_chunk: dict[str, dict[str, SemanticEntity]] = {}
    file_entities_by_source: dict[str, set[str]] = defaultdict(set)
    mention_ids: set[str] = set()
    for chunk, chunk_specs in zip(ordered_chunks, specs_per_chunk, strict=True):
        by_normalized: dict[str, SemanticEntity] = {}
        for label, kind, method in chunk_specs:
            normalized = normalize_entity_name(label)
            if not normalized:
                continue
            if normalized in declared_aliases:
                continue
            entity = get_entity(label, kind, method)
            if kind == "file":
                file_entities_by_source[chunk.source].add(entity.id)
            by_normalized.setdefault(normalized, entity)
        entity_ids_by_chunk[chunk.id] = by_normalized

    alias_candidates: dict[str, set[str]] = defaultdict(set)
    alias_text_by_chunk: dict[str, list[tuple[str, SemanticEntity]]] = defaultdict(list)
    for chunk in ordered_chunks:
        for canonical, alias in _chunk_alias_specs(chunk):
            canonical_key = normalize_entity_name(canonical)
            candidates = [
                entity_by_key[key] for key in entity_keys_by_name.get(canonical_key, [])
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

    for chunk, chunk_specs in zip(ordered_chunks, specs_per_chunk, strict=True):
        by_normalized = entity_ids_by_chunk[chunk.id]
        for label, kind, method in chunk_specs:
            normalized = normalize_entity_name(label)
            resolved_entity: SemanticEntity | None
            if normalized in alias_candidates and normalized not in ambiguous_names:
                resolved_entity = entity_by_key[
                    entity_key_by_id[next(iter(alias_candidates[normalized]))]
                ]
                by_normalized[normalized] = resolved_entity
            else:
                resolved_entity = by_normalized.get(normalized)
            if resolved_entity is None:
                continue
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": resolved_entity.id,
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
                    entity_id=resolved_entity.id,
                    chunk_id=chunk.id,
                    mention_text=label,
                    extraction_method=method,
                    metadata=_chunk_temporal_metadata(chunk),
                )
            )

    relations: dict[str, SemanticRelation] = {}
    file_targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source, entity_ids in file_entities_by_source.items():
        file_labels = {source}
        file_labels.update(
            entity_by_key[entity_key_by_id[entity_id]].canonical_name
            for entity_id in entity_ids
        )
        for file_label in file_labels:
            basename = posixpath.basename(file_label)
            stem = basename[:-3] if basename.casefold().endswith(".md") else basename
            for candidate in (file_label, basename, stem):
                file_targets[normalize_entity_name(candidate)].update(
                    (source, entity_id) for entity_id in entity_ids
                )
    file_entity_by_name: dict[str, SemanticEntity] = {}
    ambiguous_file_names: set[str] = set()
    for normalized, targets in file_targets.items():
        if len(targets) == 1:
            _source, entity_id = next(iter(targets))
            file_entity_by_name[normalized] = entity_by_key[entity_key_by_id[entity_id]]
        elif targets:
            ambiguous_file_names.add(normalized)

    # A supersession claim is an authored semantic edge as well as a trust-layer verdict.  Its
    # direction is old document to replacement, so a traversal can move from a stale hit to the
    # document that replaced it.  The trust layer still decides which of the two may be evidence.
    for chunk in ordered_chunks:
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        replacement_id = next(iter(subject_ids))
        for target in _as_labels(chunk.metadata.get("supersedes")):
            target_sources = file_targets.get(normalize_entity_name(supersedes_key(target)), set())
            if len(target_sources) != 1:
                if target_sources:
                    diagnostics.append(
                        SemanticGraphDiagnostic(
                            id=_identity(
                                "diagnostic",
                                {
                                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                    "tenant_id": tenant_id,
                                    "generation_id": generation_id,
                                    "kind": "ambiguous_entity",
                                    "reference": target,
                                    "source": chunk.source,
                                },
                            ),
                            tenant_id=tenant_id,
                            generation_id=generation_id,
                            kind="ambiguous_entity",
                            reference=target,
                            message="supersession target resolves to multiple files",
                        )
                    )
                continue
            _target_source, superseded_id = next(iter(target_sources))
            raw_temporal = {
                "effective_at": chunk.metadata.get("effective_at")
                or chunk.metadata.get("effective_date")
                or chunk.metadata.get("valid_from"),
                "valid_from": chunk.metadata.get("valid_from"),
                "valid_until": chunk.metadata.get("valid_until"),
            }
            try:
                effective_at, valid_from, valid_until = _relation_temporal_fields(raw_temporal)
            except ValueError as exc:
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
                                "value": target,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="invalid_relation",
                        reference=chunk.id,
                        message=str(exc),
                    )
                )
                continue
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": superseded_id,
                    "object_id": replacement_id,
                    "relation": "supersedes",
                    "evidence_chunk_ids": [chunk.id],
                    "effective_at": effective_at.isoformat() if effective_at else None,
                    "valid_from": valid_from.isoformat() if valid_from else None,
                    "valid_until": valid_until.isoformat() if valid_until else None,
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=superseded_id,
                object_id=replacement_id,
                relation="supersedes",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="metadata",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={
                    "source": chunk.source,
                    "target": target,
                    "edge_kind": "supersession",
                    **_temporal_metadata(effective_at, valid_from, valid_until),
                },
                effective_at=effective_at,
                valid_from=valid_from,
                valid_until=valid_until,
            )

    # `recall_graph.depends_on` is an existing authored dependency contract used by the
    # invalidation and lint layers. Project it into the typed graph as a provenance backed edge,
    # but inspect only the first chunk for each source because frontmatter is copied to every
    # chunk produced from one file.
    first_chunk_by_source: dict[str, Chunk] = {}
    for chunk in ordered_chunks:
        first_chunk_by_source.setdefault(chunk.source, chunk)
    for chunk in ordered_chunks:
        if first_chunk_by_source.get(chunk.source) is not chunk:
            continue
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        subject_id = next(iter(subject_ids))
        try:
            dependencies = dependencies_from_metadata(chunk.metadata)
        except ValueError as exc:
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
                            "field": "depends_on",
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="invalid_relation",
                    reference=chunk.id,
                    message=str(exc),
                )
            )
            continue
        for target in dependencies:
            target_sources = file_targets.get(normalize_entity_name(target), set())
            if len(target_sources) != 1:
                if len(target_sources) > 1:
                    diagnostic_kind = "ambiguous_entity"
                    message = "dependency target resolves to multiple files"
                else:
                    diagnostic_kind = "missing_evidence"
                    message = "dependency target does not resolve to a unique file"
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": diagnostic_kind,
                                "reference": chunk.id,
                                "target": target,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind=diagnostic_kind,
                        reference=chunk.id,
                        message=message,
                    )
                )
                continue
            _target_source, object_id = next(iter(target_sources))
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "relation": "depends_on",
                    "evidence_chunk_ids": [chunk.id],
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=subject_id,
                object_id=object_id,
                relation="depends_on",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="metadata",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={
                    "source": chunk.source,
                    "target": target,
                    "edge_kind": "dependency",
                },
            )

    # Markdown and wikilinks are authored source references. They are safe to project as
    # `references` only when the target resolves to exactly one file entity. External URLs,
    # missing files, and duplicate basenames are deliberately skipped and diagnosed.
    for chunk in ordered_chunks:
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        subject_id = next(iter(subject_ids))
        for target in _explicit_reference_targets(chunk.text):
            target_sources = file_targets.get(normalize_entity_name(target), set())
            if len(target_sources) != 1:
                if len(target_sources) > 1:
                    diagnostics.append(
                        SemanticGraphDiagnostic(
                            id=_identity(
                                "diagnostic",
                                {
                                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                    "tenant_id": tenant_id,
                                    "generation_id": generation_id,
                                    "kind": "ambiguous_entity",
                                    "reference": target,
                                    "source": chunk.source,
                                },
                            ),
                            tenant_id=tenant_id,
                            generation_id=generation_id,
                            kind="ambiguous_entity",
                            reference=target,
                            message="explicit file reference resolves to multiple files",
                        )
                    )
                continue
            _target_source, object_id = next(iter(target_sources))
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": object_id,
                    "chunk_id": chunk.id,
                    "mention_text": target,
                },
            )
            if mention_id not in mention_ids:
                mention_ids.add(mention_id)
                mentions.append(
                    SemanticMention(
                        id=mention_id,
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        entity_id=object_id,
                        chunk_id=chunk.id,
                        mention_text=target,
                        extraction_method="explicit_reference",
                        metadata=_chunk_temporal_metadata(chunk),
                    )
                )
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "relation": "references",
                    "evidence_chunk_ids": [chunk.id],
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=subject_id,
                object_id=object_id,
                relation="references",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="explicit_reference",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={"source": chunk.source, "target": target},
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
                    metadata=_chunk_temporal_metadata(chunk),
                )
            )

    for normalized, kinds in sorted(labels_by_key.items()):
        if len(kinds) > 1 and "unknown" not in kinds:
            kind_entity_ids = tuple(
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
                    entity_ids=kind_entity_ids,
                )
            )

    for chunk in ordered_chunks:
        local = entity_ids_by_chunk.get(chunk.id, {})
        for raw in _chunk_relation_specs(chunk):
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
            confidence_value = raw.get("confidence", 1.0)
            if (
                isinstance(confidence_value, bool)
                or not isinstance(confidence_value, (int, float))
                or not math.isfinite(confidence_value)
                or not 0.0 <= confidence_value <= 1.0
            ):
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
                                "field": "confidence",
                                "value": repr(confidence_value),
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="invalid_relation",
                        reference=chunk.id,
                        message="relation confidence must be a finite number between 0.0 and 1.0",
                    )
                )
                continue
            subject_key = normalize_entity_name(subject)
            object_key = normalize_entity_name(object_value)
            subject_entity = local.get(subject_key) or file_entity_by_name.get(subject_key)
            object_entity = local.get(object_key) or file_entity_by_name.get(object_key)
            if (
                subject_key in ambiguous_names
                or object_key in ambiguous_names
                or subject_key in ambiguous_file_names
                or object_key in ambiguous_file_names
            ):
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
                        message=(
                            "relation endpoints must be mentioned by the supporting chunk or "
                            "resolve to a unique file"
                        ),
                    )
                )
                continue
            try:
                effective_at, valid_from, valid_until = _relation_temporal_fields(raw)
            except ValueError as exc:
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
                                "field": "temporal",
                                "value": repr(raw),
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="invalid_relation",
                        reference=chunk.id,
                        message=str(exc),
                    )
                )
                continue
            structural_type = raw.get("structural_type")
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
                    "structural_type": structural_type
                    if isinstance(structural_type, str)
                    else None,
                    "effective_at": effective_at.isoformat() if effective_at else None,
                    "valid_from": valid_from.isoformat() if valid_from else None,
                    "valid_until": valid_until.isoformat() if valid_until else None,
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
                confidence=float(confidence_value),
                status="authored",
                uncertainty=tuple(item for item in raw.get("uncertainty", ()) if isinstance(item, str)),
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={
                    **_relation_metadata(raw, chunk.source),
                    **_temporal_metadata(effective_at, valid_from, valid_until),
                },
                effective_at=effective_at,
                valid_from=valid_from,
                valid_until=valid_until,
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
