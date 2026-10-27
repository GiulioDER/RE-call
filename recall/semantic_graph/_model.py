"""The semantic graph's records, vocabularies and pure helpers.

Imports no database driver, so a reader of a projection does not load one.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any, Literal, Protocol

from recall._frozen import freeze_value as _freeze
from recall.lineage import canonical_sha256

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


@dataclass
class _EntityBuildResult:
    """Mutable entity indexes produced before relation projection begins."""

    entity_by_key: dict[tuple[str, EntityKind], SemanticEntity]
    entity_key_by_id: dict[str, tuple[str, EntityKind]]
    labels_by_key: dict[str, set[EntityKind]]
    mentions: list[SemanticMention]
    diagnostics: list[SemanticGraphDiagnostic]
    entity_ids_by_chunk: dict[str, dict[str, SemanticEntity]]
    file_entities_by_source: dict[str, set[str]]
    mention_ids: set[str]
    alias_text_by_chunk: dict[str, list[tuple[str, SemanticEntity]]]
    ambiguous_names: set[str]


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
                        "effective_at": (
                            relation.effective_at.isoformat() if relation.effective_at else None
                        ),
                        "valid_from": (
                            relation.valid_from.isoformat() if relation.valid_from else None
                        ),
                        "valid_until": (
                            relation.valid_until.isoformat() if relation.valid_until else None
                        ),
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


def _identity(kind: str, payload: Mapping[str, Any]) -> str:
    return f"sg_{kind}_{canonical_sha256(dict(payload))[:24]}"


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
