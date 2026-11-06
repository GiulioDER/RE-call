"""The semantic graph's records, vocabularies and pure helpers.

Imports no database driver, so a reader of a projection does not load one.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, get_args

from recall._frozen import freeze_value as _freeze
from recall.lineage import canonical_sha256

#: Bumped whenever `build_semantic_graph` writes different output for the same chunks. The
#: marker records the version a graph was built with; a graph built by an older version keeps
#: serving under its own version until `recall graph rebuild`, and `validate` rebuilds it.
#: 3 (2026-11-06): NUM-001/002 dates, BUG-001 file identity, DOC-002 supersedes, NUM-003 dedup.
SEMANTIC_GRAPH_SCHEMA_VERSION = 3
#: The version of a marker written before markers recorded one.
LEGACY_GRAPH_SCHEMA_VERSION = 2


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


DiagnosticKind = Literal["ambiguous_entity", "invalid_relation", "missing_evidence"]


# Derived from the Literals, so a kind added to one cannot be missing from the other.
ENTITY_KINDS: tuple[EntityKind, ...] = get_args(EntityKind)


RELATION_KINDS: tuple[RelationKind, ...] = get_args(RelationKind)


RELATION_STATUSES: tuple[RelationStatus, ...] = get_args(RelationStatus)


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
    """One evidence backed edge of the semantic graph, bound to a tenant and generation.

    Traversal runs from `subject_id` to `object_id`. For `supersedes` the subject is the
    replaced document and the object its replacement, so a walk moves from a stale hit to its
    successor. `effective_at`, `valid_from` and `valid_until` are normalised to UTC, and
    `valid_until` must be after `valid_from`.
    """

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
    kind: DiagnosticKind
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
    #: The version the graph was built with (`LEGACY_GRAPH_SCHEMA_VERSION` for an old marker).
    schema_version: int | None = None

    @property
    def stale_schema(self) -> bool:
        """Whether the graph was built by an older builder and wants `recall graph rebuild`."""
        return self.schema_version is not None and self.schema_version < SEMANTIC_GRAPH_SCHEMA_VERSION


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
            schema_version=self.schema_version,
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
        """Atomically replace the graph rows for ``graph.generation_id``.

        The generation's readiness marker is not changed; a graph build writes it separately.
        """

    def load_generation_graph(
        self, tenant_id: str, generation_id: str
    ) -> SemanticGraphProjection | None:
        """Load one generation graph, or ``None`` when it has not been built."""

    def delete_generation_graph(self, generation_id: str | None = None) -> int:
        """Delete all derived graph rows and return the number of root rows removed."""

    def graph_readiness(self, generation_id: str | None = None) -> GraphReadiness:
        """Report the generation's readiness marker, without loading the graph rows.

        The marker is written by a graph build or rebuild; serving compares it with the loaded
        graph's fingerprint before using the graph.
        """


def _identity(kind: str, payload: Mapping[str, Any]) -> str:
    return f"sg_{kind}_{canonical_sha256(dict(payload))[:24]}"


def _diagnostic(
    *,
    tenant_id: str,
    generation_id: str,
    kind: DiagnosticKind,
    reference: str | None,
    message: str,
    identity: Mapping[str, Any] | None = None,
    entity_ids: Sequence[str] = (),
    relation_ids: Sequence[str] = (),
    schema_version: int | None = None,
) -> SemanticGraphDiagnostic:
    """One graph diagnostic, its id derived from the same kind and reference it stores.

    `identity` holds the extra fields that tell two diagnostics of one kind and reference apart.
    At one `SEMANTIC_GRAPH_SCHEMA_VERSION` the id hashes exactly the payload each call site
    hashed before this helper existed, so the helper changed no id; a version bump changes every
    id. `schema_version` keys a diagnostic raised while loading an older graph to that graph's
    own version.
    """
    return SemanticGraphDiagnostic(
        id=_identity(
            "diagnostic",
            {
                "schema_version": (
                    SEMANTIC_GRAPH_SCHEMA_VERSION if schema_version is None else schema_version
                ),
                "tenant_id": tenant_id,
                "generation_id": generation_id,
                "kind": kind,
                "reference": reference,
                **(identity or {}),
            },
        ),
        tenant_id=tenant_id,
        generation_id=generation_id,
        kind=kind,
        reference=reference,
        message=message,
        entity_ids=tuple(entity_ids),
        relation_ids=tuple(relation_ids),
    )


def marker_schema_version(marker: Mapping[str, Any]) -> int:
    """The builder version a graph marker records; a marker from before the field is legacy."""
    value = marker.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return LEGACY_GRAPH_SCHEMA_VERSION
    return value


def _graph_identity(
    *,
    schema_version: int | None = None,
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None,
    corpus_fingerprint: str | None,
    entity_ids: Sequence[str],
    mention_ids: Sequence[str],
    relation_ids: Sequence[str],
    diagnostic_ids: Sequence[str],
) -> str:
    """The graph id, from the one recipe the builder and the loader both use."""
    return _identity(
        "graph",
        {
            "schema_version": (
                SEMANTIC_GRAPH_SCHEMA_VERSION if schema_version is None else schema_version
            ),
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "entities": list(entity_ids),
            "mentions": list(mention_ids),
            "relations": list(relation_ids),
            "diagnostics": list(diagnostic_ids),
        },
    )


_BARE_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")


def _parse_temporal_value(value: Any, field_name: str, *, end_of_day: bool = False) -> datetime | None:
    if value is None or value == "":
        return None
    # Only a bare date runs to the end of its day: an explicit instant that happens to fall at
    # midnight UTC means that instant, as `recall_mcp.graph_expansion.temporal_bounds` reads it.
    bare_date = not isinstance(value, datetime) and bool(_BARE_DATE.fullmatch(str(value).strip()))
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
            # The caller's `end_of_day` decides, as on the ISO path: only `valid_until` runs to
            # the end of its day, so an unpadded `valid_from` still starts at midnight.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    if end_of_day and bare_date:
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
