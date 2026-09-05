"""Deterministic authority and source dependency invalidation.

The dependency graph is deliberately separate from the semantic graph.  Semantic relations are
useful for discovery, while this module answers the stricter question: may a memory still be
served when an explicitly named prerequisite is no longer trustworthy?
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, cast

from psycopg.types.json import Jsonb

from recall.frontmatter import authority_from_metadata, dependencies_from_metadata
from recall.lineage import canonical_sha256
from recall.types import Authority, Chunk, DependencyCause, InvalidationReason

AUTHORITY_VALUES: tuple[Authority, ...] = (
    "policy",
    "user_confirmed_decision",
    "tool_observation",
    "model_inference",
    "unknown",
)
AUTHORITY_RANK: dict[Authority, int] = {
    "policy": 4,
    "user_confirmed_decision": 3,
    "tool_observation": 2,
    "model_inference": 1,
    "unknown": 0,
}

BaseState = Literal[
    "current",
    "superseded",
    "expired",
    "not_yet_valid",
    "not_yet_known",
    "ambiguous",
    "invalid",
]

INVALIDATING_STATES: frozenset[str] = frozenset(
    {
        "superseded",
        "expired",
        "not_yet_valid",
        "not_yet_known",
        "ambiguous",
        "invalid",
    }
)
MAX_INVALIDATION_DEPTH = 64
MAX_INVALIDATION_CHAIN = 16


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def source_file(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    return value if isinstance(value, str) and value else chunk.source


@dataclass(frozen=True)
class DependencyEdge:
    id: str
    tenant_id: str
    generation_id: str
    dependent: str
    prerequisite: str
    asserting_chunk_id: str
    authority: Authority
    asserted_at: datetime | None = None


@dataclass(frozen=True)
class DependencyDiagnostic:
    kind: Literal[
        "malformed_metadata",
        "inconsistent_authority",
        "duplicate_dependency",
        "self_dependency",
        "unresolved_dependency",
        "dependency_cycle",
    ]
    source: str
    message: str
    dependency: str | None = None


@dataclass(frozen=True)
class DependencyProjection:
    schema_version: int
    projection_id: str
    tenant_id: str
    generation_id: str
    authorities: Mapping[str, Authority]
    dependencies: Mapping[str, tuple[str, ...]]
    edges: tuple[DependencyEdge, ...]
    invalidations: Mapping[str, InvalidationReason]
    reverse: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    diagnostics: tuple[DependencyDiagnostic, ...] = ()
    max_depth: int = MAX_INVALIDATION_DEPTH
    as_of: datetime | None = None
    known_as_of: datetime | None = None
    corpus_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorities",
            MappingProxyType({key: self.authorities[key] for key in sorted(self.authorities)}),
        )
        object.__setattr__(
            self,
            "dependencies",
            MappingProxyType(
                {key: tuple(self.dependencies[key]) for key in sorted(self.dependencies)}
            ),
        )
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda edge: edge.id)))
        object.__setattr__(
            self,
            "invalidations",
            MappingProxyType(
                {key: self.invalidations[key] for key in sorted(self.invalidations)}
            ),
        )
        object.__setattr__(
            self,
            "reverse",
            MappingProxyType(
                {key: tuple(sorted(self.reverse[key])) for key in sorted(self.reverse)}
            ),
        )
        object.__setattr__(self, "diagnostics", tuple(sorted(self.diagnostics, key=lambda d: (d.source, d.kind, d.dependency or ""))))

    def reason_for(self, source: str) -> InvalidationReason | None:
        return self.invalidations.get(source)


def write_dependency_projection(conn: Any, projection: DependencyProjection) -> None:
    """Replace the derived dependency rows for one generation inside a transaction."""
    conn.execute(
        "DELETE FROM recall_dependency_edges_v1 WHERE tenant_id = %s AND generation_id = %s",
        (projection.tenant_id, projection.generation_id),
    )
    conn.execute(
        "DELETE FROM recall_dependency_diagnostics_v1 WHERE tenant_id = %s AND generation_id = %s",
        (projection.tenant_id, projection.generation_id),
    )
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO recall_dependency_edges_v1 "
            "(tenant_id, generation_id, edge_id, dependent_source, prerequisite_source, "
            "asserting_chunk_id, authority, asserted_at, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    edge.tenant_id,
                    edge.generation_id,
                    edge.id,
                    edge.dependent,
                    edge.prerequisite,
                    edge.asserting_chunk_id,
                    edge.authority,
                    edge.asserted_at,
                    Jsonb({}),
                )
                for edge in projection.edges
            ],
        )
        cursor.executemany(
            "INSERT INTO recall_dependency_diagnostics_v1 "
            "(tenant_id, generation_id, diagnostic_id, kind, source, dependency, message) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    projection.tenant_id,
                    projection.generation_id,
                    "depdiag_" + canonical_sha256(
                        {
                            "kind": diagnostic.kind,
                            "source": diagnostic.source,
                            "dependency": diagnostic.dependency,
                        }
                    )[:24],
                    diagnostic.kind,
                    diagnostic.source,
                    diagnostic.dependency,
                    diagnostic.message,
                )
                for diagnostic in projection.diagnostics
            ],
        )


def _edge_id(
    tenant_id: str,
    generation_id: str,
    dependent: str,
    prerequisite: str,
    asserting_chunk_id: str,
    asserted_at: datetime | None,
) -> str:
    return "dep_" + canonical_sha256(
        {
            "schema_version": 1,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "dependent": dependent,
            "prerequisite": prerequisite,
            "asserting_chunk_id": asserting_chunk_id,
            "asserted_at": asserted_at.isoformat() if asserted_at else None,
        }
    )[:24]


def build_dependency_projection(
    chunks: Sequence[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
    base_states: Mapping[str, str] | None = None,
    schema_version: int = 1,
    as_of: datetime | None = None,
    known_as_of: datetime | None = None,
    asserted_at_by_source: Mapping[str, datetime | None] | None = None,
    corpus_fingerprint: str | None = None,
) -> DependencyProjection:
    """Build an immutable dependency projection from one generation's chunks.

    ``base_states`` is supplied by the current state projection or a trust caller.  The function
    never guesses a state from retrieval scores.  Missing states are treated as current, which
    keeps this pure builder useful for authoring and lint tests while callers that have temporal
    state can provide the complete map.
    """
    as_of = _utc(as_of)
    known_as_of = _utc(known_as_of)
    chunks_by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[source_file(chunk)].append(chunk)
    known_sources = set(chunks_by_source)
    authorities: dict[str, Authority] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    diagnostics: list[DependencyDiagnostic] = []
    edges: list[DependencyEdge] = []
    hard_failures: dict[str, Literal["invalid", "ambiguous", "cycle"]] = {}

    for source in sorted(chunks_by_source):
        source_chunks = sorted(chunks_by_source[source], key=lambda item: item.id)
        source_authorities: set[Authority] = set()
        source_dependencies: list[str] = []
        for chunk in source_chunks:
            authority: Authority = "unknown"
            try:
                authority = authority_from_metadata(chunk.metadata)
            except ValueError as exc:
                diagnostics.append(
                    DependencyDiagnostic("malformed_metadata", source, str(exc))
                )
                hard_failures[source] = "invalid"
            else:
                source_authorities.add(authority)
            try:
                declared = dependencies_from_metadata(chunk.metadata)
            except ValueError as exc:
                diagnostics.append(
                    DependencyDiagnostic("malformed_metadata", source, str(exc))
                )
                if authority not in {"model_inference"}:
                    hard_failures[source] = "invalid"
            else:
                source_dependencies.extend(declared)
        if len(source_authorities) > 1:
            diagnostics.append(
                DependencyDiagnostic(
                    "inconsistent_authority",
                    source,
                    "chunks from one source declare different authority tiers",
                )
            )
            hard_failures[source] = "ambiguous"
        authority = sorted(source_authorities)[0] if source_authorities else "unknown"
        authorities[source] = authority
        unique_dependencies = tuple(sorted(set(source_dependencies)))
        if len(source_dependencies) != len(unique_dependencies):
            diagnostics.append(
                DependencyDiagnostic(
                    "duplicate_dependency",
                    source,
                    "duplicate dependency declarations were collapsed",
                )
            )
        dependencies[source] = unique_dependencies
        asserting_chunk = source_chunks[0]
        asserted_at = (asserted_at_by_source or {}).get(source)
        asserted_at = _utc(asserted_at)
        for prerequisite in unique_dependencies:
            if prerequisite == source:
                diagnostics.append(
                    DependencyDiagnostic("self_dependency", source, "a source cannot depend on itself", prerequisite)
                )
                hard_failures[source] = "cycle"
                continue
            if prerequisite not in known_sources:
                diagnostics.append(
                    DependencyDiagnostic(
                        "unresolved_dependency",
                        source,
                        "dependency does not resolve to exactly one canonical source",
                        prerequisite,
                    )
                )
                continue
            edges.append(
                DependencyEdge(
                    id=_edge_id(
                        tenant_id,
                        generation_id,
                        source,
                        prerequisite,
                        asserting_chunk.id,
                        asserted_at,
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    dependent=source,
                    prerequisite=prerequisite,
                    asserting_chunk_id=asserting_chunk.id,
                    authority=authority,
                    asserted_at=asserted_at,
                )
            )

    adjacency: dict[str, tuple[str, ...]] = {
        source: tuple(sorted(value)) for source, value in dependencies.items()
    }
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        reverse[edge.prerequisite].append(edge.dependent)

    cycle_members: set[str] = set()
    visiting: list[str] = []
    active: set[str] = set()
    visited: set[str] = set()

    def find_cycles(source: str) -> None:
        if source in active:
            try:
                cycle_members.update(visiting[visiting.index(source) :])
            except ValueError:
                cycle_members.add(source)
            return
        if source in visited:
            return
        active.add(source)
        visiting.append(source)
        for prerequisite in adjacency.get(source, ()):
            if prerequisite in known_sources:
                find_cycles(prerequisite)
        visiting.pop()
        active.remove(source)
        visited.add(source)

    for source in sorted(known_sources):
        find_cycles(source)
    for source in sorted(cycle_members):
        diagnostics.append(
            DependencyDiagnostic("dependency_cycle", source, "dependency graph contains a cycle")
        )

    states = base_states or {}
    invalidations: dict[str, InvalidationReason] = {}

    def root_reason(source: str) -> InvalidationReason | None:
        state = str(states.get(source, "current"))
        authority = authorities.get(source, "unknown")
        if source in hard_failures:
            return InvalidationReason(
                source,
                hard_failures[source],
                (source,),
                authority,
                generation_id,
                as_of,
                known_as_of,
            )
        if source in cycle_members and authority != "model_inference":
            return InvalidationReason(
                source, "cycle", (source,), authority, generation_id, as_of, known_as_of
            )
        if state in INVALIDATING_STATES and authority != "model_inference":
            return InvalidationReason(
                source, cast(DependencyCause, state), (source,), authority, generation_id, as_of, known_as_of
            )
        return None

    def walk(source: str, path: tuple[str, ...], depth: int) -> InvalidationReason | None:
        if depth > MAX_INVALIDATION_DEPTH:
            return InvalidationReason(
                source,
                "cycle",
                path + (source,),
                authorities.get(source, "unknown"),
                generation_id,
                as_of,
                known_as_of,
            )
        direct = root_reason(source)
        if direct is not None:
            return direct
        for prerequisite in adjacency.get(source, ()):
            if prerequisite not in known_sources:
                return InvalidationReason(
                    prerequisite,
                    "unresolved",
                    path + (prerequisite,),
                    authorities.get(prerequisite, "unknown"),
                    generation_id,
                    as_of,
                    known_as_of,
                )
            if prerequisite in path:
                cycle = set(path[path.index(prerequisite) :])
                if all(authorities.get(member, "unknown") == "model_inference" for member in cycle):
                    return None
                return InvalidationReason(
                    prerequisite,
                    "cycle",
                    path + (prerequisite,),
                    authorities.get(prerequisite, "unknown"),
                    generation_id,
                    as_of,
                    known_as_of,
                )
            reason = walk(prerequisite, path + (prerequisite,), depth + 1)
            if reason is not None:
                if not reason.path or reason.path[0] != source:
                    reason = replace(reason, path=(source,) + reason.path)
                return reason
        return None

    for source in sorted(known_sources):
        reason = walk(source, (source,), 0)
        if reason is not None:
            invalidations[source] = reason
        elif source in cycle_members and authorities.get(source, "unknown") != "model_inference":
            invalidations[source] = reason or InvalidationReason(
                source,
                "cycle",
                (source,),
                authorities.get(source, "unknown"),
                generation_id,
                as_of,
                known_as_of,
            )

    payload = {
        "schema_version": schema_version,
        "tenant_id": tenant_id,
        "generation_id": generation_id,
        "as_of": as_of.isoformat() if as_of else None,
        "known_as_of": known_as_of.isoformat() if known_as_of else None,
        "corpus_fingerprint": corpus_fingerprint,
        "authorities": authorities,
        "dependencies": dependencies,
        "edges": [edge.id for edge in edges],
        "reverse": {
            prerequisite: tuple(sorted(dependents))
            for prerequisite, dependents in sorted(reverse.items())
        },
        "invalidations": {
            source: {
                "dependency": reason.dependency,
                "cause": reason.cause,
                "path": reason.path,
                "authority": reason.authority,
                "generation": reason.generation,
                "as_of": reason.as_of.isoformat() if reason.as_of else None,
                "known_as_of": reason.known_as_of.isoformat() if reason.known_as_of else None,
            }
            for source, reason in invalidations.items()
        },
        "diagnostics": [
            {
                "kind": diagnostic.kind,
                "source": diagnostic.source,
                "dependency": diagnostic.dependency,
            }
            for diagnostic in diagnostics
        ],
    }
    return DependencyProjection(
        schema_version=schema_version,
        projection_id="dip_" + canonical_sha256(payload)[:24],
        tenant_id=tenant_id,
        generation_id=generation_id,
        authorities=authorities,
        dependencies=dependencies,
        edges=tuple(edges),
        invalidations=invalidations,
        reverse={
            prerequisite: tuple(sorted(dependents))
            for prerequisite, dependents in reverse.items()
        },
        diagnostics=tuple(diagnostics),
        max_depth=MAX_INVALIDATION_DEPTH,
        as_of=as_of,
        known_as_of=known_as_of,
        corpus_fingerprint=corpus_fingerprint,
    )


__all__ = [
    "AUTHORITY_RANK",
    "AUTHORITY_VALUES",
    "Authority",
    "DependencyDiagnostic",
    "DependencyEdge",
    "DependencyProjection",
    "InvalidationReason",
    "MAX_INVALIDATION_CHAIN",
    "build_dependency_projection",
    "authority_from_metadata",
    "dependencies_from_metadata",
    "source_file",
    "write_dependency_projection",
]
