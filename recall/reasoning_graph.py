"""Immutable graph projection for reasoning over indexed evidence.

This module derives a graph from chunks and metadata. It does not change corpus metadata,
generation state, trust verdicts, or retrieval ranking.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol

from recall.frontmatter import supersedes_key, validity_bounds
from recall.lineage import canonical_sha256
from recall.store import EdgeCandidates, resolve_supersession_candidates
from recall.types import Chunk

GRAPH_SCHEMA_VERSION = 1

GraphNodeKind = Literal["chunk", "source"]
GraphEdgeKind = Literal["authored_supersedes", "inferred_candidate_supersedes"]
GraphDiagnosticKind = Literal[
    "unresolved_reference",
    "ambiguous_reference",
    "cycle",
    "conflicting_authored_claim",
    "orphaned_node",
    "duplicate_entity_candidate",
    "malformed_metadata",
]


def _freeze_projection_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_projection_value(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_projection_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_projection_value(item) for item in value), key=repr))
    return value


class ChunkIterable(Protocol):
    tenant: str
    generation_id: str

    def iter_chunks(self, batch_size: int = 1000) -> Any:
        ...

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        ...


@dataclass(frozen=True)
class ReasoningGraphNode:
    id: str
    kind: GraphNodeKind
    tenant_id: str
    generation_id: str
    source: str
    chunk_id: str | None = None
    file: str | None = None
    ord: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validity: Mapping[str, datetime | None] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze_projection_value(self.provenance))
        object.__setattr__(self, "validity", _freeze_projection_value(self.validity))
        object.__setattr__(self, "calibration", _freeze_projection_value(self.calibration))
        object.__setattr__(self, "metadata", _freeze_projection_value(self.metadata))


@dataclass(frozen=True)
class ReasoningGraphEdge:
    id: str
    kind: GraphEdgeKind
    tenant_id: str
    generation_id: str
    from_node_id: str
    to_node_id: str | None
    from_file: str
    to_file: str | None
    authored_reference: str | None = None
    asserted_at: datetime | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze_projection_value(self.provenance))
        object.__setattr__(self, "metadata", _freeze_projection_value(self.metadata))


@dataclass(frozen=True)
class ReasoningGraphDiagnostic:
    id: str
    kind: GraphDiagnosticKind
    tenant_id: str
    generation_id: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    reference: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ReasoningGraphProjection:
    schema_version: int
    graph_id: str
    tenant_id: str
    generation_id: str
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    nodes: tuple[ReasoningGraphNode, ...]
    authored_edges: tuple[ReasoningGraphEdge, ...]
    inferred_candidate_edges: tuple[ReasoningGraphEdge, ...]
    diagnostics: tuple[ReasoningGraphDiagnostic, ...]

    def authored_supersession_map(self) -> dict[str, str]:
        return {
            edge.from_file: edge.to_file
            for edge in self.authored_edges
            if edge.kind == "authored_supersedes" and edge.to_file is not None
        }


def _identity(kind: str, payload: dict[str, Any]) -> str:
    return f"rg_{kind}_{canonical_sha256(payload)[:24]}"


def _node_id(
    *, tenant_id: str, generation_id: str, kind: GraphNodeKind, source: str, chunk_id: str | None
) -> str:
    return _identity(
        "node",
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "kind": kind,
            "source": source,
            "chunk_id": chunk_id,
        },
    )


def _edge_id(
    *,
    tenant_id: str,
    generation_id: str,
    kind: GraphEdgeKind,
    from_file: str,
    to_file: str | None,
    authored_reference: str | None,
    asserted_at: datetime | None,
) -> str:
    return _identity(
        "edge",
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "kind": kind,
            "from_file": from_file,
            "to_file": to_file,
            "authored_reference": authored_reference,
            "asserted_at": asserted_at.isoformat() if asserted_at is not None else None,
        },
    )


def _diagnostic_id(
    *,
    tenant_id: str,
    generation_id: str,
    kind: GraphDiagnosticKind,
    node_ids: tuple[str, ...] = (),
    edge_ids: tuple[str, ...] = (),
    reference: str | None = None,
) -> str:
    return _identity(
        "diag",
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "kind": kind,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "reference": reference,
        },
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_file(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    if isinstance(value, str) and value:
        return value
    return chunk.source


def _source_nodes(
    chunks: list[Chunk], *, tenant_id: str, generation_id: str
) -> tuple[list[ReasoningGraphNode], dict[str, str]]:
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    nodes: list[ReasoningGraphNode] = []
    source_node_by_source: dict[str, str] = {}
    for source in sorted(by_source):
        first = sorted(by_source[source], key=lambda item: item.id)[0]
        file = _source_file(first)
        node_id = _node_id(
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind="source",
            source=source,
            chunk_id=None,
        )
        source_node_by_source[source] = node_id
        nodes.append(
            ReasoningGraphNode(
                id=node_id,
                kind="source",
                tenant_id=tenant_id,
                generation_id=generation_id,
                source=source,
                file=file,
                provenance={"source": source, "file": file},
            )
        )
    return nodes, source_node_by_source


def _chunk_nodes(
    chunks: list[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
) -> tuple[list[ReasoningGraphNode], list[ReasoningGraphDiagnostic]]:
    nodes: list[ReasoningGraphNode] = []
    diagnostics: list[ReasoningGraphDiagnostic] = []
    for chunk in sorted(chunks, key=lambda item: item.id):
        file = _source_file(chunk)
        valid_from: datetime | None = None
        valid_until: datetime | None = None
        try:
            valid_from, valid_until = validity_bounds(chunk.metadata)
        except ValueError as exc:
            node_id = _node_id(
                tenant_id=tenant_id,
                generation_id=generation_id,
                kind="chunk",
                source=chunk.source,
                chunk_id=chunk.id,
            )
            diagnostics.append(
                ReasoningGraphDiagnostic(
                    id=_diagnostic_id(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="malformed_metadata",
                        node_ids=(node_id,),
                        reference=file,
                    ),
                    kind="malformed_metadata",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    node_ids=(node_id,),
                    reference=file,
                    message=str(exc),
                )
            )
        nodes.append(
            ReasoningGraphNode(
                id=_node_id(
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="chunk",
                    source=chunk.source,
                    chunk_id=chunk.id,
                ),
                kind="chunk",
                tenant_id=tenant_id,
                generation_id=generation_id,
                source=chunk.source,
                chunk_id=chunk.id,
                file=file,
                ord=_as_int(chunk.metadata.get("ord")),
                provenance={
                    "source": chunk.source,
                    "file": file,
                    "ord": _as_int(chunk.metadata.get("ord")),
                },
                validity={"valid_from": valid_from, "valid_until": valid_until},
                calibration=dict(chunk.metadata.get("calibration", {}))
                if isinstance(chunk.metadata.get("calibration"), dict)
                else {},
                metadata={"text": chunk.text, **dict(sorted(chunk.metadata.items()))},
            )
        )
    return nodes, diagnostics


def _supersession_rows(chunks: list[Chunk]) -> list[tuple[str | None, str | None, datetime | None]]:
    rows: dict[tuple[str | None, str | None], datetime | None] = {}
    for chunk in chunks:
        file = chunk.metadata.get("file")
        supersedes = chunk.metadata.get("supersedes")
        if not isinstance(file, str):
            file = None
        if not isinstance(supersedes, str):
            supersedes = None
        key = (file, supersedes)
        rows.setdefault(key, None)
    ordered = sorted(rows.items(), key=lambda item: (item[0][0] or "", item[0][1] or ""))
    return [(file, supersedes, when) for (file, supersedes), when in ordered]


def _authored_edges(
    *,
    tenant_id: str,
    generation_id: str,
    chunks: list[Chunk],
    candidates: EdgeCandidates,
    source_node_by_file: dict[str, str],
) -> tuple[list[ReasoningGraphEdge], list[ReasoningGraphDiagnostic]]:
    refs_by_claim: dict[tuple[str, str], str] = {}
    for chunk in chunks:
        file = chunk.metadata.get("file")
        supersedes = chunk.metadata.get("supersedes")
        if isinstance(file, str) and isinstance(supersedes, str) and supersedes:
            refs_by_claim.setdefault((file, supersedes_key(supersedes)), supersedes)

    edges: list[ReasoningGraphEdge] = []
    diagnostics: list[ReasoningGraphDiagnostic] = []
    for target in sorted(candidates):
        for superseding, asserted_at in sorted(
            candidates[target],
            key=lambda item: (
                item[1] is None,
                item[1].isoformat() if item[1] is not None else "",
                item[0],
            ),
        ):
            authored_reference = refs_by_claim.get((superseding, supersedes_key(target)), target)
            edge_id = _edge_id(
                tenant_id=tenant_id,
                generation_id=generation_id,
                kind="authored_supersedes",
                from_file=target,
                to_file=superseding,
                authored_reference=authored_reference,
                asserted_at=asserted_at,
            )
            edges.append(
                ReasoningGraphEdge(
                    id=edge_id,
                    kind="authored_supersedes",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    from_node_id=source_node_by_file.get(target, ""),
                    to_node_id=source_node_by_file.get(superseding),
                    from_file=target,
                    to_file=superseding,
                    authored_reference=authored_reference,
                    asserted_at=asserted_at,
                    provenance={
                        "authored_in_file": superseding,
                        "authored_reference": authored_reference,
                    },
                )
            )
    by_target = Counter(edge.from_file for edge in edges)
    for target, count in sorted(by_target.items()):
        if count > 1:
            edge_ids = tuple(edge.id for edge in edges if edge.from_file == target)
            diagnostics.append(
                ReasoningGraphDiagnostic(
                    id=_diagnostic_id(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="conflicting_authored_claim",
                        edge_ids=edge_ids,
                        reference=target,
                    ),
                    kind="conflicting_authored_claim",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    edge_ids=edge_ids,
                    reference=target,
                    message=f"{target} has {count} authored supersession claims",
                )
            )
    return edges, diagnostics


def _graph_diagnostics(
    *,
    tenant_id: str,
    generation_id: str,
    source_nodes: tuple[ReasoningGraphNode, ...],
    authored_edges: tuple[ReasoningGraphEdge, ...],
    unresolved: frozenset[str],
) -> list[ReasoningGraphDiagnostic]:
    diagnostics: list[ReasoningGraphDiagnostic] = []
    node_by_file = {node.file: node.id for node in source_nodes if node.file is not None}

    for file in sorted(unresolved):
        diagnostics.append(
            ReasoningGraphDiagnostic(
                id=_diagnostic_id(
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="ambiguous_reference",
                    node_ids=(node_by_file[file],) if file in node_by_file else (),
                    reference=file,
                ),
                kind="ambiguous_reference",
                tenant_id=tenant_id,
                generation_id=generation_id,
                node_ids=(node_by_file[file],) if file in node_by_file else (),
                reference=file,
                message=f"supersession reference to {file} is ambiguous",
            )
        )

    for edge in authored_edges:
        if edge.from_node_id == "":
            diagnostics.append(
                ReasoningGraphDiagnostic(
                    id=_diagnostic_id(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="unresolved_reference",
                        edge_ids=(edge.id,),
                        reference=edge.from_file,
                    ),
                    kind="unresolved_reference",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    edge_ids=(edge.id,),
                    reference=edge.from_file,
                    message=f"supersession target {edge.from_file} is not a projected source node",
                )
            )

    outgoing: dict[str, list[str]] = {}
    edge_ids_by_pair: dict[tuple[str, str], list[str]] = {}
    for edge in authored_edges:
        if edge.to_file is None or edge.from_file == edge.to_file:
            continue
        outgoing.setdefault(edge.from_file, []).append(edge.to_file)
        edge_ids_by_pair.setdefault((edge.from_file, edge.to_file), []).append(edge.id)
    for from_file in outgoing:
        outgoing[from_file] = sorted(set(outgoing[from_file]))
    for pair in edge_ids_by_pair:
        edge_ids_by_pair[pair].sort()
    state: dict[str, str] = {}
    emitted_cycles: set[tuple[str, ...]] = set()

    def walk_cycle(start: str, path: list[str], path_index: dict[str, int]) -> None:
        state[start] = "visiting"
        path_index[start] = len(path)
        path.append(start)
        for nxt in outgoing.get(start, []):
            if state.get(nxt) == "visiting":
                cycle = tuple(path[path_index[nxt] :])
                canonical_cycle = min(
                    (cycle[index:] + cycle[:index] for index in range(len(cycle))),
                    default=cycle,
                )
                if canonical_cycle not in emitted_cycles:
                    emitted_cycles.add(canonical_cycle)
                    pairs = zip(cycle, (*cycle[1:], cycle[0]), strict=True)
                    edge_ids = tuple(
                        edge_id
                        for pair in pairs
                        for edge_id in edge_ids_by_pair.get(pair, [])
                    )
                    diagnostics.append(
                        ReasoningGraphDiagnostic(
                            id=_diagnostic_id(
                                tenant_id=tenant_id,
                                generation_id=generation_id,
                                kind="cycle",
                                edge_ids=edge_ids,
                                reference=nxt,
                            ),
                            kind="cycle",
                            tenant_id=tenant_id,
                            generation_id=generation_id,
                            edge_ids=edge_ids,
                            reference=canonical_cycle[0] if canonical_cycle else nxt,
                            message=f"authored supersession cycle includes {', '.join(cycle)}",
                        )
                    )
            elif state.get(nxt) is None:
                walk_cycle(nxt, path, path_index)
        path.pop()
        path_index.pop(start)
        state[start] = "visited"

    for start in sorted(outgoing):
        if state.get(start) is None:
            walk_cycle(start, [], {})

    touched = {edge.from_node_id for edge in authored_edges if edge.from_node_id}
    touched.update(edge.to_node_id for edge in authored_edges if edge.to_node_id)
    for node in source_nodes:
        if node.id not in touched:
            diagnostics.append(
                ReasoningGraphDiagnostic(
                    id=_diagnostic_id(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="orphaned_node",
                        node_ids=(node.id,),
                        reference=node.file,
                    ),
                    kind="orphaned_node",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    node_ids=(node.id,),
                    reference=node.file,
                    message=f"{node.file or node.source} has no authored graph edges",
                )
            )

    stems: dict[str, list[str]] = {}
    for node in source_nodes:
        if node.file:
            stems.setdefault(supersedes_key(node.file), []).append(node.file)
    for stem, files in sorted(stems.items()):
        if len(files) > 1:
            node_ids = tuple(node_by_file[file] for file in sorted(files))
            diagnostics.append(
                ReasoningGraphDiagnostic(
                    id=_diagnostic_id(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="duplicate_entity_candidate",
                        node_ids=node_ids,
                        reference=stem,
                    ),
                    kind="duplicate_entity_candidate",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    node_ids=node_ids,
                    reference=stem,
                    message=f"{stem} resolves to multiple source candidates",
                )
            )
    return diagnostics


def build_reasoning_graph(
    chunks: list[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
    authored_edge_candidates: EdgeCandidates | None = None,
    unresolved_references: frozenset[str] | None = None,
    inferred_candidate_edges: tuple[ReasoningGraphEdge, ...] = (),
) -> ReasoningGraphProjection:
    """Project chunks and authored metadata into an immutable graph value."""
    ordered_chunks = sorted(chunks, key=lambda item: item.id)
    source_nodes, source_node_by_source = _source_nodes(
        ordered_chunks, tenant_id=tenant_id, generation_id=generation_id
    )
    source_node_by_file = {
        node.file: source_node_by_source[node.source]
        for node in source_nodes
        if node.file is not None
    }
    chunk_nodes, metadata_diagnostics = _chunk_nodes(
        ordered_chunks, tenant_id=tenant_id, generation_id=generation_id
    )
    if authored_edge_candidates is None or unresolved_references is None:
        _winner, unresolved, candidates = resolve_supersession_candidates(
            _supersession_rows(ordered_chunks)
        )
    else:
        unresolved = unresolved_references
        candidates = authored_edge_candidates
    edges, edge_diagnostics = _authored_edges(
        tenant_id=tenant_id,
        generation_id=generation_id,
        chunks=ordered_chunks,
        candidates=candidates,
        source_node_by_file=source_node_by_file,
    )
    authored_edges = tuple(sorted(edges, key=lambda edge: edge.id))
    source_nodes_tuple = tuple(sorted(source_nodes, key=lambda node: node.id))
    diagnostics = tuple(
        sorted(
            [
                *metadata_diagnostics,
                *edge_diagnostics,
                *_graph_diagnostics(
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    source_nodes=source_nodes_tuple,
                    authored_edges=authored_edges,
                    unresolved=unresolved,
                ),
            ],
            key=lambda diag: diag.id,
        )
    )
    nodes = tuple(sorted([*source_nodes_tuple, *chunk_nodes], key=lambda node: node.id))
    inferred = tuple(sorted(inferred_candidate_edges, key=lambda edge: edge.id))
    graph_id = _identity(
        "graph",
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "nodes": [node.id for node in nodes],
            "authored_edges": [edge.id for edge in authored_edges],
            "inferred_candidate_edges": [edge.id for edge in inferred],
            "diagnostics": [diag.id for diag in diagnostics],
        },
    )
    return ReasoningGraphProjection(
        schema_version=GRAPH_SCHEMA_VERSION,
        graph_id=graph_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        nodes=nodes,
        authored_edges=authored_edges,
        inferred_candidate_edges=inferred,
        diagnostics=diagnostics,
    )


def _store_binding_fingerprints(
    store: ChunkIterable,
    *,
    pipeline_fingerprint: str | None,
    corpus_fingerprint: str | None,
) -> tuple[str | None, str | None]:
    if pipeline_fingerprint is not None and corpus_fingerprint is not None:
        return pipeline_fingerprint, corpus_fingerprint
    generation_binding = getattr(store, "generation_binding", None)
    if not callable(generation_binding):
        return pipeline_fingerprint, corpus_fingerprint
    binding = generation_binding()
    if pipeline_fingerprint is None:
        value = binding.get("pipeline_fingerprint")
        if isinstance(value, str):
            pipeline_fingerprint = value
    if corpus_fingerprint is None:
        value = binding.get("corpus_fingerprint")
        if isinstance(value, str):
            corpus_fingerprint = value
    return pipeline_fingerprint, corpus_fingerprint


def project_store_graph(
    store: ChunkIterable,
    *,
    pipeline_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
) -> ReasoningGraphProjection:
    """Project the chunks visible through a store into a graph without mutating the store."""
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        with snapshot() as generation_id:
            pipeline, corpus = _store_binding_fingerprints(
                store,
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
            )
            _edges, unresolved, candidates = store.supersession_all()
            return build_reasoning_graph(
                list(store.iter_chunks()),
                tenant_id=store.tenant,
                generation_id=str(generation_id),
                pipeline_fingerprint=pipeline,
                corpus_fingerprint=corpus,
                authored_edge_candidates=candidates,
                unresolved_references=unresolved,
            )
    pipeline, corpus = _store_binding_fingerprints(
        store,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
    )
    _edges, unresolved, candidates = store.supersession_all()
    return build_reasoning_graph(
        list(store.iter_chunks()),
        tenant_id=store.tenant,
        generation_id=store.generation_id,
        pipeline_fingerprint=pipeline,
        corpus_fingerprint=corpus,
        authored_edge_candidates=candidates,
        unresolved_references=unresolved,
    )
