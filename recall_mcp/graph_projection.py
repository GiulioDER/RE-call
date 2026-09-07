"""MCP graph projection cache and response boundary."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from recall.reasoning_graph import ReasoningGraphProjection, project_store_graph
from recall_mcp.models import ReasoningProjectionResult

if TYPE_CHECKING:
    from recall.store import PgVectorStore


_GRAPH_PROJECTION_LOCK = threading.Lock()
_GRAPH_PROJECTIONS: dict[tuple[str, str, bool, str | None], ReasoningGraphProjection] = {}
_GRAPH_PROJECTION_CACHE_MAX = 4


def _reset_graph_projection_cache() -> None:
    with _GRAPH_PROJECTION_LOCK:
        _GRAPH_PROJECTIONS.clear()


def _store_graph_with_readiness(
    store: PgVectorStore,
    *,
    include_text: bool,
    _project_store_graph_fn=project_store_graph,
) -> tuple[ReasoningGraphProjection, Any]:
    """Project immutable generations once while leaving mutable legacy stores uncached."""
    snapshot = getattr(store, "snapshot", None)
    lookup = getattr(store, "active_generation_id", None)
    if not callable(snapshot) and not callable(lookup):
        return _project_store_graph_fn(store, include_text=include_text), None
    scope: AbstractContextManager[Any] = snapshot() if callable(snapshot) else nullcontext(None)
    with scope as pinned:
        if pinned is not None:
            generation_id = str(pinned)
        elif not callable(lookup):
            return _project_store_graph_fn(store, include_text=include_text), None
        else:
            generation_id = str(lookup())
        readiness_reader = getattr(store, "graph_readiness", None)
        readiness = readiness_reader() if callable(readiness_reader) else None
        fingerprint = getattr(readiness, "graph_fingerprint", None) if readiness else None
        key = (store.tenant, generation_id, include_text, fingerprint)
        with _GRAPH_PROJECTION_LOCK:
            cached = _GRAPH_PROJECTIONS.get(key)
        if cached is not None:
            return cached, readiness
        graph = _project_store_graph_fn(store, include_text=include_text)
        if graph.generation_id != generation_id:
            return graph, readiness
        with _GRAPH_PROJECTION_LOCK:
            if key not in _GRAPH_PROJECTIONS:
                while len(_GRAPH_PROJECTIONS) >= _GRAPH_PROJECTION_CACHE_MAX:
                    _GRAPH_PROJECTIONS.pop(next(iter(_GRAPH_PROJECTIONS)))
            _GRAPH_PROJECTIONS[key] = graph
        return graph, readiness


def _store_graph(
    store: PgVectorStore,
    *,
    include_text: bool,
    _project_store_graph_fn=project_store_graph,
) -> ReasoningGraphProjection:
    return _store_graph_with_readiness(
        store, include_text=include_text, _project_store_graph_fn=_project_store_graph_fn
    )[0]


def reasoning_projection(
    store: PgVectorStore,
    *,
    include_text: bool = False,
    _project_store_graph_fn=project_store_graph,
) -> ReasoningProjectionResult:
    graph, readiness = _store_graph_with_readiness(
        store, include_text=include_text, _project_store_graph_fn=_project_store_graph_fn
    )
    semantic = graph.semantic_graph
    return ReasoningProjectionResult(
        schema_version=graph.schema_version,
        graph_id=graph.graph_id,
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        node_count=len(graph.nodes),
        authored_edge_count=len(graph.authored_edges),
        inferred_candidate_edge_count=len(graph.inferred_candidate_edges),
        diagnostic_count=len(graph.diagnostics),
        trust_state="trusted" if graph.generation_id != "legacy" else "degraded",
        semantic_graph_ready=bool(readiness.ready)
        if readiness is not None
        else semantic is not None,
        semantic_graph_reason=getattr(readiness, "reason", None) if readiness is not None else None,
        semantic_entity_count=len(semantic.entities) if semantic is not None else 0,
        semantic_mention_count=len(semantic.mentions) if semantic is not None else 0,
        semantic_relation_count=len(semantic.relations) if semantic is not None else 0,
        semantic_diagnostic_count=len(semantic.diagnostics) if semantic is not None else 0,
    )
