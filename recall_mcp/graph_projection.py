"""MCP graph projection cache and response boundary."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from recall.observability import current_performance_trace
from recall.reasoning_graph import ReasoningGraphProjection, project_store_graph
from recall_mcp.models import ReasoningProjectionResult
from recall.semantic_graph import relation_coverage

if TYPE_CHECKING:
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall.store import PgVectorStore


class _GraphProjectionFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: ReasoningGraphProjection | None = None
        self.error: BaseException | None = None


_GRAPH_PROJECTION_LOCK = threading.Lock()
_GRAPH_PROJECTIONS: OrderedDict[
    tuple[str, str, str | None, bool, str | None, str | None], ReasoningGraphProjection
] = OrderedDict()
_GRAPH_PROJECTION_INFLIGHT: dict[
    tuple[str, str, str | None, bool, str | None, str | None], _GraphProjectionFlight
] = {}
_GRAPH_PROJECTION_CACHE_MAX = 4


def _corpus_fingerprint(store: PgVectorStore, generation_id: str) -> str | None:
    """The generation's current corpus fingerprint, for keying caches of its content.

    An erasure deletes a source's rows from a generation in place and changes only this value,
    so a projection keyed without it would keep serving the erased text. Stores without the
    hook (legacy and test stores) key on None, which is their previous behaviour.
    """
    identity = getattr(store, "_serving_identity", None)
    return str(identity(generation_id)[1]) if callable(identity) else None


def _project_store_graph(store: PgVectorStore, *, include_text: bool) -> ReasoningGraphProjection:
    """Resolve the legacy projector dynamically so compatibility monkeypatches still observe it."""
    from recall_mcp import service

    projector = getattr(service, "project_store_graph", project_store_graph)
    return projector(store, include_text=include_text)


def _reset_graph_projection_cache() -> None:
    with _GRAPH_PROJECTION_LOCK:
        _GRAPH_PROJECTIONS.clear()
        _GRAPH_PROJECTION_INFLIGHT.clear()


def _combined_graph_policy_fingerprint(
    *,
    security_policy: SourceSecurityPolicy | None = None,
    graph_policy_fingerprint: str | None = None,
) -> str | None:
    security_fingerprint = getattr(security_policy, "digest", None)
    if not isinstance(security_fingerprint, str):
        security_fingerprint = None
    if security_fingerprint is None and graph_policy_fingerprint is None:
        return None
    if security_fingerprint is None:
        return graph_policy_fingerprint
    if graph_policy_fingerprint is None:
        return security_fingerprint
    return hashlib.sha256(
        f"security:{security_fingerprint}|graph:{graph_policy_fingerprint}".encode("utf-8")
    ).hexdigest()


def _store_graph_with_readiness(
    store: PgVectorStore,
    *,
    include_text: bool,
    policy_fingerprint: str | None = None,
) -> tuple[ReasoningGraphProjection, Any]:
    """Project immutable generations once while leaving mutable legacy stores uncached."""
    snapshot = getattr(store, "snapshot", None)
    lookup = getattr(store, "active_generation_id", None)
    if not callable(snapshot) and not callable(lookup):
        return _project_store_graph(store, include_text=include_text), None
    scope: AbstractContextManager[Any] = snapshot() if callable(snapshot) else nullcontext(None)
    with scope as pinned:
        if pinned is not None:
            generation_id = str(pinned)
        elif not callable(lookup):
            return _project_store_graph(store, include_text=include_text), None
        else:
            generation_id = str(lookup())
        readiness_reader = getattr(store, "graph_readiness", None)
        readiness = readiness_reader() if callable(readiness_reader) else None
        fingerprint = getattr(readiness, "graph_fingerprint", None) if readiness else None
        key = (
            store.tenant,
            generation_id,
            _corpus_fingerprint(store, generation_id),
            include_text,
            fingerprint,
            policy_fingerprint,
        )
        with _GRAPH_PROJECTION_LOCK:
            cached = _GRAPH_PROJECTIONS.get(key)
            if cached is not None:
                _GRAPH_PROJECTIONS.move_to_end(key)
                performance = current_performance_trace()
                if performance is not None:
                    performance.add("projection_cache_hits")
                return cached, readiness
            flight = _GRAPH_PROJECTION_INFLIGHT.get(key)
            owner = flight is None
            if owner:
                flight = _GraphProjectionFlight()
                _GRAPH_PROJECTION_INFLIGHT[key] = flight
        assert flight is not None
        performance = current_performance_trace()
        if performance is not None:
            performance.add("projection_cache_misses")
            performance.add(
                "projection_single_flight_owners" if owner else "projection_single_flight_waiters"
            )
        if not owner:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            assert flight.result is not None
            return flight.result, readiness
        try:
            graph = _project_store_graph(store, include_text=include_text)
        except BaseException as exc:
            with _GRAPH_PROJECTION_LOCK:
                flight.error = exc
                _GRAPH_PROJECTION_INFLIGHT.pop(key, None)
                flight.done.set()
            raise
        with _GRAPH_PROJECTION_LOCK:
            if graph.generation_id == generation_id:
                while len(_GRAPH_PROJECTIONS) >= _GRAPH_PROJECTION_CACHE_MAX:
                    _GRAPH_PROJECTIONS.popitem(last=False)
                _GRAPH_PROJECTIONS[key] = graph
            flight.result = graph
            _GRAPH_PROJECTION_INFLIGHT.pop(key, None)
            flight.done.set()
        return graph, readiness


def _store_graph(
    store: PgVectorStore,
    *,
    include_text: bool,
    policy_fingerprint: str | None = None,
) -> ReasoningGraphProjection:
    return _store_graph_with_readiness(
        store, include_text=include_text, policy_fingerprint=policy_fingerprint
    )[0]


def _validate_security_context(
    store: PgVectorStore,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
) -> None:
    if security_policy is None:
        return
    if access_context is None:
        raise ValueError("access_context is required when security_policy is configured")
    store_tenant = getattr(store, "tenant", None)
    if isinstance(store_tenant, str) and store_tenant != access_context.tenant:
        raise PermissionError("access context tenant does not match the serving store")


def _authorized_graph(
    store: PgVectorStore,
    graph: ReasoningGraphProjection,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
) -> ReasoningGraphProjection:
    _validate_security_context(store, security_policy, access_context)
    if security_policy is None:
        return graph
    assert access_context is not None
    visible_node_ids = {
        node.id
        for node in graph.nodes
        if security_policy.decide(node.source, access_context).allowed
    }

    def visible_edge(edge: object) -> bool:
        from_node_id = getattr(edge, "from_node_id", None)
        to_node_id = getattr(edge, "to_node_id", None)
        return from_node_id in visible_node_ids and (
            to_node_id is None or to_node_id in visible_node_ids
        )

    authored_edges = tuple(edge for edge in graph.authored_edges if visible_edge(edge))
    inferred_edges = tuple(edge for edge in graph.inferred_candidate_edges if visible_edge(edge))
    dependency_edges = tuple(edge for edge in graph.authored_dependency_edges if visible_edge(edge))
    visible_edge_ids = {edge.id for edge in (*authored_edges, *inferred_edges, *dependency_edges)}
    diagnostics = tuple(
        diagnostic
        for diagnostic in graph.diagnostics
        if set(diagnostic.node_ids) <= visible_node_ids
        and set(diagnostic.edge_ids) <= visible_edge_ids
    )
    return replace(
        graph,
        nodes=tuple(node for node in graph.nodes if node.id in visible_node_ids),
        authored_edges=authored_edges,
        inferred_candidate_edges=inferred_edges,
        authored_dependency_edges=dependency_edges,
        diagnostics=diagnostics,
        semantic_graph=None,
    )


def reasoning_projection(
    store: PgVectorStore,
    *,
    include_text: bool = False,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> ReasoningProjectionResult:
    graph, readiness = _store_graph_with_readiness(
        store,
        include_text=include_text,
        policy_fingerprint=_combined_graph_policy_fingerprint(security_policy=security_policy),
    )
    graph = _authorized_graph(store, graph, security_policy, access_context)
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
        semantic_relation_coverage=relation_coverage(semantic) if semantic is not None else {},
        semantic_diagnostic_count=len(semantic.diagnostics) if semantic is not None else 0,
    )
