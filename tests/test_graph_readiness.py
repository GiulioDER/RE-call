"""Compact semantic graph readiness reads."""

from contextvars import ContextVar

from recall.generation_store import GenerationStore
from recall.semantic_graph import read_graph_readiness


class _Result:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, row):
        self.row = row
        self.queries = []

    def execute(self, query, params):
        self.queries.append((query, params))
        return _Result(self.row)


def test_generation_store_readiness_uses_only_the_compact_marker(monkeypatch):
    """Readiness must not load graph members or transfer their payload.

    Red proof: the baseline `GenerationStore.graph_readiness` calls `load_semantic_graph` after
    reading the marker. The failure reason is that the monkeypatched loader raises, proving the
    old path still performs the repeated full graph read.
    """
    marker = {
        "semantic_graph": {
            "ready": True,
            "graph_id": "graph-1",
            "graph_fingerprint": "fingerprint-1",
            "entity_count": 3,
            "mention_count": 4,
            "relation_count": 2,
            "diagnostic_count": 1,
        }
    }
    connection = _Connection((marker,))
    store = object.__new__(GenerationStore)
    store._tenant = "tenant-1"
    store._pinned_corpus = ContextVar("pinned_corpus", default=None)
    store._generation_id = lambda: "generation-1"
    store._with_retry = lambda operation: operation(connection)

    def fail_full_graph_load(*args, **kwargs):
        raise AssertionError("full semantic graph load is forbidden during readiness")

    monkeypatch.setattr("recall.generation_store.load_semantic_graph", fail_full_graph_load)

    readiness = store.graph_readiness()

    assert readiness.ready is True
    assert readiness.graph_id == "graph-1"
    assert readiness.graph_fingerprint == "fingerprint-1"
    assert (readiness.entity_count, readiness.mention_count) == (3, 4)
    assert len(connection.queries) == 1
    assert "recall_graph_" not in connection.queries[0][0]


def test_read_graph_readiness_rejects_an_incomplete_marker():
    connection = _Connection(
        ({"semantic_graph": {"ready": True, "graph_id": "graph-1"}},)
    )

    readiness = read_graph_readiness(connection, "tenant-1", "generation-1")

    assert readiness.ready is False
    assert readiness.reason == "GRAPH_NOT_READY"


class _MarkerConnection:
    """One generation row whose ``semantic_graph`` marker the graph delete must retire."""

    def __init__(self, marker):
        self.marker = marker

    def transaction(self):
        from contextlib import nullcontext

        return nullcontext()

    def execute(self, query, params):
        if query.startswith("DELETE FROM recall_graph_entities_v1"):
            return type("_Deleted", (), {"rowcount": 3})()
        if query.startswith("UPDATE recall_generations"):
            self.marker = None
            return _Result(None)
        return _Result(({"semantic_graph": self.marker} if self.marker else {},))


def test_a_pinned_store_stops_reporting_a_deleted_graph_ready():
    """Deleting a graph must retire the ready verdict a pinned store has cached for it.

    `GenerationStore.graph_readiness` caches a ready verdict per serving identity, and a graph
    delete leaves that identity unchanged, so without a reset the same store kept answering
    ready after the marker was gone.

    Red proof (2026-09-23, base ``c7f2b9bc`` plus the marker fix), node
    ``tests/test_graph_readiness.py::test_a_pinned_store_stops_reporting_a_deleted_graph_ready``:
    removing the ``self._graph_readiness_cache = None`` reset in
    `GenerationStore.delete_generation_graph` fails ``assert store.graph_readiness().ready is
    False``.
    """
    connection = _MarkerConnection(
        {
            "ready": True,
            "graph_id": "graph-1",
            "graph_fingerprint": "fingerprint-1",
            "entity_count": 3,
            "mention_count": 4,
            "relation_count": 2,
            "diagnostic_count": 1,
        }
    )
    store = object.__new__(GenerationStore)
    store._tenant = "tenant-1"
    store._pinned_corpus = ContextVar("pinned_corpus", default=None)
    store._generation_id = lambda: "generation-1"
    store._pinned_identity = lambda generation_id: (generation_id, "corpus-1")
    store._graph_readiness_cache = None
    store._with_retry = lambda operation: operation(connection)

    assert store.graph_readiness().ready is True
    assert store.delete_generation_graph() == 3
    assert store.graph_readiness().ready is False
