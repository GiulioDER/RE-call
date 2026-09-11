"""Compact semantic graph readiness reads."""

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
