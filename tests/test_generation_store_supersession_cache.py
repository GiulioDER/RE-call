"""Generation-backed graph serving must not rescan immutable chunk metadata per request."""

from __future__ import annotations

from recall.generation_store import GenerationStore
from recall.semantic_graph import GraphReadiness


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def execute(self, _query, _params):
        self.calls += 1
        return _Result(self.rows)


class _Store(GenerationStore):
    def __init__(self, connection):
        self._tenant = "tenant-1"
        self._supersession_cache = None
        self._supersession_scans = 0
        self._connection = connection
        self.generation = "generation-1"

    def _generation_id(self):  # type: ignore[no-untyped-def]
        return self.generation

    def _with_retry(self, operation):  # type: ignore[no-untyped-def]
        return operation(self._connection)


def test_generation_store_caches_supersession_scan_per_generation() -> None:
    """The immutable serving store should transfer the full supersession result only once.

    Red proof node: ``generation-supersession-cache-001`` targets
    ``recall.generation_store.GenerationStore.supersession_all``. Mutating the implementation to
    execute the SQL on every call makes ``connection.calls`` equal 2 and fails because the old
    path repeated the full generation scan and its payload for every graph request.
    """
    connection = _Connection([("old.md", "new.md", None)])
    store = _Store(connection)

    first = store.supersession_all()
    second = store.supersession_all()

    assert first == second
    assert connection.calls == 1
    assert store._supersession_scans == 1


def test_generation_store_retires_supersession_cache_when_generation_changes() -> None:
    """A generation switch must not reuse the previous generation's closure."""
    connection = _Connection([("old.md", "new.md", None)])
    store = _Store(connection)

    store.supersession_all()
    store.generation = "generation-2"
    store.supersession_all()

    assert connection.calls == 2
    assert store._supersession_scans == 2


def test_generation_store_caches_ready_marker_without_caching_not_ready(monkeypatch) -> None:
    """Cache only a positive immutable marker so a later build can become ready.

    Red proof node: ``generation-readiness-cache-001`` targets
    ``recall.generation_store.GenerationStore.graph_readiness``. Mutating the implementation to
    cache every result makes the second call after ``GRAPH_NOT_READY`` return the stale negative
    result and fails because the serving path would refuse a graph that became ready.
    """
    calls: list[str] = []
    ready = GraphReadiness(
        ready=True,
        tenant_id="tenant-1",
        generation_id="generation-1",
        graph_id="graph-1",
        graph_fingerprint="fingerprint-1",
        entity_count=1,
        mention_count=1,
        relation_count=1,
        diagnostic_count=0,
    )
    not_ready = GraphReadiness(
        ready=False,
        tenant_id="tenant-1",
        generation_id="generation-1",
        graph_id=None,
        graph_fingerprint=None,
        entity_count=0,
        mention_count=0,
        relation_count=0,
        diagnostic_count=0,
        reason="GRAPH_NOT_READY",
    )
    outcomes = iter((ready, ready))

    def read(_connection, _tenant, generation):  # type: ignore[no-untyped-def]
        calls.append(generation)
        return next(outcomes)

    monkeypatch.setattr("recall.generation_store.read_graph_readiness", read)
    store = _Store(_Connection([]))

    assert store.graph_readiness() is ready
    assert store.graph_readiness() is ready
    assert calls == ["generation-1"]

    store._graph_readiness_cache = None
    outcomes = iter((not_ready, ready))
    assert store.graph_readiness() is not_ready
    assert store.graph_readiness() is ready
    assert calls == ["generation-1", "generation-1", "generation-1"]
