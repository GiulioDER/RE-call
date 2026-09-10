"""Request-local performance attribution and graph cache accounting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import recall_mcp.service as service
from recall.observability import PerformanceTrace, performance_trace_scope
from recall.semantic_graph import SemanticGraphProjection


def _empty_semantic_graph() -> SemanticGraphProjection:
    return SemanticGraphProjection(
        schema_version=2,
        graph_id="graph-1",
        tenant_id="acme",
        generation_id="gen-1",
        pipeline_fingerprint=None,
        corpus_fingerprint=None,
        entities=(),
        mentions=(),
        relations=(),
        diagnostics=(),
    )


class _Readiness:
    graph_fingerprint = "fp-1"


class _SemanticStore:
    tenant = "acme"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def load_semantic_graph(self, generation_id: str) -> SemanticGraphProjection:
        assert generation_id == "gen-1"
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=2)
        return _empty_semantic_graph()


def test_performance_trace_has_stable_attribution_shape() -> None:
    trace = PerformanceTrace()
    with performance_trace_scope(trace):
        with trace.span("baseline_retrieval_ms"):
            time.sleep(0.001)
        trace.add("db_statement_count")
        trace.add("db_parameter_bytes", 3)
        trace.add("db_result_bytes", 5)

    payload = trace.snapshot()
    assert payload["spans_ms"]["baseline_retrieval_ms"] > 0
    assert payload["spans_ms"]["graph_readiness_check_ms"] == 0.0
    assert payload["counters"]["db_statement_count"] == 1
    assert payload["counters"]["db_transferred_bytes"] == 8
    assert payload["values"]["wire_bytes_available"] is False


def test_semantic_graph_projection_misses_share_one_single_flight(monkeypatch) -> None:
    """Concurrent lazy graph loads expose one owner and one waiter.

    Red proof: removing `_SEMANTIC_GRAPH_INFLIGHT` coordination from `_cached_semantic_graph`
    makes the store loader run twice. The failure reason is duplicate immutable generation reads.
    """
    service._reset_graph_projection_cache()
    store = _SemanticStore()
    traces = [PerformanceTrace(), PerformanceTrace()]

    def load(index: int):
        with performance_trace_scope(traces[index]):
            return service._cached_semantic_graph(store, "gen-1", _Readiness(), "policy")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(load, 0)
        assert store.started.wait(timeout=2)
        second = executor.submit(load, 1)
        store.release.set()
        assert first.result(timeout=2) is not None
        assert second.result(timeout=2) is not None

    assert store.calls == 1
    counters = [trace.snapshot()["counters"] for trace in traces]
    assert sorted(item["projection_single_flight_owners"] for item in counters) == [0, 1]
    assert sorted(item["projection_single_flight_waiters"] for item in counters) == [0, 1]
    assert all(item["projection_cache_misses"] == 1 for item in counters)


def test_counting_db_boundary_records_statement_parameters_and_results() -> None:
    """The database accounting remains application-payload scoped and never claims wire bytes."""
    from recall.store import _observed_db_call

    class Cursor:
        def fetchall(self):
            return [("abc", "def")]

    class Connection:
        def execute(self, query, params=None):
            assert query == "SELECT 1"
            assert params == ("x",)
            return Cursor()

    trace = PerformanceTrace()
    with performance_trace_scope(trace):
        rows = _observed_db_call(Connection(), lambda conn: conn.execute("SELECT 1", ("x",)).fetchall())

    assert rows == [("abc", "def")]
    counters = trace.snapshot()["counters"]
    assert counters["db_statement_count"] == 1
    assert counters["db_parameter_bytes"] == 1
    assert counters["db_result_bytes"] == 6
    assert counters["db_transferred_bytes"] == 7
