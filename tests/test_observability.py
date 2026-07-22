"""Logging and metrics.

The point of this module is that an operator can answer questions the logs alone cannot:
how often does retrieval abstain, what is it demoting, is latency drifting, is the database
flapping. So the tests check the counters move on the real code path — not just that the
registry can add up numbers.
"""
from __future__ import annotations

import io
import json
import logging
import threading

import pytest

from recall.observability import LOGGER_NAME, Metrics, configure_logging, get_logger

from tests.conftest import requires_db


@pytest.fixture
def metrics():
    return Metrics()


def test_counters_and_labels(metrics):
    metrics.increment("searches")
    metrics.increment("searches")
    metrics.increment("verdicts", verdict="ok")
    metrics.increment("verdicts", verdict="superseded", other="x")
    snap = metrics.snapshot()["counters"]
    assert snap["searches"] == 2
    assert snap["verdicts{verdict=ok}"] == 1
    assert snap["verdicts{other=x,verdict=superseded}"] == 1  # labels sorted -> stable key


def test_histogram_reports_percentiles(metrics):
    for value in range(1, 101):
        metrics.observe("latency", float(value))
    h = metrics.snapshot()["histograms"]["latency"]
    assert h["count"] == 100
    assert h["p50"] == pytest.approx(51, abs=2)
    assert h["p99"] == pytest.approx(100, abs=2)


def test_histogram_memory_is_bounded(metrics):
    """A long-running process must not grow a list per observation."""
    from recall.observability import HISTOGRAM_CAPACITY

    for i in range(HISTOGRAM_CAPACITY * 3):
        metrics.observe("latency", float(i))
    assert metrics.snapshot()["histograms"]["latency"]["count"] == HISTOGRAM_CAPACITY


def test_timer_records_even_when_the_body_raises(metrics):
    """The slow path worth finding is usually the one that fails."""
    with pytest.raises(ValueError):
        with metrics.timer("op"):
            raise ValueError("boom")
    assert metrics.snapshot()["histograms"]["op"]["count"] == 1


def test_counters_are_thread_safe(metrics):
    """The MCP server runs tool bodies in a worker-thread pool, so increments really are
    concurrent — and `d[k] = d[k] + 1` is not atomic."""
    def bump():
        for _ in range(500):
            metrics.increment("hits")

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metrics.snapshot()["counters"]["hits"] == 4000


def test_empty_snapshot_has_no_fabricated_entries(metrics):
    assert metrics.snapshot() == {"counters": {}, "histograms": {}}


# --------------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------------


def test_json_format_emits_one_object_per_line_with_extra_fields():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    try:
        get_logger("t").info("indexed", extra={"chunks": 7, "tenant": "acme"})
        payload = json.loads(stream.getvalue().strip())
        assert payload["message"] == "indexed"
        assert payload["level"] == "INFO"
        assert payload["chunks"] == 7
        assert payload["tenant"] == "acme"
        assert payload["logger"] == "recall.t"
    finally:
        logging.getLogger(LOGGER_NAME).handlers.clear()


def test_configure_logging_is_idempotent():
    """Calling it twice must not double every line — a handler leak looks like duplicated logs."""
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream)
    configure_logging(level="INFO", fmt="text", stream=stream)
    try:
        get_logger("t").info("once")
        assert stream.getvalue().count("once") == 1
    finally:
        logging.getLogger(LOGGER_NAME).handlers.clear()


def test_logging_does_not_propagate_to_the_root_logger():
    """stdout carries JSON-RPC in the MCP server: a root handler must not re-emit our records."""
    root_stream = io.StringIO()
    root = logging.getLogger()
    root_handler = logging.StreamHandler(root_stream)
    root.addHandler(root_handler)
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream)
    try:
        get_logger("t").info("private")
        assert "private" in stream.getvalue()
        assert "private" not in root_stream.getvalue()
    finally:
        root.removeHandler(root_handler)
        logging.getLogger(LOGGER_NAME).handlers.clear()


def test_library_does_not_configure_logging_on_import():
    """A library that attaches handlers on import hijacks the host application's logging."""
    import importlib

    logging.getLogger(LOGGER_NAME).handlers.clear()
    importlib.reload(importlib.import_module("recall.store"))
    assert logging.getLogger(LOGGER_NAME).handlers == []


# --------------------------------------------------------------------------------------------
# The counters must move on the REAL path, not only in the registry's own unit tests
# --------------------------------------------------------------------------------------------


@requires_db
def test_trusted_search_counts_searches_verdicts_and_abstentions(make_store):
    """Instrumentation that is never wired to the code path is worse than none: it reports zero
    forever and reads as 'nothing is going wrong'."""
    from recall.embeddings import HashingEmbedder
    from recall.observability import METRICS
    from recall.trust import trusted_search
    from recall.types import Chunk

    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    store.upsert(
        [Chunk("a", "notes.md", "the caching layer decision", {"file": "notes.md", "ord": 0})],
        emb.embed(["the caching layer decision"]),
    )

    before = METRICS.snapshot()["counters"]
    trusted_search(store, emb, "caching layer decision", k=5)
    trusted_search(store, emb, "utterly unrelated penguins on mars", k=5)
    after = METRICS.snapshot()["counters"]

    def delta(key: str) -> int:
        return after.get(key, 0) - before.get(key, 0)

    assert delta("recall_searches_total") == 2
    assert delta("recall_abstentions_total") >= 1  # the unanswerable one
    assert sum(v - before.get(k, 0) for k, v in after.items()
               if k.startswith("recall_verdicts_total")) >= 1
