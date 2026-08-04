"""The store's own latency, and the proof that the instrument measuring it can be wrong.

Why these tests are shaped the way they are: the number this instrument produces is expected to
be SMALL. That is the prediction it exists to test, and it is also exactly what a broken
instrument reports. A histogram nailed to zero and a store that costs nothing are the same
reading, so `test_recorded_latency_tracks_injected_delay` is the load-bearing test here — it
injects a known delay and requires the instrument to report it. Without that, a low reading is
not evidence of a fast store.
"""
from __future__ import annotations

import time

import pytest

from recall.observability import HISTOGRAM_CAPACITY, METRICS, Metrics
from recall.store import LEG_DENSE, LEG_SPARSE, STORE_QUERY_METRIC
from recall.types import Chunk

DIM = 8


def _vec(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            source=f"/corpus/doc{i}.md",
            text=f"alpha beta gamma document number {i}",
            metadata={"file": f"doc{i}.md", "ord": 0},
        )
        for i in range(n)
    ]


@pytest.fixture
def store(make_store):
    s = make_store(DIM)
    chunks = _chunks(5)
    s.upsert(chunks, [_vec(0.1 + i / 100) for i in range(len(chunks))])
    METRICS.reset()
    return s


def _series(leg: str) -> tuple[list[float], int]:
    return METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)


def test_query_dense_records_one_sample_per_call(store):
    for _ in range(3):
        store.query_dense(_vec(), k=2)

    samples, total = _series(LEG_DENSE)
    assert total == 3
    assert len(samples) == 3
    assert all(ms > 0 for ms in samples), "a real round trip cannot take zero measurable time"


def test_the_two_legs_are_separate_series(store):
    store.query_dense(_vec(), k=2)
    store.query_sparse("alpha beta", k=2)

    dense_samples, dense_total = _series(LEG_DENSE)
    sparse_samples, sparse_total = _series(LEG_SPARSE)
    assert (dense_total, sparse_total) == (1, 1)
    assert len(dense_samples) == 1 and len(sparse_samples) == 1


def test_recorded_latency_tracks_injected_delay(store, monkeypatch):
    """THE falsification test: make the query provably slow, require the instrument to say so.

    The delay is injected at `_with_retry`, the single funnel through which `_query_dense`
    executes its statements. A timer that only spanned parameter setup, or one that recorded a
    constant, would pass every other test in this file and fail this one.
    """
    baseline_ms = 250.0
    original = type(store)._with_retry

    def _slow(self, op):
        time.sleep(baseline_ms / 1000.0)
        return original(self, op)

    store.query_dense(_vec(), k=2)
    fast, _ = _series(LEG_DENSE)

    monkeypatch.setattr(type(store), "_with_retry", _slow)
    store.query_dense(_vec(), k=2)
    slow, _ = _series(LEG_DENSE)

    assert slow[0] >= baseline_ms, (
        f"injected {baseline_ms} ms of query time and the instrument reported {slow[0]:.1f} ms; "
        "it is not spanning the query"
    )
    assert slow[0] - fast[0] >= baseline_ms * 0.9, (
        "the delta between a normal and a deliberately slowed query must show up in the metric"
    )


def test_a_rejected_call_records_nothing(store):
    """`k <= 0` issues no statement, so it must not contribute a ~0 ms sample."""
    with pytest.raises(ValueError):
        store.query_dense(_vec(), k=0)
    with pytest.raises(ValueError):
        store.query_sparse("alpha", k=0)

    assert _series(LEG_DENSE) == ([], 0)
    assert _series(LEG_SPARSE) == ([], 0)


def test_a_failing_query_is_still_timed(store):
    """A timer that records only on success hides the slow path worth finding."""
    boom = RuntimeError("connection died")

    def _raise(self, op):
        raise boom

    original = type(store)._with_retry
    try:
        type(store)._with_retry = _raise
        with pytest.raises(RuntimeError):
            store.query_dense(_vec(), k=2)
    finally:
        type(store)._with_retry = original

    _, total = _series(LEG_DENSE)
    assert total == 1


def test_drain_isolates_consecutive_measurements(store):
    """Draining is what stops configuration B's mean from including configuration A's samples."""
    store.query_dense(_vec(), k=2)
    first, _ = _series(LEG_DENSE)
    assert len(first) == 1

    store.query_dense(_vec(), k=2)
    store.query_dense(_vec(), k=2)
    second, total = _series(LEG_DENSE)
    assert (len(second), total) == (2, 2), "the drain did not clear the previous series"


def test_drain_reports_evicted_samples_rather_than_hiding_them():
    """Past the ring's capacity the retained samples are a suffix, and the caller must be told.

    A mean over the last 1024 of 2000 calls is a different statistic from a mean over the run.
    Reporting it under the same name is the silent truncation this return pair exists to expose.
    """
    metrics = Metrics()
    overflow = HISTOGRAM_CAPACITY + 500
    for i in range(overflow):
        metrics.observe("m", float(i))

    samples, total = metrics.drain_histogram("m")
    assert total == overflow
    assert len(samples) == HISTOGRAM_CAPACITY
    assert total > len(samples), "eviction happened and the counts must disagree to reveal it"
    assert metrics.drain_histogram("m") == ([], 0), "drain must clear the total, not just the ring"
