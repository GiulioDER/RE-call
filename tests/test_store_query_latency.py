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


def test_no_subclass_overrides_the_timed_public_query_methods():
    """A subclass overriding the PUBLIC pair silently loses the timing, and nothing errors.

    `GenerationStore` did exactly this, and `RECALL_ENV=production` selects it — so the metric
    fired on a laptop and recorded nothing in production. An absent series and a free store are
    the same reading, so the loss is invisible precisely where it matters most. Timing lives on
    the public wrapper; subclasses override the `_`-prefixed twin.

    Enumerating the method names inline is what let this recur: the first version of this guard
    listed `query_dense` and `query_sparse`, and `newest_indexed_at` became a timed method the
    same day — so the subclass dropped the timing again and the guard could not see it. It
    iterates `TIMED_PUBLIC_METHODS` now. That tuple is a hand-maintained declaration, so it is
    only as wide as the hazard while it stays in step with the actual timer call sites; that is
    what `test_timed_public_methods_matches_the_actual_timer_call_sites` checks.
    """
    import importlib
    import inspect
    import pkgutil

    import recall
    from recall.store import TIMED_PUBLIC_METHODS

    # Discovered by walking the package and matching the MRO BY NAME, not by
    # `PgVectorStore.__subclasses__()`.
    #
    # Two reasons, and the second one is why this exists in this shape. (a) `__subclasses__()` is
    # direct-only, so a grandchild would be invisible. (b) It is keyed on CLASS IDENTITY, and any
    # test in the session that reloads or re-imports `recall.store` rebinds `PgVectorStore` to a
    # NEW class object whose `__subclasses__()` is empty, while `GenerationStore` still inherits
    # from the old one. That is not hypothetical: this guard passed locally and failed in CI's
    # `floor` job, which collects ~120 more tests, with "found no subclasses" — the vacuity
    # assertion catching the guard rather than the hazard. Name-matching the MRO survives it.
    subclasses: list[type] = []
    for mod in pkgutil.walk_packages(recall.__path__, prefix="recall."):
        try:
            module = importlib.import_module(mod.name)
        except ImportError:
            continue  # optional-extra module; not installed in every environment
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != mod.name:
                continue  # imported into this module, not defined here
            if any(base.__name__ == "PgVectorStore" for base in obj.__mro__[1:]):
                subclasses.append(obj)

    assert subclasses, "found no PgVectorStore subclasses; an import regression made this vacuous"
    offenders = [
        f"{cls.__name__}.{name}"
        for cls in subclasses
        for name in TIMED_PUBLIC_METHODS
        if name in vars(cls)
    ]
    assert not offenders, (
        f"{offenders} override the TIMED public methods, so their store latency is never "
        "recorded. Override _query_dense / _query_sparse instead."
    )


def test_timed_public_methods_matches_the_actual_timer_call_sites():
    """The declaration must equal what the code actually does, or the guard is only as good as
    someone's memory.

    `TIMED_PUBLIC_METHODS` sets the width of the subclass guard above. A hand-maintained tuple
    that nothing checks is the SAME failure one level up: add a fourth timed public method, forget
    the tuple, and a subclass can drop the timing again with every test still green. That is
    exactly how this defect recurred twice. So derive the set from the source — every method whose
    body opens `METRICS.timer(STORE_QUERY_METRIC, ...)` — and require the declaration to match it.
    """
    import ast
    import inspect

    import recall.store as store_mod

    tree = ast.parse(inspect.getsource(store_mod))
    timed: set[str] = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr == "timer"):
                    continue
                if any(
                    isinstance(a, ast.Name) and a.id == "STORE_QUERY_METRIC" for a in call.args
                ):
                    timed.add(fn.name)

    assert timed, "found no METRICS.timer(STORE_QUERY_METRIC, ...) call sites; the parse is wrong"
    assert timed == set(store_mod.TIMED_PUBLIC_METHODS), (
        f"TIMED_PUBLIC_METHODS is {sorted(store_mod.TIMED_PUBLIC_METHODS)} but the timers are "
        f"actually on {sorted(timed)}. Update the tuple: the subclass guard is only as wide as it."
    )


def test_snapshot_reveals_truncation_like_the_drain_does():
    """The operator-facing reader must not report a suffix statistic under the run's name."""
    metrics = Metrics()
    for i in range(HISTOGRAM_CAPACITY + 10):
        metrics.observe("m", float(i))

    entry = metrics.snapshot()["histograms"]["m"]
    assert entry["count"] == HISTOGRAM_CAPACITY
    assert entry["observed"] == HISTOGRAM_CAPACITY + 10
    assert entry["truncated"] is True


def test_newest_indexed_at_is_timed_as_a_store_leg(store):
    """It is a real round trip on every search; untimed, it books store cost as Python glue."""
    store.newest_indexed_at()

    samples, total = _series("meta")
    assert total == 1
    assert samples and samples[0] > 0


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
