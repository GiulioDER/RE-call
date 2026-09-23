"""The storage read path's per-search round trips, and the instrument that counts them.

Invariants:

1. `db_statement_count` counts every statement a store operation sends, including the `BEGIN`
   and `COMMIT` of a `conn.transaction()` block, which `execute` never sees.
2. `GenerationStore._newest_indexed_at` reads `max(indexed_at)` once per
   `(generation, corpus fingerprint)`, not once per search: the aggregate has no index to use, so
   every uncached call read every row of the generation. An erasure changes the fingerprint and
   retires the entry.
3. `GenerationStore.generation_binding` is served from a cache inside `snapshot()`, where the
   fingerprint is pinned, and still reads the row outside one.

Red proof, recorded 2026-09-23 against this branch's base (`f4031c0c`, P2) by restoring the
pre-change `recall/store.py` and `recall/generation_store.py`:

* `test_a_transaction_block_counts_its_begin_and_commit` failed with `assert 1 == 3`: the
  counter saw the `SELECT` and not the two statements around it.
* `test_a_failed_transaction_block_still_counts_its_rollback` failed with `assert 1 == 3`.
* `test_newest_indexed_at_is_read_once_per_corpus_state` failed with `assert 2 == 1` at the
  first cache assertion.
* `test_the_binding_is_cached_only_inside_a_snapshot` failed with `assert 2 == 1` at the
  pinned-snapshot assertion.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from recall.generation_store import GenerationStore
from recall.observability import PerformanceTrace, performance_trace_scope
from recall.store import _observed_db_call


class _Rows:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row

    def fetchall(self):
        return [self.row]


def test_a_transaction_block_counts_its_begin_and_commit() -> None:
    class _Connection:
        @contextmanager
        def transaction(self):
            yield None

        def execute(self, sql, params=None):
            return _Rows((1,))

    def op(conn):  # type: ignore[no-untyped-def]
        with conn.transaction():
            return conn.execute("SELECT 1").fetchall()

    trace = PerformanceTrace()
    with performance_trace_scope(trace):
        _observed_db_call(_Connection(), op)

    assert trace.snapshot()["counters"]["db_statement_count"] == 3


def test_a_failed_transaction_block_still_counts_its_rollback() -> None:
    class _Connection:
        @contextmanager
        def transaction(self):
            yield None

        def execute(self, sql, params=None):
            raise RuntimeError("boom")

    def op(conn):  # type: ignore[no-untyped-def]
        with conn.transaction():
            conn.execute("SELECT 1")

    trace = PerformanceTrace()
    with performance_trace_scope(trace):
        try:
            _observed_db_call(_Connection(), op)
        except RuntimeError:
            pass

    assert trace.snapshot()["counters"]["db_statement_count"] == 3


class _Store(GenerationStore):
    """A real `GenerationStore` whose database is a statement counter."""

    NEWEST = datetime(2026, 9, 1, tzinfo=UTC)

    def __init__(self) -> None:  # deliberately does not call super().__init__
        self._tenant = "acme"
        self._pinned_generation = ContextVar("pinned_generation", default=None)
        self._pinned_corpus = ContextVar("pinned_corpus", default=None)
        self._fixed_generation = "gen-1"
        self._binding_cache = None
        self._newest_indexed_at_cache = None
        self.corpus = "corpus-1"
        self.sql: list[str] = []

    def _with_retry(self, op):  # type: ignore[no-untyped-def]
        return op(self)

    def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
        self.sql.append(sql)
        if "max(indexed_at)" in sql:
            return _Rows((self.NEWEST,))
        if "pipeline_identity" in sql:
            return _Rows(("pipe-1", self.corpus, {"embedder": {"model": "m", "dimension": 4}}))
        if "SELECT corpus_fingerprint" in sql:
            return _Rows((self.corpus,))
        raise AssertionError(f"unexpected statement: {sql}")

    @contextmanager
    def pinned(self):  # type: ignore[no-untyped-def]
        """What `snapshot()` does, without the active-pointer query."""
        token = self._pinned_generation.set("gen-1")
        corpus_token = self._pinned_corpus.set(("gen-1", self.corpus))
        try:
            yield
        finally:
            self._pinned_corpus.reset(corpus_token)
            self._pinned_generation.reset(token)

    def count(self, fragment: str) -> int:
        return sum(fragment in sql for sql in self.sql)


def test_newest_indexed_at_is_read_once_per_corpus_state() -> None:
    store = _Store()

    with store.pinned():
        assert store._newest_indexed_at() == _Store.NEWEST
        assert store._newest_indexed_at() == _Store.NEWEST
    assert store.count("max(indexed_at)") == 1

    store.corpus = "corpus-2"  # an erasure landed
    with store.pinned():
        store._newest_indexed_at()
    assert store.count("max(indexed_at)") == 2


def test_the_binding_is_cached_only_inside_a_snapshot() -> None:
    store = _Store()

    with store.pinned():
        first = store.generation_binding()
        second = store.generation_binding()
    assert store.count("pipeline_identity") == 1
    assert first == second
    assert first["corpus_fingerprint"] == "corpus-1"
    assert first["embedder_model"] == "m"
    first["corpus_fingerprint"] = "mutated by a caller"
    with store.pinned():
        assert store.generation_binding()["corpus_fingerprint"] == "corpus-1"

    store.generation_binding()  # outside a snapshot: always read
    assert store.count("pipeline_identity") == 2

    store.corpus = "corpus-2"
    with store.pinned():
        assert store.generation_binding()["corpus_fingerprint"] == "corpus-2"
    assert store.count("pipeline_identity") == 3
