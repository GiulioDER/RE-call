"""Pooled-connection mode: the store must be safe to share between concurrent callers.

The single-connection store is correct for a CLI and wrong for a server. One `psycopg`
connection serialises every caller, and `_with_retry`'s reconnect REPLACES `self._conn` while
another thread may be using it. These tests pin the pooled alternative: many threads, correct
results, no cross-talk, and the statement timeout still fails closed rather than being retried
onto a fresh session that no longer carries it.
"""
from __future__ import annotations

import concurrent.futures as cf
import uuid

import psycopg
import pytest

from recall.store import PgVectorStore
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


def _vec(i: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


@pytest.fixture
def pooled_store():
    table = "p_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=8, table=table, pool_size=4)
    store.ensure_schema()
    yield store
    try:
        store.drop_table()
    finally:
        store.close()


@requires_db
def test_pooled_store_does_basic_crud(pooled_store):
    pooled_store.upsert([Chunk("a", "s.md", "hello world")], [_vec(0)])
    assert pooled_store.count() == 1
    hits = pooled_store.query_dense(_vec(0), k=1)
    assert hits and hits[0].chunk.id == "a"


@requires_db
def test_concurrent_reads_return_each_thread_its_own_correct_answer(pooled_store):
    """The failure this guards: results crossing between threads sharing one connection."""
    chunks, vecs = [], []
    for i in range(8):
        chunks.append(Chunk(f"c{i}", f"s{i}.md", f"document number {i}"))
        vecs.append(_vec(i))
    pooled_store.upsert(chunks, vecs)

    def _query(i: int) -> str:
        hits = pooled_store.query_dense(_vec(i), k=1)
        return hits[0].chunk.id

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(_query, range(8)))

    assert got == [f"c{i}" for i in range(8)]


@requires_db
def test_concurrent_writes_all_land(pooled_store):
    def _write(i: int) -> None:
        pooled_store.upsert([Chunk(f"w{i}", f"s{i}.md", f"text {i}")], [_vec(i)])

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_write, range(16)))

    assert pooled_store.count() == 16


@requires_db
def test_pool_serves_more_callers_than_it_has_connections(pooled_store):
    """A pool of 4 must not deadlock or drop work when 16 callers arrive at once."""
    pooled_store.upsert([Chunk("a", "s.md", "hello")], [_vec(0)])

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: pooled_store.count(), range(16)))

    assert results == [1] * 16


@requires_db
def test_statement_timeout_fails_closed_and_is_not_retried():
    """A cancelled statement must propagate, not be retried onto a session without the timeout.

    `QueryCanceled` subclasses `OperationalError`, so a retry keyed on the exception TYPE would
    re-run the statement on a fresh connection that no longer carries the setting which killed
    it — silently escaping the guard. The pooled path must keep that distinction.
    """
    table = "p_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=8, table=table, pool_size=2, statement_timeout_ms=100)
    try:
        store.ensure_schema()
        with pytest.raises(psycopg.errors.QueryCanceled):
            store._with_retry(lambda conn: conn.execute("SELECT pg_sleep(3)").fetchone())
    finally:
        store.close()
        # Clean up on a SEPARATE connection: this store's 100 ms timeout applies to its own
        # teardown too, and DROP TABLE is not reliably under 100 ms on slower disks — the test
        # failed in CI for that reason while passing locally. The timeout is the subject of the
        # assertion, not something the cleanup should have to survive.
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")


@requires_db
def test_close_is_sticky_for_a_pooled_store():
    table = "p_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=8, table=table, pool_size=2)
    store.ensure_schema()
    store.drop_table()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.count()


@requires_db
def test_drop_table_replaces_reaching_into_the_private_connection(make_store):
    """`drop_table()` exists so callers stop doing `store._conn.execute("DROP TABLE ...")`,
    which breaks outright once the connection is a pool."""
    store = make_store(8)
    store.upsert([Chunk("a", "s.md", "x")], [_vec(0)])
    assert store.count() == 1
    store.drop_table()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (store.table,)).fetchone()
    assert row[0] is None


@requires_db
def test_single_connection_mode_is_still_the_default(make_store):
    """The default must keep the carefully-tested single-connection semantics: pooling is opt-in
    so a CLI does not silently acquire a background pool thread."""
    store = make_store(8)
    assert store._pool is None
    assert store._conn is not None
