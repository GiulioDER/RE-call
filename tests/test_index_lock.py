"""One indexer per corpus, and one bound on what an embedder is allowed to allocate.

The guard was broken on purpose and these tests watched to go red, which is the only thing that
makes a concurrency test evidence rather than decoration. Measured 2026-08-22:

* `single_writer` mutated into a no-op (`dsn = None` before it connects): **8 of these fail**;
* `Indexer.__init__` mutated to ignore `_batch_chunks_from_env()`:
  `test_the_host_bound_applies_to_a_caller_that_never_heard_of_it` fails.

⚠️ **And one mutation that nothing here catches, recorded rather than papered over.** Deleting the
explicit `pg_advisory_unlock` in `single_writer`'s `finally` leaves every test green. That is not a
gap in the tests, it is what the lock's design says: the release that matters is CLOSING the
connection, which the outer `finally` always does and which
`test_the_lock_is_released_when_the_process_dies` covers. The explicit unlock is belt and braces
for a future in which that connection is reused rather than closed, and no test can distinguish it
today. Anyone tempted to assert on it should change the design first.
"""
from __future__ import annotations

import threading
import uuid

import psycopg
import pytest

from recall.embeddings import HashingEmbedder
from recall.index import DEFAULT_BATCH_CHUNKS, ENV_BATCH_CHUNKS, Indexer, _batch_chunks_from_env
from recall.index_lock import (
    ENV_ALLOW_CONCURRENT,
    ConcurrentIndex,
    lock_name,
    single_writer,
)
from recall.store import PgVectorStore

from tests.conftest import TEST_DSN, requires_db

DIM = 64


@pytest.fixture
def store():
    table = "lk_" + uuid.uuid4().hex[:8]
    s = PgVectorStore(TEST_DSN, dim=DIM, table=table)
    s.ensure_schema()
    yield s
    try:
        s.drop_table()
    finally:
        s.close()


def _corpus(tmp_path, n: int = 3):
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(n):
        (root / f"note{i}.md").write_text(f"memory number {i}", encoding="utf-8")
    return root


# --- the lock itself -----------------------------------------------------------------------------


@requires_db
def test_a_second_indexer_refuses_while_the_first_holds_the_lock(store):
    """The headline. Refuses PROMPTLY: an unbounded wait on a command reads as a hang."""
    with single_writer(store) as taken:
        assert taken is True
        with pytest.raises(ConcurrentIndex):
            with single_writer(store, wait_seconds=0.0):
                pytest.fail("two writers held one corpus's lock at the same time")


@requires_db
def test_the_refusal_names_who_holds_the_lock(store):
    """Whose run is it? An error that cannot answer that sends the reader to `docker ps`.

    The holder's `application_name` carries host and pid, so the answer survives the message
    being copied into a chat window.
    """
    with single_writer(store):
        with pytest.raises(ConcurrentIndex) as excinfo:
            with single_writer(store, wait_seconds=0.0):
                pass
    message = str(excinfo.value)
    assert "pid " in message
    assert store._table in message and store._tenant in message
    # The way out is in the message, not only in the docs.
    assert ENV_ALLOW_CONCURRENT in message


@requires_db
def test_the_lock_is_released_when_the_run_finishes(store):
    with single_writer(store) as taken:
        assert taken
    with single_writer(store) as taken:
        assert taken, "the lock outlived the block that took it"


@requires_db
def test_the_lock_is_released_when_the_run_raises(store):
    """A failed index must not wedge the corpus against every later one."""
    with pytest.raises(RuntimeError):
        with single_writer(store):
            raise RuntimeError("index blew up")
    with single_writer(store) as taken:
        assert taken


@requires_db
def test_two_tenants_are_not_each_other_s_business(store):
    """Different corpora index concurrently ON PURPOSE. Serialising them would be a bug."""
    other = PgVectorStore(TEST_DSN, dim=DIM, table=store._table, tenant="another-tenant")
    try:
        with single_writer(store):
            with single_writer(other, wait_seconds=0.0) as taken:
                assert taken
    finally:
        other.close()


@requires_db
def test_indexing_the_same_corpus_twice_at_once_is_refused(tmp_path, store):
    """End to end through `Indexer.index_path`, which is where every caller arrives.

    The first run is held open inside the embedder, so the second one meets a lock that a real
    run is holding rather than one a test took by hand.
    """
    root = _corpus(tmp_path)
    inside = threading.Event()
    release = threading.Event()

    class _BlockingEmbedder(HashingEmbedder):
        def embed(self, texts):
            inside.set()
            release.wait(timeout=30)
            return super().embed(texts)

    first = Indexer(store, _BlockingEmbedder(dim=DIM))
    failure: list[BaseException] = []

    def _run():
        try:
            first.index_path(root)
        except BaseException as exc:  # noqa: BLE001 - reported through the assertion below
            failure.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        assert inside.wait(timeout=30), "the first index never reached the embedder"
        second = Indexer(
            PgVectorStore(TEST_DSN, dim=DIM, table=store._table), HashingEmbedder(dim=DIM)
        )
        with pytest.raises(ConcurrentIndex):
            second.index_path(root)
    finally:
        release.set()
        worker.join(timeout=60)
    assert not failure, f"the first index failed: {failure[0]!r}"


@requires_db
def test_the_escape_hatch_lets_a_second_run_through_and_says_so(store, monkeypatch, caplog):
    """An escape nobody can see is an escape nobody audits, so it logs."""
    with single_writer(store):
        monkeypatch.setenv(ENV_ALLOW_CONCURRENT, "1")
        with caplog.at_level("WARNING"):
            with single_writer(store, wait_seconds=0.0) as taken:
                assert taken is False
    assert any(ENV_ALLOW_CONCURRENT in record.getMessage() for record in caplog.records)


def test_a_store_with_no_dsn_locks_nothing_rather_than_failing():
    """A stubbed store is process-local: there is no shared database to serialise against."""

    class _FakeStore:
        _table = "chunks"
        _tenant = "default"

    with single_writer(_FakeStore()) as taken:
        assert taken is False


def test_the_lock_name_cannot_be_forged_by_a_tenant_that_contains_the_separator():
    assert lock_name("chunks", "a") != lock_name("chunks\x1fa", "")


@requires_db
def test_the_lock_is_released_when_the_process_dies(store):
    """Why there is no `--force` for a stale lock: a session lock cannot outlive its backend.

    Killing the connection is the closest a test gets to killing the indexer, and it is the same
    mechanism — the backend goes away, Postgres drops its session locks.
    """
    key = lock_name(store._table, store._tenant)
    conn = psycopg.connect(TEST_DSN, autocommit=True)
    row = conn.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)).fetchone()
    assert row is not None and row[0]
    conn.close()
    with single_writer(store, wait_seconds=0.0) as taken:
        assert taken, "a dead process's lock was still standing"


# --- the allocation bound ------------------------------------------------------------------------


def test_the_batch_default_is_unchanged_when_the_host_says_nothing(monkeypatch):
    monkeypatch.delenv(ENV_BATCH_CHUNKS, raising=False)
    assert DEFAULT_BATCH_CHUNKS == 64
    assert _batch_chunks_from_env() == DEFAULT_BATCH_CHUNKS


def test_the_host_bound_applies_to_a_caller_that_never_heard_of_it(monkeypatch):
    """The point of the variable: bound every embedding run on a host, not the polite ones."""
    monkeypatch.setenv(ENV_BATCH_CHUNKS, "64")
    assert Indexer(object(), HashingEmbedder(dim=DIM))._batch_chunks == 64


def test_an_explicit_batch_beats_the_environment(monkeypatch):
    """A caller who names a batch has sized that run; the environment must not overrule it."""
    monkeypatch.setenv(ENV_BATCH_CHUNKS, "64")
    assert Indexer(object(), HashingEmbedder(dim=DIM), batch_chunks=8)._batch_chunks == 8


@pytest.mark.parametrize("raw", ["0", "-1", "", "sixty-four", "64.5"])
def test_a_malformed_bound_falls_back_rather_than_being_clamped(monkeypatch, raw, caplog):
    """Substituting a different bound for the configured one is how a host gets OOM-killed by a
    setting somebody believed was in force. Ignore it, and say so."""
    monkeypatch.setenv(ENV_BATCH_CHUNKS, raw)
    with caplog.at_level("WARNING"):
        assert _batch_chunks_from_env() == DEFAULT_BATCH_CHUNKS
    assert any(ENV_BATCH_CHUNKS in record.getMessage() for record in caplog.records)


def test_an_explicit_zero_is_still_a_caller_bug(monkeypatch):
    monkeypatch.delenv(ENV_BATCH_CHUNKS, raising=False)
    with pytest.raises(ValueError):
        Indexer(object(), HashingEmbedder(dim=DIM), batch_chunks=0)
