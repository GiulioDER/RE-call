import uuid
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from recall.store import PgVectorStore, resolve_supersession, warn_if_insecure_dsn
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


# --- resolve_supersession: pure, DB-free (the supersession-keying rule) ---------------------


def test_resolve_supersession_basic_basename():
    rows = [("v1.md", None), ("v2.md", "v1.md")]
    assert resolve_supersession(rows) == {"v1.md": "v2.md"}


def test_resolve_supersession_empty_when_no_supersedes():
    assert resolve_supersession([("a.md", None), ("b.md", None)]) == {}


def test_resolve_supersession_keys_on_relpath_no_basename_collision():
    # Two files share the basename old.md in different directories. A memo supersedes old.md by
    # basename; ONLY the file it actually points to may be marked superseded — but the basename
    # is ambiguous, so neither is (skip beats a silent mis-map). The unrelated sibling stays valid.
    rows = [
        ("a/old.md", None),
        ("b/old.md", None),
        ("a/new.md", "old.md"),
    ]
    assert resolve_supersession(rows) == {}


def test_resolve_supersession_unique_nested_target_resolves():
    # Unambiguous basename in a nested layout resolves to its root-relative path.
    rows = [("sub/old.md", None), ("sub/new.md", "old.md")]
    assert resolve_supersession(rows) == {"sub/old.md": "sub/new.md"}


def test_resolve_supersession_dangling_falls_back_to_raw_basename():
    # supersedes points at a basename absent from the corpus (predecessor never indexed, or
    # deleted). Not ambiguous -- nothing to disambiguate -- so it resolves via the raw basename
    # rather than being silently dropped.
    assert resolve_supersession([("a/new.md", "ghost.md")]) == {"ghost.md": "a/new.md"}


# --- warn_if_insecure_dsn: pure, DB-free (the default-credentials footgun guard) ---------------


def test_warn_insecure_dsn_flags_default_creds_on_remote_host(capsys):
    msg = warn_if_insecure_dsn("postgresql://recall:recall@db.prod.internal:5432/recall")
    assert msg is not None
    err = capsys.readouterr().err
    assert "WARNING" in err and "db.prod.internal" in err


def test_warn_insecure_dsn_silent_on_localhost():
    assert warn_if_insecure_dsn("postgresql://recall:recall@localhost:5432/recall") is None
    assert warn_if_insecure_dsn("postgresql://recall:recall@127.0.0.1:5432/recall") is None


def test_warn_insecure_dsn_silent_when_creds_are_not_default():
    assert warn_if_insecure_dsn("postgresql://recall:s3cret@db.prod.internal:5432/recall") is None


# --- _with_retry: DB-free (broken-connection reconnect-and-retry-once) --------------------------


def _bare_store() -> PgVectorStore:
    """A PgVectorStore instance WITHOUT running __init__ (no real DB connection)."""
    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "chunks"
    store._supersession_cache = None
    return store


def test_with_retry_reconnects_once_on_broken_connection():
    class _BrokenThenGood:
        def __init__(self):
            self.calls = 0

        def op(self):
            self.calls += 1
            if self.calls == 1:
                raise psycopg.OperationalError("server closed the connection unexpectedly")
            return "recovered"

    store = _bare_store()
    reconnects = {"n": 0}
    fresh_conn = object()

    def fake_reconnect():
        reconnects["n"] += 1
        store._conn = fresh_conn

    store._conn = object()
    store._reconnect = fake_reconnect  # type: ignore[method-assign]
    target = _BrokenThenGood()

    result = store._with_retry(lambda conn: target.op())
    assert result == "recovered"
    assert reconnects["n"] == 1  # reconnected exactly once
    assert target.calls == 2  # original attempt + one retry


def test_with_retry_propagates_second_failure():
    store = _bare_store()
    store._conn = object()
    store._reconnect = lambda: setattr(store, "_conn", object())  # type: ignore[method-assign]

    def always_broken(_conn):
        raise psycopg.InterfaceError("connection already closed")

    with pytest.raises(psycopg.InterfaceError):
        store._with_retry(always_broken)


def test_with_retry_does_not_retry_non_connection_errors():
    store = _bare_store()
    store._conn = object()
    reconnects = {"n": 0}
    store._reconnect = lambda: reconnects.__setitem__("n", reconnects["n"] + 1)  # type: ignore

    def bad_query(_conn):
        raise psycopg.errors.UndefinedColumn("no such column")

    with pytest.raises(psycopg.errors.UndefinedColumn):
        store._with_retry(bad_query)
    assert reconnects["n"] == 0  # a data/query error must NOT trigger a reconnect


@requires_db
def test_upsert_and_dense_query_ranks_by_cosine(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "alpha"), Chunk("b", "f.md", "beta")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.query_dense([1.0, 0.0, 0.0], k=2)
    assert hits[0].chunk.id == "a"
    assert hits[0].score > 0.99  # cosine ~1.0


@requires_db
def test_upsert_is_idempotent_on_id(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "first")], [[1.0, 0.0, 0.0]])
    store.upsert([Chunk("a", "f.md", "second")], [[0.0, 1.0, 0.0]])
    hits = store.query_dense([0.0, 1.0, 0.0], k=5)
    ids = [h.chunk.id for h in hits]
    assert ids.count("a") == 1
    assert hits[0].chunk.text == "second"


@requires_db
def test_touch_files_bumps_indexed_at_without_reembedding(make_store):
    # deferred PERF-004/DAT-003: the trust eval's stale-touch re-embedded identical text to
    # refresh a timestamp; a store-level touch makes "only indexed_at changes" true by
    # construction (and works for nested corpora, which keyed the old path-join variant out)
    import time

    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "alpha", metadata={"file": "a.md", "ord": 0}),
         Chunk("b", "f.md", "beta", metadata={"file": "b.md", "ord": 0})],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    before = {h.chunk.id: (h.indexed_at, h.score)
              for h in store.query_dense([1.0, 0.0, 0.0], k=5)}
    time.sleep(0.01)
    touched = store.touch_files(["a.md"])
    assert touched == 1
    after = {h.chunk.id: (h.indexed_at, h.score)
             for h in store.query_dense([1.0, 0.0, 0.0], k=5)}
    assert after["a"][0] > before["a"][0]          # timestamp moved
    assert after["b"][0] == before["b"][0]         # untouched file untouched
    assert after["a"][1] == before["a"][1]         # embedding (hence score) unchanged


@requires_db
def test_sparse_query_matches_keyword(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "the caching layer decision"), Chunk("b", "f.md", "unrelated text")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.query_sparse("caching", k=5)
    assert [h.chunk.id for h in hits] == ["a"]


@requires_db
def test_source_filter(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a", "one.md", "alpha"), Chunk("b", "two.md", "alpha")],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    hits = store.query_dense([1.0, 0.0, 0.0], k=5, source="two.md")
    assert [h.chunk.id for h in hits] == ["b"]


@requires_db
def test_newest_indexed_at_none_when_empty(make_store):
    store = make_store(3)
    assert store.newest_indexed_at() is None


@requires_db
def test_fresh_database_bootstraps_vector_extension():
    """A brand-new database (no `vector` extension yet) must work out of the box.

    Regression guard: register_vector needs the `vector` type, so PgVectorStore.__init__ must
    install the extension itself — otherwise the README quickstart crashes on a fresh DB.
    """
    parts = urlsplit(TEST_DSN)
    fresh_name = "recall_fresh_" + uuid.uuid4().hex[:8]
    admin = urlunsplit(parts._replace(path="/recall"))  # manage from the default db
    fresh = urlunsplit(parts._replace(path="/" + fresh_name))
    conn = psycopg.connect(admin, autocommit=True)
    try:
        conn.execute(f'CREATE DATABASE "{fresh_name}"')  # NO CREATE EXTENSION — store must self-bootstrap
        with PgVectorStore(fresh, dim=8) as store:
            store.ensure_schema()
            store.upsert([Chunk("a", "f", "hello")], [[0.1] * 8])
            assert store.count() == 1
    finally:
        conn.execute(f'DROP DATABASE IF EXISTS "{fresh_name}" WITH (FORCE)')
        conn.close()


@requires_db
def test_hits_carry_indexed_at(make_store):
    from datetime import datetime, timedelta, timezone

    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    dense = store.query_dense([1.0, 0.0, 0.0], k=1)
    sparse = store.query_sparse("cats", k=1)
    for hit in (dense[0], sparse[0]):
        assert hit.indexed_at is not None
        assert hit.indexed_at.tzinfo is not None
        assert datetime.now(timezone.utc) - hit.indexed_at < timedelta(minutes=5)


@requires_db
def test_query_sparse_with_vec_returns_true_cosine(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    qvec = [0.6, 0.8, 0.0]
    dense_score = store.query_dense(qvec, k=1)[0].score
    sparse_hit = store.query_sparse("cats", k=1, vec=qvec)[0]
    assert abs(sparse_hit.score - dense_score) < 1e-6
    # without vec the score is still the ts_rank (unchanged behavior)
    plain = store.query_sparse("cats", k=1)[0]
    assert plain.score != sparse_hit.score or plain.score >= 0


@requires_db
def test_supersession_map_roundtrip(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("old", "v1.md", "old policy", metadata={"file": "v1.md", "ord": 0}),
            Chunk(
                "new",
                "v2.md",
                "new policy",
                metadata={"file": "v2.md", "ord": 0, "supersedes": "v1.md"},
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert store.supersession_map() == {"v1.md": "v2.md"}


@requires_db
def test_supersession_map_empty_when_no_supersedes(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    assert store.supersession_map() == {}


@requires_db
def test_delete_sources_removes_all_rows_for_a_source(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a1", "f.md", "one"), Chunk("a2", "f.md", "two"), Chunk("b1", "g.md", "keep")],
        [[1.0, 0.0, 0.0]] * 3,
    )
    removed = store.delete_sources(["f.md"])
    assert removed == 2
    assert store.count() == 1


@requires_db
def test_supersession_map_cache_invalidated_by_writes(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("old", "v1.md", "old", metadata={"file": "v1.md", "ord": 0})], [[1.0, 0.0, 0.0]]
    )
    assert store.supersession_map() == {}  # primes the cache
    store.upsert(
        [Chunk("new", "v2.md", "new",
               metadata={"file": "v2.md", "ord": 0, "supersedes": "v1.md"})],
        [[0.0, 1.0, 0.0]],
    )
    assert store.supersession_map() == {"v1.md": "v2.md"}  # upsert invalidated the cache
    store.delete_sources(["v2.md"])
    assert store.supersession_map() == {}  # delete invalidated it too
