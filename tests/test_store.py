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
    assert resolve_supersession(rows) == ({"v1.md": "v2.md"}, frozenset())


def test_resolve_supersession_empty_when_no_supersedes():
    assert resolve_supersession([("a.md", None), ("b.md", None)]) == ({}, frozenset())


def test_resolve_supersession_keys_on_relpath_no_basename_collision():
    # Two files share the basename old.md in different directories. A memo supersedes old.md by
    # basename; ONLY the file it actually points to may be marked superseded — but the basename
    # is ambiguous, so neither is (skip beats a silent mis-map). The unrelated sibling stays valid.
    rows = [
        ("a/old.md", None),
        ("b/old.md", None),
        ("a/new.md", "old.md"),
    ]
    # ...and both candidates are NAMED, so the read path can fail closed instead of serving
    # a possibly-superseded memo as healthy.
    assert resolve_supersession(rows) == ({}, frozenset({"a/old.md", "b/old.md"}))


def test_resolve_supersession_unique_nested_target_resolves():
    # Unambiguous basename in a nested layout resolves to its root-relative path.
    rows = [("sub/old.md", None), ("sub/new.md", "old.md")]
    assert resolve_supersession(rows) == ({"sub/old.md": "sub/new.md"}, frozenset())


def test_resolve_supersession_dangling_falls_back_to_raw_basename():
    # supersedes points at a basename absent from the corpus (predecessor never indexed, or
    # deleted). Not ambiguous -- nothing to disambiguate -- so it resolves via the raw basename
    # rather than being silently dropped.
    assert resolve_supersession([("a/new.md", "ghost.md")]) == ({"ghost.md": "a/new.md"}, frozenset())


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


class _FakeConn:
    """Stands in for a psycopg connection; models the liveness flags the retry consults.

    `_with_retry` only reconnects when the connection is observably dead, so a fake must say
    whether it is — a bare object would make every connection look alive.
    """

    def __init__(self, closed: bool = True, broken: bool = False) -> None:
        self.closed = closed
        self.broken = broken


def _bare_store(conn: _FakeConn | None = None) -> PgVectorStore:
    """A PgVectorStore instance WITHOUT running __init__ (no real DB connection)."""
    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "chunks"
    store._dim = 3
    store._supersession_cache = None
    store._closed = False
    store._conn = conn if conn is not None else _FakeConn()
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

    store = _bare_store()  # conn reports closed -> a genuine drop
    reconnects = {"n": 0}
    fresh_conn = _FakeConn(closed=False)

    def fake_reconnect():
        reconnects["n"] += 1
        store._conn = fresh_conn

    store._reconnect = fake_reconnect  # type: ignore[method-assign]
    target = _BrokenThenGood()

    result = store._with_retry(lambda conn: target.op())
    assert result == "recovered"
    assert reconnects["n"] == 1  # reconnected exactly once
    assert target.calls == 2  # original attempt + one retry


def test_with_retry_propagates_second_failure():
    store = _bare_store()
    store._reconnect = lambda: setattr(store, "_conn", _FakeConn())  # type: ignore[method-assign]

    def always_broken(_conn):
        raise psycopg.InterfaceError("connection already closed")

    with pytest.raises(psycopg.InterfaceError):
        store._with_retry(always_broken)


def test_with_retry_does_not_retry_non_connection_errors():
    store = _bare_store()
    reconnects = {"n": 0}
    store._reconnect = lambda: reconnects.__setitem__("n", reconnects["n"] + 1)  # type: ignore

    def bad_query(_conn):
        raise psycopg.errors.UndefinedColumn("no such column")

    with pytest.raises(psycopg.errors.UndefinedColumn):
        store._with_retry(bad_query)
    assert reconnects["n"] == 0  # a data/query error must NOT trigger a reconnect


class _Cursor:
    def fetchone(self):
        return (3,)


class _RecordingConn:
    """Records the SQL it is asked to run; optionally dies on its first statement.

    Records the statements themselves rather than a count, so the assertions below can say
    "the same work replayed" instead of pinning a number that drifts every time
    `ensure_schema` gains a statement.
    """

    def __init__(self, *, fail_first: bool = False):
        self.fail_first = fail_first
        self.sql: list[str] = []
        # the retry only reconnects when the connection reports itself dead; a dropped socket
        # is exactly that
        self.closed = fail_first
        self.broken = False

    @property
    def calls(self) -> int:
        return len(self.sql)

    def execute(self, sql="", *_args, **_kwargs):
        self.sql.append(" ".join(str(sql).split()))
        if self.fail_first:
            self.fail_first = False
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        return _Cursor()


def _ensure_schema_statements() -> list[str]:
    """The statements a CLEAN ensure_schema issues — the baseline to replay against.

    Derived by running the real method, so it cannot drift from the implementation.
    """
    store = _bare_store(_RecordingConn())
    store.ensure_schema()
    return store._conn.sql


def test_ensure_schema_uses_reconnect_retry():
    """A broken connection mid-ensure_schema replays the WHOLE operation on a fresh one.

    Asserted against a baseline captured from the method itself, not a hard-coded count:
    adding a statement to `ensure_schema` moves both sides together, so this test keeps
    testing the invariant instead of needing to be re-numbered.
    """
    expected = _ensure_schema_statements()

    store = _bare_store(_RecordingConn(fail_first=True))
    first = store._conn
    second = _RecordingConn()
    reconnects = {"n": 0}

    def fake_reconnect():
        reconnects["n"] += 1
        store._conn = second

    store._reconnect = fake_reconnect  # type: ignore[method-assign]

    store.ensure_schema()

    assert reconnects["n"] == 1                 # reconnected exactly once
    assert first.sql == expected[:1]            # died on its first statement
    assert second.sql == expected               # the whole operation replayed, in order


def test_ensure_schema_replay_baseline_is_not_trivially_empty():
    # guards the guard: an ensure_schema that issued nothing would make the assertion above
    # vacuously true
    stmts = _ensure_schema_statements()
    assert len(stmts) > 3
    assert any(s.startswith("CREATE TABLE") for s in stmts)
    assert sum(s.startswith("CREATE INDEX") for s in stmts) >= 2


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


@requires_db
def test_reconnect_does_not_swallow_a_statement_timeout(make_store):
    """QueryCanceled is an OperationalError raised on a LIVE connection.

    Retrying it re-runs the statement on a fresh session that no longer carries the limit which
    killed it — the guard is escaped rather than reported.
    """
    store = make_store(3)
    store._with_retry(lambda c: c.execute("SET statement_timeout = '150ms'"))
    with pytest.raises(psycopg.errors.QueryCanceled):
        store._with_retry(lambda c: c.execute("SELECT pg_sleep(0.5)"))


@requires_db
def test_closed_store_stays_closed(make_store):
    # close() must be final: reconnecting on use silently leaks a connection nobody owns
    store = make_store(3)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.count()


@requires_db
def test_reconnect_is_reported_to_stderr(make_store, capsys):
    # a silent reconnect hides an outage: the unit stays 'active', NRestarts never moves
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    capsys.readouterr()
    store._conn.close()
    assert store.count() == 1
    assert "reconnect" in capsys.readouterr().err.lower()


def test_redacted_dsn_removes_the_password():
    from recall.store import redacted_dsn

    out = redacted_dsn("postgresql://recall:sup3rs3cret@db.example.com:5432/recall")
    assert "sup3rs3cret" not in out and "db.example.com" in out


def test_percent_encoded_default_password_is_still_detected():
    from recall.store import warn_if_insecure_dsn

    # urlsplit returns the RAW encoded form; "recal%6C" IS the password "recall"
    assert warn_if_insecure_dsn("postgresql://recall:recal%6C@db.example.com/recall")


def test_loopback_range_and_socket_dsns_are_treated_as_local():
    from recall.store import warn_if_insecure_dsn

    for dsn in (
        "postgresql://recall:recall@127.0.0.2:5432/recall",
        "postgresql://recall:recall@0.0.0.0:5432/recall",
        "postgresql://recall:recall@host.docker.internal:5432/recall",
        "postgresql://recall:recall@%2Fvar%2Frun%2Fpostgresql/recall",
    ):
        assert warn_if_insecure_dsn(dsn) is None, dsn


def test_with_retry_does_not_retry_a_conn_error_on_a_live_connection():
    """A connection-class error raised while the socket is FINE is not a dropped connection.

    QueryCanceled (statement_timeout), DeadlockDetected and SerializationFailure all subclass
    OperationalError. Reconnecting would re-run the statement on a fresh session without the
    setting that killed it — escaping the guard instead of reporting it.
    """
    store = _bare_store(_FakeConn(closed=False))
    reconnects = {"n": 0}
    store._reconnect = lambda: reconnects.__setitem__("n", reconnects["n"] + 1)  # type: ignore

    def cancelled(_conn):
        raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    with pytest.raises(psycopg.errors.QueryCanceled):
        store._with_retry(cancelled)
    assert reconnects["n"] == 0


@requires_db
def test_ensure_schema_indexes_every_hot_access_path(make_store):
    """The hot paths must be indexed, not just the vector and full-text ones.

    `newest_indexed_at()` runs a max() on EVERY search, a source-filtered search cannot use
    HNSW without an index on `source`, and the supersession map groups on metadata->>'file'.
    Left unindexed each is a sequential scan that grows with the corpus.
    """
    store = make_store(3)
    rows = store._with_retry(
        lambda c: c.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = %s", (store.table,)
        ).fetchall()
    )
    defs = " ".join(r[0] for r in rows)
    assert "indexed_at" in defs, defs
    assert "(source)" in defs, defs
    assert "'file'" in defs, defs


@requires_db
def test_upsert_uses_one_round_trip_per_batch_not_per_row(make_store, monkeypatch):
    # 6k chunks at one execute() each is ~1-3s of pure round-trip latency on every re-index
    store = make_store(3)
    calls = {"execute": 0, "executemany": 0}
    real_cursor = store._conn.cursor

    class CountingCursor:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

        def execute(self, *a, **k):
            calls["execute"] += 1
            return self._inner.execute(*a, **k)

        def executemany(self, *a, **k):
            calls["executemany"] += 1
            return self._inner.executemany(*a, **k)

    monkeypatch.setattr(store._conn, "cursor", lambda *a, **k: CountingCursor(real_cursor()))
    store.upsert(
        [Chunk(f"c{i}", "f.md", f"text {i}") for i in range(50)],
        [[1.0, 0.0, 0.0]] * 50,
    )
    assert calls["executemany"] == 1
    assert calls["execute"] == 0
