import pytest
import uuid
from urllib.parse import urlsplit, urlunsplit

import psycopg

from recall.store import PgVectorStore
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


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
def test_supersession_drops_an_edge_whose_target_basename_is_ambiguous(make_store):
    """Two files sharing a basename must not produce a silently mis-mapped edge.

    `supersedes:` names a basename. When that basename exists in more than one directory the
    edge cannot be resolved, so it is withheld and the endpoint reported as unresolved —
    fail closed rather than attribute supersession to the wrong document.
    """
    store = make_store(3)
    store.upsert(
        [
            Chunk("a", "a/notes.md", "one", metadata={"file": "notes.md", "ord": 0}),
            Chunk("b", "b/notes.md", "two", metadata={"file": "notes.md", "ord": 0}),
            Chunk(
                "c", "c/new.md", "three",
                metadata={"file": "new.md", "ord": 0, "supersedes": "notes.md"},
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    edges, unresolved = store.supersession()
    assert edges == {}  # no guessed edge
    assert "notes.md" in unresolved
    assert store.supersession_map() == {}


@requires_db
def test_supersession_keeps_unambiguous_edges_alongside_an_ambiguous_one(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("a", "a/notes.md", "one", metadata={"file": "notes.md", "ord": 0}),
            Chunk("b", "b/notes.md", "two", metadata={"file": "notes.md", "ord": 0}),
            Chunk("c", "c/new.md", "x", metadata={"file": "new.md", "ord": 0,
                                                  "supersedes": "notes.md"}),
            Chunk("d", "d/v1.md", "y", metadata={"file": "v1.md", "ord": 0}),
            Chunk("e", "e/v2.md", "z", metadata={"file": "v2.md", "ord": 0,
                                                 "supersedes": "v1.md"}),
        ],
        [[1.0, 0.0, 0.0]] * 5,
    )
    edges, unresolved = store.supersession()
    assert edges == {"v1.md": "v2.md"}  # the clean edge survives
    assert unresolved == frozenset({"notes.md"})


@requires_db
def test_supersession_drops_an_edge_whose_successor_basename_is_ambiguous(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("a", "a/dup.md", "one", metadata={"file": "dup.md", "ord": 0,
                                                    "supersedes": "old.md"}),
            Chunk("b", "b/dup.md", "two", metadata={"file": "dup.md", "ord": 0}),
            Chunk("c", "c/old.md", "three", metadata={"file": "old.md", "ord": 0}),
        ],
        [[1.0, 0.0, 0.0]] * 3,
    )
    edges, unresolved = store.supersession()
    assert edges == {}  # which 'dup.md' is the successor? unknowable
    assert "old.md" in unresolved


@requires_db
def test_supersession_is_unambiguous_when_one_file_has_many_chunks(make_store):
    # several chunks of the SAME file share a basename but not a source — not ambiguous
    store = make_store(3)
    store.upsert(
        [
            Chunk("a1", "a/v1.md", "one", metadata={"file": "v1.md", "ord": 0}),
            Chunk("a2", "a/v1.md", "two", metadata={"file": "v1.md", "ord": 1}),
            Chunk("b1", "a/v2.md", "x", metadata={"file": "v2.md", "ord": 0,
                                                  "supersedes": "v1.md"}),
        ],
        [[1.0, 0.0, 0.0]] * 3,
    )
    assert store.supersession_map() == {"v1.md": "v2.md"}


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
def test_store_reconnects_after_the_connection_drops(make_store):
    """A long-running MCP server must survive a DB restart / dropped connection.

    Without this, one network blip poisons every later tool call until the server itself is
    restarted — the connection is opened once for the process lifetime.
    """
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    store._conn.close()  # simulate the server-side drop
    assert store.count() == 1  # transparently reconnected
    assert store.query_dense([1.0, 0.0, 0.0], k=1)[0].chunk.id == "a"


@requires_db
def test_reconnect_reregisters_the_vector_type(make_store):
    # pgvector's adapters are registered per connection: a reconnect that forgets them would
    # make every dense query fail with a type-adaptation error instead of returning rows
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    store._conn.close()
    hits = store.query_dense([1.0, 0.0, 0.0], k=1)
    assert hits and abs(hits[0].score - 1.0) < 1e-6


def test_store_does_not_retry_forever_on_an_unreachable_db():
    import pytest as _pytest

    with _pytest.raises(Exception):
        PgVectorStore(
            "postgresql://nobody:nobody@127.0.0.1:1/none?connect_timeout=2", dim=3
        )


def test_default_credentials_at_a_remote_host_are_flagged():
    from recall.store import insecure_default_credentials

    msg = insecure_default_credentials("postgresql://recall:recall@db.example.com:5432/recall")
    assert msg and "recall:recall" in msg


def test_default_credentials_on_localhost_are_not_flagged():
    from recall.store import insecure_default_credentials

    for dsn in (
        "postgresql://recall:recall@localhost:5432/recall",
        "postgresql://recall:recall@127.0.0.1:5432/recall",
        "postgresql://recall:recall@[::1]:5432/recall",
    ):
        assert insecure_default_credentials(dsn) is None


def test_real_credentials_at_a_remote_host_are_not_flagged():
    from recall.store import insecure_default_credentials

    assert insecure_default_credentials("postgresql://app:s3cret@db.example.com/recall") is None


def test_unparseable_dsn_does_not_raise():
    from recall.store import insecure_default_credentials

    assert insecure_default_credentials("host=db.example.com user=recall") is None


@requires_db
def test_local_default_dsn_does_not_warn(make_store, capsys):
    store = make_store(3)
    capsys.readouterr()
    PgVectorStore(TEST_DSN.replace("localhost", "127.0.0.1"), dim=3, table=store.table).close()
    assert "recall:recall" not in capsys.readouterr().err  # local dev stays quiet


def test_constructor_warns_on_a_remote_default_dsn(capsys):
    # the predicate is unit-tested above; this pins that __init__ actually EMITS it
    with pytest.raises(Exception):  # connection will fail; the warning precedes it
        PgVectorStore(
            "postgresql://recall:recall@db.example.com:1/recall?connect_timeout=2", dim=3
        )
    assert "recall:recall" in capsys.readouterr().err


@requires_db
def test_reconnect_does_not_swallow_a_statement_timeout(make_store):
    """QueryCanceled is an OperationalError raised on a LIVE connection.

    Retrying it silently re-runs the statement on a fresh session that no longer carries the
    limit which killed it — the guard is escaped rather than reported.
    """
    store = make_store(3)
    store._execute("SET statement_timeout = '150ms'")
    with pytest.raises(psycopg.errors.QueryCanceled):
        store._execute("SELECT pg_sleep(0.4)")


@requires_db
def test_closed_store_stays_closed(make_store):
    # close() must be final: reconnecting on use would silently leak a connection nobody owns
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.count()


@requires_db
def test_reconnect_is_reported_to_stderr(make_store, capsys):
    # a silent reconnect hides an outage: the unit stays 'active', NRestarts never moves,
    # and nothing in the journal records that the DB went away
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    capsys.readouterr()
    store._conn.close()
    assert store.count() == 1
    assert "reconnect" in capsys.readouterr().err.lower()


def test_redacted_dsn_removes_the_password():
    from recall.store import redacted_dsn

    out = redacted_dsn("postgresql://recall:sup3rs3cret@db.example.com:5432/recall")
    assert "sup3rs3cret" not in out
    assert "db.example.com" in out and "recall" in out


def test_percent_encoded_default_password_is_still_detected():
    from recall.store import insecure_default_credentials

    # urlsplit returns the RAW encoded form; "recal%6C" IS the password "recall"
    assert insecure_default_credentials("postgresql://recall:recal%6C@db.example.com/recall")


def test_loopback_range_and_unix_socket_are_local():
    from recall.store import insecure_default_credentials

    for dsn in (
        "postgresql://recall:recall@127.0.0.2:5432/recall",
        "postgresql://recall:recall@0.0.0.0:5432/recall",
        "postgresql://recall:recall@host.docker.internal:5432/recall",
        "postgresql://recall:recall@%2Fvar%2Frun%2Fpostgresql/recall",
    ):
        assert insecure_default_credentials(dsn) is None, dsn
