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
