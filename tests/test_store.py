from recall.types import Chunk

from tests.conftest import requires_db


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
