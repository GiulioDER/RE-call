from __future__ import annotations

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from tests.conftest import requires_db


@requires_db
def test_query_sparse_with_rank_returns_ts_rank_not_cosine(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    (tmp_path / "b.md").write_text("indexing decision two", encoding="utf-8")
    (tmp_path / "c.md").write_text("caching appears here once", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    qvec = emb.embed(["caching"])[0]

    hits, ranks = store.query_sparse("caching", k=5, vec=qvec, with_rank=True)

    assert len(ranks) == len(hits)
    assert ranks == sorted(ranks, reverse=True)          # ts_rank order, descending
    assert all(r >= 0.0 for r in ranks)
    assert any(r > 0.0 for r in ranks)                   # a real lexical match scored
    # the whole point: ts_rank is a DIFFERENT quantity from the cosine in `score`
    assert [h.score for h in hits] != ranks


@requires_db
def test_query_sparse_default_return_is_unchanged(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    qvec = emb.embed(["caching"])[0]

    plain = store.query_sparse("caching", k=5, vec=qvec)
    hits, _ = store.query_sparse("caching", k=5, vec=qvec, with_rank=True)

    assert isinstance(plain, list)                        # not a tuple
    assert [h.chunk.id for h in plain] == [h.chunk.id for h in hits]
    assert [h.score for h in plain] == [h.score for h in hits]


@requires_db
def test_query_sparse_with_rank_without_vec_returns_score_as_rank(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)

    hits, ranks = store.query_sparse("caching", k=5, with_rank=True)

    # with no vec, `score` IS ts_rank already — the two must agree exactly
    assert [h.score for h in hits] == ranks
