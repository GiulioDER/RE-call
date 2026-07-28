from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.retriever import HybridRetriever
from tests.conftest import requires_db


def _index(tmp_path, make_store):
    for i in range(12):
        (tmp_path / f"doc{i}.md").write_text(
            f"caching decision {i} about retrieval and indexing", encoding="utf-8"
        )
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    return store, emb


@requires_db
def test_default_fusion_is_unchanged(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    assert HybridRetriever(store, emb)._fusion == "rrf"


@requires_db
def test_wrrf_returns_the_same_candidate_SET_as_rrf(tmp_path, make_store):
    """Weighting reorders; it must never add or drop a candidate. If the sets differ, the
    weighting has changed retrieval rather than ranking, and every downstream comparison
    between the two arms would be confounded."""
    store, emb = _index(tmp_path, make_store)
    a = HybridRetriever(store, emb, candidate_k=20).search("caching retrieval", k=10)
    b = HybridRetriever(store, emb, candidate_k=20, fusion="wrrf").search("caching retrieval", k=10)
    assert {h.chunk.id for h in a.hits} == {h.chunk.id for h in b.hits}


@requires_db
def test_wrrf_keeps_score_as_the_dense_cosine(tmp_path, make_store):
    """The trust layer reads `score` as a cosine. Fusion weights must not leak into it."""
    store, emb = _index(tmp_path, make_store)
    r = HybridRetriever(store, emb, fusion="wrrf").search("caching retrieval", k=5)
    assert all(-1.0 <= h.score <= 1.0 for h in r.hits)


@requires_db
def test_unknown_fusion_is_rejected(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    with pytest.raises(ValueError):
        HybridRetriever(store, emb, fusion="bogus")
