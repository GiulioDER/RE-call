from __future__ import annotations

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.retriever import HybridRetriever, LegProbe
from tests.conftest import requires_db


def _index(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one about caching", encoding="utf-8")
    (tmp_path / "b.md").write_text("indexing decision two", encoding="utf-8")
    (tmp_path / "c.md").write_text("caching appears here once", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    return store, emb


@requires_db
def test_attaching_a_probe_does_not_change_the_result(tmp_path, make_store):
    """The apparatus guarantee in miniature: instrumentation that perturbs the retrieved set
    has broken the thing it was measuring."""
    store, emb = _index(tmp_path, make_store)
    plain = HybridRetriever(store, emb).search("caching", k=5)
    seen: list[LegProbe] = []
    probed = HybridRetriever(store, emb, probe=seen.append).search("caching", k=5)

    assert [h.chunk.id for h in plain.hits] == [h.chunk.id for h in probed.hits]
    assert [h.score for h in plain.hits] == [h.score for h in probed.hits]
    assert plain.gap_warning == probed.gap_warning
    assert len(seen) == 1


@requires_db
def test_probe_reports_sparse_ts_rank_not_cosine(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    seen: list[LegProbe] = []
    HybridRetriever(store, emb, probe=seen.append).search("caching", k=5)
    p = seen[0]

    assert p.query == "caching"
    assert len(p.sparse_ranks) == len(p.sparse)
    assert p.sparse_ranks == sorted(p.sparse_ranks, reverse=True)
    assert [h.score for h in p.sparse] != p.sparse_ranks   # cosine != ts_rank
    assert p.dense and p.fused


@requires_db
def test_probe_fires_once_per_search_with_empty_sparse_when_disabled(tmp_path, make_store):
    store, emb = _index(tmp_path, make_store)
    seen: list[LegProbe] = []
    HybridRetriever(store, emb, use_sparse=False, probe=seen.append).search("caching", k=5)

    assert len(seen) == 1
    assert seen[0].sparse == [] and seen[0].sparse_ranks == []
