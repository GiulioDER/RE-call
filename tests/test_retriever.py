from recall.retriever import HybridRetriever
from recall.types import Chunk

from tests.conftest import requires_db


class DictEmbedder:
    """Deterministic embedder mapping known texts to known vectors."""

    dim = 3
    name = "dict"

    def __init__(self, mapping: dict[str, list[float]], default: list[float]) -> None:
        self._mapping = mapping
        self._default = default

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(t, self._default) for t in texts]


@requires_db
def test_search_returns_relevant_hit_without_gap(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "cats"), Chunk("b", "f.md", "dogs")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = HybridRetriever(store, emb, candidate_k=10).search("cats")
    assert result.gap_warning is False
    assert result.hits[0].chunk.id == "a"


@requires_db
def test_search_sets_gap_warning_when_no_semantic_match(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "cats"), Chunk("b", "f.md", "dogs")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])  # query orthogonal to everything
    result = HybridRetriever(store, emb, candidate_k=10).search("unicorns")
    assert result.gap_warning is True


@requires_db
def test_search_reports_staleness(make_store):
    from datetime import timedelta

    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    # max_age=0 forces "stale" for a just-written row.
    result = HybridRetriever(store, emb, max_age=timedelta(0)).search("cats")
    assert result.staleness.stale is True


@requires_db
def test_sparse_only_hit_carries_true_dense_cosine(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("a", "f.md", "felines"),          # dense match, no lexical overlap
            Chunk("b", "g.md", "cats cats cats"),   # lexical match only
        ],
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
    )
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    # candidate_k=1: the dense leg only surfaces "felines", so "cats cats cats" is sparse-only
    result = HybridRetriever(store, emb, candidate_k=1).search("cats", k=5)
    by_id = {h.chunk.id: h for h in result.hits}
    assert "b" in by_id
    assert abs(by_id["b"].score - 0.6) < 1e-6  # true cosine, not the old 0.0 placeholder
    assert by_id["b"].indexed_at is not None


@requires_db
def test_reranker_can_rescue_a_candidate_below_the_top_k_cut(make_store):
    """The cross-encoder must see more than the final top-k, so it can rescue a buried hit.

    Otherwise a relevant doc sitting just below the k boundary can never be rescued — the
    rerank stage would only reorder what fusion already chose. The pool it sees is bounded by
    `rerank_k` (cross-encoder cost is one forward pass per candidate), not by the fused length.
    """
    store = make_store(3)
    # 5 docs; "target" is the LAST one by fused rank, so a k=2 pre-cut would hide it
    chunks, vecs = [], []
    for i in range(4):
        chunks.append(Chunk(f"d{i}", "f.md", f"filler document {i}"))
        vecs.append([1.0, 0.0, 0.0])
    chunks.append(Chunk("target", "f.md", "the buried answer"))
    vecs.append([0.0, 1.0, 0.0])
    store.upsert(chunks, vecs)

    class TargetReranker:
        """Ranks the buried answer first — what a real cross-encoder would do."""

        def rerank(self, query, hits):
            return sorted(hits, key=lambda h: h.chunk.id != "target")

    emb = DictEmbedder({}, default=[1.0, 0.0, 0.0])  # query matches the filler, not the target
    retr = HybridRetriever(store, emb, reranker=TargetReranker(), candidate_k=10)
    result = retr.search("filler", k=2)
    assert len(result.hits) == 2
    assert result.hits[0].chunk.id == "target"


@requires_db
def test_search_rejects_nonpositive_k(make_store):
    import pytest

    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({}, default=[1.0, 0.0, 0.0])
    retr = HybridRetriever(store, emb)
    with pytest.raises(ValueError):
        retr.search("cats", k=0)
    with pytest.raises(ValueError):
        retr.search("cats", k=-3)


@requires_db
def test_reranker_depth_is_bounded_not_the_whole_fused_pool(make_store):
    # a cross-encoder costs one forward pass per candidate; the pool it scores must be bounded
    # by rerank_k, not by however many candidates retrieval happened to fan out to
    store = make_store(3)
    chunks = [Chunk(f"d{i}", "f.md", f"doc {i}") for i in range(30)]
    store.upsert(chunks, [[1.0, 0.0, 0.0]] * 30)

    seen = {}

    class CountingReranker:
        def rerank(self, query, hits):
            seen["n"] = len(hits)
            return hits

    emb = DictEmbedder({}, default=[1.0, 0.0, 0.0])
    HybridRetriever(
        store, emb, reranker=CountingReranker(), candidate_k=30, rerank_k=8
    ).search("doc", k=3)
    assert seen["n"] == 8
