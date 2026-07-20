from datetime import datetime, timezone

from recall.retriever import HybridRetriever
from recall.types import Chunk, ScoredChunk

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


class _FakeStore:
    """DB-free store stub: returns a fixed dense ranking, no sparse hits."""

    def __init__(self, dense: list[ScoredChunk]) -> None:
        self._dense = dense

    def query_dense(self, vector, k, source=None):
        return self._dense[:k]

    def query_sparse(self, query, k, source=None, vec=None):
        return []

    def newest_indexed_at(self):
        return datetime.now(timezone.utc)


class _PromoteReranker:
    """Reranker that moves the hit with the given id to the front (simulating a rescue)."""

    def __init__(self, winner: str) -> None:
        self._winner = winner

    def rerank(self, query, hits):
        return sorted(hits, key=lambda h: h.chunk.id != self._winner)


def test_reranker_rescues_doc_outside_naive_top_k():
    # "c" is last in the fused ranking (fused rank 3), so a naive top-k=1 slice before reranking
    # would drop it. Reranking the full candidate pool THEN truncating lets it be rescued.
    now = datetime.now(timezone.utc)
    dense = [
        ScoredChunk(Chunk("a", "f.md", "alpha"), score=0.9, indexed_at=now),
        ScoredChunk(Chunk("b", "f.md", "beta"), score=0.8, indexed_at=now),
        ScoredChunk(Chunk("c", "f.md", "gamma"), score=0.7, indexed_at=now),
    ]
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])
    retr = HybridRetriever(
        _FakeStore(dense), emb, reranker=_PromoteReranker("c"), candidate_k=3, use_sparse=False
    )
    result = retr.search("q", k=1)
    assert [h.chunk.id for h in result.hits] == ["c"]


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
