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
