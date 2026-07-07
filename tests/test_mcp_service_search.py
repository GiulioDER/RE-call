from recall.types import Chunk
from recall_mcp.service import search_memory

from tests.conftest import requires_db


class DictEmbedder:
    dim = 3
    name = "dict"

    def __init__(self, mapping, default):
        self._mapping, self._default = mapping, default

    def embed(self, texts):
        return [self._mapping.get(t, self._default) for t in texts]


@requires_db
def test_search_memory_answerable(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = search_memory(store, emb, "cats")
    assert result.gap_warning is False
    assert result.hits and result.hits[0].source == "notes.md"
    assert "relevant" in result.advice.lower()


@requires_db
def test_search_memory_gap(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])  # query orthogonal -> gap
    result = search_memory(store, emb, "unicorns")
    assert result.gap_warning is True
    assert "gap" in result.advice.lower() or "unreliable" in result.advice.lower()
