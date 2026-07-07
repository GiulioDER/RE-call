from recall.types import Chunk
from recall_mcp.service import memory_stats

from tests.conftest import requires_db


@requires_db
def test_store_count(make_store):
    store = make_store(3)
    assert store.count() == 0
    store.upsert([Chunk("a", "f", "x")], [[1.0, 0.0, 0.0]])
    assert store.count() == 1


@requires_db
def test_memory_stats(make_store):
    store = make_store(3)
    assert memory_stats(store)["chunks"] == 0
    store.upsert([Chunk("a", "f", "x")], [[1.0, 0.0, 0.0]])
    stats = memory_stats(store)
    assert stats["chunks"] == 1
    assert stats["newest_indexed_at"] is not None
