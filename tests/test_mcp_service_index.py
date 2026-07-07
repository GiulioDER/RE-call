from recall.embeddings import HashingEmbedder
from recall_mcp.service import index_memory, search_memory

from tests.conftest import requires_db


@requires_db
def test_index_then_search(tmp_path, make_store):
    (tmp_path / "note.md").write_text("the caching decision was adopted", encoding="utf-8")
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    stats = index_memory(store, emb, str(tmp_path))
    assert stats["chunks"] == 1
    assert stats["files"] == 1
    result = search_memory(store, emb, "caching")
    assert any("caching" in h.text for h in result.hits)
