import pytest

from recall.embeddings import HashingEmbedder
from recall_mcp.service import index_memory, search_memory

from tests.conftest import requires_db


@requires_db
def test_index_then_search(tmp_path, make_store, monkeypatch):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "note.md").write_text("the caching decision was adopted", encoding="utf-8")
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    stats = index_memory(store, emb, str(tmp_path))
    assert stats["chunks"] == 1
    assert stats["files"] == 1
    result = search_memory(store, emb, "caching")
    assert any("caching" in h.text for h in result.hits)


@requires_db
def test_index_rejects_path_outside_root(tmp_path, make_store, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    # tmp_path is the PARENT of the allowed root -> must be rejected before any read.
    with pytest.raises(ValueError, match="outside the allowed index root"):
        index_memory(store, emb, str(tmp_path))
