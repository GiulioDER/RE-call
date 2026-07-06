from recall.index import Indexer, chunk_text
from recall.embeddings import HashingEmbedder

from tests.conftest import requires_db


def test_chunk_text_splits_on_blank_lines():
    text = "para one\n\npara two\n\npara three"
    assert chunk_text(text, max_chars=10) == ["para one", "para two", "para three"]


def test_chunk_text_packs_small_paragraphs_together():
    text = "a\n\nb\n\nc"
    assert chunk_text(text, max_chars=800) == ["a\n\nb\n\nc"]


def test_chunk_text_ignores_empty_input():
    assert chunk_text("   \n\n  ") == []


@requires_db
def test_index_path_ingests_markdown(tmp_path, make_store):
    (tmp_path / "a.md").write_text("caching decision one", encoding="utf-8")
    (tmp_path / "b.md").write_text("indexing decision two", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    stats = Indexer(store, emb).index_path(tmp_path)
    assert stats.files == 2
    assert stats.chunks == 2
    hits = store.query_sparse("caching", k=5)
    assert hits and hits[0].chunk.text == "caching decision one"
