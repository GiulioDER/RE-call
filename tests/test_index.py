from recall.index import Indexer, chunk_code, chunk_text
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


def test_chunk_code_keeps_top_level_blocks_whole():
    src = "import os\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    chunks = chunk_code(src, max_chars=20)
    assert chunks == ["import os", "def alpha():\n    return 1", "def beta():\n    return 2"]


def test_chunk_code_keeps_methods_with_their_class():
    src = "class Foo:\n    def bar(self):\n        return 1\n"
    chunks = chunk_code(src, max_chars=800)
    assert len(chunks) == 1  # an indented method does not start a new top-level block
    assert "class Foo:" in chunks[0] and "def bar" in chunks[0]


@requires_db
def test_index_code_with_code_chunker(tmp_path, make_store):
    (tmp_path / "mod.py").write_text(
        "import os\n\n\ndef reciprocal_rank_fusion(rankings):\n"
        "    return {}\n\n\ndef unrelated_helper():\n    return 0\n",
        encoding="utf-8",
    )
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    stats = Indexer(store, emb, chunker=chunk_code).index_path(tmp_path, glob="**/*.py")
    assert stats.files == 1
    hits = store.query_sparse("reciprocal rank fusion", k=5)
    assert hits and "reciprocal_rank_fusion" in hits[0].chunk.text
