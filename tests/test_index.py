import pytest

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


def test_chunk_text_force_splits_oversized_block_under_cap():
    # a single block (no blank lines) longer than max_chars must not become one oversized chunk
    # that the embedder would silently truncate — it is force-split into pieces under the cap
    text = " ".join(f"word{i:03d}" for i in range(400))  # ~3200 chars, no blank lines
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_oversized_split_has_overlap():
    # adjacent pieces of a force-split block share `overlap` characters so a fact on the cut
    # boundary survives in both neighbours
    text = "x" * 500 + "y" * 500  # 1000 chars, single block
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert chunks[0][-50:] == chunks[1][:50]  # tail of one == head of the next


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


@requires_db
def test_index_path_stores_validity_frontmatter_in_metadata(tmp_path, make_store):
    (tmp_path / "policy_v2.md").write_text(
        "---\nvalid_until: 2099-12-31\nsupersedes: policy_v1.md\n---\nnew policy body",
        encoding="utf-8",
    )
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    Indexer(store, emb).index_path(tmp_path)
    hits = store.query_sparse("policy", k=5)
    assert hits
    md = hits[0].chunk.metadata
    assert md["supersedes"] == "policy_v1.md"
    assert md["valid_until"] == "2099-12-31"
    assert md["file"] == "policy_v2.md" and md["ord"] == 0
    assert "---" not in hits[0].chunk.text  # frontmatter block is not indexed


@requires_db
def test_index_path_malformed_date_raises_with_filename(tmp_path, make_store):
    (tmp_path / "bad.md").write_text("---\nvalid_until: soonish\n---\nbody", encoding="utf-8")
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    with pytest.raises(ValueError, match="bad.md"):
        Indexer(store, emb).index_path(tmp_path)


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


@requires_db
def test_reindex_replaces_rows_no_orphans_and_no_stale_supersedes(tmp_path, make_store):
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    big = tmp_path / "doc.md"
    # two chunks + a supersedes claim
    big.write_text(
        "---\nsupersedes: other.md\n---\n" + ("alpha " * 100) + "\n\n" + ("beta " * 100),
        encoding="utf-8",
    )
    Indexer(store, emb).index_path(tmp_path)
    assert store.count() == 2
    assert store.supersession_map() == {"other.md": "doc.md"}
    # shrink the doc AND withdraw the claim: no orphan chunk, no stale supersession
    big.write_text("gamma only", encoding="utf-8")
    Indexer(store, emb).index_path(tmp_path)
    assert store.count() == 1
    assert store.supersession_map() == {}


@requires_db
def test_bom_file_frontmatter_still_parsed(tmp_path, make_store):
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    (tmp_path / "bom.md").write_bytes(
        b"\xef\xbb\xbf---\nsupersedes: old.md\n---\nbody words here"
    )
    Indexer(store, emb).index_path(tmp_path)
    hits = store.query_sparse("body words", k=1)
    assert hits and hits[0].chunk.metadata.get("supersedes") == "old.md"


@requires_db
def test_failed_embedding_leaves_existing_rows_intact(tmp_path, make_store):
    # re-index must not destroy memory when embedding fails: embed runs BEFORE the
    # delete+insert transaction, so the old rows survive an embedder outage
    class BoomEmbedder:
        dim = 64
        name = "boom"

        def embed(self, texts):
            raise RuntimeError("embedder outage")

    store = make_store(64)
    doc = tmp_path / "doc.md"
    doc.write_text("original memory content", encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)
    assert store.count() == 1

    doc.write_text("updated memory content", encoding="utf-8")
    with pytest.raises(RuntimeError):
        Indexer(store, BoomEmbedder()).index_path(tmp_path)
    assert store.count() == 1  # old row still present
    hits = store.query_sparse("original memory", k=1)
    assert hits and "original" in hits[0].chunk.text


def _repeated_chars(chunks: list[str]) -> int:
    """Characters duplicated across consecutive chunk boundaries."""
    total = 0
    for a, b in zip(chunks, chunks[1:]):
        aw, bw = a.split(), b.split()
        best = 0
        for j in range(1, min(len(aw), len(bw)) + 1):
            if aw[-j:] == bw[:j]:
                best = j
        total += len(" ".join(aw[-best:])) if best else 0
    return total


def test_force_split_breaks_on_whitespace_not_mid_word():
    """A fixed-stride window cuts through words, corrupting the tokens the embedder sees.

    'word18' becoming 'wor' + 'd18' indexes two tokens that mean nothing and loses the one
    that did.
    """
    text = " ".join(f"word{i}" for i in range(400))
    for c in chunk_text(text, max_chars=200):
        for tok in c.split():
            assert tok.startswith("word") and tok[4:].isdigit(), tok


def test_force_split_does_not_inflate_the_corpus():
    """Stride must stay proportional to the cap, whatever the overlap.

    With a fixed stride of `max_chars - overlap`, an overlap at or above max_chars collapses
    the stride to 1 and re-emits a near-identical window per character.
    """
    for mc in (800, 200, 100, 50):
        chunks = chunk_text("x" * 5000, max_chars=mc)
        emitted = sum(len(c) for c in chunks)
        assert emitted < 2 * 5000, f"max_chars={mc}: {emitted / 5000:.1f}x inflation"


def test_force_split_survives_overlap_larger_than_the_cap():
    chunks = chunk_text("x" * 2000, max_chars=100, overlap=500)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) < 2 * 2000


def test_force_split_terminates_for_small_max_chars():
    for mc in range(1, 12):
        chunks = chunk_text("ab cdefghij klm", max_chars=mc)
        assert chunks and all(len(c) <= mc for c in chunks), (mc, chunks)


def test_force_split_overlap_scales_with_its_own_value():
    text = " ".join(f"word{i}" for i in range(400))
    small = _repeated_chars(chunk_text(text, max_chars=200, overlap=20))
    large = _repeated_chars(chunk_text(text, max_chars=200, overlap=160))
    assert large > small


@requires_db
def test_reindexing_the_same_dir_under_a_different_spelling_does_not_duplicate(
    tmp_path, make_store, monkeypatch
):
    """`source` is the row key `replace_sources` deletes by.

    Left as typed, indexing one corpus as `corpus` and then `/abs/corpus` writes a SECOND copy
    of every chunk instead of replacing it.
    """
    (tmp_path / "v1.md").write_text("old rate policy one hundred", encoding="utf-8")
    (tmp_path / "v2.md").write_text(
        "---\nsupersedes: v1.md\n---\nnew rate policy twenty", encoding="utf-8"
    )
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    Indexer(store, emb).index_path(tmp_path)
    rows = store.count()

    monkeypatch.chdir(tmp_path.parent)
    Indexer(store, emb).index_path(tmp_path.name)  # SAME dir, relative spelling

    assert store.count() == rows  # replaced, not duplicated
