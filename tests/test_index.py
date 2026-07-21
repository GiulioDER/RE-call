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


def test_chunk_text_hard_splits_a_block_longer_than_max_chars():
    # a wall of prose with no blank line must not become one oversized chunk that the
    # embedder would silently truncate — the tail would be lost from the index
    text = " ".join(f"word{i}" for i in range(400))  # ~2.7k chars, no blank line
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_hard_split_keeps_every_word():
    text = " ".join(f"word{i}" for i in range(400))
    joined = " ".join(chunk_text(text, max_chars=200))
    for i in range(400):
        assert f"word{i} " in joined + " "


def test_chunk_text_hard_split_breaks_on_whitespace_not_mid_word():
    text = " ".join(f"word{i}" for i in range(400))
    for c in chunk_text(text, max_chars=200):
        for tok in c.split():
            assert tok.startswith("word") and tok[4:].isdigit()


def _repeated_chars(chunks: list[str]) -> int:
    """Total characters duplicated across consecutive chunk boundaries."""
    total = 0
    for a, b in zip(chunks, chunks[1:]):
        aw, bw = a.split(), b.split()
        best = 0
        for j in range(1, min(len(aw), len(bw)) + 1):
            if aw[-j:] == bw[:j]:
                best = j
        total += len(" ".join(aw[-best:])) if best else 0
    return total


def test_chunk_text_hard_split_overlaps_forced_pieces():
    # a forced split severs a sentence; a small overlap keeps the boundary retrievable
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 1
    tail = chunks[0].split()[-1]
    assert tail in chunks[1].split()


def test_chunk_text_overlap_scales_with_its_own_value():
    """A knob that ignores its own value is not a feature.

    Guards the direction of the step-back search: seeking the LAST whitespace before the cut
    yields one word regardless of `overlap`, which reads as working but repeats ~nothing.
    """
    text = " ".join(f"word{i}" for i in range(400))
    small = _repeated_chars(chunk_text(text, max_chars=200, overlap=20))
    large = _repeated_chars(chunk_text(text, max_chars=200, overlap=160))
    assert large > small * 2


def test_chunk_text_overlap_applies_to_newline_delimited_blocks():
    # chunk_code's blocks are newline-delimited; the cut accepts \n, so the step-back must too
    text = "\n".join(f"token{i}" for i in range(300))
    assert _repeated_chars(chunk_text(text, max_chars=200, overlap=80)) > 0


def test_chunk_text_hard_split_does_not_emit_slivers():
    """Whitespace landing exactly on the cap must not produce 1-character chunks.

    Each chunk costs a full embedding row and an index entry, so dust is pure overhead that
    also crowds the top-k. Only the trailing remainder may be short — that is leftover
    content, not a splitting artifact.
    """
    chunks = chunk_text(("B" * 800 + " ") * 6, max_chars=800, overlap=80)
    assert all(len(c) > 8 for c in chunks[:-1]), [len(c) for c in chunks]


def test_chunk_text_splits_an_unbreakable_block_without_whitespace():
    # no whitespace to break on (a base64 blob / long token): still must be capped
    chunks = chunk_text("x" * 1000, max_chars=200)
    assert len(chunks) == 5 and all(len(c) <= 200 for c in chunks)


def test_chunk_code_hard_splits_an_oversized_function():
    src = "def huge():\n" + "\n".join(f"    step_{i}()" for i in range(200))
    chunks = chunk_code(src, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


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
    chunks = chunk_code(src, max_chars=30)  # each block fits; none is force-split
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


@requires_db
def test_reindexing_the_same_dir_under_a_different_spelling_does_not_duplicate(
    tmp_path, make_store, monkeypatch
):
    """Re-indexing one corpus by a different path spelling must not double its rows.

    `source` is the row key `replace_sources` deletes by, so an unnormalized spelling
    (relative vs absolute) writes a SECOND copy of every chunk. That alone is waste — but it
    also makes each basename look like it belongs to two documents, which withdraws every
    supersession edge touching it and turns a correct answer into an abstention.
    """
    (tmp_path / "v1.md").write_text("old rate policy one hundred", encoding="utf-8")
    (tmp_path / "v2.md").write_text(
        "---\nsupersedes: v1.md\n---\nnew rate policy twenty", encoding="utf-8"
    )
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    Indexer(store, emb).index_path(tmp_path)  # absolute
    assert store.supersession_map() == {"v1.md": "v2.md"}
    rows = store.count()

    monkeypatch.chdir(tmp_path.parent)
    Indexer(store, emb).index_path(tmp_path.name)  # SAME dir, relative spelling

    assert store.count() == rows  # replaced, not duplicated
    edges, unresolved = store.supersession()
    assert edges == {"v1.md": "v2.md"}  # the edge survives
    assert unresolved == frozenset()



def test_chunk_text_terminates_for_small_max_chars():
    """A cut landing exactly on `start` must still force progress.

    Guards a boundary the minimum-piece rule can silently swallow: when max_chars is small the
    piece floor rounds to 0, so a `cut == start` no longer trips it and the walk stalls.
    """
    for mc in range(1, 12):
        chunks = chunk_text("ab cdefghij klm", max_chars=mc)
        assert chunks and all(len(c) <= mc for c in chunks), (mc, chunks)


def test_chunk_text_overlap_does_not_inflate_the_corpus():
    """Overlap repeats context; it must not re-emit almost the same piece over and over.

    The step-back has to advance the walk by a real amount, not just satisfy the piece floor —
    otherwise each frame emits a near-duplicate and the index (and embedding bill) balloons.
    """
    text = ("x" * 180 + " " + "y " * 40) * 40
    emitted = sum(len(c) for c in chunk_text(text, max_chars=200, overlap=160))
    assert emitted < 2 * len(text), f"{emitted / len(text):.1f}x inflation"
