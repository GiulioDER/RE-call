"""Incremental indexing: only touch what changed, bound memory, and notice deletions.

Three defects, one code path:

1. **Every re-index rewrote the whole corpus.** `replace_sources` was handed every file, so
   changing one note deleted and re-inserted every row — WAL and lock cost proportional to
   corpus size rather than to the change.
2. **The whole corpus was materialised in memory** before a single row was written: read every
   file, chunk everything, embed everything, then write. A large corpus exhausts RAM and writes
   nothing until it finishes (observed: 2.3 GB resident and 30+ minutes with zero rows visible).
3. **A file deleted from disk kept its rows forever** — the glob only lists files that still
   exist, so vanished ones were never replaced and never removed. That one is not a performance
   bug: the trust layer went on serving a deleted memory with verdict `ok`.
"""
from __future__ import annotations

import uuid

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.store import PgVectorStore

from tests.conftest import TEST_DSN, requires_db

DIM = 64


class _CountingEmbedder:
    """Wraps a real embedder and records how many texts it was asked to embed."""

    def __init__(self, dim: int = DIM) -> None:
        self._inner = HashingEmbedder(dim=dim)
        self.embedded: list[str] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return self._inner.embed(texts)


@pytest.fixture
def store():
    table = "ix_" + uuid.uuid4().hex[:8]
    s = PgVectorStore(TEST_DSN, dim=DIM, table=table)
    s.ensure_schema()
    yield s
    try:
        s.drop_table()
    finally:
        s.close()


def _corpus(tmp_path, n: int):
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(n):
        (root / f"note{i}.md").write_text(f"memory number {i}", encoding="utf-8")
    return root


@requires_db
def test_unchanged_files_are_not_re_embedded(tmp_path, store):
    """The headline: re-indexing an untouched corpus must cost (almost) nothing."""
    root = _corpus(tmp_path, 5)
    emb = _CountingEmbedder()
    ix = Indexer(store, emb)

    ix.index_path(root)
    assert len(emb.embedded) == 5

    emb.embedded.clear()
    stats = ix.index_path(root)
    assert emb.embedded == [], "re-embedded files whose content had not changed"
    assert stats.skipped == 5
    assert stats.chunks == 0


@requires_db
def test_only_the_changed_file_is_re_embedded(tmp_path, store):
    root = _corpus(tmp_path, 5)
    emb = _CountingEmbedder()
    ix = Indexer(store, emb)
    ix.index_path(root)

    (root / "note2.md").write_text("memory number 2, revised", encoding="utf-8")
    emb.embedded.clear()
    stats = ix.index_path(root)

    assert emb.embedded == ["memory number 2, revised"]
    assert stats.skipped == 4
    assert stats.files == 1
    assert store.count() == 5  # the other four rows are still there, untouched


@requires_db
def test_a_changed_file_replaces_its_own_rows_only(tmp_path, store):
    """Re-indexing must still be a REPLACE for the file that changed: a shrinking file leaves
    no stale chunks behind to poison retrieval or the supersession map."""
    root = tmp_path / "corpus"
    root.mkdir()
    long_text = "\n\n".join(f"paragraph {i} " + "x" * 700 for i in range(4))
    (root / "big.md").write_text(long_text, encoding="utf-8")
    (root / "other.md").write_text("an unrelated memory", encoding="utf-8")
    ix = Indexer(store, HashingEmbedder(dim=DIM))
    ix.index_path(root)
    assert store.count() > 2

    (root / "big.md").write_text("now it is short", encoding="utf-8")
    ix.index_path(root)
    assert store.count() == 2


@requires_db
def test_rows_of_a_file_deleted_from_disk_are_pruned(tmp_path, store):
    """Not a performance bug: the trust layer served a deleted memory with verdict `ok`."""
    root = _corpus(tmp_path, 3)
    ix = Indexer(store, HashingEmbedder(dim=DIM))
    ix.index_path(root)
    assert store.count() == 3

    (root / "note1.md").unlink()
    stats = ix.index_path(root)

    assert stats.deleted == 1
    assert store.count() == 2
    texts = store._with_retry(
        lambda c: c.execute(f"SELECT text FROM {store.table}").fetchall()
    )
    assert "memory number 1" not in [t[0] for t in texts]


@requires_db
def test_pruning_is_scoped_to_the_indexed_root(tmp_path, store):
    """Indexing a subdirectory must not delete everything indexed from elsewhere."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "one.md").write_text("memory in a", encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "two.md").write_text("memory in b", encoding="utf-8")

    ix = Indexer(store, HashingEmbedder(dim=DIM))
    ix.index_path(a)
    ix.index_path(b)
    assert store.count() == 2

    ix.index_path(a)  # re-index only a
    assert store.count() == 2, "re-indexing one root pruned another root's rows"


@requires_db
def test_indexing_a_single_file_never_prunes(tmp_path, store):
    root = _corpus(tmp_path, 3)
    ix = Indexer(store, HashingEmbedder(dim=DIM))
    ix.index_path(root)
    ix.index_path(root / "note0.md")
    assert store.count() == 3


@requires_db
def test_writes_are_batched_rather_than_one_giant_transaction(tmp_path, store):
    """Memory is bounded by the batch, not by the corpus.

    Asserted through the number of write calls: the previous implementation made exactly one,
    holding every chunk and every embedding in memory until it did.
    """
    root = _corpus(tmp_path, 40)
    ix = Indexer(store, HashingEmbedder(dim=DIM), batch_chunks=8)

    calls: list[int] = []
    real_replace = store.replace_sources

    def counting_replace(sources, chunks, embeddings):
        calls.append(len(chunks))
        return real_replace(sources, chunks, embeddings)

    store.replace_sources = counting_replace  # type: ignore[method-assign]
    try:
        ix.index_path(root)
    finally:
        store.replace_sources = real_replace  # type: ignore[method-assign]

    assert len(calls) > 1, "still writing the whole corpus in one transaction"
    assert max(calls) <= 8 + 4, f"a batch exceeded its bound: {max(calls)}"
    assert store.count() == 40


@requires_db
def test_an_embedding_failure_leaves_earlier_batches_and_is_resumable(tmp_path, store):
    """Batching trades all-or-nothing for progress. The trade is only acceptable because a retry
    is cheap and idempotent: already-written files are skipped by their content hash."""
    root = _corpus(tmp_path, 20)

    class _FailsAfterAWhile(_CountingEmbedder):
        def embed(self, texts):
            if len(self.embedded) >= 8:
                raise RuntimeError("embedding backend is down")
            return super().embed(texts)

    emb = _FailsAfterAWhile()
    ix = Indexer(store, emb, batch_chunks=4)
    with pytest.raises(RuntimeError):
        ix.index_path(root)

    written = store.count()
    assert 0 < written < 20, f"expected partial progress, got {written}"

    good = _CountingEmbedder()
    Indexer(store, good, batch_chunks=4).index_path(root)
    assert store.count() == 20
    assert len(good.embedded) == 20 - written, "re-embedded files that were already written"


@requires_db
def test_content_hash_is_recorded_for_every_chunk(tmp_path, store):
    """The skip decision is only as trustworthy as the fingerprint it reads."""
    root = _corpus(tmp_path, 2)
    Indexer(store, HashingEmbedder(dim=DIM)).index_path(root)
    rows = store._with_retry(
        lambda c: c.execute(
            f"SELECT metadata->>'content_hash' FROM {store.table}"
        ).fetchall()
    )
    assert all(r[0] and len(r[0]) == 64 for r in rows)  # sha256 hex


@requires_db
def test_a_file_whose_content_reverts_is_recognised(tmp_path, store):
    """Hash equality, not mtime: touching a file or reverting an edit must not force work."""
    root = _corpus(tmp_path, 1)
    emb = _CountingEmbedder()
    ix = Indexer(store, emb)
    ix.index_path(root)

    (root / "note0.md").write_text("temporarily different", encoding="utf-8")
    ix.index_path(root)
    (root / "note0.md").write_text("memory number 0", encoding="utf-8")  # reverted
    emb.embedded.clear()
    stats = ix.index_path(root)

    assert emb.embedded == ["memory number 0"]
    assert stats.skipped == 0  # content differs from what is stored, so it is re-indexed


@requires_db
def test_a_nul_byte_in_one_file_does_not_abort_the_whole_corpus(tmp_path, store, caplog):
    """Found by indexing a real 792-file memory corpus: ONE file carried two stray NUL bytes and
    psycopg aborted the entire run with an error naming neither the file nor the chunk.

    0.13% of the corpus took down 100% of the indexing — and with batching it discards every
    batch after the bad one too. Stripping is correct (a NUL in markdown is corruption, not
    content) but must be logged, since quietly rewriting a user's document is its own problem.
    """
    root = _corpus(tmp_path, 3)
    (root / "note1.md").write_text("a memory with a \x00 stray byte", encoding="utf-8")

    with caplog.at_level("WARNING", logger="recall.index"):
        stats = Indexer(store, HashingEmbedder(dim=DIM)).index_path(root)

    assert stats.files == 3, "the whole corpus must still index"
    assert store.count() == 3
    assert any("NUL" in r.getMessage() for r in caplog.records), "stripping must not be silent"
    assert any("note1.md" in r.getMessage() for r in caplog.records), "must name the file"


@requires_db
def test_upserting_a_nul_byte_directly_fails_with_an_actionable_message(store):
    """The direct-API path cannot strip silently, but it can say which chunk is at fault."""
    from recall.types import Chunk

    with pytest.raises(ValueError, match="NUL"):
        store.upsert([Chunk("bad", "s.md", "text with \x00 inside")], [[0.0] * DIM])
