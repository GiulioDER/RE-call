"""Sources the indexer could not reach are not the same as sources that are gone.

The prune guard exists because "the corpus disappeared" and "the corpus was deleted" look
identical at the filesystem level. These tests pin the distinction one level down: a file the
scan could not *stat* — an unreadable parent directory, a dropped network mount, a symlink loop
— is neither present nor deleted, and the only safe reading is "leave it alone".

That distinction is easy to lose, because `Path.exists()` answers "does this exist?" and
"could I stat this?" with the same `False`. Code that asks it cannot tell the two apart, and an
`except OSError` wrapped around it never fires. These tests fail against that shape.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from recall.index import Indexer


class _NullEmbedder:
    dim = 2
    name = "null"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _RecordingStore:
    """The Indexer's collaborator interface, without a database."""

    def __init__(self, known: dict[str, str] | None = None):
        self.known = known or {}
        self.chunks = []
        self.deleted = []

    def source_content_hashes(self):
        return dict(self.known)

    def delete_sources(self, sources):
        self.deleted.extend(sources)
        return len(sources)

    def replace_sources(self, sources, chunks, embeddings):
        self.chunks.extend(chunks)
        return len(chunks)


def _unstattable_source(root: Path) -> str:
    """A stored source path under `root` that `os.stat` refuses, portably and deterministically.

    Stands in for the realistic triggers — EACCES on an unreadable parent, EIO/ESTALE on a
    dropped mount, ELOOP on a symlink cycle — none of which can be synthesised the same way on
    every platform. An embedded NUL reaches the identical branch: `os.stat` raises, so the
    question "is this file gone?" has no answer.

    Note this is a stored SOURCE STRING (the row key the store hands back), not a file that has
    to exist. That is exactly the shape the prune path sees: strings out of the database,
    checked against a filesystem that may or may not still be able to answer.
    """
    return str(root / "memo\x00corrupt.md")


def test_a_source_that_cannot_be_stat_ed_is_not_pruned(tmp_path):
    """"I could not look" is not "it is gone". Looking away must never delete.

    Fails against `Path.exists()`, which reports False for a path it merely could not stat —
    and, since Python 3.13, does so from a C accelerator that swallows the error before any
    `except OSError` in Python could see it.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(9):
        (root / f"memo{i}.md").write_text(f"memo {i}", encoding="utf-8")

    walked = sorted(root.glob("*.md"))
    unreachable = _unstattable_source(root)
    # Precondition: the interpreter really cannot answer for this path, and `Path.exists()`
    # really does turn that into "absent". If either stops being true the test is meaningless.
    with pytest.raises((OSError, ValueError)):
        os.stat(unreachable)
    assert Path(unreachable).exists() is False

    known = {str(p): "hash" for p in walked} | {unreachable: "hash"}
    store = _RecordingStore(known=known)
    indexer = Indexer(store, _NullEmbedder())

    deleted = indexer._prune_vanished(root, walked, store.source_content_hashes())

    assert deleted == 0, "a source that could not be stat'd was treated as deleted"
    assert store.deleted == []
    # 1/10 = 10%, well under DEFAULT_MAX_PRUNE_FRACTION — the fraction guard cannot catch this,
    # which is why the per-source question has to be answered correctly in the first place.


def test_a_source_that_is_really_gone_is_still_pruned(tmp_path):
    """The control: fixing the above must not disable pruning itself."""
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(10):
        (root / f"memo{i}.md").write_text(f"memo {i}", encoding="utf-8")

    all_sources = sorted(str(p) for p in root.glob("*.md"))
    gone = Path(all_sources[0])
    gone.unlink()
    walked = [Path(s) for s in all_sources[1:]]

    store = _RecordingStore(known={s: "hash" for s in all_sources})
    indexer = Indexer(store, _NullEmbedder())

    deleted = indexer._prune_vanished(root, walked, store.source_content_hashes())

    assert deleted == 1
    assert store.deleted == [str(gone)]


def test_a_file_that_vanishes_before_it_is_read_does_not_abort_the_run(tmp_path):
    """A pre-walked list is stale by construction; one dead entry must not void the rest.

    The caller may already have debited a byte budget for the whole measured set, and earlier
    batches may already be committed, so aborting mid-run leaves a partial index AND a spent
    quota. Skipping the file keeps both.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    keep = root / "keep.md"
    keep.write_text("still here", encoding="utf-8")
    doomed = root / "doomed.md"
    doomed.write_text("about to vanish", encoding="utf-8")

    walked = sorted([keep, doomed])
    doomed.unlink()  # gone between the walk and the read

    store = _RecordingStore()
    stats = Indexer(store, _NullEmbedder()).index_path(root, files=walked)

    assert stats.files == 1, "the surviving file was not indexed"
    assert [c.source for c in store.chunks] == [str(keep)]


def test_a_file_that_vanishes_at_read_time_is_skipped_not_fatal(tmp_path, monkeypatch):
    """Exercises the READ-LOOP tolerance, on the walked path where nothing can pre-empt it.

    The `files=` variant below cannot test this: `_confined_to` filters on `is_file()`, so a file
    already gone is dropped before the loop ever sees it, and the test would pass with the
    `try/except` deleted entirely. The window this fix exists for is narrower than that — the file
    is present at the walk and gone by the time `read_text` runs — so it has to be simulated at
    the read, not at the walk.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "keep.md").write_text("still here", encoding="utf-8")
    (root / "doomed.md").write_text("about to vanish", encoding="utf-8")

    real_read_text = Path.read_text

    def vanishing(self, *args, **kwargs):
        if self.name == "doomed.md":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanishing)

    store = _RecordingStore()
    stats = Indexer(store, _NullEmbedder()).index_path(root)

    assert stats.files == 1, "one file vanishing at read time took the whole run down"
    assert [c.source for c in store.chunks] == [str(root / "keep.md")]


def test_a_path_that_does_not_exist_raises_rather_than_reporting_zero_files(tmp_path):
    """The operator-facing case: `recall index` with a typo'd path must not exit 0.

    This is what the vanished-count is FOR. If the counter never increments, the total-failure
    check compares 0 against 1, stays quiet, and a mistyped path reports "indexed 0 files"
    successfully — indistinguishable from an empty corpus.
    """
    with pytest.raises(FileNotFoundError):
        Indexer(_RecordingStore(), _NullEmbedder()).index_path(tmp_path / "typo.md")


def test_an_unreadable_file_still_aborts_the_run(tmp_path, monkeypatch):
    """Tolerating *disappearance* must not become tolerating *any* read failure.

    The distinction is the same one `gone_from_disk` draws: ENOENT means the file is gone,
    EACCES/EIO mean it is there and could not be read. Swallowing the second class turns "I could
    not read your corpus" into "indexed 0 files", exit 0 — which is how a permissions or mount
    problem gets mistaken for an empty directory.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(3):
        (root / f"memo{i}.md").write_text(f"memo {i}", encoding="utf-8")

    def deny(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_text", deny)

    with pytest.raises(PermissionError):
        Indexer(_RecordingStore(), _NullEmbedder()).index_path(root)


def test_a_corpus_that_vanished_entirely_raises_instead_of_reporting_success(tmp_path):
    """A total disappearance is not an empty corpus, and must not exit 0.

    Same principle as the prune guard: at the filesystem level "everything is gone" and "there
    was never anything here" look identical, and the safe reading of the former is that something
    is wrong. Covers the walked path.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(4):
        (root / f"memo{i}.md").write_text(f"memo {i}", encoding="utf-8")
    walked = sorted(root.glob("*.md"))
    for f in walked:
        f.unlink()  # the whole corpus goes after the walk — an unmounted volume, a bad sync

    with pytest.raises(FileNotFoundError, match="none of the"):
        Indexer(_RecordingStore(), _NullEmbedder()).index_path(root, files=walked)


def test_an_empty_directory_is_not_treated_as_a_vanished_corpus(tmp_path):
    """The control for the above: nothing to index is a legitimate no-op, not a failure."""
    root = tmp_path / "empty"
    root.mkdir()

    stats = Indexer(_RecordingStore(), _NullEmbedder()).index_path(root)

    assert stats.files == 0 and stats.chunks == 0


def test_a_pre_walked_list_is_re_confined_to_the_root(tmp_path):
    """`files=` is re-filtered, not trusted: a docstring precondition is not a check.

    A single-file root has no `relative_to` to raise on the way past, so without this an
    out-of-root path is read and embedded.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    inside = root / "inside.md"
    inside.write_text("in the corpus", encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("private notes from outside the root", encoding="utf-8")

    store = _RecordingStore()
    stats = Indexer(store, _NullEmbedder()).index_path(root, files=[inside, outside])

    assert stats.files == 1
    assert [c.source for c in store.chunks] == [str(inside)]
    assert all("private notes" not in c.text for c in store.chunks)


def test_a_single_file_root_does_not_smuggle_in_a_foreign_path(tmp_path):
    """The `rel` fallback to a bare basename is the reason the directory case is not enough."""
    inside = tmp_path / "only.md"
    inside.write_text("the one file", encoding="utf-8")
    outside = tmp_path / "elsewhere" / "secret.md"
    outside.parent.mkdir()
    outside.write_text("private notes from outside the root", encoding="utf-8")

    store = _RecordingStore()
    stats = Indexer(store, _NullEmbedder()).index_path(inside, files=[inside, outside])

    assert stats.files == 1
    assert [c.source for c in store.chunks] == [str(inside)]


def test_passing_both_glob_and_files_is_refused(tmp_path):
    """Silently ignoring one of two conflicting descriptions of "what to index" is the bug."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("a", encoding="utf-8")

    with pytest.raises(ValueError, match="not both"):
        Indexer(_RecordingStore(), _NullEmbedder()).index_path(
            root, glob="**/*.py", files=[root / "a.md"]
        )
