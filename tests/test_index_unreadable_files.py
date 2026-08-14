"""One file this package cannot read must not cost the operator the whole corpus.

`recall index` is the command that WRITES, and it was the one command without the per-file
guard its siblings all have. `cli.py`, `lint.py`, `fix.py` and `rewrite.py` each catch the
undecodable memo and keep going; `index` aborted, discarding every file already read and
leaving a partial index behind whatever batches had committed.

Two different things are unreadable, and they are counted apart because only one of them is
fixed by re-running:

* The NAME is not valid UTF-8. A POSIX filename is bytes, and one that is not valid UTF-8
  arrives as a lone surrogate through `Path.glob`'s surrogateescape. It crashed the md5 chunk
  id, and had that been the only fix it would then have reached `source` and `metadata["file"]`,
  both bound to TEXT columns psycopg encodes as UTF-8: the abort would have moved, not gone.
* The CONTENTS are not valid UTF-8. `UnicodeDecodeError` is a ValueError, not an OSError, so it
  fell straight through a clause catching `(FileNotFoundError, NotADirectoryError)`.

Properties, one test each:

1. A file whose NAME is not UTF-8 does not abort the run, and the other memos are still indexed.
2. It is counted as `unrepresentable` rather than silently dropped.
3. Everything the indexer emits is UTF-8 encodable. This is the property that keeps the abort
   from simply moving to the database, and it is checked on the values, not on the exception.
4. A file whose CONTENTS are not UTF-8 does not abort the run, and is counted as `undecodable`.
5. The two counters do not stand in for each other.
6. A corpus where nothing at all could be read RAISES, rather than reporting "indexed 0" and
   exiting 0. Tolerating a file is right per file and wrong for a whole corpus, and the guard
   for that already existed for files that vanished: without extending it, this change would
   have replaced a loud crash with silence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.index import Indexer

#: A lone surrogate, which is how a filename that is not valid UTF-8 reaches Python.
BAD_NAME = "bad\udcff.md"
#: latin-1 for "café", which `read_text(encoding="utf-8-sig")` cannot decode.
LATIN_BYTES = b"# caf\xe9\n\nNot utf-8 content.\n"


class _NullEmbedder:
    dim = 2
    name = "null"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _RecordingStore:
    """The Indexer's collaborator interface, without a database.

    A real store would fail on these inputs at the psycopg boundary, which is exactly why the
    assertions here are about the VALUES the indexer emits: a stub that accepts anything cannot
    prove the database would, so property 3 checks encodability directly.
    """

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

    def analyze_if_stale(self, modified):
        return True


def _corpus(tmp_path: Path, *, bad_name: bool = False, bad_bytes: bool = False) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "good_2026-01-01.md").write_text(
        "# Good\n\nA plain memo about retrieval.\n", encoding="utf-8", newline="\n"
    )
    if bad_name:
        (root / BAD_NAME).write_bytes(b"# Bad\n\nAnother memo about retrieval.\n")
    if bad_bytes:
        (root / "latin.md").write_bytes(LATIN_BYTES)
    return root


def _index(root: Path):
    store = _RecordingStore()
    return Indexer(store, _NullEmbedder()).index_path(root), store


def test_a_filename_that_is_not_utf8_does_not_abort_the_run(tmp_path):
    """It crashed at the md5 chunk id, before anything was committed."""
    root = _corpus(tmp_path, bad_name=True)
    # Precondition: the name really is unencodable, so this test cannot pass vacuously on a
    # filesystem that normalised it away.
    assert any(_unencodable(p.name) for p in root.iterdir()), "the corpus has no bad name in it"

    stats, store = _index(root)

    assert stats.files == 1, "the awkward name discarded the rest of the corpus"
    assert [c.metadata["file"] for c in store.chunks] == ["good_2026-01-01.md"]


def test_a_filename_that_is_not_utf8_is_counted_not_silently_dropped(tmp_path):
    """A file that is skipped and not reported is a file the operator does not know is missing."""
    stats, _ = _index(_corpus(tmp_path, bad_name=True))

    assert stats.unrepresentable == 1
    assert stats.undecodable == 0, "a name problem was reported as a contents problem"


def test_everything_the_indexer_emits_can_be_stored(tmp_path):
    """The property that stops the abort simply MOVING to the database.

    Fixing only the chunk id left `source` and `metadata["file"]` carrying the surrogate, and
    both are bound to TEXT columns that psycopg encodes as UTF-8. A stub store accepts them
    happily, so asserting "the run completed" would have proved nothing about a real one.
    """
    _, store = _index(_corpus(tmp_path, bad_name=True))

    for chunk in store.chunks:
        assert not _unencodable(chunk.source), f"source {chunk.source!r} cannot reach postgres"
        assert not _unencodable(chunk.metadata["file"]), "metadata file cannot reach postgres"


def test_contents_that_are_not_utf8_do_not_abort_the_run(tmp_path):
    """`UnicodeDecodeError` is a ValueError, so `except (FileNotFoundError, ...)` never saw it."""
    stats, store = _index(_corpus(tmp_path, bad_bytes=True))

    assert stats.files == 1, "one undecodable memo discarded the corpus"
    assert stats.undecodable == 1
    assert [c.metadata["file"] for c in store.chunks] == ["good_2026-01-01.md"]


def test_the_two_unreadable_counters_are_independent(tmp_path):
    """Both at once, because a single counter would let either failure impersonate the other."""
    stats, _ = _index(_corpus(tmp_path, bad_name=True, bad_bytes=True))

    assert (stats.unrepresentable, stats.undecodable) == (1, 1)
    assert stats.files == 1


def test_a_corpus_that_is_entirely_unreadable_raises(tmp_path):
    """Per file, tolerating is right. For a whole corpus it is the silence the guard exists for.

    This is the half that is easy to miss: adding the per-file guards without extending this
    check would have turned a loud crash into `indexed 0 files`, exit 0, which is strictly worse
    than the crash it replaced. The pre-existing check covered files that VANISHED only.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / BAD_NAME).write_bytes(b"# Bad\n\nA memo.\n")
    (root / "latin.md").write_bytes(LATIN_BYTES)

    # `match=` on the two COUNTS, not just the exception type: the pre-existing message said
    # "every one of them vanished between the scan and the read", which would have been a lie
    # about these files. Pinning the numbers keeps the message honest about which failure it saw.
    with pytest.raises(FileNotFoundError, match=r"1 are not valid UTF-8, 1 have a name"):
        _index(root)


@pytest.mark.parametrize(
    "stats_kwargs,expected",
    [
        ({"undecodable": 2}, "2 file(s) whose contents are not UTF-8"),
        ({"unrepresentable": 3}, "3 file(s) whose NAME is not UTF-8"),
    ],
)
def test_the_run_summary_reports_what_it_refused(stats_kwargs, expected):
    """A skip only the log knows about is a memo silently missing from search.

    The summary is the one line an operator reads, and `deleted` is already there for exactly
    this reason. Asserted through `_index_summary`, which is a pure function so this needs no
    database; the alternative was leaving the only operator-visible signal untested.
    """
    from recall.cli import _index_summary
    from recall.index import IndexStats

    assert expected in _index_summary(IndexStats(files=1, chunks=1, **stats_kwargs))


def test_the_run_summary_stays_quiet_when_nothing_was_refused():
    """The counts are conditional, so a clean run reads exactly as it did before."""
    from recall.cli import _index_summary
    from recall.index import IndexStats

    assert _index_summary(IndexStats(files=2, chunks=7)) == "indexed 7 chunks from 2 files"


def _unencodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False
