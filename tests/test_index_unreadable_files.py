"""Unreadable markdown files must not abort an otherwise readable indexing run."""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.index import Indexer

BAD_NAME = "bad\udcff.md"


class _Embedder:
    dim = 2
    name = "test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Store:
    def __init__(self) -> None:
        self.chunks = []

    def source_content_hashes(self) -> dict[str, str]:
        return {}

    def delete_sources(self, sources: list[str]) -> int:
        return len(sources)

    def replace_sources(self, sources, chunks, embeddings) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def analyze_if_stale(self, modified: int) -> bool:
        return True


def _index(root: Path) -> tuple[object, _Store]:
    store = _Store()
    return Indexer(store, _Embedder()).index_path(root), store


def _unencodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def test_one_invalid_utf8_file_does_not_discard_readable_files(tmp_path: Path) -> None:
    """A malformed file is skipped per file, while valid files still reach the store.

    Red proof: the baseline raises `UnicodeDecodeError` while reading the malformed file. The
    production line under test is the markdown read in `Indexer._index_path`.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "good.md").write_text("# Good\n\nreadable memo\n", encoding="utf-8")
    (root / "bad.md").write_bytes(b"# Bad\n\nnot utf-8: \xe9\n")

    stats, store = _index(root)

    assert stats.files == 1
    assert stats.undecodable == 1
    assert [chunk.metadata["file"] for chunk in store.chunks] == ["good.md"]


def test_unrepresentable_name_is_counted_and_skipped(tmp_path: Path) -> None:
    """A filename that cannot be encoded must not reach chunk ids or database text fields.

    Red proof: the baseline raises `UnicodeEncodeError` while deriving the chunk id. The
    production line under test is the filename boundary in `Indexer._index_path`.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "good.md").write_text("# Good\n\nreadable memo\n", encoding="utf-8")
    bad = root / BAD_NAME
    bad.write_bytes(b"# Bad\n\nname cannot be represented\n")
    assert _unencodable(bad.name)

    stats, store = _index(root)

    assert stats.files == 1
    assert stats.unrepresentable == 1
    assert all(not _unencodable(str(chunk.source)) for chunk in store.chunks)


def test_an_entirely_unreadable_corpus_raises(tmp_path: Path) -> None:
    """Skipping every candidate must remain loud instead of reporting a successful empty run."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "bad.md").write_bytes(b"# Bad\n\nnot utf-8: \xe9\n")

    with pytest.raises(FileNotFoundError, match=r"1 are not valid UTF-8"):
        _index(root)


def test_index_summary_names_refused_files() -> None:
    """The operator facing result must identify files skipped before storage.

    Red proof: a mutation that omits either refused count makes the exact assertion fail. The
    production line under test is `_index_summary` in `recall.cli_commands.index_search`.
    """
    from recall.cli_commands.index_search import _index_summary
    from recall.index import IndexStats

    summary = _index_summary(
        IndexStats(files=1, chunks=2, undecodable=1, unrepresentable=1)
    )

    assert "1 file(s) whose contents are not UTF-8" in summary
    assert "1 file(s) whose NAME is not UTF-8" in summary
