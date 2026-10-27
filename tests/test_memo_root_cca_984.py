"""FIX-005 / BUG-004: a memo cache must preserve the requested directory's identity."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import recall.records_files as records_files
from recall.records_files import MemoRecord, load_memo_records, match_memo_records
from tests.audit_memory_support import EXPECTED, REPO


NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
ANCHOR = {"file": "pkg/cache.py", "symbol": "Cache.read", "code_sha": "a" * 64}
GOLDEN_BUG = EXPECTED["RE-call__0123456789ab__BUG-001.md"]


@pytest.fixture(autouse=True)
def _isolated_memo_cache() -> Iterator[None]:
    # Existing records-files tests use this reset to model a fresh hook process.
    records_files._CACHE.clear()
    yield
    records_files._CACHE.clear()


def _write_memo(path: Path, number: int, summary: str, *, supersedes: str | None = None) -> str:
    """Write a synthetic memo using the existing admitted golden record's complete schema."""
    record = deepcopy(GOLDEN_BUG)
    record_id = f"RE-call:0123456789ab:BUG-{number:03d}"
    record.update(
        id=record_id,
        bug_id=record_id,
        finding_id=f"BUG-{number:03d}",
        summary=summary,
        anchors=[dict(ANCHOR)],
        caused_by=[],
    )
    header = ["---", f"name: memo-{number}", "description: synthetic cache record", "valid_from: 2026-01-01"]
    if supersedes is not None:
        header.append(f"supersedes: {supersedes}")
    header.append("recall_record: " + json.dumps(record, sort_keys=True, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join([*header, "---", "", "Synthetic memo.", ""])).encode("utf-8"))
    return record_id


def _matches(memos: Sequence[MemoRecord]) -> dict[str, dict[str, Any]]:
    return {
        entry["id"]: entry
        for entry in match_memo_records(
            memos, repo=REPO, anchors=[dict(ANCHOR)], kinds=["bug"], now=NOW
        )
    }


@pytest.mark.parametrize("cache_mode", ["memory", "persisted"])
def test_overlapping_roots_preserve_qualified_supersession(tmp_path: Path, cache_mode: str) -> None:
    """Nodes:
    tests/test_memo_root_cca_984.py::test_overlapping_roots_preserve_qualified_supersession[memory]
    tests/test_memo_root_cca_984.py::test_overlapping_roots_preserve_qualified_supersession[persisted]

    Target: recall.records_files.load_memo_records cache reuse and root-relative MemoRecord.file.
    Baseline e6d9ee13a52292ec7d63feda471235b095fb7240 caches the parent's paths, then reuses
    them for a child-root load. Expected RED: "warm cache retired the wrong record" because
    supersedes: sub/old.md must name the child's nested memo, not its root-level memo.
    Both in-memory reuse and a simulated new process reusing the same cache file are covered.
    The cold child load is an independent public-API control. Observed 2026-10-26 on the isolated testbench: intended assertion RED on the baseline above, then the same case GREEN in the 199-test run (zero failures, errors or skips).
    """
    parent = tmp_path / "memos"
    child = parent / "sub"
    outer_id = _write_memo(child / "old.md", 101, "Outer root record.")
    inner_id = _write_memo(child / "sub" / "old.md", 102, "Inner root record.")
    fix_id = _write_memo(child / "fix.md", 103, "Qualified successor.", supersedes="sub/old.md")
    cache = tmp_path / "cache" / "memos.json" if cache_mode == "persisted" else None

    first = load_memo_records(parent, cache_file=cache)
    assert len(first) == 3 and not first.skipped, first.skipped
    parent_matches = _matches(first)
    assert parent_matches[outer_id]["superseded_by"] == "sub/fix.md"
    assert parent_matches[inner_id]["superseded_by"] is None
    if cache is not None:
        assert cache.is_file(), "the persisted-cache route was not established"
        records_files._CACHE.clear()

    warm = load_memo_records(child, cache_file=cache)
    warm_matches = _matches(warm)
    records_files._CACHE.clear()
    cold = load_memo_records(child)
    cold_matches = _matches(cold)
    assert set(cold_matches) == {outer_id, inner_id, fix_id}
    assert cold_matches[inner_id]["superseded_by"] == "fix.md"
    assert cold_matches[outer_id]["superseded_by"] is None

    assert warm_matches[inner_id]["superseded_by"] == "fix.md", (
        "warm cache retired the wrong record after changing the loading root", warm_matches
    )
    assert warm_matches == cold_matches, "warm and cold child-root listings disagree"
    assert {memo.file for memo in warm} == {memo.file for memo in cold} == {
        "old.md", "sub/old.md", "fix.md"
    }


@pytest.mark.parametrize("cache_mode", ["memory", "persisted"])
def test_relative_root_after_chdir_reads_its_own_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_mode: str
) -> None:
    """Nodes:
    tests/test_memo_root_cca_984.py::test_relative_root_after_chdir_reads_its_own_memo[memory]
    tests/test_memo_root_cca_984.py::test_relative_root_after_chdir_reads_its_own_memo[persisted]

    Target: recall.records_files.load_memo_records cache keys for relative directory arguments.
    Baseline e6d9ee13a52292ec7d63feda471235b095fb7240 aliases two physically different
    memos/record.md files after chdir when their byte size and mtime_ns match. Expected RED:
    "relative root reused another directory's admitted memo", at the public loaded-summary
    assertion. Merely reconstructing parsed.file does not fix this stale record payload.
    The persisted case clears memory while retaining the same absolute cache-file path.
    A cold second-directory load proves its different admitted content. Observed 2026-10-26 on the isolated testbench: intended assertion RED on the baseline above, then the same case GREEN in the 199-test run (zero failures, errors or skips).
    """
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_path, second_path = first_dir / "memos" / "record.md", second_dir / "memos" / "record.md"
    first_id = _write_memo(first_path, 201, "Directory A.")
    second_id = _write_memo(second_path, 202, "Directory B.")
    stamp_ns = 1_790_000_000_000_000_000
    for path in (first_path, second_path):
        os.utime(path, ns=(stamp_ns, stamp_ns))
    first_stat, second_stat = first_path.stat(), second_path.stat()
    assert first_path.read_bytes() != second_path.read_bytes()
    assert (first_stat.st_mtime_ns, first_stat.st_size) == (second_stat.st_mtime_ns, second_stat.st_size)
    cache = tmp_path / "cache" / "memos.json" if cache_mode == "persisted" else None

    monkeypatch.chdir(first_dir)
    first = load_memo_records(Path("memos"), cache_file=cache)
    assert [memo.record["id"] for memo in first] == [first_id]
    assert not first.skipped
    if cache is not None:
        assert cache.is_file(), "the persisted-cache route was not established"
        records_files._CACHE.clear()

    monkeypatch.chdir(second_dir)
    warm = load_memo_records(Path("memos"), cache_file=cache)
    records_files._CACHE.clear()
    cold = load_memo_records(Path("memos"))
    assert [memo.record["id"] for memo in cold] == [second_id]
    assert not cold.skipped
    assert set(_matches(cold)) == {second_id}

    assert [memo.record["summary"] for memo in warm] == ["Directory B."], (
        "relative root reused another directory's admitted memo", warm
    )
    assert _matches(warm) == _matches(cold), "warm and cold second-directory listings disagree"
    assert [memo.record for memo in warm] == [memo.record for memo in cold]
