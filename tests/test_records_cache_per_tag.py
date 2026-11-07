"""`recall.records_files`: one cache file per reading-code version (CCA BUG-101, run 20261107T105648Z).

Invariants and the failure each one catches:
- T1 two versions of the reading code (two cache tags) sharing one memo directory and one
  configured cache path keep each other warm. Before the fix they shared one file, and a load
  under a deadline compacted the other version's rows away, so alternating launches parsed the
  whole directory cold every time (measured with a probe: tags A,A,B,A,B,A parsed 9,0,9,9,9,9).
- T4 (review of BUG-101) a version that only reads (its memos unchanged, so it writes nothing)
  marks its file as used, so another version's cleanup does not remove it a week on.
- T3 (owner request) `recall records` labels `<ambiguous>` "supersession ambiguous", as the brief does.
- T2 a sibling cache file no load has written for a week is removed when a load saves its own, the
  pre-change untagged file included; one written recently is kept, since another version may
  still be reading it. Without the cleanup every upgrade would leave its cache on disk for good.

Red proof, recorded 2026-11-07 on the workstation against the pre-fix `recall/records_files.py`
(master ffa918217), each failing in its named assertion, then green on the fix:
- T1 `...::test_two_versions_keep_each_other_warm`: "a second code version evicted the first".
- T2 `...::test_a_cache_unused_for_a_week_is_removed`: "a week-old cache file was left on disk".
- T4 `...::test_a_version_that_only_reads_keeps_its_cache`, before the mark in
  `_remove_stale_versions`: "a version that only read its cache had it removed by another version".
- T3 `...::test_the_cli_labels_an_ambiguous_supersession`: on master's `records_cmd._print`: "the
  CLI did not label an ambiguous supersession".
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

import recall.records_files as records_files
from recall.records_files import load_memo_records
from tests.audit_memory_support import EXPECTED, copy_memos

WEEK = 7 * 24 * 3600


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    records_files._CACHE.clear()


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: list[str] = []
    real_read = records_files._read

    def counting_read(path: Path, file: str) -> Any:
        reads.append(path.name)
        return real_read(path, file)

    monkeypatch.setattr(records_files, "_read", counting_read)
    return reads


def _load_as(tag: str, memos: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_files, "_cache_tag", lambda: tag)
    records_files._CACHE.clear()  # a new hook process
    load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)


def test_two_versions_keep_each_other_warm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T1: version A, then B, then A again: the third load parses nothing."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    _load_as("4:aaaaaaaaaaaaaaaa", memos, cache, monkeypatch)
    _load_as("4:bbbbbbbbbbbbbbbb", memos, cache, monkeypatch)
    reads = _counting_reads(monkeypatch)
    _load_as("4:aaaaaaaaaaaaaaaa", memos, cache, monkeypatch)
    assert reads == [], "a second code version evicted the first"
    _load_as("4:bbbbbbbbbbbbbbbb", memos, cache, monkeypatch)
    assert reads == [], "the first code version evicted the second"


def test_a_cache_unused_for_a_week_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: the untagged file an older RE-call wrote, and another version's file, both a week
    stale, are removed; a third version's file written an hour ago is kept."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    cache.parent.mkdir()
    stale = [cache, cache.with_name(cache.name + ".delta"), cache.with_name("memos.3-0000000000000000.json")]
    fresh = cache.with_name("memos.4-1111111111111111.json")
    unrelated = cache.with_name("other.json")
    old, recent = time.time() - WEEK - 3600, time.time() - 3600
    for path, when in [*((p, old) for p in stale), (fresh, recent), (unrelated, old)]:
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (when, when))
    _load_as("4:aaaaaaaaaaaaaaaa", memos, cache, monkeypatch)
    left = [p.name for p in stale if p.exists()]
    assert left == [], f"a week-old cache file was left on disk: {left}"
    assert fresh.exists(), "a cache another version wrote an hour ago was removed"
    assert unrelated.exists(), "a file of another memo directory was removed"
    records_files._CACHE.clear()
    reads = _counting_reads(monkeypatch)
    loaded = load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    assert reads == [] and len(loaded) == len(EXPECTED), "the version's own cache was not kept"


def test_the_cli_labels_an_ambiguous_supersession(capsys: pytest.CaptureFixture[str]) -> None:
    """T3 (owner request 2026-11-07): `recall records` says "supersession ambiguous" for
    `superseded_by: "<ambiguous>"`, as the auditor brief does, not "superseded by <ambiguous>"."""
    import argparse

    from recall.cli_commands import records_cmd
    from recall.records import AMBIGUOUS

    record = {"superseded_by": AMBIGUOUS, "valid_until": None, "caused": [], "anchor_match": "exact",
              "kind": "lesson", "id": "RE-call:lesson:1", "source": "/memos/a/l.md"}
    records_cmd._print({"records": [record], "generation_id": "g"}, argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "supersession ambiguous" in out, "the CLI did not label an ambiguous supersession"
    assert "superseded by" not in out


def test_a_version_that_only_reads_keeps_its_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T4 (review of BUG-101): a version whose memos did not change writes nothing, so a completed
    load marks its file as in use; another version's cleanup then leaves it, a week on."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    _load_as("4:aaaaaaaaaaaaaaaa", memos, cache, monkeypatch)
    mine = cache.parent / "memos.4-aaaaaaaaaaaaaaaa.json.delta"
    assert mine.exists()
    old = time.time() - WEEK - 3600
    os.utime(mine, (old, old))
    _load_as("4:aaaaaaaaaaaaaaaa", memos, cache, monkeypatch)  # reads only: nothing to write
    _load_as("4:bbbbbbbbbbbbbbbb", memos, cache, monkeypatch)
    assert mine.exists(), "a version that only read its cache had it removed by another version"
