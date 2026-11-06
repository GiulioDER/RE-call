"""`recall.records_files`: fixes from the CCA DEEP hunt of the records modules (run 20261106T223009Z).

Invariants and the failure each one catches:
- H1 (CCA STAKES-001) a file reached through a Windows directory junction is kept only when it
  resolves inside the memo directory, as the indexer's `_confined_with_stats` keeps it; before the
  fix only a file symlink was confined, so a junction to another directory put its memos in the
  auditor brief while no store held them. Two tests: one through the `_is_junction` seam on every
  platform, one with a real junction on Windows (no privilege needed for `mklink /J`).
- H2 (CCA SEC-002) a file symlink whose TARGET the indexer's `_safe_link_target` refuses (here one
  under `node_modules`, and a dot-named one) is not read; before the fix only containment was
  checked. Needs symlink privilege, so it skips on a Windows host without it; it runs on POSIX.
- A control test pins the restated link-target names and suffixes to the indexer's.
- H3 (CCA STAKES-003) a memo whose read raised an `OSError` is not cached as a refusal: the next
  load reads it again. Before the fix a passing lock skipped the memo until the file changed.
- H4 (CCA PERF-001) a load under a deadline that reads the whole directory over a cache holding
  another tag's rows rewrites the cache, so a tag change does not leave one more dead copy of the
  directory in the journal for every later launch to read.

- H5 (architect REVISE of PERF-001) a rewrite under a deadline that is killed before its replace
  keeps this load's parses: they are appended first.

Red proof, recorded 2026-11-07, each test failing in its named assertion, then green on the fix:
- H1 `...::test_a_file_under_a_junction_is_confined` and `...::test_a_real_junction_out_of_the_directory_is_not_read`:
  pre-fix `recall/records_files.py` (`cca_checks tautology verify`, Windows): "a file under a
  junction leaving the root was admitted" / "a memo outside the directory was read through a junction".
- H2 `...::test_a_link_to_a_target_the_indexer_refuses_is_not_read`, on the Linux test host (no
  symlink privilege on the Windows workstation): `_memo_files` mutated to
  `if target is None:` (the `_safe_target` check removed): "a link into node_modules was read".
- H3 `...::test_a_read_error_is_retried_by_the_next_load`: pre-fix (tautology verify): "a passing
  read error was cached as a refusal".
- H4 `...::test_a_tag_change_does_not_grow_the_journal`: pre-fix (tautology verify): "a tag change
  left the old tag's copy of the directory in the journal".
- H5 `...::test_a_killed_rewrite_keeps_the_parses`: `load_memo_records` mutated to append only when
  the cache holds no dead rows (the first version of this fix): "a killed rewrite lost every memo
  the load had parsed".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

import recall.records_files as records_files
from recall.records_files import load_memo_records
from tests.audit_memory_support import EXPECTED, copy_memos

BUG = "RE-call__0123456789ab__BUG-001.md"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    records_files._CACHE.clear()


def _golden(memos: Path, new_id: str) -> bytes:
    return (memos / BUG).read_bytes().replace(b"BUG-001", new_id.encode())


def test_a_file_under_a_junction_is_confined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H1, every platform: a file under a directory the walk takes for a junction goes through
    `within`, and is dropped when `within` says it leaves the root."""
    memos = copy_memos(tmp_path / "memos")
    (memos / "j").mkdir()
    (memos / "j" / "x.md").write_bytes(_golden(memos, "BUG-901"))
    real_within = records_files.within
    monkeypatch.setattr(records_files, "_is_junction", lambda entry: entry.name == "j", raising=False)
    monkeypatch.setattr(
        records_files,
        "within",
        lambda root, path: None if "j" in Path(path).parts else real_within(root, path),
    )
    names = {Path(m.source).relative_to(memos).as_posix() for m in load_memo_records(memos)}
    assert "j/x.md" not in names, "a file under a junction leaving the root was admitted"
    assert names == set(EXPECTED)


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions exist on Windows only")
def test_a_real_junction_out_of_the_directory_is_not_read(tmp_path: Path) -> None:
    """H1 on Windows: `memos/j` is a junction to a sibling directory holding a valid memo."""
    import _winapi

    memos = copy_memos(tmp_path / "memos")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.md").write_bytes(_golden(memos, "BUG-902"))
    _winapi.CreateJunction(str(outside), str(memos / "j"))
    names = {Path(m.source).relative_to(memos).as_posix() for m in load_memo_records(memos)}
    assert "j/x.md" not in names, "a memo outside the directory was read through a junction"
    assert names == set(EXPECTED)


def test_a_link_to_a_target_the_indexer_refuses_is_not_read(tmp_path: Path) -> None:
    """H2: in-root links to `node_modules/b.md` and to `sub/.draft.md` are refused, as
    `recall.index._safe_link_target` refuses them; a link to a plain in-root memo is kept."""
    memos = copy_memos(tmp_path / "memos")
    (memos / "node_modules").mkdir()
    (memos / "node_modules" / "b.md").write_bytes(_golden(memos, "BUG-903"))
    (memos / "sub").mkdir()
    (memos / "sub" / ".draft.md").write_bytes(_golden(memos, "BUG-904"))
    (memos / "sub" / "plain.md").write_bytes(_golden(memos, "BUG-905"))
    try:
        (memos / "tool.md").symlink_to(memos / "node_modules" / "b.md")
        (memos / "draft.md").symlink_to(memos / "sub" / ".draft.md")
        (memos / "kept.md").symlink_to(memos / "sub" / "plain.md")
    except OSError:
        pytest.skip("no symlink privilege on this host")
    names = {Path(m.source).relative_to(memos).as_posix() for m in load_memo_records(memos)}
    assert "tool.md" not in names, "a link into node_modules was read"
    assert "draft.md" not in names, "a link to a dot-named memo was read"
    assert names == {*EXPECTED, "kept.md", "sub/plain.md"}


def test_link_target_names_match_the_indexer() -> None:
    """Control, not a behaviour test: the restated sets equal the indexer's."""
    import recall.index

    assert records_files._EXCLUDED_NAMES == recall.index._DEFAULT_INDEX_EXCLUDED_NAMES
    assert records_files._EXCLUDED_SUFFIXES == recall.index._DEFAULT_INDEX_EXCLUDED_SUFFIXES


def test_a_read_error_is_retried_by_the_next_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H3: a sharing violation on one memo skips it once; a new process reads it again."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    real_read_text = Path.read_text
    failed: list[str] = []

    def locked_once(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == BUG and not failed:
            failed.append(self.name)
            raise PermissionError(32, "The process cannot access the file")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", locked_once)
    first = load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    assert failed == [BUG]
    assert any(Path(source).name == BUG for source, _why in first.skipped)

    records_files._CACHE.clear()  # a new process
    second = load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    assert BUG in {Path(m.source).name for m in second}, "a passing read error was cached as a refusal"
    assert second.skipped == []


def test_a_tag_change_does_not_grow_the_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H4: three upgrades (three cache tags), each followed by a hook load under a deadline,
    leave the cache holding one copy of the directory, not three."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    delta = cache.with_name(cache.name + ".delta")
    for tag in ("3:aaaaaaaaaaaaaaaa", "3:bbbbbbbbbbbbbbbb", "3:cccccccccccccccc"):
        monkeypatch.setattr(records_files, "_cache_tag", lambda tag=tag: tag)
        records_files._CACHE.clear()  # a new process
        load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    rows = [line for line in delta.read_text(encoding="utf-8").splitlines() if line] if delta.exists() else []
    persisted = cache.read_text(encoding="utf-8") if cache.exists() else ""
    assert not any("aaaaaaaa" in row or "bbbbbbbb" in row for row in rows), (
        "a tag change left the old tag's copy of the directory in the journal"
    )
    assert "aaaaaaaa" not in persisted and "bbbbbbbb" not in persisted
    entries, dead = records_files._read_cache_file(cache, "3:cccccccccccccccc")
    assert len(entries) == len(EXPECTED) and dead == 0


def test_a_killed_rewrite_keeps_the_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H5 (architect REVISE of PERF-001): a rewrite under a deadline that is killed before its
    replace (the hook's process exits) still leaves this load's parses in the journal, so the next
    launch parses nothing again instead of repeating the same work forever."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    monkeypatch.setattr(records_files, "_cache_tag", lambda: "3:aaaaaaaaaaaaaaaa")
    load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    monkeypatch.setattr(records_files, "_cache_tag", lambda: "3:bbbbbbbbbbbbbbbb")  # an upgrade
    records_files._CACHE.clear()

    def killed(*args: object, **kwargs: object) -> None:
        raise SystemExit("the hook's process exited during the rewrite")

    real_write = records_files._write_cache_file
    monkeypatch.setattr(records_files, "_write_cache_file", killed)
    try:
        load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    except SystemExit:
        pass
    monkeypatch.setattr(records_files, "_write_cache_file", real_write)
    records_files._CACHE.clear()  # a new process
    reads: list[str] = []
    real_read = records_files._read
    monkeypatch.setattr(records_files, "_read", lambda path, file: reads.append(path.name) or real_read(path, file))
    load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    assert reads == [], "a killed rewrite lost every memo the load had parsed"
