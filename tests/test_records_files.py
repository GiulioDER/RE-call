"""`recall.records_files`: audit records read from a memo directory (Task 8.1 of the audit memory plan).

Invariants and the failure each one catches:
- F1 the C4 golden memos give the same C5 entries through a memo directory as through an indexed
  store, for exact, symbol_changed and file_only queries, before and after a dated retraction.
  `chunk_id` and `chunk_ids` are left out: a file has no chunk.
- F2 a memo whose record does not validate, or is not JSON, is skipped and listed, never raised
  and never returned.
- F3 a retraction dated in the future does not hide its target yet; inside its window it does.
- F4 the per-file cache is keyed on `(path, mtime_ns, size)`: a changed file is read again, an
  unchanged one is not.
- F5 with `cache_file` the cache outlives the process: a new process parses nothing unchanged.
- F6 a load past its deadline raises `TimeoutError` and keeps what it parsed, so a cold cache
  warms over several calls rather than never.
- F7 (CCA DAT-001) a change to a module that reads memos invalidates the persisted cache.
- F8 (CCA DAT-002) a record holding a lone surrogate is cached and reads back equal.
- F9 (CCA DAT-003) the walk reads what the indexer reads: no hidden or excluded directory, no
  hidden file, no symlink out of the directory; a nested memo is read.
- F10 (CCA PERF-001) a load under a deadline only appends to the cache, never rewrites it whole.
- A control test pins the restated excluded-directory set to the indexer's.

Red proof, recorded 2026-10-22 on the test host inside the test bounds, each mutation of
`recall/records_files.py` failing in the named assertion, then restored and green:
- F1 `tests/test_records_files.py::test_matches_like_lookup_records`: `memo_supersession_graph`
  mutated to ignore `supersedes` (`for target in (None,)`): the `superseded_by` of the superseded
  bug and the retracted false positive differ, so the two-path equality fails.
- F2 `...::test_invalid_memo_is_skipped`: `_read` mutated to skip admission
  (`admitted = dict(meta)`): the invalid and the not-JSON records are returned
  (`assert len(loaded) == len(EXPECTED)`).
- F3 `...::test_future_retraction_does_not_hide_yet`: the graph rows built without the validity
  window (four-element rows): "a future retraction hid the lesson".
- F4 `...::test_cache_rereads_a_changed_file`: the cache keyed on the path alone (the stamp
  mutated to `(0, 0)`): "the cache re-read unchanged files, or missed the changed one". A first
  mutation of only the second stamp check stayed green: the in-memory check above it masks it.
- F5 `...::test_cache_file_serves_a_new_process`: the persisted entries ignored
  (`cached = None` for `persisted.get(key)`): "a new process re-parsed memos the cache file holds".
- F6 `...::test_deadline_keeps_progress`: the cache not saved when the deadline passes (`pass` for
  that save; since the CCA audit fixes it is `_append_cache_delta`, re-proved the same way):
  "the progress made before the deadline was not kept".

Recorded 2026-10-22 for the CCA audit fixes (run 20261022T185308Z), same host and bounds:
- F7 `...::test_cache_tag_follows_the_reader_code`: `_cache_tag` mutated to ignore the reader
  files (`for file in []:`): "a change to the reading code kept serving the old parses".
- F8 `...::test_a_lone_surrogate_is_cached`: `_dumps` mutated to `ensure_ascii=False`: the write
  fails silently, "a record holding a lone surrogate was never cached".
- F9 `...::test_walk_reads_what_the_indexer_reads`: the hidden-name check removed from
  `_memo_files`: `.hidden.md` and `.trash/old.md` are read (`assert names == ...`).
- `...::test_excluded_dirs_match_the_indexer` (control): `venv` removed from `_EXCLUDED_DIRS`.
- F10 `...::test_a_deadline_load_only_appends`: the end-of-load append mutated to `if False:`, so
  the load rewrites the cache: "a load under a deadline rewrote the whole cache".
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import recall.records_files as records_files
from recall.records import lookup_records
from recall.records_files import load_memo_records, match_memo_records
from tests.audit_memory_support import EXPECTED, REPO, copy_memos, indexed_lite_store

ALL_KINDS = ["bug", "false_positive", "false_positive_retraction", "lesson", "lesson_retraction"]
#: Before the lesson retraction's `valid_from` (2026-10-23), after every other golden date.
BEFORE_RETRACTION = datetime(2026, 10, 22, 12, tzinfo=UTC)
AFTER_ALL = datetime(2026, 11, 1, tzinfo=UTC)
LESSON_ID = EXPECTED["RE-call__lesson__6a3ae35062b9.md"]["id"]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    records_files._CACHE.clear()


def _queries(variant: str) -> list[dict[str, Any]]:
    anchors = [dict(a) for record in EXPECTED.values() for a in record.get("anchors", [])]
    for anchor in anchors:
        if variant != "exact":
            anchor["code_sha"] = "0" * 64
        if variant == "file_only":
            anchor["symbol"] = "Nowhere.at_all"
    return anchors


def _without_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in ("chunk_id", "chunk_ids")} for r in records]


@pytest.mark.parametrize("variant", ["exact", "symbol_changed", "file_only"])
@pytest.mark.parametrize("now", [BEFORE_RETRACTION, AFTER_ALL], ids=["before", "after"])
def test_matches_like_lookup_records(tmp_path: Path, variant: str, now: datetime) -> None:
    """F1."""
    memos = copy_memos(tmp_path / "memos")
    store = indexed_lite_store(tmp_path, memos=memos)
    query = _queries(variant)
    through_store = lookup_records(store, repo=REPO, anchors=query, kinds=ALL_KINDS, now=now)["records"]
    through_files = match_memo_records(
        load_memo_records(memos), repo=REPO, anchors=query, kinds=ALL_KINDS, now=now
    )
    assert through_store, "the golden memos matched nothing: the comparison would be vacuous"
    assert {r["anchor_match"] for r in through_files} == {variant}
    assert _without_chunks(through_files) == _without_chunks(through_store)
    assert all(r["chunk_id"] is None and r["chunk_ids"] == [] for r in through_files)


def test_invalid_memo_is_skipped(tmp_path: Path) -> None:
    """F2."""
    memos = copy_memos(tmp_path / "memos")
    golden = (memos / "RE-call__0123456789ab__BUG-001.md").read_text(encoding="utf-8")
    # A required field removed: valid JSON, an invalid record.
    (memos / "invalid.md").write_text(golden.replace('"category":"concurrency",', ""), encoding="utf-8")
    broken_line = next(line for line in golden.splitlines() if line.startswith("recall_record:"))
    (memos / "not-json.md").write_text(golden.replace(broken_line, "recall_record: {not json"), encoding="utf-8")

    loaded = load_memo_records(memos)

    assert len(loaded) == len(EXPECTED)
    assert sorted(Path(source).name for source, _reason in loaded.skipped) == ["invalid.md", "not-json.md"]
    assert all(Path(m.source).name not in {"invalid.md", "not-json.md"} for m in loaded)
    assert {m.record["id"] for m in loaded} == {r["id"] for r in EXPECTED.values()}


def test_future_retraction_does_not_hide_yet(tmp_path: Path) -> None:
    """F3: the lesson's retraction is valid from 2026-10-23."""
    loaded = load_memo_records(copy_memos(tmp_path / "memos"))
    anchor = EXPECTED["RE-call__lesson__6a3ae35062b9.md"]["anchors"][0]

    def lesson_at(now: datetime) -> dict[str, Any]:
        found = match_memo_records(loaded, repo=REPO, anchors=[anchor], kinds=["lesson"], now=now)
        return next(r for r in found if r["id"] == LESSON_ID)

    assert lesson_at(BEFORE_RETRACTION)["superseded_by"] is None, "a future retraction hid the lesson"
    assert lesson_at(AFTER_ALL)["superseded_by"] == "RE-call__lesson__6a3ae35062b9~retract.md"


def test_cache_rereads_a_changed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F4."""
    memos = copy_memos(tmp_path / "memos")
    reads: list[str] = []
    real_read = records_files._read

    def counting_read(path: Path, file: str) -> Any:
        reads.append(path.name)
        return real_read(path, file)

    monkeypatch.setattr(records_files, "_read", counting_read)
    first = {m.record["id"]: m for m in load_memo_records(memos)}
    assert len(reads) == len(EXPECTED)

    target = memos / "RE-call__0123456789ab__BUG-001.md"
    before = target.stat()
    raw = target.read_bytes()
    old = b"so the liveness check raced."
    # Same length, so only the modification time tells the cache the file changed. Bytes, so a
    # Windows text write cannot turn the golden LF endings into CRLF and change the size.
    target.write_bytes(raw.replace(old, b"so the liveness check broke."))
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert target.stat().st_size == before.st_size

    reads.clear()
    second = {m.record["id"]: m for m in load_memo_records(memos)}
    bug_id = EXPECTED[target.name]["id"]
    assert reads == [target.name], "the cache re-read unchanged files, or missed the changed one"
    assert first[bug_id].record["summary"].endswith("raced.")
    assert second[bug_id].record["summary"].endswith("broke."), "a changed file was served from the cache"


def test_cache_file_serves_a_new_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F5: with `cache_file`, a new process (an empty in-memory cache) parses nothing unchanged,
    and returns the same records."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    first = load_memo_records(memos, cache_file=cache)
    assert cache.is_file()

    records_files._CACHE.clear()
    reads: list[str] = []
    real_read = records_files._read
    monkeypatch.setattr(records_files, "_read", lambda path, file: reads.append(path.name) or real_read(path, file))
    second = load_memo_records(memos, cache_file=cache)

    assert reads == [], "a new process re-parsed memos the cache file holds"
    assert list(second) == list(first)


def test_deadline_keeps_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F6: past the deadline the load raises `TimeoutError` and saves what it parsed, so the next
    call parses only the rest."""
    import time

    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "memos.json"
    reads: list[str] = []
    real_read = records_files._read

    def slow_read(path: Path, file: str) -> Any:
        reads.append(path.name)
        time.sleep(0.2)
        return real_read(path, file)

    monkeypatch.setattr(records_files, "_read", slow_read)
    with pytest.raises(TimeoutError):
        load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 0.5)
    first_pass = list(reads)
    assert 1 <= len(first_pass) < len(EXPECTED)

    records_files._CACHE.clear()
    reads.clear()
    loaded = load_memo_records(memos, cache_file=cache)
    assert len(loaded) == len(EXPECTED)
    assert sorted(reads) == sorted(set(p.name for p in memos.glob("*.md")) - set(first_pass)), (
        "the progress made before the deadline was not kept"
    )


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: list[str] = []
    real_read = records_files._read

    def counting_read(path: Path, file: str) -> Any:
        reads.append(path.name)
        return real_read(path, file)

    monkeypatch.setattr(records_files, "_read", counting_read)
    return reads


def test_cache_tag_follows_the_reader_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F7 (DAT-001): changing a module that reads memos re-reads every memo, not only a schema change."""
    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"
    reader = tmp_path / "reader.py"
    reader.write_text("RULE = 1\n", encoding="utf-8")
    monkeypatch.setattr(records_files, "_reader_files", lambda: [str(reader)])
    load_memo_records(memos, cache_file=cache)

    reader.write_text("RULE = 22\n", encoding="utf-8")
    records_files._CACHE.clear()
    reads = _counting_reads(monkeypatch)
    load_memo_records(memos, cache_file=cache)
    assert len(reads) == len(EXPECTED), "a change to the reading code kept serving the old parses"


def test_a_lone_surrogate_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F8 (DAT-002): a record holding a lone surrogate is written to the cache and read back equal."""
    memos = copy_memos(tmp_path / "memos")
    target = memos / "RE-call__0123456789ab__BUG-001.md"
    text = target.read_text(encoding="utf-8")
    assert '"summary":"The pool' in text
    target.write_text(text.replace('"summary":"The pool', '"summary":"\\udc00The pool'), encoding="utf-8")
    cache = tmp_path / "cache" / "memos.json"
    first = {m.record["id"]: m for m in load_memo_records(memos, cache_file=cache)}
    bug_id = EXPECTED[target.name]["id"]
    assert first[bug_id].record["summary"].startswith("\udc00")

    records_files._CACHE.clear()
    reads = _counting_reads(monkeypatch)
    second = {m.record["id"]: m for m in load_memo_records(memos, cache_file=cache)}
    assert reads == [], "a record holding a lone surrogate was never cached"
    assert second[bug_id] == first[bug_id]


def test_walk_reads_what_the_indexer_reads(tmp_path: Path) -> None:
    """F9 (DAT-003): hidden files and directories, excluded directories and a symlink out of the
    directory are not read; a nested memo is."""
    memos = copy_memos(tmp_path / "memos")
    golden = (memos / "RE-call__0123456789ab__BUG-001.md").read_bytes()
    for relative in (".hidden.md", ".trash/old.md", "node_modules/x.md", "sub/.hidden.md"):
        path = memos / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(golden)
    (memos / "nested").mkdir()
    (memos / "nested" / "kept.md").write_bytes(golden.replace(b"BUG-001", b"BUG-777"))
    outside = tmp_path / "outside.md"
    outside.write_bytes(golden)
    try:
        (memos / "link.md").symlink_to(outside)
    except OSError:
        pass  # no symlink privilege on this platform; the other cases still hold

    loaded = load_memo_records(memos)
    names = sorted(Path(m.source).relative_to(memos).as_posix() for m in loaded)
    assert names == sorted([*EXPECTED, "nested/kept.md"]), names


def test_excluded_dirs_match_the_indexer() -> None:
    """Control, not a behaviour test: the restated set equals the indexer's, so the walks agree."""
    import recall.index

    assert records_files._EXCLUDED_DIRS == recall.index._DEFAULT_INDEX_EXCLUDED_DIRS


def test_a_deadline_load_only_appends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F10 (PERF-001): under a deadline the cache is only appended to, never rewritten whole (a
    rewrite grows with the cache and a hook's exit can kill it), and the next call reads it."""
    import time

    memos = copy_memos(tmp_path / "memos")
    cache = tmp_path / "cache" / "memos.json"

    def no_rewrite(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a load under a deadline rewrote the whole cache")

    monkeypatch.setattr(records_files, "_write_cache_file", no_rewrite)
    load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)

    records_files._CACHE.clear()
    reads = _counting_reads(monkeypatch)
    loaded = load_memo_records(memos, cache_file=cache, deadline=time.monotonic() + 60)
    assert reads == [], "what the first call parsed was not kept"
    assert len(loaded) == len(EXPECTED)
