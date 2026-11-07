"""`recall.records` and `recall.records_files`: fixes from the second CCA hunt of the records modules
(run 20261107T105648Z, over the pool the first hunt, run 20261106T223009Z, deferred).

Invariants and the failure each one catches:
- A1 (CCA BUG-002) a record whose memo a `supersedes:` reference names among several files is
  reported `<ambiguous>` by the memo reader, never None: before the fix both lessons a retraction
  named ambiguously read as current, while the trust gate refuses the same documents.
- A2 (BUG-002) the store reader (`lookup_records` over a lite store) reports the same.
- A3 (BUG-002) the marker holds at every instant, before the retraction's `valid_from` too, as the
  trust gate's `ambiguous_supersession` does: the ambiguity is not window-aware, and erring towards
  "may be retracted" is the safe side.
- A4 (BUG-002) the auditor brief labels such a record "supersession ambiguous".
- A5 (CCA DAT-003) a record memo with front matter and no body is skipped with a reason, because
  the indexer stores no chunk for it and so no store holds its record.
- A6 (CCA DEPLOY-101) the cache tag follows the reader modules' content, not their modification
  time: the same code rewritten with a new mtime keeps the tag, changed code changes it.
- A8 (architect residual of BUG-002) ambiguity wins over a successor another claim names by
  path, as the trust gate checks `unresolved` first.
- A7 (CCA CODE-005) a memo with no root-relative `file` is keyed by its source, as the store keys
  a chunk with none, not by its basename, so two such memos in different folders stay two.
- A control (CCA CODE-001) pins `RECORD_KINDS` to the schema's kinds.

Red proof, recorded 2026-11-07, each test failing in its named assertion, then green on the fix:
- A1 to A4 and A8: the `unresolved` check in `reported_successor` removed: "an ambiguously
  retracted lesson read as current (memo reader)" / "(store reader)" / "the brief did not flag".
- A3: the same mutation: "an ambiguously retracted lesson read as current before valid_from".
- A8: also red before the reorder (successor checked first): "a named successor hid an
  ambiguous supersession".
- A5: the empty-body check in `_read` removed: "a memo with no body was admitted".
- A6: `_cache_tag` mutated back to `st_size:st_mtime_ns`: "rewriting identical code changed the tag".
- A7: `_file` mutated back to `Path(memo.source).name`: "two memos with one basename shared a key".
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

import recall.records_files as records_files
from recall.records import AMBIGUOUS, RECORD_KINDS, lookup_records, record_schema
from recall.records_files import MemoRecord, load_memo_records, match_memo_records, memo_supersession_graph
from recall_hooks import auditor_brief
from tests.audit_memory_support import EXPECTED, REPO, copy_memos, indexed_lite_store

LESSON = "RE-call__lesson__6a3ae35062b9.md"
RETRACTION = "RE-call__lesson__6a3ae35062b9~retract.md"
LESSON_ID = EXPECTED[LESSON]["id"]
#: After the retraction's `valid_from` (2026-10-23).
AFTER = datetime(2026, 11, 1, tzinfo=UTC)
#: Before it.
BEFORE = datetime(2026, 10, 22, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    records_files._CACHE.clear()


def _ambiguous_memos(root: Path) -> Path:
    """The golden memos with the lesson in two folders, so the retraction's bare stem names both."""
    memos = copy_memos(root)
    (memos / "a").mkdir()
    (memos / "b").mkdir()
    shutil.copyfile(memos / LESSON, memos / "b" / LESSON)
    (memos / LESSON).replace(memos / "a" / LESSON)
    return memos


def _lesson_entries(memos: Path, now: datetime) -> list[dict]:
    anchors = [dict(a) for a in EXPECTED[LESSON]["anchors"]]
    entries = match_memo_records(load_memo_records(memos), repo=REPO, anchors=anchors, kinds=["lesson"], now=now)
    return [e for e in entries if e["id"] == LESSON_ID]


def test_an_ambiguous_retraction_is_reported_by_the_memo_reader(tmp_path: Path) -> None:
    """A1."""
    entries = _lesson_entries(_ambiguous_memos(tmp_path / "memos"), AFTER)
    assert len(entries) == 2
    assert [e["superseded_by"] for e in entries] == [AMBIGUOUS, AMBIGUOUS], (
        "an ambiguously retracted lesson read as current (memo reader)"
    )


def test_an_ambiguous_retraction_is_reported_by_the_store_reader(tmp_path: Path) -> None:
    """A2: the same memos indexed into a lite store, looked up by key."""
    store = indexed_lite_store(tmp_path, memos=_ambiguous_memos(tmp_path / "memos"))
    anchors = [dict(a) for a in EXPECTED[LESSON]["anchors"]]
    found = lookup_records(store, repo=REPO, anchors=anchors, kinds=["lesson"], now=AFTER)["records"]
    lessons = [r for r in found if r["id"] == LESSON_ID]
    assert len(lessons) == 2
    assert [r["superseded_by"] for r in lessons] == [AMBIGUOUS, AMBIGUOUS], (
        "an ambiguously retracted lesson read as current (store reader)"
    )


def test_the_marker_holds_before_the_retraction_is_valid(tmp_path: Path) -> None:
    """A3: parity with the trust gate, whose `ambiguous_supersession` is not rewound by time."""
    entries = _lesson_entries(_ambiguous_memos(tmp_path / "memos"), BEFORE)
    assert [e["superseded_by"] for e in entries] == [AMBIGUOUS, AMBIGUOUS], (
        "an ambiguously retracted lesson read as current before valid_from"
    )


def test_the_brief_flags_an_ambiguous_record(tmp_path: Path) -> None:
    """A4: the hook's own renderer, given the entries it would render."""
    memos = _ambiguous_memos(tmp_path / "memos")
    records = load_memo_records(memos)
    anchors = [dict(a) for a in EXPECTED[LESSON]["anchors"]]
    entries = match_memo_records(records, repo=REPO, anchors=anchors, kinds=list(auditor_brief.KINDS), now=AFTER)
    retractions = {m.file for m in records if m.record.get("kind", "").endswith("_retraction")}
    block = auditor_brief.render(entries, now=AFTER, retraction_files=retractions)
    lesson_lines = [line for line in block.splitlines() if LESSON_ID in line]
    assert lesson_lines, "the lesson was not rendered at all"
    assert all("supersession ambiguous" in line for line in lesson_lines), (
        "the brief did not flag an ambiguously retracted lesson"
    )


def test_a_memo_with_no_body_is_skipped(tmp_path: Path) -> None:
    """A5."""
    memos = copy_memos(tmp_path / "memos")
    target = memos / "RE-call__0123456789ab__BUG-001.md"
    raw = target.read_bytes()
    head, sep, _body = raw.partition(b"\n---\n")
    assert sep, "the golden memo has no front matter end"
    target.write_bytes(head + sep)
    loaded = load_memo_records(memos)
    assert target.name not in {Path(m.source).name for m in loaded}, "a memo with no body was admitted"
    assert any(Path(source).name == target.name and "no body" in why for source, why in loaded.skipped)


def test_the_cache_tag_follows_content_not_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6: reader modules stood in for by copies the test can rewrite."""
    copies = []
    for name in records_files._READER_MODULES:
        module = __import__(name, fromlist=["_"])
        copy = tmp_path / (name + ".py")
        copy.write_bytes(Path(module.__file__).read_bytes())
        copies.append(str(copy))
    monkeypatch.setattr(records_files, "_reader_files", lambda: list(copies))
    before = records_files._cache_tag()
    first = Path(copies[0])
    stat = first.stat()
    first.write_bytes(first.read_bytes())
    os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert records_files._cache_tag() == before, "rewriting identical code changed the tag"
    first.write_bytes(first.read_bytes() + b"\n# changed\n")
    assert records_files._cache_tag() != before, "changed reader code kept the tag"


def test_a_memo_without_a_file_key_is_keyed_by_its_source() -> None:
    """A7: two memos named `n.md` in different folders, both built without `file`."""
    def memo(source: str) -> MemoRecord:
        return MemoRecord(source=source, record={"id": source}, valid_from=None, supersedes=())

    first, second = memo("/x/a/n.md"), memo("/x/b/n.md")
    keys = {records_files._file(first), records_files._file(second)}
    assert len(keys) == 2, "two memos with one basename shared a key"
    assert memo_supersession_graph([first, second]).collided == frozenset()


def test_record_kinds_match_the_schema() -> None:
    """Control, not a behaviour test (CCA CODE-001): the restated kinds equal the schema's."""
    kinds: set[str] = set()
    for branch in record_schema()["oneOf"]:
        kind = branch["properties"]["kind"]
        kinds.update([kind["const"]] if "const" in kind else kind["enum"])
    assert RECORD_KINDS - {"code"} == kinds


def test_ambiguity_wins_over_a_named_successor(tmp_path: Path) -> None:
    """A8 (architect residual of BUG-002): a record that one claim names by path and another
    names ambiguously reads `<ambiguous>`, not the path claimant, as the trust gate checks
    `unresolved` before any successor (`recall/trust_gate.py`)."""
    memos = _ambiguous_memos(tmp_path / "memos")
    raw = (memos / RETRACTION).read_bytes()
    stem = b"supersedes: RE-call__lesson__6a3ae35062b9\n"
    assert stem in raw
    (memos / "r2.md").write_bytes(raw.replace(stem, b"supersedes: a/RE-call__lesson__6a3ae35062b9\n"))
    entries = {Path(e["source"]).parent.name: e for e in _lesson_entries(memos, AFTER)}
    assert entries["a"]["superseded_by"] == AMBIGUOUS, "a named successor hid an ambiguous supersession"
    assert entries["b"]["superseded_by"] == AMBIGUOUS
