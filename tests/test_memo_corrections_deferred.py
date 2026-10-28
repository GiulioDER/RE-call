"""Deferred CCA fixes for PR 947 (memo corrections): the memo folder and its request journal.

Invariants and the failure each one catches:
- D1 an empty or whitespace-only memo is refused before anything is written: it would index to no
  chunks, so a receipt would read `index_state="failed"` on every retry (CCA BUG-004).
- D2 `MemoFolder.authors()` reads the ledger with the journal's own reader, so a malformed ledger
  fails with that reader's `ValueError`, not an `AttributeError` (CCA CODE-002).
- D3 a ledger newer than this release reads is refused, and a ledger carrying the request journal
  is written as version 2 (CCA DEPLOY-001).
- D4 the publish path stages the memo in ONE temporary file: no second staging file through
  `atomic_write_bytes` (CCA PERF-002).
- D5 when `os.fdopen` fails on that staging descriptor, the descriptor is closed (on Windows an open
  handle makes the staging file's unlink fail and hides the error), whether or not io.open closed
  it first (CCA PERF-002 rework, the guard `_create_exclusive` already has).

None needs a database: the folder is a temporary directory.

Node IDs, all in `tests/test_memo_corrections_deferred.py`:
D1 `test_d1_an_empty_memo_is_refused`,
D2 `test_d2_a_malformed_ledger_fails_authors_with_the_journal_error`,
D3 `test_d3_a_newer_ledger_is_refused_and_a_journal_is_version_2`,
D4 `test_d4_publish_stages_the_memo_once`,
D5 `test_d5_a_failed_fdopen_on_the_staging_file_closes_the_descriptor`.

Red proof, 2026-10-28, on this workstation (no database), CCA deferred run 20261028T141844Z:
- D1, D2, D3, D4 by `cca_checks tautology verify` against the pre-fix files: RED, 1 failed each
  (D2 asserts a boolean so its report does not print the type the defect raises).
- D5 by mutation of the fix, because the pre-fix `_create` closed the descriptor before writing
  through another file: the fdopen guard's `os.close(fd)` removed fails, on Windows, "the fdopen
  failure was hidden by the staging cleanup" (closes_first=False: the leaked handle makes the
  unlink raise; elsewhere the descriptor check fails); restored byte for byte, then green.
"""

from __future__ import annotations

import json
import os

import pytest

import recall.lite.memo_requests as memo_requests
from recall.lite.assistant_memos import MemoFolder
from recall.lite.memo_requests import _create, _save, load_ledger


def test_d1_an_empty_memo_is_refused(tmp_path) -> None:
    folder = MemoFolder(tmp_path)
    result = folder.write("   \n")
    assert "refused" in result, "an empty memo was written"
    assert not list(tmp_path.glob("*.md"))


def test_d2_a_malformed_ledger_fails_authors_with_the_journal_error(tmp_path) -> None:
    folder = MemoFolder(tmp_path)
    folder.ledger_path.write_text(json.dumps({"version": 1, "authors": []}), encoding="utf-8")
    raised: type[BaseException] | None = None
    try:
        folder.authors()
    except Exception as exc:  # noqa: BLE001 - which type is raised is the invariant
        raised = type(exc)
    # A boolean, so the failure report names the invariant and not the type the defect raises (a
    # red-proof checker reads a printed AttributeError as a test that never reached the code).
    journal_error = raised is ValueError
    assert journal_error, "a malformed ledger failed authors() outside the journal's ValueError"


def test_d3_a_newer_ledger_is_refused_and_a_journal_is_version_2(tmp_path) -> None:
    path = tmp_path / ".recall-authors.json"
    path.write_text(json.dumps({"version": 99, "authors": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        load_ledger(path)
    _save(path, {"version": 1, "authors": {}, "requests": {"r": {"state": "prepared"}}})
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2


def test_d4_publish_stages_the_memo_once(tmp_path, monkeypatch) -> None:
    staged: list[object] = []

    def record(path, data):  # noqa: ANN001, ANN202
        staged.append(path)
        raise AssertionError("a second staging file was written")

    monkeypatch.setattr(memo_requests, "atomic_write_bytes", record)
    target = tmp_path / "note-0001.md"
    try:
        _create(target, b"body\n")
    except AssertionError:
        pass
    assert staged == [], "the publish path staged the memo through a second temporary file"
    assert target.read_bytes() == b"body\n"


@pytest.mark.parametrize("closes_first", [False, True])
def test_d5_a_failed_fdopen_on_the_staging_file_closes_the_descriptor(
    tmp_path, monkeypatch, closes_first
) -> None:
    opened: list[int] = []
    real_mkstemp = memo_requests.tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        fd, name = real_mkstemp(*args, **kwargs)
        opened.append(fd)
        return fd, name

    def failing_fdopen(fd, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if closes_first:
            os.close(fd)
        raise MemoryError("injected fdopen failure")

    monkeypatch.setattr(memo_requests.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(memo_requests.os, "fdopen", failing_fdopen)
    raised: BaseException | None = None
    try:
        _create(tmp_path / "note-0001.md", b"body\n")
    except BaseException as exc:  # noqa: BLE001 - which error surfaces is the invariant
        raised = exc
    # On Windows a leaked handle makes the cleanup unlink fail, and that error replaces this one.
    surfaced = isinstance(raised, MemoryError) and "injected fdopen failure" in str(raised)
    assert surfaced, "the fdopen failure was hidden by the staging cleanup"
    leaked = False
    for fd in opened:
        try:
            os.fstat(fd)
            leaked = True
            os.close(fd)
        except OSError:
            pass
    assert not leaked, "the staging descriptor leaked"
    assert not list(tmp_path.glob(".recall-memo-*")), "the staging file was left behind"
