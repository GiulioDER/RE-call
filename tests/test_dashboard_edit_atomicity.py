"""A failed dashboard mutation preserves the memo and a coherent, durable undo history.

Failure injection uses the public edit/undo entry points and the existing storage boundaries.
No test depends on a proposed journal schema or a newly added recovery helper.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import edit

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
ORIGINAL = b"\xef\xbb\xbf# Memo\r\nA person's unchanged prose.\r\n"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "memo.md").write_bytes(ORIGINAL)
    return tmp_path


def _apply(root: Path) -> edit.EditRecord:
    plan = edit.plan_edit(root, "memo.md", valid_until="2027-01-01")
    return edit.apply_edit(
        root, "memo.md", editor="reviewer", note="set expiry", shown_sha=plan.before_sha,
        now=NOW, valid_until="2027-01-01",
    )


def _undo(root: Path, record: edit.EditRecord) -> str:
    return edit.undo_edit(root, record.edit_id, editor="reviewer", note="restore", now=NOW)


def _inject_ledger_failure(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    if failure in {"constructor", "record", "mark_undone"}:
        def unavailable(*args: object, **kwargs: object) -> None:
            raise sqlite3.OperationalError("injected ledger failure")

        method = "__init__" if failure == "constructor" else failure
        monkeypatch.setattr(edit.EditLedger, method, unavailable)
        return

    real_connect = sqlite3.connect

    class FailingCommit(sqlite3.Connection):
        armed = False
        failed = False

        def execute(self, sql: str, *args: object, **kwargs: object) -> sqlite3.Cursor:
            if sql.lstrip().upper().startswith(("INSERT INTO EDITS", "UPDATE EDITS")):
                self.armed = True
            return super().execute(sql, *args, **kwargs)

        def commit(self) -> None:
            if self.armed and not self.failed:
                self.failed = True
                if failure == "after_commit":
                    super().commit()
                raise sqlite3.OperationalError("injected ledger commit failure")
            super().commit()

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        return real_connect(*args, factory=FailingCommit, **kwargs)

    monkeypatch.setattr(edit.sqlite3, "connect", connect)


@pytest.mark.parametrize("failure", ["constructor", "record", "before_commit"])
def test_failed_edit_ledger_preserves_original_memo_and_history(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """A ledger refusal leaves exact original bytes and no successful edit in history."""
    failure_error: Exception | None = None
    with monkeypatch.context() as patch:
        _inject_ledger_failure(patch, failure)
        try:
            _apply(corpus)
        except (edit.EditRefused, sqlite3.Error, OSError) as exc:
            failure_error = exc

    assert (corpus / "memo.md").read_bytes() == ORIGINAL
    assert edit.history(corpus, "memo.md") == []
    assert isinstance(failure_error, edit.EditRefused)
    record = _apply(corpus)
    assert _undo(corpus, record) == "memo.md"
    assert (corpus / "memo.md").read_bytes() == ORIGINAL


@pytest.mark.parametrize(
    "failure", ["constructor", "mark_undone", "record", "before_commit"],
)
def test_failed_undo_ledger_preserves_edit_and_can_be_retried(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """A failed undo preserves edited bytes, the live original record, and retryability.

    Marking the original undone and recording its inverse form one outcome: neither may
    survive alone when the undo refuses.
    """
    record = _apply(corpus)
    edited = (corpus / "memo.md").read_bytes()
    before_history = edit.history(corpus, "memo.md")

    failure_error: Exception | None = None
    with monkeypatch.context() as patch:
        _inject_ledger_failure(patch, failure)
        try:
            _undo(corpus, record)
        except (edit.EditRefused, sqlite3.Error, OSError) as exc:
            failure_error = exc

    assert (corpus / "memo.md").read_bytes() == edited
    assert edit.history(corpus, "memo.md") == before_history
    assert isinstance(failure_error, edit.EditRefused)
    assert _undo(corpus, record) == "memo.md"
    assert (corpus / "memo.md").read_bytes() == ORIGINAL
    rows = edit.history(corpus, "memo.md")
    assert len(rows) == 2
    assert next(row for row in rows if row.edit_id == record.edit_id).undone_at is not None
    with edit.EditLedger(corpus) as ledger:
        inverse = next(row for row in rows if row.edit_id != record.edit_id)
        assert ledger.bytes_of(inverse.edit_id) == ("memo.md", edited, ORIGINAL, None)


@pytest.mark.parametrize("operation", [pytest.param("edit", id="edit-compatibility-control"), "undo"])
def test_uncertain_commit_reopens_as_one_coherent_outcome(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """An acknowledged commit failure leaves a rollback or a fully recorded completion.

    A SQLite commit can persist before an I/O error reaches its caller. Reopening must
    reconcile that result, and an undo must never retain only half its history changes.

    The edit parameter is a compatibility control expected to pass the baseline: its
    single committed row already contains recovery bytes. A mutation omitting that record
    while retaining the file replacement must fail the durable-history assertion.
    """
    record = _apply(corpus) if operation == "undo" else None
    before = (corpus / "memo.md").read_bytes()
    before_history = edit.history(corpus, "memo.md")
    with monkeypatch.context() as patch:
        _inject_ledger_failure(patch, "after_commit")
        try:
            if record is None:
                _apply(corpus)
            else:
                _undo(corpus, record)
        except (edit.EditRefused, sqlite3.Error):
            pass

    rows = edit.history(corpus, "memo.md")
    current = (corpus / "memo.md").read_bytes()
    if current == before:
        assert rows == before_history
    elif record is None:
        assert len(rows) == 1
        with edit.EditLedger(corpus) as ledger:
            assert ledger.bytes_of(rows[0].edit_id) == ("memo.md", ORIGINAL, current, None)
        assert _undo(corpus, rows[0]) == "memo.md"
    else:
        assert current == ORIGINAL
        assert len(rows) == 2
        assert next(row for row in rows if row.edit_id == record.edit_id).undone_at is not None
        inverse = next(row for row in rows if row.edit_id != record.edit_id)
        with edit.EditLedger(corpus) as ledger:
            assert ledger.bytes_of(inverse.edit_id) == ("memo.md", before, ORIGINAL, None)


@pytest.mark.parametrize("operation", ["edit", "undo"])
def test_failed_file_replacement_preserves_memo_and_history(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """A file replacement failure cannot create a successful edit or consume an undo.

    After storage is writable again, the same public operation remains usable.
    """
    record = _apply(corpus) if operation == "undo" else None
    before = (corpus / "memo.md").read_bytes()
    before_history = edit.history(corpus, "memo.md")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("injected replacement failure")

    failure_error: Exception | None = None
    with monkeypatch.context() as patch:
        patch.setattr(edit, "atomic_write_bytes", unavailable)
        try:
            if record is None:
                _apply(corpus)
            else:
                _undo(corpus, record)
        except (edit.EditRefused, sqlite3.Error, OSError) as exc:
            failure_error = exc

    assert (corpus / "memo.md").read_bytes() == before
    assert edit.history(corpus, "memo.md") == before_history
    assert isinstance(failure_error, edit.EditRefused)
    if record is None:
        assert _apply(corpus).name == "memo.md"
    else:
        assert _undo(corpus, record) == "memo.md"


@pytest.mark.parametrize("operation", ["edit", "undo"])
def test_interrupted_replacement_has_durable_recovery_on_reopen(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """An interruption after replacement leaves old bytes or a recoverable recorded change.

    The next public history/undo session must reconcile durable state without the failed
    call's Python objects. A retained undo must mark the original and record its inverse.
    """
    record = _apply(corpus) if operation == "undo" else None
    before = (corpus / "memo.md").read_bytes()
    before_history = edit.history(corpus, "memo.md")
    real_write = edit.atomic_write_bytes

    class Interrupted(BaseException):
        pass

    def interrupted(path: Path, data: bytes) -> None:
        real_write(path, data)
        raise Interrupted("process interrupted after replacement")

    with monkeypatch.context() as patch:
        patch.setattr(edit, "atomic_write_bytes", interrupted)
        with pytest.raises(Interrupted):
            if record is None:
                _apply(corpus)
            else:
                _undo(corpus, record)

    rows = edit.history(corpus, "memo.md")
    current = (corpus / "memo.md").read_bytes()
    if current == before:
        assert rows == before_history
    elif record is None:
        assert len(rows) == 1
        with edit.EditLedger(corpus) as ledger:
            assert ledger.bytes_of(rows[0].edit_id) == ("memo.md", ORIGINAL, current, None)
        assert _undo(corpus, rows[0]) == "memo.md"
        assert (corpus / "memo.md").read_bytes() == ORIGINAL
    else:
        assert current == ORIGINAL
        assert len(rows) == 2
        assert next(row for row in rows if row.edit_id == record.edit_id).undone_at is not None
        inverse = next(row for row in rows if row.edit_id != record.edit_id)
        with edit.EditLedger(corpus) as ledger:
            assert ledger.bytes_of(inverse.edit_id) == ("memo.md", before, ORIGINAL, None)


def test_another_process_cannot_recover_a_live_edit(
    corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery in another process waits until the active ledger closes.

    A committed recovery entry still belongs to its live writer until replacement and
    history publication finish. A child opening public history must not discard it merely
    because the memo still contains the before bytes. Readiness is signalled after imports;
    completion is signalled only after history has opened and closed its ledger.
    """
    ready = corpus / "child-ready"
    completed = corpus / "child-completed"
    real_write = edit.atomic_write_bytes
    child: subprocess.Popen[str] | None = None
    source_root = Path(edit.__file__).resolve().parents[2]
    child_script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from recall.dashboard import edit
root, ready, completed = map(Path, sys.argv[2:])
ready.write_text('ready', encoding='utf-8')
edit.history(root, 'memo.md')
completed.write_text('completed', encoding='utf-8')
"""

    def paused_replacement(path: Path, data: bytes) -> None:
        nonlocal child
        child = subprocess.Popen(
            [sys.executable, "-c", child_script, str(source_root), str(corpus), str(ready), str(completed)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child did not reach the ledger-open boundary"
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        assert not completed.exists(), "another process recovered the active writer's journal"
        assert child.poll() is None, "child failed instead of waiting for the active ledger"
        real_write(path, data)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(edit, "atomic_write_bytes", paused_replacement)
            record = _apply(corpus)
        assert child is not None
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, f"child history failed: {stdout}\n{stderr}"
        assert completed.exists(), "child did not finish after the writer's ledger closed"
        assert [row.edit_id for row in edit.history(corpus, "memo.md")] == [record.edit_id]
        assert _undo(corpus, record) == "memo.md"
        assert (corpus / "memo.md").read_bytes() == ORIGINAL
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)
