"""A folder's ``.recall`` state stays on the machine that wrote it.

Invariant: every RE-call writer that creates ``<root>/.recall/`` leaves a ``.gitignore`` of ``*``
in it, so a ``git add .`` in a user's memo repository cannot stage memo text, vectors or review
ledgers; and the desktop image's build context excludes ``.recall`` at any depth.

Red proof, 2026-10-27, each run against the named mutation and restored green after:

* ``test_a_state_directory_gets_an_ignore_file_that_git_honours`` and
  ``test_every_writer_leaves_the_ignore_file`` (all five cases): ``make_parent`` in
  ``recall/state_dir.py`` mutated to return right after ``mkdir``; they fail on the ``is_file()``
  assertion.
* ``test_every_writer_leaves_the_ignore_file[stale-reports]``: ``StaleReportQueue.__init__``
  reverted to ``self._path.parent.mkdir(...)``; fails on ``is_file()``, the other four stay green.
* ``test_an_existing_ignore_file_is_never_rewritten``: the ``"x"`` open mode mutated to ``"w"``;
  fails on the equality with the user's text.
* ``test_the_docker_context_excludes_recall_state_at_any_depth``: the ``.recall`` and
  ``**/.recall`` lines removed from ``.dockerignore``; fails on the subset assertion.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from recall.dashboard.edit import EditLedger
from recall.dashboard.triage import DismissalLedger, default_dismissal_path
from recall.lite.store import LiteStore
from recall.rewrite import RejectionLedger, default_ledger_path
from recall.stale_reports import StaleReportQueue, default_queue_path
from recall.state_dir import IGNORE_TEXT, make_parent

REPO = Path(__file__).resolve().parents[1]


def test_a_state_directory_gets_an_ignore_file_that_git_honours(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    store = tmp_path / ".recall" / "memory.db"
    make_parent(store)
    assert (tmp_path / ".recall" / ".gitignore").is_file()
    store.write_bytes(b"SQLite format 3\x00")
    (tmp_path / ".recall" / "memory.db-wal").write_bytes(b"\x00")
    (tmp_path / "note.md").write_text("a memo\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert status.splitlines() == ["?? note.md"]


def test_an_existing_ignore_file_is_never_rewritten(tmp_path: Path) -> None:
    (tmp_path / ".recall").mkdir()
    mine = "# tracked on purpose\n"
    (tmp_path / ".recall" / ".gitignore").write_text(mine, encoding="utf-8")
    make_parent(tmp_path / ".recall" / "edits.sqlite3")
    assert (tmp_path / ".recall" / ".gitignore").read_text(encoding="utf-8") == mine


def test_a_directory_not_named_recall_is_left_alone(tmp_path: Path) -> None:
    """A control, not a behaviour guard: a lite store at a path the user chose gets no file."""
    make_parent(tmp_path / "elsewhere" / "memory.db")
    assert (tmp_path / "elsewhere").is_dir()
    assert not (tmp_path / "elsewhere" / ".gitignore").exists()


def _stale_reports(root: Path) -> None:
    StaleReportQueue(default_queue_path(root)).close()


def _rejections(root: Path) -> None:
    RejectionLedger(default_ledger_path(root)).close()


def _edits(root: Path) -> None:
    with EditLedger(root):
        pass


def _dismissals(root: Path) -> None:
    connection = DismissalLedger(default_dismissal_path(root))._connect(write=True)
    assert connection is not None
    with closing(connection):
        pass


def _lite_store(root: Path) -> None:
    LiteStore(root / ".recall" / "memory.db", dim=4).close()


@pytest.mark.parametrize(
    "open_writer",
    [_stale_reports, _rejections, _edits, _dismissals, _lite_store],
    ids=["stale-reports", "rejections", "edits", "dismissals", "lite-store"],
)
def test_every_writer_leaves_the_ignore_file(tmp_path: Path, open_writer: Callable[[Path], None]) -> None:
    open_writer(tmp_path)
    ignore = tmp_path / ".recall" / ".gitignore"
    assert ignore.is_file()
    assert ignore.read_text(encoding="utf-8") == IGNORE_TEXT


def test_the_docker_context_excludes_recall_state_at_any_depth() -> None:
    lines = set((REPO / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".recall", "**/.recall"} <= lines

