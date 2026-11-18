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

Invariant added 2026-11-18 (CCA DEPLOY-003): on POSIX a ``.recall`` directory a writer creates is
owner-only (0700), so the store, its ``-wal``/``-shm``, lanes and ledgers inside it are unreadable by
other local accounts whatever the umask; an existing directory, and a directory not named
``.recall``, keep their mode; on Windows the mode is not asked for, because Python 3.13+ turns 0o700
into an ACL that drops every other account. Red proof, 2026-11-18, on a Linux test host at umask
022, each run against the named mutation of ``make_parent`` in ``recall/state_dir.py`` and restored
green after:

* ``test_every_writer_makes_the_state_directory_owner_only`` (all five cases): the mode argument
  mutated to ``0o777``; fails on the ``== 0o700`` assertion with ``0o755``.
* ``test_an_existing_state_directory_keeps_its_mode``: ``os.chmod(parent, 0o700)`` added after the
  ``mkdir``; fails on the ``== 0o755`` assertion with ``0o700`` (as does the Windows test, since
  the chmod ignores ``os.name``).
* ``test_a_directory_the_user_chose_keeps_the_umask_mode``: the early return for a directory not
  named ``.recall`` changed to ``mkdir(mode=0o700, ...)``; fails on ``== 0o755`` with ``0o700``.
* ``test_windows_is_not_asked_for_a_private_mode``: the mode expression mutated to a bare
  ``0o700``; fails on ``== 0o755`` with ``0o700``.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import pytest

from recall.dashboard.edit import EditLedger
from recall.dashboard.triage import DismissalLedger, default_dismissal_path
from recall.lite.store import LiteStore
from recall.rewrite import RejectionLedger, default_ledger_path
from recall.stale_reports import StaleReportQueue, default_queue_path
from recall import state_dir
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


_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX; Windows uses the folder's ACL")


@pytest.fixture
def umask_022() -> Iterator[None]:
    """The usual umask, which on its own would leave a new directory 0755."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
@pytest.mark.parametrize(
    "open_writer",
    [_stale_reports, _rejections, _edits, _dismissals, _lite_store],
    ids=["stale-reports", "rejections", "edits", "dismissals", "lite-store"],
)
def test_every_writer_makes_the_state_directory_owner_only(tmp_path: Path, open_writer: Callable[[Path], None]) -> None:
    open_writer(tmp_path)
    assert _mode(tmp_path / ".recall") == 0o700


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
def test_an_existing_state_directory_keeps_its_mode(tmp_path: Path) -> None:
    """A directory its owner loosened on purpose (a store shared with a group) is not tightened."""
    (tmp_path / ".recall").mkdir()
    LiteStore(tmp_path / ".recall" / "memory.db", dim=4).close()
    assert _mode(tmp_path / ".recall") == 0o755


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
def test_a_directory_the_user_chose_keeps_the_umask_mode(tmp_path: Path) -> None:
    make_parent(tmp_path / "elsewhere" / "memory.db")
    assert _mode(tmp_path / "elsewhere") == 0o755


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
def test_windows_is_not_asked_for_a_private_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Observed on POSIX: with ``os.name`` reading ``nt`` the mode asked for is the default one."""
    with monkeypatch.context() as patch:
        patch.setattr(state_dir.os, "name", "nt")
        make_parent(tmp_path / ".recall" / "memory.db")
    assert _mode(tmp_path / ".recall") == 0o755


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
def test_a_rescue_root_written_first_makes_the_state_directory_owner_only(tmp_path: Path) -> None:
    """CCA ENV-001 (run 20261118T205956Z): on a PostgreSQL install the atomic rescue's default root
    ``<project>/.recall/atomic`` can be the first thing written under ``.recall``, and every later
    writer keeps the mode it got. Red proof (2026-11-18, Linux test host, umask 022): with
    ``build_lock`` in ``recall/generation_rescue.py`` reverted to ``root.mkdir(parents=True,
    exist_ok=True)`` this failed on ``== 0o700`` with ``0o755``.
    """
    from recall.generation_rescue import build_lock

    with build_lock(tmp_path / ".recall" / "atomic", "generation-1"):
        pass
    assert _mode(tmp_path / ".recall") == 0o700
    assert (tmp_path / ".recall" / ".gitignore").is_file()


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_022")
def test_make_dir_leaves_a_directory_outside_recall_at_the_umask(tmp_path: Path) -> None:
    """A rescue root the user pointed outside ``.recall`` keeps the umask, as ``make_parent`` does.
    Red proof (2026-11-18, Linux test host, umask 022): with ``make_dir``'s final ``mkdir`` mutated
    to make every missing component with ``mode=0o700`` this failed on ``== 0o755`` with
    ``0o700``."""
    state_dir.make_dir(tmp_path / "rescue" / "atomic")
    assert _mode(tmp_path / "rescue") == 0o755


def test_the_docker_context_excludes_recall_state_at_any_depth() -> None:
    lines = set((REPO / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".recall", "**/.recall"} <= lines

