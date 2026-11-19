"""The session-end memory index skips an unchanged folder and runs detached (#1454).

Red proof, one mutation per test, each run against this file with the named production line
changed, each failing in its own assertion (recorded 2026-11-19, Python 3.14, Windows 11):

- `test_an_unchanged_folder_is_not_indexed_again`: `needs_index` returning True unconditionally,
  which is the pre-#1454 behaviour (index at every session end). Fails `calls == 1`.
- `test_every_kind_of_change_is_indexed[add|rename|backdate|delete]`: a high-water mark, i.e.
  `walk` returning only the newest mtime and `needs_index` indexing only when it is newer than the
  recorded one. Fails `calls == 2` for those four; `edit` stays green, which is why the cases are
  separate.
- `test_rows_gone_from_an_unchanged_folder_are_indexed_again`: the `refresh_stats(config) == 0`
  line of `needs_index` replaced by `False`. Fails `calls == 2`.
- `test_a_failed_index_is_retried`: `_run` recording the walk even when the index returned False.
  Fails `calls == 2`.
- `test_another_store_is_indexed_even_when_the_folder_is_unchanged`: `tenant` dropped from
  `folder_key`'s material. Fails `calls == 2`.
- `test_a_walk_holding_a_just_written_memo_is_not_recorded`: `if not racy:` removed in `_run`.
  Fails `calls == 2`.
- `test_the_hook_starts_a_detached_child_and_does_not_index_itself`: `_index_and_refresh` calling
  `memory_refresh.run([str(cwd)])` in place of `start_background_index`. Fails `indexed == []`.
- `test_a_job_that_refuses_breakaway_still_gets_its_child`: the `except OSError` fallback in
  `start_background_index` removed. Fails at `len(launched) == 2` (the OSError propagates out of
  the hook's fail-open catch, so nothing is retried).
- `test_a_child_that_cannot_start_fails_open`: the `except Exception` around the spawn in
  `_index_and_refresh` removed. Fails with the OSError raised out of `session_end`.
- `test_a_second_child_waits_for_the_first`: `_try_lock` returning True unconditionally. Fails
  `pytest.raises(_Waited)`.
- `test_memos_written_during_an_index_are_indexed_by_the_same_child`: `MAX_PASSES = 1`. Fails
  `calls == 2`.
- `test_a_child_skips_a_folder_another_child_indexed_while_it_waited`: the initial `done` forced
  to None in `_run`. Fails `calls == 0`.
- `test_the_child_is_bounded_and_disarms_its_bound`: `watchdog.start()` removed. Fails
  `timers[0].started`.
- `test_the_child_rechecks_the_project_boundary`: `_run` using `Path(args[0]) / "memory"` in
  place of `_project_memory_dir`. Fails `calls == 0`.
- `test_index_memory_is_dispatched_before_stdin_is_read`: the `index-memory` branch of
  `_dispatch` removed. Fails `seen == [[str(tmp_path)]]`.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import recall_hooks
from recall_hooks import memory_refresh

#: Bound at import time, before `tests/conftest.py::_memory_index_runs_inline` replaces it.
REAL_START = memory_refresh.start_background_index
PAST = time.time() - 3600


def _project(tmp_path: Path, monkeypatch: Any, **values: Any) -> Path:
    config = tmp_path / "recall-hook.json"
    values.setdefault("dsn", "postgresql://h/db")
    values.setdefault("embedder", "hashing")
    values.setdefault("project_root", str(tmp_path))
    config.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(recall_hooks, "config_path", lambda: config)
    (tmp_path / "memory").mkdir(exist_ok=True)
    return tmp_path / "memory"


def _memo(path: Path, text: str, mtime: float = PAST) -> Path:
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


class _Index:
    """Stands in for `index_memory_directory`: counts calls, returns `result`."""

    def __init__(self, monkeypatch: Any, result: bool = True) -> None:
        self.calls = 0
        self.result = result
        monkeypatch.setattr("recall.memory_index.index_memory_directory", self)

    def __call__(self, **_kwargs: Any) -> bool:
        self.calls += 1
        return self.result


@pytest.mark.parametrize("event", ["session_end", "pre_compact"])
def test_an_unchanged_folder_is_not_indexed_again(tmp_path: Path, monkeypatch: Any, event: str) -> None:
    """Invariant: a second session end over an unchanged folder does not index; it still refreshes.

    Both Claude Code events and Codex's SessionEnd share `_index_and_refresh`."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    refreshed: list[int] = []
    def refresh(config: Any = None) -> int:
        refreshed.append(1)
        return 5

    monkeypatch.setattr(recall_hooks, "refresh_stats", refresh)
    monkeypatch.setattr("recall_hooks.mcp_cleanup.close_own_mcp_transports", lambda *a, **kw: None)

    hook = getattr(recall_hooks, event)
    assert hook({"cwd": str(tmp_path)}) == 0
    assert hook({"cwd": str(tmp_path)}) == 0

    assert index.calls == 1
    assert len(refreshed) >= 2, "the cached count must still be refreshed when nothing is indexed"


@pytest.mark.parametrize("change", ["edit", "add", "rename", "backdate", "delete"])
def test_every_kind_of_change_is_indexed(tmp_path: Path, monkeypatch: Any, change: str) -> None:
    """Invariant: any change to the memo set indexes again, whatever it does to the newest mtime."""
    memory = _project(tmp_path, monkeypatch)
    old = _memo(memory / "old.md", "older fact\n", PAST - 600)
    new = _memo(memory / "new.md", "newer fact\n", PAST)
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0

    if change == "edit":
        _memo(new, "newer fact, edited\n", PAST + 60)
    elif change == "add":
        _memo(memory / "third.md", "a third fact\n", PAST - 1200)
    elif change == "rename":
        new.rename(memory / "renamed.md")
    elif change == "backdate":
        # A clock that ran fast, then was corrected: the edit lands EARLIER than the recorded one.
        _memo(new, "newer fact\n", PAST - 300)
    else:
        old.unlink()
    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0

    assert index.calls == 2


def test_rows_gone_from_an_unchanged_folder_are_indexed_again(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: an unchanged folder beside a tenant with no rows (a recreated database) indexes."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    counts = iter([5, 0, 0])
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: next(counts))

    recall_hooks.pre_compact({"cwd": str(tmp_path)})
    recall_hooks.pre_compact({"cwd": str(tmp_path)})

    assert index.calls == 2


def test_a_failed_index_is_retried(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: a walk is recorded only after an index that completed."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch, result=False)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)

    recall_hooks.pre_compact({"cwd": str(tmp_path)})
    recall_hooks.pre_compact({"cwd": str(tmp_path)})

    assert index.calls == 2


def test_another_store_is_indexed_even_when_the_folder_is_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Invariant: a walk recorded for one tenant says nothing about another."""
    memory = _project(tmp_path, monkeypatch, tenant="one")
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    recall_hooks.pre_compact({"cwd": str(tmp_path)})

    _project(tmp_path, monkeypatch, tenant="two")
    recall_hooks.pre_compact({"cwd": str(tmp_path)})

    assert index.calls == 2


def test_a_walk_holding_a_just_written_memo_is_not_recorded(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: a memo written within the timestamp window of the walk is indexed again later,
    because a second write in the same tick would leave its size and mtime unchanged."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n", time.time())
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)

    recall_hooks.pre_compact({"cwd": str(tmp_path)})
    recall_hooks.pre_compact({"cwd": str(tmp_path)})

    assert index.calls == 2


class _Popen:
    def __init__(self, refuse_flag: int = 0) -> None:
        self.launched: list[tuple[list[str], dict[str, Any]]] = []
        self.refuse_flag = refuse_flag

    def __call__(self, argv: list[str], **kwargs: Any) -> None:
        self.launched.append((argv, kwargs))
        if self.refuse_flag and kwargs.get("creationflags", 0) & self.refuse_flag:
            raise PermissionError("the job does not allow breakaway")


def test_the_hook_starts_a_detached_child_and_does_not_index_itself(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Invariant: the hook never indexes in its own process; it launches one detached child."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    indexed: list[Any] = []
    monkeypatch.setattr("recall.memory_index.index_memory_directory", lambda **kw: indexed.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    monkeypatch.setattr("recall_hooks.mcp_cleanup.close_own_mcp_transports", lambda *a, **kw: None)
    monkeypatch.setattr(memory_refresh, "start_background_index", REAL_START)
    popen = _Popen()
    monkeypatch.setattr(memory_refresh.subprocess, "Popen", popen)

    assert recall_hooks.session_end({"cwd": str(tmp_path), "reason": "exit"}) == 0

    assert indexed == []
    assert len(popen.launched) == 1
    argv, kwargs = popen.launched[0]
    assert argv == [sys.executable, "-P", "-m", "recall_hooks", "index-memory", str(tmp_path)]
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["close_fds"] is True
    assert kwargs["cwd"] == str(memory_refresh._home())
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.DETACHED_PROCESS
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True


def test_a_job_that_refuses_breakaway_still_gets_its_child(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: when the client's job forbids breakaway, the child is launched without it."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    monkeypatch.setattr(memory_refresh, "start_background_index", REAL_START)
    breakaway = 0x01000000
    monkeypatch.setattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", breakaway, raising=False)
    popen = _Popen(refuse_flag=breakaway)
    monkeypatch.setattr(memory_refresh.subprocess, "Popen", popen)

    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0

    assert len(popen.launched) == 2
    assert not popen.launched[1][1]["creationflags"] & breakaway


def test_a_child_that_cannot_start_fails_open(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: a launch failure never fails the session end."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    monkeypatch.setattr("recall_hooks.mcp_cleanup.close_own_mcp_transports", lambda *a, **kw: None)

    def refuse(cwd: str) -> None:
        raise OSError("no such interpreter")

    monkeypatch.setattr(memory_refresh, "start_background_index", refuse)

    assert recall_hooks.session_end({"cwd": str(tmp_path), "reason": "exit"}) == 0


class _Waited(Exception):
    pass


def test_a_second_child_waits_for_the_first(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: never two indexers at once; a child that finds the lock held waits for it."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    home = memory_refresh._home()
    home.mkdir(parents=True, exist_ok=True)

    def waited(_seconds: float) -> None:
        raise _Waited

    monkeypatch.setattr(memory_refresh.time, "sleep", waited)
    with (home / memory_refresh.LOCK_NAME).open("a+b") as held:
        assert memory_refresh._try_lock(held)
        with pytest.raises(_Waited):
            memory_refresh.run([str(tmp_path)])

    assert index.calls == 0


def test_memos_written_during_an_index_are_indexed_by_the_same_child(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Invariant: the child does not record a walk older than the folder it leaves behind."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    calls: list[int] = []

    def index(**_kwargs: Any) -> bool:
        if not calls:
            _memo(memory / "b.md", "written while indexing\n")
        calls.append(1)
        return True

    monkeypatch.setattr("recall.memory_index.index_memory_directory", index)

    memory_refresh.run([str(tmp_path)])

    assert len(calls) == 2


def test_a_child_skips_a_folder_another_child_indexed_while_it_waited(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Invariant: a child that waited does not repeat an index finished after it was started."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    config = recall_hooks.load_config()
    digest, _count, _racy = memory_refresh.walk(memory.resolve())
    home = memory_refresh._home()
    home.mkdir(parents=True, exist_ok=True)
    (home / memory_refresh.STATE_NAME).write_text(json.dumps({
        "version": 1,
        "folders": {memory_refresh.folder_key(config, memory.resolve()): {
            "digest": digest, "indexed_at": time.time() + 60}},
    }), encoding="utf-8")

    memory_refresh.run([str(tmp_path)])

    assert index.calls == 0


class _Timer:
    def __init__(self, interval: float, function: Any, args: tuple[Any, ...] = ()) -> None:
        self.interval, self.function, self.args = interval, function, args
        self.daemon = False
        self.started = self.cancelled = False
        timers.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


timers: list[_Timer] = []


def test_the_child_is_bounded_and_disarms_its_bound(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: a child is killed at TIME_LIMIT_SECONDS, and a run that returns disarms that."""
    memory = _project(tmp_path, monkeypatch)
    _memo(memory / "a.md", "a fact\n")
    _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)
    timers.clear()
    monkeypatch.setattr(memory_refresh.threading, "Timer", _Timer)

    memory_refresh.run([str(tmp_path)])

    assert len(timers) == 1
    assert timers[0].interval == memory_refresh.TIME_LIMIT_SECONDS
    assert timers[0].function is os._exit and timers[0].daemon
    assert timers[0].started
    assert timers[0].cancelled


def test_the_child_rechecks_the_project_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: the child indexes only a memo folder inside the configured project root."""
    owned = tmp_path / "owned"
    owned.mkdir()
    _project(tmp_path, monkeypatch, project_root=str(owned))
    _memo(tmp_path / "memory" / "a.md", "a fact\n")
    index = _Index(monkeypatch)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 5)

    memory_refresh.run([str(tmp_path)])

    assert index.calls == 0


def test_index_memory_is_dispatched_before_stdin_is_read(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant: `python -m recall_hooks index-memory <cwd>` runs the child, reading no payload."""
    seen: list[list[str]] = []
    def run(args: list[str]) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(memory_refresh, "run", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO("not a payload"))

    assert recall_hooks.main(["index-memory", str(tmp_path)]) == 0

    assert seen == [[str(tmp_path)]]
    assert sys.stdin.read() == "not a payload"
