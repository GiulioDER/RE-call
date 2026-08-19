"""Tests for the hook processes themselves.

Two properties matter more than any output these produce. They must **fail open**, because a hook
that runs before every session must never be the reason Claude does not start, and they must not
import the `recall` package, because that costs about a second of every session launch.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import recall_hooks


def _configure(tmp_path: Path, monkeypatch: Any, **values: Any) -> Path:
    config = tmp_path / "recall-hook.json"
    config.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(recall_hooks, "config_path", lambda: config)
    return config


def test_session_start_says_nothing_without_a_config(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(recall_hooks, "config_path", lambda: tmp_path / "absent.json")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0
    assert sys.stdout.getvalue() == ""


def test_session_start_says_nothing_when_the_corpus_is_empty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=0)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0
    assert sys.stdout.getvalue() == ""


def test_session_start_emits_the_documented_envelope(tmp_path: Path, monkeypatch: Any) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=1847)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert recall_hooks.session_start({"cwd": "C:/proj/demo"}) == 0

    emitted = json.loads(sys.stdout.getvalue())["hookSpecificOutput"]
    assert emitted["hookEventName"] == "SessionStart"
    assert "1847" in emitted["additionalContext"]
    assert "demo" in emitted["additionalContext"]


def test_session_start_does_not_touch_the_database(tmp_path: Path, monkeypatch: Any) -> None:
    """The count comes from the cache. A DSN pointing at a closed port must cost nothing."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://u:p@127.0.0.1:1/db", chunks=12)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("session_start must not open a connection")

    monkeypatch.setattr(recall_hooks, "refresh_stats", explode)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": "C:/proj/demo"}) == 0


def test_a_corrupt_config_is_survived_rather_than_raised(tmp_path: Path, monkeypatch: Any) -> None:
    config = tmp_path / "recall-hook.json"
    config.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setattr(recall_hooks, "config_path", lambda: config)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.load_config() == {}
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0


def test_refresh_stats_keeps_the_last_known_count_when_the_database_is_down(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A stale number beats zero: reporting no memory is worse than reporting yesterday's."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://u:p@127.0.0.1:1/db", chunks=99)
    assert recall_hooks.refresh_stats() == 99


def test_session_end_is_a_noop_outside_a_project(tmp_path: Path, monkeypatch: Any) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db")
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)
    assert recall_hooks.session_end({"cwd": str(tmp_path)}) == 0


def test_main_survives_stdin_that_is_not_json(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.main(["session-start"]) == 0


def test_main_ignores_an_event_it_does_not_handle(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert recall_hooks.main(["PreToolUse"]) == 0
    assert recall_hooks.main([]) == 0


def test_the_hook_path_does_not_import_the_recall_package() -> None:
    """The reason `recall_hooks` is a separate top-level package.

    `recall/__init__.py` eagerly imports the calibration, evidence and lineage modules, so an
    accidental `from recall.x import y` at module scope here would put roughly a second onto every
    session launch. Nothing in the module's own text would reveal that, so it is asserted.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import recall_hooks, sys; print('recall' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
