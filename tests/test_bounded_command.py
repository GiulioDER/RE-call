"""Regression proofs for the bounded command helper's platform and signal handling."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "bounded.py"


def test_windows_timeout_terminates_the_process_tree(monkeypatch):
    """BUG-001: Windows SIGTERM availability must not select Unix cleanup APIs.

    Red proof: on baseline a58fb833, this exact node fails the taskkill call
    assertion because main.kill_tree selects the absent os.killpg API. The
    production exception is handled by main and exits 2 instead of 124.
    Node: tests/test_bounded_command.py::test_windows_timeout_terminates_the_process_tree.
    """
    spec = importlib.util.spec_from_file_location("bounded_under_test", HELPER)
    assert spec is not None and spec.loader is not None
    bounded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bounded)
    timer_actions = []
    calls = []

    class Timer:
        def __init__(self, seconds, action):
            timer_actions.append(action)

        def start(self):
            pass

        def cancel(self):
            pass

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self):
            timer_actions[0]()
            return 0

    def taskkill(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bounded, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(bounded, "signal", SimpleNamespace(SIGTERM=15))
    monkeypatch.setattr(bounded.threading, "Timer", Timer)
    monkeypatch.setattr(
        bounded,
        "subprocess",
        SimpleNamespace(DEVNULL=subprocess.DEVNULL, Popen=lambda *a, **k: Process(), run=taskkill),
    )
    monkeypatch.setattr(sys, "argv", [str(HELPER), "example-command"])

    with pytest.raises(SystemExit) as exit_info:
        bounded.main()

    assert calls == [["taskkill", "/T", "/F", "/PID", "123"]]
    assert exit_info.value.code == 124


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_a_command_that_ignores_sigterm(tmp_path):
    """BUG-002: a command cannot opt out of the helper's wall-clock deadline.

    Red proof: on baseline a58fb833, this exact node reaches the final assertion
    with missed_deadline=True after independent cleanup, because main.kill_tree
    sends only SIGTERM. The failure is an assertion, not a test timeout.
    Node: tests/test_bounded_command.py::test_timeout_kills_a_command_that_ignores_sigterm.
    """
    pidfile = tmp_path / "child.pid"
    command = (
        "import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    wrapper = subprocess.Popen(
        [sys.executable, str(HELPER), sys.executable, "-c", command, str(pidfile)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    missed_deadline = False
    try:
        try:
            stdout, stderr = wrapper.communicate(timeout=7)
        except subprocess.TimeoutExpired:
            missed_deadline = True
            stdout = stderr = ""
    finally:
        if pidfile.exists():
            child_pid = int(pidfile.read_text())
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if wrapper.poll() is None:
            os.killpg(wrapper.pid, signal.SIGKILL)
        wrapper.communicate(timeout=2)

    assert not missed_deadline, "bounded.py failed to enforce its deadline on a SIGTERM-ignoring child"
    assert wrapper.returncode == 124, stderr
    assert stdout.strip() == "TIMEOUT"
