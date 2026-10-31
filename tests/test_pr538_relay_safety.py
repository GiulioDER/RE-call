"""PR538 regressions, red against 76a0b36ea on the exact nodes below.

SEC-001: a changed config must not publish an endpoint under the parent's claim.
SEC-002: detached module startup must not execute a same-named cwd package.
BUG-003: console signalling errors must not count as a Windows process exit.
The Windows API is simulated on Linux; no native Windows termination claim is made.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from recall_hooks import relay, write_time


@pytest.mark.parametrize("changed", ["dsn", "tenant", "project_root"])
def test_child_refuses_a_different_configuration(tmp_path, monkeypatch, changed):
    """SEC-001 red: baseline opens a socket for B under A's fingerprint."""
    config = {"dsn": "postgresql://example/a", "tenant": "a", "project_root": "/project-a"}
    state = relay._initial_state("token", relay._fingerprint(config, write_time.settings(config)))
    path = tmp_path / "relay.json"
    relay._atomic_write(path, state)
    current = {**config, changed: config[changed] + "-other"}
    monkeypatch.setattr(relay, "load_config", lambda: current)
    sockets = []

    class Listener:
        def setsockopt(self, *args): pass
        def bind(self, *args): pass
        def listen(self, *args): pass
        def settimeout(self, *args): pass
        def getsockname(self): return ("127.0.0.1", 12345)
        def close(self): pass

    def socket(*args):
        sockets.append(1)
        return Listener()

    monkeypatch.setattr(relay.socket, "socket", socket)
    monkeypatch.setattr(relay, "_publish_startup_state", lambda *args: False)
    assert relay._serve(path, relay._token_digest("token")) == 0
    assert sockets == [], "configuration B was allowed to open an endpoint for A"
    current = config
    assert relay._serve(path, relay._token_digest("token")) == 0
    assert sockets == [1], "the matching configuration must still reach startup"


def test_detached_child_does_not_import_the_project_package(tmp_path, monkeypatch):
    """SEC-002 red: real baseline child writes the poison cwd package's marker."""
    project = tmp_path / "project"
    poison = project / "recall_hooks"
    poison.mkdir(parents=True)
    marker = tmp_path / "shadow-executed"
    (poison / "__init__.py").write_text("", encoding="utf-8")
    (poison / "__main__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('shadow')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONSAFEPATH", None)
    # Stats profiling requires mutmut configuration in cwd, which this child leaves.
    # Keep real mutant selections so the launcher's security regression still runs.
    if environment.get("MUTANT_UNDER_TEST") == "stats":
        environment.pop("MUTANT_UNDER_TEST")
    # The trusted snapshot is importable after cwd. Baseline -m still puts cwd first.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["RECALL_HOOK_CONFIG_HOME"] = str(tmp_path / "empty-config")
    real_popen = subprocess.Popen
    returns = []

    def spawn(command, **kwargs):
        child = real_popen(command, cwd=project, env=environment, **kwargs)
        returns.append(child.wait(timeout=10))
        return child

    monkeypatch.setattr(relay.subprocess, "Popen", spawn)
    monkeypatch.setattr(relay, "START_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(relay.RelayUnavailable, match="did not become ready"):
        relay._spawn(tmp_path / "state.json", "token", "fingerprint")
    assert returns == [0], "the child must reach its installed entry point"
    assert not marker.exists(), "the detached child executed the project-local package"


@pytest.mark.parametrize("changed", ["token", "fingerprint", "both"])
def test_spawner_never_adopts_a_replacement_claim(tmp_path, monkeypatch, changed):
    """SEC-001 red: baseline pings and returns a different startup claim."""
    path = tmp_path / "state.json"
    expected = {"token": "original", "fingerprint": "original", "port": 12345}
    replacement = dict(expected)
    for field in ("token", "fingerprint") if changed == "both" else (changed,):
        replacement[field] = "replacement"
    relay._atomic_write(path, replacement)
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(relay, "START_TIMEOUT_SECONDS", 0.01)
    endpoints = []

    def ping(state, *args, **kwargs):
        endpoints.append(state)
        return {"status": "ok"}

    monkeypatch.setattr(relay, "_endpoint_request", ping)
    try:
        result = relay._spawn(path, "original", "original", initialize=False)
    except relay.RelayUnavailable:
        result = None
    assert result is None, "the spawner adopted a different token or corpus claim"
    assert endpoints == []
    assert relay._read(path) == replacement
    assert not relay._stop_marker_path(path).exists()
    relay._atomic_write(path, expected)
    assert relay._spawn(path, "original", "original", initialize=False) == expected
    assert endpoints == [expected]


def test_windows_cleanup_waits_for_process_status(tmp_path, monkeypatch):
    """BUG-003 red: a console error falsely reports a live process exited."""
    status = {"wait": 258, "handle": 123, "error": 87}
    calls = []

    class Function:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def __call__(self, *args):
            calls.append((self.name, args))
            return self.result()

    kernel = SimpleNamespace(
        OpenProcess=Function("open", lambda: status["handle"]),
        WaitForSingleObject=Function("wait", lambda: status["wait"]),
        CloseHandle=Function("close", lambda: 1),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: status["error"], raising=False)
    monkeypatch.setattr(relay, "sys", SimpleNamespace(platform="win32"))

    def console_signal(*args):
        raise OSError("caller has no console")

    monkeypatch.setattr(relay.os, "kill", console_signal)
    monkeypatch.setattr(relay, "STOP_WAIT_SECONDS", 0.0)
    assert relay._pid_alive(42), "a live process is not a failed console event"
    assert not relay._wait_for_exit(42)
    path = tmp_path / "relay.json"
    state = {"pid": 42, "port": 0, "token": "token"}
    relay._atomic_write(path, state)
    assert relay._stop_state(path, state) is False
    assert relay._stop_marker_path(path).exists()
    assert all(args == (0x00100000, False, 42) for name, args in calls if name == "open")
    assert sum(name == "open" for name, _ in calls) == sum(name == "close" for name, _ in calls)
    status["wait"] = 0
    assert not relay._pid_alive(42)
    assert relay._wait_for_exit(42)
    status["handle"] = 0
    assert not relay._pid_alive(42)
    status["error"] = 5
    assert relay._pid_alive(42), "access denied is not proof of exit"
