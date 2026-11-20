"""Fixtures: the fake Hermes `agent` package (unless HERMES_SRC names a real checkout) and a
provider wired to the scripted server in tests/fake_mcp_server.py."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parent
HERMES_SRC = os.environ.get("HERMES_SRC", "")
sys.path.insert(0, HERMES_SRC or str(HERE / "fakes"))
sys.path.insert(0, str(ROOT))


def wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class Harness:
    """A Hermes home with a provider configured to launch the fake server."""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "hermes-home"
        self.data = self.home / "re-call"
        self.data.mkdir(parents=True)
        self.scenario_path = tmp_path / "scenario.json"
        self.log_path = tmp_path / "server-events.jsonl"
        store = self.data / "memory.db"
        store.write_bytes(b"")  # the provider only checks that the store exists
        quoted = f'"{sys.executable}" "{HERE / "fake_mcp_server.py"}"'
        self.config = {"recall_command": quoted, "dsn": "sqlite:///" + store.as_posix()}
        self.provider = None

    def scenario(self, tools, **extra):
        self.scenario_path.write_text(json.dumps({"tools": tools, **extra}), encoding="utf-8")

    def start(self, *, agent_context="primary", ready=True, **config):
        from re_call_hermes.provider import ReCallMemoryProvider

        (self.data / "config.json").write_text(json.dumps({**self.config, **config}), encoding="utf-8")
        os.environ["FAKE_SCENARIO"] = str(self.scenario_path)
        os.environ["FAKE_LOG"] = str(self.log_path)
        self.provider = ReCallMemoryProvider()
        init = lambda: self.provider.initialize(  # noqa: E731
            "s1", hermes_home=str(self.home), platform="cli", agent_context=agent_context)
        if not ready:
            # initialize() waits for the server, so a test of the starting state runs it aside.
            threading.Thread(target=init, daemon=True).start()
            assert wait_for(lambda: self.provider._settings is not None)
            return self.provider
        init()
        assert wait_for(self.provider._ready.is_set), self.provider._start_error
        return self.provider

    def events(self, kind=None):
        if not self.log_path.exists():
            return []
        rows = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if kind is None or r["event"] == kind]


@pytest.fixture(autouse=True)
def private_shared_store(tmp_path, monkeypatch):
    """The per-user shared store directory, inside the test's own folder: no test reads the real one."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "re-call"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # The scripted server reads these two; they pass through the provider's environment filter only
    # because the test adds them to the passthrough list, never in production.
    from re_call_hermes import config

    monkeypatch.setattr(config, "PASSTHROUGH_ENV", (*config.PASSTHROUGH_ENV, "FAKE_SCENARIO", "FAKE_LOG"))
    h = Harness(tmp_path)
    yield h
    if h.provider is not None:
        h.provider.shutdown()
