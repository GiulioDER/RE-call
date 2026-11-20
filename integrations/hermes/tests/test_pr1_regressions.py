"""Regression boundaries for the PR review; red against PR head fb0b460."""

from __future__ import annotations

import json
import sys
import threading

import pytest

from re_call_hermes import provider as module
from re_call_hermes.mcp_client import McpError


def _start(home, monkeypatch, call, *, ready=True, **config):
    data = home / "re-call"
    data.mkdir(parents=True, exist_ok=True)
    (data / "memory.db").touch()
    (data / "config.json").write_text(json.dumps({"recall_command": sys.executable, **config}), encoding="utf-8")
    monkeypatch.setattr(module.ReCallMemoryProvider, "_call", call)
    monkeypatch.setattr(module.ReCallMemoryProvider, "_launch", lambda self, name: self._ready.set() if ready else None)
    provider = module.ReCallMemoryProvider()
    provider.initialize("test", hermes_home=home)
    return provider


@pytest.mark.parametrize("eviction", ["age", "count"])
def test_unknown_outcome_is_not_replayed_after_request_eviction(tmp_path, monkeypatch, eviction):
    """BUG-001: absent server request records, whether age or count evicted, never permit replay.

    The transport models a committed write whose answer was lost, followed by forgetting and
    request eviction. Red at fb0b460: startup submits the forgotten text again (calls is nonempty).
    The assertion is at the provider boundary; no embeddings or live store are needed.
    """
    requests = {}

    def lost(self, tool, arguments, timeout):
        requests[arguments["request_id"]] = "cancelled"
        raise McpError("answer lost after commit")

    provider = _start(tmp_path, monkeypatch, lost)
    provider.on_memory_write("add", "user", "Forgotten private fact")
    provider.shutdown()
    assert requests
    # Both supported pruning mechanisms have the same observable result: no request record.
    requests.clear()
    calls = []

    def after_eviction(self, tool, arguments, timeout):
        calls.append((tool, arguments))
        assert arguments.get("request_id") not in requests
        return {"memo_id": "note-0002"}

    provider = _start(tmp_path, monkeypatch, after_eviction)
    provider.shutdown()
    assert calls == [], f"unsafe replay after {eviction} eviction"
    state = json.loads((tmp_path / "re-call/mirror.json").read_text(encoding="utf-8"))
    assert state["pending"], "the unknown outcome must remain available for reconciliation"


@pytest.mark.parametrize("change", ["store", "folder", "neither"])
def test_pending_removal_stays_bound_to_its_destination(tmp_path, monkeypatch, change):
    """BUG-002: a pending deletion cannot cross a store/folder switch, even if IDs collide.

    Red at fb0b460: both changed destinations receive recall_forget_memo. The unchanged
    destination control must still process a known, not-yet-submitted operation.
    """
    calls = []

    def call(self, tool, arguments, timeout):
        calls.append((tool, arguments))
        return {"memo_id": "note-0001"} if tool == "recall_remember" else {"deleted": True}

    provider = _start(tmp_path, monkeypatch, call)
    provider.on_memory_write("add", "user", "Original fact")
    provider.shutdown()
    path = tmp_path / "re-call/mirror.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["pending"] = [["remove-original", "remove", "user", "", "Original fact"]]
    path.write_text(json.dumps(state), encoding="utf-8")
    calls.clear()
    config = {}
    if change == "store":
        store = tmp_path / "other.db"
        store.touch()
        config["dsn"] = "sqlite:///" + store.as_posix()
    elif change == "folder":
        config["memo_folder"] = str(tmp_path / "other-memos")
    provider = _start(tmp_path, monkeypatch, call, **config)
    provider.shutdown()
    expected = [("recall_forget_memo", {"memo_id": "note-0001"})] if change == "neither" else []
    assert calls == expected
    if change != "neither":
        assert json.loads(path.read_text(encoding="utf-8")) == state


def test_shutdown_interrupts_an_idle_writer_waiting_for_startup(tmp_path, monkeypatch):
    """BUG-003: shutdown wakes an idle writer even when startup never becomes ready.

    Red at fb0b460: the shutdown thread remains in its ten-second join. Cleanup releases
    readiness before asserting, so the red run does not leak a blocked thread.
    """
    provider = _start(tmp_path, monkeypatch, lambda *args: {}, ready=False)
    writer = provider._writer
    stopped = threading.Thread(target=provider.shutdown)
    stopped.start()
    stopped.join(2)
    finished = not stopped.is_alive()
    provider._ready.set()
    stopped.join(3)
    assert finished, "shutdown did not interrupt the startup wait"
    assert writer is not None and not writer.is_alive()
