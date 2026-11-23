"""The stdio client: answers are matched by id, and a late answer is dropped, not kept."""

from __future__ import annotations

import json
import sys
import time

import pytest
from conftest import HERE

from re_call_hermes.mcp_client import McpError, StdioMcpClient


def test_a_late_answer_is_dropped_and_the_next_call_gets_its_own(tmp_path, monkeypatch):
    """Invariant: a timed-out request's answer neither lingers nor answers a later request.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    LATE-ANSWER`): in re_call_hermes/mcp_client.py, `if message["id"] in self._abandoned:` became `if
    False:`; the test failed at `assert {2: {'jsonrpc...ror': False}}} == {}`. Restored, green.
    """
    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({"tools": {"slow": [{"n": 1}], "fast": [{"n": 2}]},
                                    "delay": {"slow": 1}}), encoding="utf-8")
    env = {"FAKE_SCENARIO": str(scenario), "FAKE_LOG": str(tmp_path / "log.jsonl"), "PYTHONIOENCODING": "utf-8"}
    for key in ("SYSTEMROOT", "PATH", "TEMP", "TMP"):
        if key in __import__("os").environ:
            env[key] = __import__("os").environ[key]
    client = StdioMcpClient([sys.executable, str(HERE / "fake_mcp_server.py")], env=env, cwd=tmp_path,
                            stderr_path=tmp_path / "err.log")
    client.start(20)
    try:
        with pytest.raises(McpError):
            client.call_tool("slow", {}, timeout=0.2)
        assert client.call_tool("fast", {}, timeout=10) == {"n": 2}
        assert client._pending == {}
    finally:
        client.close()


def _client(tmp_path, **scenario):
    import os

    path = tmp_path / "s.json"
    path.write_text(json.dumps({"tools": {}, **scenario}), encoding="utf-8")
    env = {"FAKE_SCENARIO": str(path), "FAKE_LOG": str(tmp_path / "log.jsonl"), "PYTHONIOENCODING": "utf-8"}
    env.update({k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ})
    return StdioMcpClient([sys.executable, str(HERE / "fake_mcp_server.py")], env=env, cwd=tmp_path,
                          stderr_path=tmp_path / "err.log")


def test_a_wedged_server_cannot_hold_a_caller_past_its_deadline(tmp_path):
    """Invariant: when the child stops reading, a call still returns at its deadline.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    WEDGED-SEND`): in re_call_hermes/mcp_client.py, `self._outbox.put((json.dumps(message, separators=(",",
    ":...` became `proc.stdin.write((json.dumps(message) + "/n").e...`; the test failed at `AssertionError:
    the call blocked past its deadline`. Restored, green.
    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    WEDGED-CLOSE`): in re_call_hermes/mcp_client.py, `self._thread_factory(lambda: _close_quietly(stdin),
    name=...` became `_close_quietly(stdin)`; the test failed at `AssertionError: close() took 119.0s on a
    wedged server`. Restored, green.
    """
    import threading

    client = _client(tmp_path, stall_after_init=True)
    client.start(20)
    outcome = []

    def call():
        try:
            client.call_tool("recall_remember", {"text": "x" * 4_000_000}, timeout=1)
        except McpError as exc:
            outcome.append(exc)

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    caller.join(4)
    try:
        assert not caller.is_alive(), "the call blocked past its deadline"
        assert outcome
    finally:
        started = time.monotonic()
        client.close(timeout=2)
        closed_in = time.monotonic() - started
    assert closed_in < 8, f"close() took {closed_in:.1f}s on a wedged server"


def test_a_malformed_answer_is_an_mcp_error(tmp_path):
    """Invariant: an answer without a content list surfaces as McpError, which callers handle.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py MALFORMED`):
    in re_call_hermes/mcp_client.py, `parts = content if isinstance(content, list) else []` became `parts =
    content`; the test failed at `AssertionError: TypeError("'NoneType' object is not iterable")`. Restored,
    green.
    """
    client = _client(tmp_path, tools={"recall_search": [{"raw_result": {"content": None}}]})
    client.start(20)
    try:
        try:
            client.call_tool("recall_search", {"query": "q"}, timeout=10)
            raised = None
        except Exception as exc:
            raised = exc
        assert isinstance(raised, McpError), repr(raised)
    finally:
        client.close()


def test_a_refusal_without_content_is_still_a_refusal(tmp_path):
    """Invariant: an isError answer is a ToolError (never retried), with or without content.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    REFUSAL-NO-CONTENT`): in re_call_hermes/mcp_client.py, `if result.get("isError"): # a refusal stays a
    refusal` became `if False: # a refusal stays a refusal`; the test failed at `AssertionError:
    McpError("recall_remember answered with text that is not JSON...`. Restored, green.
    """
    from re_call_hermes.mcp_client import ToolError

    client = _client(tmp_path, tools={"recall_remember": [{"raw_result": {"isError": True}}]})
    client.start(20)
    try:
        try:
            client.call_tool("recall_remember", {"text": "x"}, timeout=10)
            raised = None
        except Exception as exc:
            raised = exc
        assert isinstance(raised, ToolError), repr(raised)
    finally:
        client.close()


def test_pending_requests_are_bounded(tmp_path, monkeypatch):
    """A full pool refuses a third call before sending it."""
    from concurrent.futures import ThreadPoolExecutor

    from conftest import wait_for

    from re_call_hermes import mcp_client

    client = _client(tmp_path, tools={"ready": [{"ok": True}]}, no_answer_text=["None"])
    client.start(20)
    log = tmp_path / "log.jsonl"

    def call():
        with pytest.raises(McpError):
            client.call_tool("recall_search", {}, timeout=12)

    try:
        # A reply proves the initialization notification has left the FIFO before lowering its limit.
        assert client.call_tool("ready", {"text": "probe"}, timeout=10) == {"ok": True}
        monkeypatch.setattr(mcp_client, "MAX_PENDING", 2, raising=False)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call) for _ in range(2)]
            try:
                assert wait_for(
                    lambda: log.read_text(encoding="utf-8").count('"tool": "recall_search"') == 2, timeout=5
                )
                with pytest.raises(mcp_client.NotSent, match="too many outstanding"):
                    client.call_tool("recall_search", {}, timeout=0.1)
            finally:
                client.close()
                for future in futures:
                    future.result()
            assert log.read_text(encoding="utf-8").count('"tool": "recall_search"') == 2
    finally:
        client.close()


def test_a_queued_deadline_reports_the_request_unsent(tmp_path):
    """An expired frame that never reached the writer has a known outcome."""
    from types import SimpleNamespace

    from re_call_hermes.mcp_client import NotSent

    client = _client(tmp_path)
    # Model a live child before the writer can take this frame.
    client._proc = SimpleNamespace(stdin=object(), poll=lambda: None)
    with pytest.raises(NotSent, match="got no answer"):
        client.call_tool("recall_remember", {"text": "memo"}, timeout=0)


def test_a_known_late_reply_resets_the_timeout_streak(tmp_path):
    """A late reply proves the client is responsive without becoming another call's answer."""
    client = _client(
        tmp_path,
        tools={"slow": [{"n": 1}], "fast": [{"n": 2}]},
        delay={"slow": 0.4},
        no_answer_text=["silent"],
    )
    client.start(20)
    try:
        with pytest.raises(McpError, match="got no answer"):
            client.call_tool("slow", {}, timeout=0.1)
        # The slow reply arrives before this separate, unanswered call expires.
        with pytest.raises(McpError, match="got no answer"):
            client.call_tool("silent", {"text": "silent"}, timeout=1.5)
        assert client.healthy
        assert client.call_tool("fast", {}, timeout=3) == {"n": 2}
        assert client._pending == {}
    finally:
        client.close()
