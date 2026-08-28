"""Unit tests for the session relay boundary.

These tests do not open PostgreSQL.  The live transport benchmark and the hook tests cover the
database query and fail-open paths separately; this file pins token use, state isolation, and
restart behavior of the local process boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from recall_hooks import relay


def test_state_names_are_stable_and_do_not_include_the_session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(relay, "claude_config_home", lambda: tmp_path)
    first = relay.state_path("session/with secrets")
    second = relay.state_path("session/with secrets")
    assert first == second
    assert "session" not in first.name
    assert first.parent == tmp_path / relay.STATE_DIR_NAME


def test_search_uses_authenticated_query_and_returns_typed_hits(monkeypatch, tmp_path):
    state_path = tmp_path / "relay.json"
    state = {"host": "127.0.0.1", "port": 1234, "token": "secret"}
    calls: list[dict] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: (state_path, state))
    monkeypatch.setattr(
        relay,
        "_endpoint_request",
        lambda _state, message: calls.append(message) or {
            "status": "ok",
            "hits": [["memo.md", "text", "0.75"]],
        },
    )
    rows = relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2})
    assert rows == [("memo.md", "text", 0.75)]
    assert calls == [{"op": "query", "token": "secret", "query": "draft"}]


def test_crashed_relay_is_restarted_once(monkeypatch, tmp_path):
    first = {"host": "127.0.0.1", "port": 1234, "token": "old"}
    second = {"host": "127.0.0.1", "port": 5678, "token": "new"}
    states = iter([(tmp_path / "relay.json", first), (tmp_path / "relay.json", second)])
    calls: list[dict] = []
    stopped: list[Path] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: next(states))
    monkeypatch.setattr(relay, "_stop_state", lambda path, state=None: stopped.append(path))

    def request(state, message):
        calls.append({"port": state["port"], **message})
        if state["token"] == "old":
            raise OSError("broken pipe")
        return {"status": "ok", "hits": []}

    monkeypatch.setattr(relay, "_endpoint_request", request)
    assert relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2}) == []
    assert [call["port"] for call in calls] == [1234, 5678]
    assert stopped == [tmp_path / "relay.json"]


def test_failure_is_not_hidden_as_an_empty_result(monkeypatch, tmp_path):
    state = {"host": "127.0.0.1", "port": 1234, "token": "secret"}
    monkeypatch.setattr(relay, "_ensure", lambda *args: (tmp_path / "relay.json", state))
    monkeypatch.setattr(
        relay,
        "_endpoint_request",
        lambda *args: {"status": "unavailable", "error": "OperationalError"},
    )
    with pytest.raises(relay.RelayUnavailable, match="OperationalError"):
        relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2})


def test_unreachable_database_keeps_followup_calls_fast_and_stops_cleanly(monkeypatch, tmp_path):
    """The production process boundary, not just a mocked socket, remains fail-open."""

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    config = {"dsn": "postgresql://u:p@127.0.0.1:1/db", "tenant": "default"}
    (tmp_path / "recall-hook.json").write_text(json.dumps(config), encoding="utf-8")
    options = {"k": 5, "connect_timeout": 0.2}

    with pytest.raises(relay.RelayUnavailable):
        relay.search("session-check", "a sufficiently long draft", config, options)
    started = time.perf_counter()
    with pytest.raises(relay.RelayUnavailable):
        relay.search("session-check", "a sufficiently long draft", config, options)
    elapsed_ms = (time.perf_counter() - started) * 1000
    relay.stop("session-check")
    assert elapsed_ms < 500
    assert not relay.state_path("session-check").exists()
