"""Unit tests for the session relay boundary.

These tests do not open PostgreSQL.  The live transport benchmark and the hook tests cover the
database query and fail-open paths separately; this file pins token use, state isolation, and
restart behavior of the local process boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from recall_hooks import relay


def test_state_names_are_stable_and_do_not_include_the_session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(relay, "claude_config_home", lambda: tmp_path)
    first = relay.state_path("session/with secrets")
    second = relay.state_path("session/with secrets")
    assert first == second
    assert "session" not in first.name
    assert first.parent == tmp_path / relay.STATE_DIR_NAME


def test_endpoint_must_be_loopback_before_any_socket_is_opened(monkeypatch):
    def unexpected_socket(*args, **kwargs):
        raise AssertionError("a non-loopback endpoint must be rejected before connect")

    monkeypatch.setattr(relay.socket, "create_connection", unexpected_socket)
    with pytest.raises(relay.RelayUnavailable, match="loopback"):
        relay._endpoint_request({"host": "192.0.2.1", "port": 1234}, {"op": "ping"})
    with pytest.raises(relay.RelayUnavailable, match="loopback"):
        relay._endpoint_request({"host": "127.0.0.1", "port": 65536}, {"op": "ping"})


def test_stop_state_waits_for_the_owned_helper_before_removing_state(monkeypatch, tmp_path):
    path = tmp_path / "relay.json"
    state = {"host": "127.0.0.1", "port": 1234, "pid": 42, "token": "token"}
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(relay, "_endpoint_request", lambda *_args: (_ for _ in ()).throw(OSError()))
    alive = iter([True, True, False])
    monkeypatch.setattr(relay, "_pid_alive", lambda _pid: next(alive))
    assert relay._stop_state(path, state) is True
    assert not path.exists()
    assert not relay._stop_marker_path(path).exists()


def test_start_timeout_leaves_abort_marker_for_a_detached_child(monkeypatch, tmp_path):
    path = tmp_path / "relay.json"
    monkeypatch.setattr(relay, "START_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *_args, **_kwargs: object())
    with pytest.raises(relay.RelayUnavailable, match="did not become ready"):
        relay._spawn(path, "token", "fingerprint")
    assert not path.exists()
    marker = relay._read(relay._stop_marker_path(path))
    assert marker == {"token": "token"}


def test_start_timeout_does_not_remove_replacement_state(monkeypatch, tmp_path):
    path = tmp_path / "relay.json"
    replacement = {"token": "replacement", "fingerprint": "new", "port": 1234}
    monkeypatch.setattr(relay, "START_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *_args, **_kwargs: object())

    class TakeoverLock:
        def __enter__(self):
            relay._atomic_write(path, replacement)
            return None

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(relay, "_state_lock", lambda *_args, **_kwargs: TakeoverLock())
    with pytest.raises(relay.RelayUnavailable, match="did not become ready"):
        relay._spawn(path, "token", "fingerprint")
    assert relay._read(path) == replacement


def test_stop_does_not_remove_replacement_stop_marker(monkeypatch, tmp_path):
    path = tmp_path / "relay.json"
    state = {"host": "127.0.0.1", "port": 0, "pid": 42, "token": "old"}
    replacement = {"host": "127.0.0.1", "port": 1234, "pid": 43, "token": "new"}
    path.write_text(json.dumps(state), encoding="utf-8")

    def wait_for_exit(_pid):
        relay._atomic_write(path, replacement)
        relay._atomic_write(relay._stop_marker_path(path), {"token": "new"})
        return True

    monkeypatch.setattr(relay, "_wait_for_exit", wait_for_exit)
    assert relay._stop_state(path, state) is True
    assert relay._read(path) == replacement
    assert relay._read(relay._stop_marker_path(path)) == {"token": "new"}


def test_search_uses_authenticated_query_and_returns_typed_hits(monkeypatch, tmp_path):
    state_path = tmp_path / "relay.json"
    state = {"host": "127.0.0.1", "port": 1234, "token": "secret"}
    calls: list[dict] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: (state_path, state))
    monkeypatch.setattr(
        relay,
        "_endpoint_request",
        lambda _state, message, **_kwargs: calls.append(message) or {
            "status": "ok",
            "hits": [["memo.md", "text", "0.75"]],
        },
    )
    rows = relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2})
    assert rows == [("memo.md", "text", 0.75)]
    assert calls == [{"op": "query", "token": "secret", "query": "draft"}]


def test_search_bounds_query_payload_before_serializing(monkeypatch, tmp_path):
    state = {"host": "127.0.0.1", "port": 1234, "token": "secret"}
    seen: list[dict] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: (tmp_path / "relay.json", state))
    monkeypatch.setattr(
        relay,
        "_endpoint_request",
        lambda _state, message, **_kwargs: seen.append(message) or {"status": "ok", "hits": []},
    )
    relay.search("session-1", "x" * 5000, {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2})
    assert len(seen[0]["query"]) == 4096


def test_crashed_relay_is_restarted_once(monkeypatch, tmp_path):
    first = {"host": "127.0.0.1", "port": 1234, "token": "old"}
    second = {"host": "127.0.0.1", "port": 5678, "token": "new"}
    states = iter([(tmp_path / "relay.json", first), (tmp_path / "relay.json", second)])
    calls: list[dict] = []
    stopped: list[Path] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: next(states))
    monkeypatch.setattr(relay, "_retire_state", lambda path, state=None, **_kwargs: stopped.append(path))

    def request(state, message, **_kwargs):
        calls.append({"port": state["port"], **message})
        if state["token"] == "old":
            raise OSError("broken pipe")
        return {"status": "ok", "hits": []}

    monkeypatch.setattr(relay, "_endpoint_request", request)
    assert relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2}) == []
    assert [call["port"] for call in calls] == [1234, 5678]
    assert stopped == [tmp_path / "relay.json"]


def test_transport_eof_during_query_restarts_the_relay(monkeypatch, tmp_path):
    first = {"host": "127.0.0.1", "port": 1234, "token": "old"}
    second = {"host": "127.0.0.1", "port": 5678, "token": "new"}
    states = iter([(tmp_path / "relay.json", first), (tmp_path / "relay.json", second)])
    stopped: list[Path] = []
    monkeypatch.setattr(relay, "_ensure", lambda *args: next(states))
    monkeypatch.setattr(relay, "_retire_state", lambda path, state=None, **_kwargs: stopped.append(path))

    def request(state, message, **_kwargs):
        if state["token"] == "old":
            raise relay.RelayUnavailable("relay closed the connection")
        return {"status": "ok", "hits": []}

    monkeypatch.setattr(relay, "_endpoint_request", request)
    assert relay.search("session-1", "draft", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2}) == []
    assert stopped == [tmp_path / "relay.json"]


def test_ensure_serializes_concurrent_first_start(monkeypatch, tmp_path):
    path = tmp_path / "relay.json"
    monkeypatch.setattr(relay, "state_path", lambda _session_id: path)
    state = {
        "host": "127.0.0.1",
        "port": 1234,
        "pid": 1,
        "token": "token",
        "fingerprint": relay._fingerprint({"dsn": "dsn"}, {"k": 5, "connect_timeout": 2}),
    }
    starts = 0

    def spawn(path_value, token, fingerprint, *_args):
        nonlocal starts
        starts += 1
        relay._atomic_write(path_value, {**state, "token": token, "fingerprint": fingerprint})
        time.sleep(0.05)
        return _state_with_token(path_value, token, fingerprint)

    def _state_with_token(_path, token, fingerprint):
        return {**state, "token": token, "fingerprint": fingerprint}

    monkeypatch.setattr(relay, "_spawn", spawn)
    monkeypatch.setattr(relay, "_endpoint_request", lambda *_args: {"status": "ok"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: relay._ensure("session-1", {"dsn": "dsn"}, {"k": 5, "connect_timeout": 2}),
            range(2),
        ))
    assert starts == 1
    assert results[0][1]["fingerprint"] == results[1][1]["fingerprint"]


def test_failure_is_not_hidden_as_an_empty_result(monkeypatch, tmp_path):
    state = {"host": "127.0.0.1", "port": 1234, "token": "secret"}
    monkeypatch.setattr(relay, "_ensure", lambda *args: (tmp_path / "relay.json", state))
    monkeypatch.setattr(
        relay,
        "_endpoint_request",
        lambda *args, **kwargs: {"status": "unavailable", "error": "OperationalError"},
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
