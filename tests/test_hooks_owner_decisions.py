"""The eight owner decisions of 2026-11-04 on the `recall_hooks` CCA residue, as behaviour.

NUM-002, STAKES-004 (which also settles DAT-004), SEC-005, ENV-005, PERF-004, SEC-006 and BUG-004.
Red proof: each test was run on the test host against the pre-change modules, failed in its own
assertion, and passed on the change; the pull request records the run. No test imports a symbol
the change introduced (an attribute set with `raising=False` exists before and after), and a call
that used to behave differently is observed through its effect, never through a new name.
Exceptions are caught and asserted as booleans, so no exception object reaches a red run's output.
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import recall_hooks
from recall_hooks import general_knowledge as gk
from recall_hooks import prompt_time, relay


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path / "home"))


def _raises(call: Any) -> tuple[Any, bool]:
    try:
        return call(), False
    except Exception:  # noqa: BLE001 - asserted by the caller
        return None, True


# --------------------------------------------------------------------------- NUM-002


def test_num002_the_relay_budget_counts_from_hook_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """NUM-002. Invariant: a relay search ends by the hook's own budget, counted from when the
    hook process started, however late in the process the search begins.

    Defect caught: the 4.5 s budget started at `relay.search()` entry, so a search that began
    late (after a slow start, or after the general knowledge pass) ran past the client's 5 s
    timeout, and a killed hook writes no cooldown. Production line: the deadline in `relay.search`.
    """
    monkeypatch.setattr(recall_hooks, "HOOK_STARTED", time.monotonic() - 4.0, raising=False)
    seen: list[float] = []

    def fake_ensure(_session: str, _config: Any, _options: Any, deadline: float | None = None) -> Any:
        seen.append(deadline if deadline is not None else float("inf"))
        raise relay.RelayUnavailable("stop here")

    monkeypatch.setattr(relay, "_ensure", fake_ensure)
    _raises(lambda: relay.search("session", "a draft", {"dsn": "x"}, {"k": 5, "connect_timeout": 2.0}))
    remaining = (seen[0] - time.monotonic()) if seen else 0.0
    assert remaining <= 0.5, f"a search 4 s into the hook still had {remaining:.2f} s"


# --------------------------------------------------------------------------- STAKES-004


def test_stakes004_hosted_sync_refuses_a_session_outside_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-004. Invariant: hosted sync uploads nothing for an event whose cwd is outside the
    configured `project_root`, as the local index already refuses.

    Defect caught: the hosted branch uploaded `memory/` from whatever directory a session ended
    in, a cloned third-party repository included, and an upload cannot be taken back.
    Production line: the project boundary in the hosted branch of `_index_and_refresh`.
    """
    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    foreign = tmp_path / "cloned-elsewhere"
    (foreign / "memory").mkdir(parents=True)
    (foreign / "memory" / "theirs.md").write_text("# Theirs\n\nNot yours.\n", encoding="utf-8")
    config = {"endpoint": "https://mcp.example.test/mcp", "tenant": "t", "project_root": str(project)}
    monkeypatch.setattr(recall_hooks, "load_config", lambda: config)
    from recall_hooks import hosted

    synced: list[str] = []
    monkeypatch.setattr(hosted, "sync_memory_roots", lambda roots, _c, **_k: synced.append(str(roots)))
    recall_hooks._index_and_refresh({"cwd": str(foreign)})
    assert synced == [], "a session outside project_root uploaded its memory"
    recall_hooks._index_and_refresh({"cwd": str(project)})
    assert len(synced) == 1, "control: a session inside project_root still syncs"


def test_stakes004_a_hosted_config_without_a_project_root_says_so() -> None:
    """STAKES-004, its notice. Invariant: SessionStart warns when a hosted config names no
    `project_root`, since hosted sync then uploads nothing at all.

    Production line: the `project_root` check in `recall_hooks._unsynced_notice`.
    """
    notice = recall_hooks._unsynced_notice({"endpoint": "https://mcp.example.test/mcp", "tenant": "t"})
    assert "project_root" in notice, notice


# --------------------------------------------------------------------------- SEC-005


def test_sec005_the_client_sends_nothing_secret_to_an_unproven_listener() -> None:
    """SEC-005. Invariant: the relay client sends neither the session token nor the draft to a
    listener that cannot prove it holds the token.

    Defect caught: the client trusted any process on the recorded loopback port, so a listener
    that took the port of a dead helper received the token and the draft text.
    Production line: the challenge in `relay._endpoint_request`.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received = bytearray()

    def impostor() -> None:
        listener.settimeout(5)
        try:
            client, _ = listener.accept()
        except OSError:
            return
        with client:
            client.settimeout(1)
            try:
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    received.extend(chunk)
                    if b"\n" in chunk:
                        client.sendall(b'{"status":"ok","hits":[]}\n')
            except OSError:
                pass

    thread = threading.Thread(target=impostor, daemon=True)
    thread.start()
    token = "the-session-token-0123456789"
    state = {"host": "127.0.0.1", "port": listener.getsockname()[1], "token": token}
    message = {"op": "query", "token": token, "query": "SECRET DRAFT TEXT"}
    _response, raised = _raises(lambda: relay._endpoint_request(state, message, timeout=2.0))
    thread.join(5)
    listener.close()
    leaked = token.encode() in received or b"SECRET DRAFT TEXT" in received
    assert not leaked, "the token or the draft reached a listener that proved nothing"
    assert raised, "an unproven listener's answer was accepted"


def test_sec005_a_state_whose_helper_is_gone_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-005, liveness half. Invariant: a saved relay state whose helper process is dead is
    retired and a new helper is started, never reused.

    Defect caught: a matching fingerprint and a port were enough, so the port of a helper that
    died uncleanly was contacted again. Production line: the `_pid_alive` check in `relay._ensure`.
    """
    config = {"dsn": "postgresql://h/db"}
    options = {"k": 5, "connect_timeout": 2.0}
    path = relay.state_path("session-x")
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = relay._fingerprint(config, options)
    path.write_text(json.dumps({"token": "old", "fingerprint": fingerprint, "port": 6553,
                                "host": "127.0.0.1", "pid": 999999}), encoding="utf-8")
    monkeypatch.setattr(relay, "_pid_alive", lambda _pid: False)
    spawned: list[str] = []

    def fake_spawn(_path: Path, token: str, *_args: Any, **_kwargs: Any) -> Any:
        spawned.append(token)
        return {"token": token, "port": 1, "host": "127.0.0.1"}

    monkeypatch.setattr(relay, "_spawn", fake_spawn)
    _result, _raised = _raises(lambda: relay._ensure("session-x", config, options, time.monotonic() + 3))
    assert spawned and spawned[0] != "old", "the dead helper's state was reused"


# --------------------------------------------------------------------------- ENV-005


def test_env005_memory_root_is_the_store_override_for_the_prompt_hook(tmp_path: Path) -> None:
    """ENV-005. Invariant: the prompt hook reads the top-level `memory_root`, the key the sync
    and index path read, and `prompt_time.store` only when `memory_root` is unset.

    Defect caught: two keys, one per path, so setting one made the prompt hook and SessionEnd
    use different stores. Production line: the `store` entry of `prompt_time.settings`.
    """
    both = {"memory_root": "/one/store", "prompt_time": {"store": "/old/alias"}}
    assert prompt_time.settings(both)["store"] == "/one/store"
    assert prompt_time.settings({"prompt_time": {"store": "/old/alias"}})["store"] == "/old/alias"


def test_env005_a_linked_store_override_is_refused(tmp_path: Path) -> None:
    """ENV-005, link half. Invariant: an override that is a link to another folder is refused,
    as the default store path already refuses one.

    Production line: the override branch of `prompt_time.find_store`.
    """
    real = tmp_path / "somewhere-else"
    real.mkdir()
    link = tmp_path / "store-link"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a directory link without privileges")
    assert prompt_time.find_store(str(tmp_path), str(link)) is None
    assert prompt_time.find_store(str(tmp_path), str(real)) == real, "control: a real folder is used"


# --------------------------------------------------------------------------- PERF-004


def test_perf004_an_index_compiles_at_most_256_regexes(tmp_path: Path) -> None:
    """PERF-004. Invariant: one index contributes at most 256 `command_regex` triggers; the rest
    are dropped at load, where every regex is still compiled and validated.

    Defect caught: the per-entry caps admitted 8,192 regexes, all compiled on every shell call
    (3.1 s measured at the caps). Production line: the regex cap in `general_knowledge.load_index`.
    """
    entries = [
        {"id": f"gk-e{i}", "body": "b", "scope": "global",
         "triggers": [{"command_regex": f"^tool{i}x{j}$"} for j in range(30)]}
        for i in range(10)
    ]
    (tmp_path / "index.json").write_text(json.dumps({"schema": gk.INDEX_SCHEMA, "entries": entries}),
                                         encoding="utf-8")
    loaded = gk.load_index(tmp_path)
    count = sum(1 for entry in loaded for trigger in entry.triggers if trigger.kind == "command_regex")
    assert count <= 256, count
    assert count == 256, f"control: the first 256 are kept, got {count}"


# --------------------------------------------------------------------------- SEC-006


def test_sec006_a_same_named_clone_with_another_remote_gets_no_project_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-006. Invariant: a `project:` entry is matched on the origin remote's `owner/name`, so
    a checkout that only shares the folder name receives nothing.

    Defect caught: matching on the folder name gave a fork, a scratch clone or another owner's
    project with the same name that project's curated operational entries.
    Production line: the project passed to `in_scope` by `general_knowledge.pre_tool_use_context`.
    """
    knowledge = tmp_path / "gk"
    knowledge.mkdir()
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(knowledge))
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(tmp_path / "gk.jsonl"))
    (knowledge / "index.json").write_text(json.dumps({"schema": gk.INDEX_SCHEMA, "entries": [
        {"id": "gk-owner-only", "body": "Owner-only procedure.", "scope": "project:alpha",
         "triggers": [{"command": "pytest"}]},
        {"id": "gk-by-remote", "body": "Matched by remote.", "scope": "project:owner/alpha",
         "triggers": [{"command": "pytest"}]},
    ]}), encoding="utf-8")
    clone = tmp_path / "alpha"
    (clone / ".git").mkdir(parents=True)
    (clone / ".git" / "config").write_text(
        '[core]\n\tbare = false\n[remote "origin"]\n\turl = https://github.com/owner/alpha.git\n',
        encoding="utf-8")
    payload = {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "cwd": str(clone),
               "session_id": "s-1"}
    text = gk.pre_tool_use_context(payload) or ""
    assert "gk-owner-only" not in text, "a folder-name scope matched a checkout with a remote"
    assert "gk-by-remote" in text, "control: the owner/name scope matches its remote"


# --------------------------------------------------------------------------- BUG-004


def _cleanup(monkeypatch: pytest.MonkeyPatch, table: Path, killed: Path) -> Any:
    for name in ("CLAUDE_PID", "RECALL_MCP_CLIENT", "RECALL_MCP_SESSION_ID", "RECALL_MCP_PATTERN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_MCP_PS_FILE", str(table))
    monkeypatch.setenv("RECALL_MCP_KILL_FILE", str(killed))
    import recall_hooks.mcp_cleanup as cleanup

    return importlib.reload(cleanup)


def test_bug004_a_launcher_parent_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-004, launcher half. Invariant: when the hook's parent is a launcher running this same
    command (a Windows venv `python.exe`), the client is the launcher's parent.

    Defect caught: nothing descends from the launcher, so Codex cleanup closed nothing and the
    MCP servers leaked. Production line: the launcher step in `close_own_mcp_transports`.
    """
    me = str(os.getpid())
    table = tmp_path / "ps.txt"
    table.write_text(
        "100 1 codex.exe\n"
        "150 100 C:/venv/Scripts/python.exe -P -m recall_hooks.codex session-end\n"
        f"{me} 150 C:/Python/python.exe -P -m recall_hooks.codex session-end\n"
        "101 100 python -m recall_mcp.server\n",
        encoding="utf-8",
    )
    killed = tmp_path / "killed.txt"
    cleanup = _cleanup(monkeypatch, table, killed)
    cleanup.close_own_mcp_transports("150", cwd=str(tmp_path))
    done = killed.read_text(encoding="utf-8").split() if killed.exists() else []
    assert done == ["101"], done


def test_bug004_a_marked_transport_of_another_session_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-004, shared-client half. Invariant: with a session ID given, a transport marked with a
    DIFFERENT session ID is left alone even when it descends from the same client.

    Defect caught: one client process hosting several sessions closed its siblings' servers.
    Production line: the session-ID check beside `by_pid` in `close_own_mcp_transports`.
    """
    table = tmp_path / "ps.txt"
    table.write_text(
        "100 1 codex.exe\n"
        "101 100 python -m recall_mcp.server RECALL_MCP_SESSION_ID=aaa\n"
        "102 100 python -m recall_mcp.server RECALL_MCP_SESSION_ID=bbb\n",
        encoding="utf-8",
    )
    killed = tmp_path / "killed.txt"
    cleanup = _cleanup(monkeypatch, table, killed)
    cleanup.close_own_mcp_transports("100", session_id="aaa", cwd=str(tmp_path))
    done = killed.read_text(encoding="utf-8").split() if killed.exists() else []
    assert done == ["101"], done


def test_sec005_control_the_real_helper_completes_the_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-005 control, labelled nonbehavioral for red proof: it passes before the change too.

    It exists because every other relay test replaces the transport. It runs the real helper
    loop (`relay._serve`) in a thread and asserts that the real client completes the challenge
    and a `ping` and a `shutdown` round trip against it, so the handshake both sides now speak is
    proved compatible end to end. Neither operation touches a database.
    """
    from recall_hooks import write_time

    config = {"dsn": "postgresql://unused/db"}
    monkeypatch.setattr(relay, "load_config", lambda: config)
    token = "control-token-0123456789abcdef"
    options = {**write_time.settings(config), "k": 5, "connect_timeout": 2.0}
    path = tmp_path / "relay-state.json"
    relay._atomic_write(path, relay._initial_state(token, relay._fingerprint(config, options), options))
    server = threading.Thread(target=relay._serve, args=(path, relay._token_digest(token)), daemon=True)
    server.start()
    deadline = time.monotonic() + 10
    state: Any = {}
    while time.monotonic() < deadline:
        state = relay._read(path) or {}
        if int(state.get("port", 0) or 0) > 0:
            break
        time.sleep(0.02)
    assert int(state.get("port", 0) or 0) > 0, "the helper never published its port"
    assert relay._endpoint_request(state, {"op": "ping", "token": token}) == {"status": "ok"}
    assert relay._endpoint_request(state, {"op": "shutdown", "token": token}) == {"status": "ok"}
    server.join(10)
    assert not server.is_alive(), "the helper did not stop after shutdown"
