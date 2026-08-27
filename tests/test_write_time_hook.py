"""The `PreToolUse` memo-injection hook: the properties that stop it breaking a session.

This hook is on the critical path of EVERY tool call, so its failure modes are not "it returns a
worse answer", they are "every Write in every session is a second slower" and "a tool call was
denied by a memory layer". Those are what is asserted here.

Nothing here needs a database: `search` is substituted, because what is under test is the
decision logic around it. The live path is verified separately against a real corpus, and the
latency table in `recall_hooks/write_time.py` records those measurements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from recall_hooks import write_time


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """An isolated CLAUDE_CONFIG_DIR, so no test can read or write the developer's own config."""

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    return tmp_path


def write_config(root: Path, **overrides) -> None:
    config = {"dsn": "postgresql://example/db", "tenant": "default"}
    config.update(overrides)
    (root / "recall-hook.json").write_text(
        json.dumps(config) + "\n", encoding="utf-8", newline="\n"
    )


HIT = [("python-write-text-crlf-churn.md", "pass newline='\\n' to write_text", 0.83)]


def payload(tool: str = "Write", **tool_input) -> dict:
    return {"tool_name": tool, "tool_input": tool_input or {"content": "x" * 200}}


def test_it_can_never_deny_a_tool_call(hook_env, monkeypatch, capsys):
    """A memory layer that can veto a write can wedge a session. It may only add context."""

    write_config(hook_env)
    monkeypatch.setattr(write_time, "search", lambda *a, **k: HIT)

    assert write_time.pre_tool_use(payload()) == 0
    out = capsys.readouterr().out
    document = json.loads(out)
    assert document["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "additionalContext" in document["hookSpecificOutput"]
    assert "permissionDecision" not in out
    assert "deny" not in out


def test_a_retrieval_failure_is_silent_and_starts_a_cooldown(hook_env, monkeypatch, capsys):
    """The expensive failure is an unreachable corpus, which otherwise costs ~3s on EVERY call."""

    write_config(hook_env)

    def explode(*args, **kwargs):
        raise RuntimeError("no database")

    monkeypatch.setattr(write_time, "search", explode)
    assert write_time.pre_tool_use(payload()) == 0
    assert capsys.readouterr().out == ""

    stamp = hook_env / write_time.COOLDOWN_NAME
    assert stamp.exists(), "a failed connection must start the cooldown"
    assert float(stamp.read_text(encoding="utf-8")) > time.time()


def test_the_cooldown_short_circuits_before_search_is_reached(hook_env, monkeypatch, capsys):
    """The whole point is to not pay the timeout again, so `search` must not even be called."""

    write_config(hook_env)
    (hook_env / write_time.COOLDOWN_NAME).write_text(
        str(time.time() + 300), encoding="utf-8", newline="\n"
    )

    called = []
    monkeypatch.setattr(write_time, "search", lambda *a, **k: called.append(1) or HIT)
    assert write_time.pre_tool_use(payload()) == 0
    assert called == [], "search ran while the corpus was known to be unreachable"
    assert capsys.readouterr().out == ""


def test_an_expired_cooldown_lets_the_hook_try_again(hook_env, monkeypatch, capsys):
    """A cooldown that never expires is a feature that silently turns itself off forever."""

    write_config(hook_env)
    (hook_env / write_time.COOLDOWN_NAME).write_text(
        str(time.time() - 1), encoding="utf-8", newline="\n"
    )
    monkeypatch.setattr(write_time, "search", lambda *a, **k: HIT)
    assert write_time.pre_tool_use(payload()) == 0
    assert "additionalContext" in capsys.readouterr().out


def test_a_success_clears_the_cooldown(hook_env, monkeypatch, capsys):
    write_config(hook_env)
    stamp = hook_env / write_time.COOLDOWN_NAME
    stamp.write_text(str(time.time() - 1), encoding="utf-8", newline="\n")
    monkeypatch.setattr(write_time, "search", lambda *a, **k: HIT)
    write_time.pre_tool_use(payload())
    assert not stamp.exists()


def test_a_corrupt_cooldown_file_does_not_suppress_the_feature(hook_env, monkeypatch, capsys):
    """This file's failure mode must be to TRY the database, never to disable the hook."""

    write_config(hook_env)
    (hook_env / write_time.COOLDOWN_NAME).write_text("not a number", encoding="utf-8")
    monkeypatch.setattr(write_time, "search", lambda *a, **k: HIT)
    write_time.pre_tool_use(payload())
    assert "additionalContext" in capsys.readouterr().out


def test_unconfigured_is_silent(hook_env, monkeypatch, capsys):
    """A checkout that never ran the installer must behave exactly as if the hook did not exist."""

    (hook_env / "recall-hook.json").write_text("{}\n", encoding="utf-8")

    # ⛔ Recorded, NOT raised. `pre_tool_use` catches every exception from `search` on
    # purpose, so a stub that raises is swallowed and the test passes against a hook that DID
    # call it. Two mutations survived this file before that was understood.
    calls: list[int] = []
    monkeypatch.setattr(write_time, "search", lambda *a, **k: calls.append(1) or HIT)
    assert write_time.pre_tool_use(payload()) == 0
    assert calls == [], "search ran without a configured dsn"
    assert capsys.readouterr().out == ""


def test_disabled_is_honoured(hook_env, monkeypatch, capsys):
    write_config(hook_env, write_time={"enabled": False})

    calls: list[int] = []
    monkeypatch.setattr(write_time, "search", lambda *a, **k: calls.append(1) or HIT)
    assert write_time.pre_tool_use(payload()) == 0
    assert calls == [], "search ran while the feature was disabled"
    assert capsys.readouterr().out == ""


def test_an_absent_block_means_enabled(hook_env):
    """An upgraded config predating the feature should still get what it was upgraded for."""

    write_config(hook_env)
    assert write_time.settings()["enabled"] is True


def test_short_payloads_never_reach_the_database(hook_env, monkeypatch, capsys):
    """This early return is what keeps `ls` at 0.19s instead of 1.0s, by skipping the import."""

    write_config(hook_env)

    calls: list[int] = []
    monkeypatch.setattr(write_time, "search", lambda *a, **k: calls.append(1) or HIT)
    assert write_time.pre_tool_use(payload("Bash", command="ls")) == 0
    assert calls == [], "search ran on a payload below min_chars"
    assert capsys.readouterr().out == ""


def test_the_query_is_the_draft_not_the_goal():
    """Measured: draft text surfaces the governing memo for 11 of 11 sessions that needed it,
    goal-shaped queries for 1 of 14. Extracting the wrong field silently reverts that."""

    assert write_time.payload_of("Write", {"content": "the draft"}) == "the draft"
    assert write_time.payload_of("Edit", {"new_string": "the draft"}) == "the draft"
    assert write_time.payload_of("Bash", {"command": "pytest -q"}) == "pytest -q"
    # A Write carries `content`, never `command`; reading the wrong one returns empty, which
    # would make the hook silently stop firing on the tool it exists for.
    assert write_time.payload_of("Write", {"command": "pytest"}) == ""
    assert write_time.payload_of("Read", {"file_path": "/etc/hosts"}) == ""


def test_a_malformed_event_is_survivable(hook_env, capsys):
    """`[]` and `3` are valid json without a `.get`. The hook is fed by a client, and an
    AttributeError escaping into the session is the one outcome it may never produce."""

    write_config(hook_env)
    for event in ({"tool_name": "Write"}, {"tool_name": "Write", "tool_input": None},
                  {"tool_input": {"content": "x" * 200}}):
        assert write_time.pre_tool_use(event) == 0
    assert capsys.readouterr().out == ""


def test_no_hits_prints_nothing(hook_env, monkeypatch, capsys):
    write_config(hook_env)
    monkeypatch.setattr(write_time, "search", lambda *a, **k: [])
    assert write_time.pre_tool_use(payload()) == 0
    assert capsys.readouterr().out == ""


def test_the_injected_text_warns_that_most_hits_are_irrelevant():
    """Measured: this fires on 29 of 36 sessions that did not need a memo. Saying so is what
    makes an irrelevant hit cheap to dismiss rather than something to reconcile."""

    rendered = write_time.render(HIT)
    assert "most searches return nothing that applies" in rendered
    assert "python-write-text-crlf-churn" in rendered
    assert "write_text" in rendered


def test_the_installer_entry_never_carries_a_permission_decision():
    """The settings the installer writes are what the client actually runs."""

    from recall.claude_code import hook_entries

    entries = hook_entries("python")
    assert "PreToolUse" in entries
    group = entries["PreToolUse"][0]
    assert group["matcher"] == "Write|Edit|MultiEdit|NotebookEdit|Bash"
    handler = group["hooks"][0]
    assert handler["args"] == ["-m", "recall_hooks", "pre-tool-use"]
    # Synchronous by necessity: additionalContext delivered after the tool ran is context the
    # model never saw, so an `async: True` here would silently make the feature inert.
    assert handler.get("async") is not True
    assert "permissionDecision" not in json.dumps(entries)
