"""Tests for the Claude Code wiring.

The bar here is higher than "does it write the file", because this code edits
`~/.claude/settings.json`, which the user shares with every project on the machine and did not
create. The three properties worth holding are that a re-run does not duplicate what a previous run
wrote, that a hook the user added themselves survives both install and uninstall, and that a DSN
never reaches a log line. Everything else is detail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recall import claude_code
from recall.claude_code import (
    _is_recall_handler,
    _redacted,
    _strip_recall_groups,
    hook_entries,
    merge_hooks,
    uninstall,
)

PYTHON = "C:/Program Files/Python311/python.exe"


def _user_settings() -> dict[str, Any]:
    """A settings document with the user's own work in it, including inside our own event."""
    return {
        "permissions": {"allow": ["Bash(git *)"]},
        "hooks": {
            "SessionStart": [
                {"matcher": "startup", "hooks": [{"type": "command", "command": "mine.sh"}]}
            ]
        },
    }


def test_merge_is_idempotent() -> None:
    entries = hook_entries(PYTHON)
    once = merge_hooks(_user_settings(), entries)
    twice = merge_hooks(once, entries)
    assert once == twice
    assert merge_hooks(twice, entries) == twice


def test_merge_does_not_mutate_the_callers_document() -> None:
    original = _user_settings()
    snapshot = json.dumps(original, sort_keys=True)
    merge_hooks(original, hook_entries(PYTHON))
    assert json.dumps(original, sort_keys=True) == snapshot


def test_merge_preserves_unrelated_settings_and_hooks() -> None:
    merged = merge_hooks(_user_settings(), hook_entries(PYTHON))
    assert merged["permissions"] == {"allow": ["Bash(git *)"]}
    commands = [h.get("command") for g in merged["hooks"]["SessionStart"] for h in g["hooks"]]
    assert "mine.sh" in commands


def test_a_users_handler_inside_our_own_group_survives_stripping() -> None:
    """The case that makes group-level removal wrong: someone edits the group we wrote."""
    ours = hook_entries(PYTHON)["SessionEnd"][0]["hooks"][0]
    theirs = {"type": "command", "command": "theirs.sh"}
    groups = [{"matcher": "other", "hooks": [ours, theirs]}]
    assert _strip_recall_groups(groups) == [{"matcher": "other", "hooks": [theirs]}]


def test_empty_groups_are_dropped_rather_than_accumulated() -> None:
    ours = hook_entries(PYTHON)["SessionEnd"][0]["hooks"][0]
    assert _strip_recall_groups([{"matcher": "other", "hooks": [ours]}]) == []


def test_handlers_are_recognised_by_the_module_they_invoke() -> None:
    ours = hook_entries(PYTHON)["SessionStart"][0]["hooks"][0]
    assert _is_recall_handler(ours)
    assert not _is_recall_handler({"type": "command", "command": "recall-ish.sh"})


def test_session_end_runs_async_and_session_start_does_not() -> None:
    """SessionEnd cannot block termination, so a synchronous index is a promise nobody keeps.

    SessionStart is the opposite: its context has to be injected before the first turn, so it must
    not be async, and its timeout must be far below the documented 600 second default.
    """
    entries = hook_entries(PYTHON)
    assert entries["SessionEnd"][0]["hooks"][0]["async"] is True
    start = entries["SessionStart"][0]["hooks"][0]
    assert "async" not in start
    assert start["timeout"] <= 30


def test_matchers_use_only_documented_values() -> None:
    documented = {
        "SessionStart": {"startup", "resume", "clear", "compact", "fork"},
        "SessionEnd": {"clear", "resume", "logout", "prompt_input_exit", "other"},
    }
    for event, groups in hook_entries(PYTHON).items():
        for group in groups:
            assert set(group["matcher"].split("|")) <= documented[event]


def test_matchers_stay_on_the_exact_list_path_rather_than_the_regex_one() -> None:
    """A matcher of letters, digits, `_`, `-`, spaces, `,` and `|` is a list of exact strings.

    Any other character moves the whole value onto the regular-expression path, where it is tested
    unanchored and matches more events than it names. A stray `.` or `*` would do it, and nothing
    about the resulting behaviour would look like a typo.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ,|")
    for groups in hook_entries(PYTHON).values():
        for group in groups:
            assert set(group["matcher"]) <= allowed


def test_interpreter_path_is_passed_as_argv_not_concatenated() -> None:
    """A Windows interpreter path contains a space; a shell would re-split a joined string."""
    handler = hook_entries(PYTHON)["SessionStart"][0]["hooks"][0]
    assert handler["command"] == PYTHON
    assert handler["args"][:2] == ["-m", "recall_hooks"]


def test_uninstall_removes_only_our_handlers(tmp_path: Path, monkeypatch: Any) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(merge_hooks(_user_settings(), hook_entries(PYTHON))), encoding="utf-8"
    )
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)
    monkeypatch.setattr(claude_code, "hook_config_path", lambda: tmp_path / "recall-hook.json")

    uninstall(path=settings, print_fn=lambda *a, **k: None)

    after = json.loads(settings.read_text(encoding="utf-8"))
    assert after["permissions"] == {"allow": ["Bash(git *)"]}
    remaining = [h["command"] for g in after["hooks"]["SessionStart"] for h in g["hooks"]]
    assert remaining == ["mine.sh"]
    assert "SessionEnd" not in after["hooks"]


def test_uninstall_leaves_no_empty_hooks_key_behind(tmp_path: Path, monkeypatch: Any) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(merge_hooks({}, hook_entries(PYTHON))), encoding="utf-8")
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)
    monkeypatch.setattr(claude_code, "hook_config_path", lambda: tmp_path / "recall-hook.json")

    uninstall(path=settings, print_fn=lambda *a, **k: None)

    assert json.loads(settings.read_text(encoding="utf-8")) == {}


def test_install_backs_up_before_editing(tmp_path: Path, monkeypatch: Any) -> None:
    settings = tmp_path / "settings.json"
    original = json.dumps(_user_settings())
    settings.write_text(original, encoding="utf-8")
    monkeypatch.setattr(claude_code, "hook_config_path", lambda: tmp_path / "recall-hook.json")
    monkeypatch.setattr(claude_code, "refresh_stats", lambda config: 0)

    claude_code.install_hooks(dsn="postgresql://u:p@h/db", path=settings, print_fn=lambda *a: None)

    backups = list(tmp_path.glob("settings.json.recall-backup-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original


def test_the_user_scope_file_is_the_sibling_of_the_config_home_not_a_child(
    monkeypatch: Any,
) -> None:
    """`~/.claude.json` is beside `~/.claude/`, not inside it.

    Getting this wrong writes `~/.claude/.claude.json`, a file the client never reads, so the
    no-CLI fallback would report a successful registration and register nothing. It would have
    failed only on machines without `claude` on PATH, which is the case the fallback exists for.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert claude_code.user_config_file() == Path.home() / ".claude.json"
    assert claude_code.user_config_file() != claude_code.claude_config_home() / ".claude.json"
    assert claude_code.user_config_file().parent == claude_code.claude_config_home().parent


def test_the_user_scope_file_follows_an_overridden_config_home(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert claude_code.user_config_file() == tmp_path / ".claude.json"


def test_the_fallback_writes_a_user_scope_server_where_the_client_reads_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Top level `mcpServers` is user scope. Under `projects[dir]` it would be local scope."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"projects": {"C:/somewhere": {"allowedTools": []}}}), "utf-8")
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)

    claude_code.register_mcp_server(
        dsn="postgresql://u:p@127.0.0.1:5432/recall", print_fn=lambda *a, **k: None
    )

    written = json.loads(config.read_text(encoding="utf-8"))
    server = written["mcpServers"]["recall"]
    assert server["args"] == ["-m", "recall_mcp.server"]
    assert server["env"]["RECALL_TRUST_MODE"] == "development"
    # The user's other keys are not ours to drop.
    assert written["projects"] == {"C:/somewhere": {"allowedTools": []}}


def test_a_password_never_reaches_a_log_line() -> None:
    assert _redacted("postgresql://recall:hunter2@127.0.0.1:5432/recall") == (
        "postgresql://recall:***@127.0.0.1:5432/recall"
    )
    assert "hunter2" not in _redacted("postgresql://recall:hunter2@h/db")
    # Shapes that carry no password must survive unchanged rather than be mangled into one.
    assert _redacted("postgresql://127.0.0.1:5432/recall") == "postgresql://127.0.0.1:5432/recall"


def test_server_env_always_sets_the_trust_mode() -> None:
    """Omitting it is how a correct install returns INDEX_NOT_READY on the user's first search."""
    env = claude_code.server_env(dsn="postgresql://h/db", tenant="default", trust_mode="development")
    assert env["RECALL_TRUST_MODE"] == "development"
