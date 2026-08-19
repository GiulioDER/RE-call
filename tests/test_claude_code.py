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


def test_precompact_is_registered_async_and_never_blocking() -> None:
    """Exit code 2 on `PreCompact` BLOCKS compaction.

    A memory tool that can wedge a session whose context window is already full is worse than one
    that occasionally misses an index, so the handler is async and its entry point returns 0 on
    every path. Async also keeps a cold embedder from delaying a compaction the user is waiting on.
    """
    group = hook_entries(PYTHON)["PreCompact"][0]
    handler = group["hooks"][0]
    assert group["matcher"] == "manual|auto"
    assert handler["async"] is True
    assert handler["args"] == ["-m", "recall_hooks", "pre-compact"]


def test_session_start_reinjects_after_a_compaction() -> None:
    """Reverses an earlier exclusion of the `compact` matcher.

    The original reasoning was that re-injecting mid-conversation restates what was already said.
    A compaction is precisely the event that may have discarded it, and it is the only moment the
    client offers to put it back, since neither compaction hook supports `additionalContext`.
    """
    matcher = hook_entries(PYTHON)["SessionStart"][0]["matcher"]
    assert "compact" in matcher.split("|")
    assert "fork" not in matcher.split("|"), "a fork inherits its parent's context"


def test_matchers_use_only_documented_values() -> None:
    documented = {
        "SessionStart": {"startup", "resume", "clear", "compact", "fork"},
        "SessionEnd": {"clear", "resume", "logout", "prompt_input_exit", "other"},
        "PreCompact": {"manual", "auto"},
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


def _fallback_register(tmp_path: Path, monkeypatch: Any, **kwargs: Any) -> dict[str, Any]:
    """Drive the no-CLI fallback and return the resulting `.claude.json`."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"projects": {"C:/somewhere": {"allowedTools": []}}}), "utf-8")
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)
    claude_code.register_mcp_server(
        dsn="postgresql://u:p@127.0.0.1:5432/recall", print_fn=lambda *a, **k: None, **kwargs
    )
    return json.loads(config.read_text(encoding="utf-8"))


def test_the_fallback_defaults_to_local_scope_under_the_project_key(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Local scope is `projects[<dir>].mcpServers`, keyed by the project's own absolute path.

    The corpus boundary is why this is the default: a recall server carries one tenant and one
    DSN, so at user scope it would follow the user into every unrelated checkout and answer about
    somewhere else's corpus, confidently and without erroring.
    """
    project = tmp_path / "proj"
    project.mkdir()
    written = _fallback_register(tmp_path, monkeypatch, project_root=project)

    server = written["projects"][str(project.resolve())]["mcpServers"]["recall"]
    assert server["args"] == ["-m", "recall_mcp.server"]
    assert server["env"]["RECALL_TRUST_MODE"] == "development"
    assert "mcpServers" not in written, "local scope must not write the user-scope key"
    # The user's other projects are not ours to touch.
    assert written["projects"]["C:/somewhere"] == {"allowedTools": []}


def test_user_scope_is_still_available_and_writes_the_top_level_key(
    tmp_path: Path, monkeypatch: Any
) -> None:
    written = _fallback_register(tmp_path, monkeypatch, scope="user")
    assert written["mcpServers"]["recall"]["args"] == ["-m", "recall_mcp.server"]
    assert "mcpServers" not in written["projects"]["C:/somewhere"]


def test_every_spelling_of_the_project_directory_is_registered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The client keeps several keys for one directory and normalises none of them.

    Measured on a real config: 313 project keys, 7 directories carrying two spellings each, one
    from a native launch and one from Git Bash. An entry under one spelling is invisible to a
    session launched the other way, with no error, which looks exactly like a failed install.
    """
    project = tmp_path / "proj"
    project.mkdir()
    native = str(project)
    posix = project.as_posix()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    config = tmp_path / ".claude.json"
    config.write_text(
        json.dumps({"projects": {native: {"allowedTools": []}, posix: {"allowedTools": []}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)

    claude_code.register_mcp_server(
        dsn="postgresql://h/db", project_root=project, print_fn=lambda *a, **k: None
    )

    written = json.loads(config.read_text(encoding="utf-8"))
    registered = [k for k, v in written["projects"].items() if "mcpServers" in v]
    assert len(registered) == 2, f"both spellings must be registered, got {registered}"
    # And neither spelling lost the keys it already had.
    assert all(written["projects"][k]["allowedTools"] == [] for k in (native, posix))


def test_a_project_the_client_has_never_seen_gets_exactly_one_key(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Inventing a key is right when there is none; inventing a second one is not."""
    project = tmp_path / "fresh"
    project.mkdir()
    written = _fallback_register(tmp_path, monkeypatch, project_root=project)
    registered = [k for k, v in written["projects"].items() if "mcpServers" in v]
    assert registered == [str(project.resolve())]


def test_the_cli_is_never_decoded_with_the_console_codec() -> None:
    """`text=True` decodes with cp1252 on Windows and this CLI emits bytes it cannot represent.

    Measured here: `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f`. The failure is
    worse than a raise, because it happens in a `subprocess` reader thread: nothing propagates,
    `stdout` arrives as None, and an error path reporting `stderr or stdout or ""` reports an
    empty reason for a real failure.
    """
    captured: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    import subprocess as subprocess_module

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(claude_code, "_claude_cli", lambda: "claude")
        monkeypatch.setattr(subprocess_module, "run", fake_run)
        claude_code._run_claude(["mcp", "list"])
    finally:
        monkeypatch.undo()

    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"
    assert "text" not in captured, "text=True would decode with the console codec"


def test_an_unknown_scope_is_refused_rather_than_guessed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(claude_code, "_claude_cli", lambda: None)
    try:
        claude_code.register_mcp_server(dsn="postgresql://h/db", scope="project")
    except ValueError as exc:
        assert "local" in str(exc) and "user" in str(exc)
    else:  # pragma: no cover - the assertion below is the failure message
        raise AssertionError("an unrecognised scope must raise rather than silently pick one")


def test_every_cli_call_is_made_from_the_project_root(tmp_path: Path, monkeypatch: Any) -> None:
    """`--scope local` and `mcp get` resolve the project from the working directory.

    Called from wherever the wizard was started, the entry lands under a directory nobody will
    open: the command succeeds, the entry exists, and the tools never appear.
    """
    project = tmp_path / "proj"
    project.mkdir()
    calls: list[tuple[list[str], Any]] = []

    def fake_run(args: list[str], *, cwd: Any = None, timeout: float = 30.0) -> Any:
        calls.append((args, cwd))

        class Result:
            returncode = 1 if args[:2] == ["mcp", "get"] else 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(claude_code, "_claude_cli", lambda: "claude")
    monkeypatch.setattr(claude_code, "_run_claude", fake_run)

    claude_code.register_mcp_server(
        dsn="postgresql://h/db",
        project_root=project,
        prefer_cli=True,
        print_fn=lambda *a, **k: None,
    )

    assert calls, "the CLI path must have been taken"
    assert all(cwd == project.resolve() for _args, cwd in calls)
    add = next(args for args, _ in calls if args[:2] == ["mcp", "add"])
    assert add[:4] == ["mcp", "add", "--scope", "local"]


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
