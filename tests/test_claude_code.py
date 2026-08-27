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

import pytest

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


@pytest.fixture(autouse=True)
def never_touch_the_real_client_config(tmp_path, monkeypatch):
    """Pin the client config somewhere disposable for EVERY test in this module.

    Not a precaution. `test_every_cli_call_is_made_from_the_project_root` wrote a real local-scope
    `recall` entry into the developer's own `~/.claude.json`, pointing at a pytest temp directory,
    and it survived there until another session's snapshot found it.

    The mechanism is worth stating because no individual test looks wrong. That test pinned nothing
    because it exercised the CLI arm, which is faked and writes no file. Flipping the primary path
    to the direct merge meant it stopped taking that arm, fell through to `_write_server_entry`,
    and `user_config_file()` resolved to the real home. A test that had been safe became unsafe
    because the code under it changed branches, which is exactly the case a per-test opt-in cannot
    cover.

    `CLAUDE_CONFIG_DIR` rather than a patched `Path.home`, because a patched `Path.home` does not
    survive into a subprocess: anything that actually shells out to `claude` resolves the real home
    itself. An environment variable is inherited. Raised by the wizard session, whose own conftest
    guard has the same limit.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "client-config"))


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
    # ⚠️ Two different kinds of value live in this table. A lifecycle event's matcher names the
    # REASON it fired (`startup`, `manual`); a tool event's matcher names the TOOLS it applies to.
    # Listing them together is deliberate: the property under test is that no matcher can name
    # something the client will never send, and for PreToolUse that means a tool that does not
    # exist, which would make the hook silently inert.
    documented = {
        "SessionStart": {"startup", "resume", "clear", "compact", "fork"},
        "SessionEnd": {"clear", "resume", "logout", "prompt_input_exit", "other"},
        "PreCompact": {"manual", "auto"},
        "PreToolUse": {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"},
    }
    assert set(hook_entries(PYTHON)) <= set(documented), (
        "a new hook event was added without documenting the matcher values it may use, so this "
        "test would skip it silently"
    )
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


def test_a_password_never_reaches_a_log_line() -> None:
    assert _redacted("postgresql://recall:hunter2@127.0.0.1:5432/recall") == (
        "postgresql://recall:***@127.0.0.1:5432/recall"
    )
    assert "hunter2" not in _redacted("postgresql://recall:hunter2@h/db")
    # Shapes that carry no password must survive unchanged rather than be mangled into one.
    assert _redacted("postgresql://127.0.0.1:5432/recall") == "postgresql://127.0.0.1:5432/recall"


def test_server_env_carries_the_embedder_and_a_non_default_table() -> None:
    """⛔ **A same-width different embedder does not raise; it returns a confident wrong answer.**

    `recall setup` asks which embedder and writes it to `.env`. `recall_mcp.server` never calls
    `load_dotenv`, so an embedder recorded only there silently falls back to fastembed. This
    project's own CLAUDE.md records the consequence: three 1024-dimension models were queried
    against each other's rows and nothing raised, because pgvector computes a cosine between any
    two same-width vectors happily.

    The second half pins the DELIBERATE omission. `RECALL_TABLE` is emitted only when it differs
    from the default, so an existing registration is byte-identical to before this change; without
    an assertion that says so, "absent because default" and "absent because broken" look the same.
    """
    from recall.store import DEFAULT_TABLE

    rich = claude_code.server_env(
        dsn="postgresql://example/recall",
        tenant="acme",
        trust_mode="strict",
        table="quickstart_chunks",
        embedder="voyage:voyage-3",
    )
    assert rich["RECALL_EMBEDDER"] == "voyage:voyage-3"
    assert rich["RECALL_TABLE"] == "quickstart_chunks"

    plain = claude_code.server_env(
        dsn="postgresql://example/recall", tenant="acme", trust_mode="strict"
    )
    assert "RECALL_TABLE" not in plain, "the default table must not be written into the env block"
    assert "RECALL_EMBEDDER" not in plain
    assert set(plain) == {"RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_TRUST_MODE"}
    assert DEFAULT_TABLE == "chunks"


def test_server_env_always_sets_the_trust_mode() -> None:
    """Omitting it is how a correct install returns INDEX_NOT_READY on the user's first search."""
    env = claude_code.server_env(dsn="postgresql://h/db", tenant="default", trust_mode="development")
    assert env["RECALL_TRUST_MODE"] == "development"


# ------------------------------------------------------------------------------------------------
# What this module still decides, now that `recall.wizard.wiring` owns the mechanism.
#
# The writer, the project-key resolution and the platform case rule are tested in
# `tests/test_wizard_wiring.py`, against the code that performs them. Re-testing them here would
# assert a second time that somebody else's function works. What is left is this layer's own
# contribution: one block, the right environment on it, and a report the user can act on.
# ------------------------------------------------------------------------------------------------


class _Registration:
    """Stand-in for `LocalScopeRegistration`, so these tests do not touch a config file."""

    def __init__(self, **kwargs: Any) -> None:
        self.config_path = kwargs.get("config_path", Path("/somewhere/.claude.json"))
        self.project_keys = kwargs.get("project_keys", ())
        self.registered = kwargs.get("registered", ())
        self.conflicts = kwargs.get("conflicts", ())
        self.skipped_reason = kwargs.get("skipped_reason", "")


def _capture(monkeypatch: Any, result: _Registration) -> tuple[list[Any], list[str]]:
    """Register against a faked mechanism, returning the blocks it was given and what was printed."""
    seen: list[Any] = []
    printed: list[str] = []

    def fake_register(blocks: Any, **kwargs: Any) -> _Registration:
        seen.append((blocks, kwargs))
        return result

    import recall.wizard.wiring as wiring

    monkeypatch.setattr(wiring, "register_local_scope", fake_register)
    claude_code.register_mcp_server(
        dsn="postgresql://recall:hunter2@127.0.0.1:5432/recall",
        tenant="default",
        project_root=Path.cwd(),
        print_fn=lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    )
    return seen, printed


def test_one_logical_server_is_passed_as_a_one_element_tuple_of_blocks(monkeypatch: Any) -> None:
    """A scalar door into the mechanism would be a second shape kept alive for one caller."""
    seen, _ = _capture(monkeypatch, _Registration(registered=("recall",)))
    blocks, kwargs = seen[0]
    assert len(blocks) == 1
    assert blocks[0].name == "recall"
    assert blocks[0].tenant == "default"
    assert kwargs["project_root"] == Path.cwd()
    assert kwargs["interpreter"], "the interpreter must be named, not left to the client's PATH"


def test_the_block_carries_the_trust_mode(monkeypatch: Any) -> None:
    """Omitting it is the likeliest way for a correct install to look broken on the first search.

    A stdio server launched with an explicit `env` inherits nothing, and a fresh corpus is
    uncalibrated, which a strict server correctly refuses with `INDEX_NOT_READY`.
    """
    seen, _ = _capture(monkeypatch, _Registration(registered=("recall",)))
    env = seen[0][0][0].env
    assert env["RECALL_TRUST_MODE"] == "development"
    assert env["RECALL_TENANT"] == "default"
    assert env["RECALL_SERVING_DSN"].endswith("/recall")


def test_the_registered_keys_are_reported_because_the_entry_is_path_keyed(
    monkeypatch: Any,
) -> None:
    """Moving or renaming the project orphans the entry silently, so the keys are printed."""
    result = _Registration(registered=("recall",), project_keys=("C:/a/proj", "C:/A/proj"))
    _, printed = _capture(monkeypatch, result)
    assert any("C:/a/proj" in line and "C:/A/proj" in line for line in printed)


def test_a_conflict_is_reported_and_not_claimed_as_a_success(monkeypatch: Any) -> None:
    """Silently repointing an existing install at another corpus is worse than declining."""
    result = _Registration(registered=(), conflicts=(("recall", "another corpus"),))
    _, printed = _capture(monkeypatch, result)
    assert any("already exists" in line and "another corpus" in line for line in printed)
    assert not any("Registered MCP server" in line for line in printed)


def test_a_skip_is_reported_with_its_reason(monkeypatch: Any) -> None:
    result = _Registration(skipped_reason="no project root")
    _, printed = _capture(monkeypatch, result)
    assert any("not registered" in line and "no project root" in line for line in printed)


def test_the_dsn_password_never_reaches_the_report(monkeypatch: Any) -> None:
    result = _Registration(registered=("recall",), project_keys=("C:/a/proj",))
    _, printed = _capture(monkeypatch, result)
    assert printed, "the success path must report something"
    assert not any("hunter2" in line for line in printed)


def test_registration_actually_reaches_the_config_file(tmp_path: Path, monkeypatch: Any) -> None:
    """The one test here that does NOT fake the mechanism, and the only one that catches this.

    Every other test in this section substitutes `register_local_scope`, which means none of them
    can tell whether it was called correctly, only that it was called. The first version of the
    delegation passed all of them while writing nothing: `wiring.claude_config_path` hardcodes
    `Path.home()`, so with `CLAUDE_CONFIG_DIR` set the entry went somewhere the client does not
    read, and `register_mcp_server` reported success.

    Faking the collaborator is right for asserting what this layer decides. It cannot substitute
    for one path that goes all the way to the file.
    """
    config_dir = tmp_path / "cfg"
    project = tmp_path / "proj"
    config_dir.mkdir()
    project.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    (config_dir / ".claude.json").write_text(json.dumps({"projects": {}}), encoding="utf-8")

    printed: list[str] = []
    status = claude_code.register_mcp_server(
        dsn="postgresql://recall:hunter2@127.0.0.1:5432/recall",
        project_root=project,
        print_fn=lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    )

    document = json.loads((config_dir / ".claude.json").read_text(encoding="utf-8"))
    registered = {
        key: list(entry.get("mcpServers", {}))
        for key, entry in document["projects"].items()
        if entry.get("mcpServers")
    }
    assert status == "registered"
    assert registered, "reported success while writing nothing, which is the bug this test exists for"
    server = document["projects"][next(iter(registered))]["mcpServers"]["recall"]
    assert server["args"] == ["-m", "recall_mcp.server"]
    assert server["env"]["RECALL_TRUST_MODE"] == "development"
    assert not any("hunter2" in line for line in printed)

    claude_code.uninstall(
        path=tmp_path / "settings.json", project_root=project, print_fn=lambda *a, **k: None
    )
    after = json.loads((config_dir / ".claude.json").read_text(encoding="utf-8"))
    assert not any(e.get("mcpServers") for e in after["projects"].values())
