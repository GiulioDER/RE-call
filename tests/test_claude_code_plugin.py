"""The Claude Code plugin under `plugin/`, and the marketplace entry that distributes it.

A plugin is written once and shipped to every user, which is what makes these checks worth having:
none of the failures below are visible on the machine that authored the files. `claude plugin
validate` passes on all of them, because they are not schema errors. They are references that point
at nothing.

**`${user_config.KEY}` fails SILENTLY when KEY is misspelled.** The substitution resolves to an
empty string rather than erroring, so a typo in the manifest launches the MCP server with
`RECALL_SERVING_DSN=""`, and the user sees a server that starts and then cannot connect. That is
the same symptom as a wrong DSN, a stopped database and a missing pgvector extension, so it is
diagnosed last.

**The hooks name a console script that lives in a different file.** `hooks.json` invokes
`recall-hooks`, which exists only because `[project.scripts]` in `pyproject.toml` declares it.
Nothing links the two, so deleting or renaming the entry point leaves a plugin whose hooks fail on
every session start, on every machine, with the manifest still validating.

**The version is written in three places.** `recall.__version__`, the plugin manifest, and the
marketplace entry. A release that bumps one and not the others advertises a version nobody can
install, and marketplace entries are cached by users, so the wrong number persists.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import recall

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "plugin"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
HOOKS = PLUGIN / "hooks" / "hooks.json"
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"

#: The hook events the plugin subscribes to, and the subcommand each must pass. Hardcoded rather
#: than derived, because deriving them from `recall.claude_code.hook_entries` would let a change
#: there silently rewrite what this test considers correct, and the two are deliberately allowed to
#: differ: the installer knows the machine's interpreter and the plugin cannot.
EXPECTED_HOOKS = {
    "SessionStart": "session-start",
    "PreCompact": "pre-compact",
    "SessionEnd": "session-end",
}

#: The console script the plugin's hooks invoke.
HOOK_SCRIPT = "recall-hooks"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# The reference that fails silently.
# --------------------------------------------------------------------------------------------


def test_every_user_config_reference_names_a_declared_key() -> None:
    """⛔ A misspelled `${user_config.KEY}` resolves to an empty string, not to an error.

    So `RECALL_SERVING_DSN` would be set to `""` and the MCP server would start, then fail to
    connect, which is indistinguishable from four other causes the user is far more likely to
    suspect. This is the single check in this file most likely to catch a real mistake.
    """
    declared = set(_json(MANIFEST).get("userConfig", {}))
    referenced = set()
    for path in (MANIFEST, HOOKS):
        referenced |= set(re.findall(r"\$\{user_config\.([A-Za-z0-9_]+)\}", path.read_text("utf-8")))

    assert referenced, "no ${user_config.*} references found; the DSN is no longer being passed"
    undeclared = referenced - declared
    assert not undeclared, f"referenced but not declared in userConfig: {sorted(undeclared)}"


def test_declared_user_config_is_actually_used() -> None:
    """A key nobody references is a question asked of the user for no reason.

    Not merely untidy: `required: true` on an unused key blocks the install behind a prompt whose
    answer is discarded.
    """
    declared = set(_json(MANIFEST).get("userConfig", {}))
    referenced = set()
    for path in (MANIFEST, HOOKS):
        referenced |= set(re.findall(r"\$\{user_config\.([A-Za-z0-9_]+)\}", path.read_text("utf-8")))

    unused = declared - referenced
    assert not unused, f"declared in userConfig but never referenced: {sorted(unused)}"


def test_the_dsn_is_marked_sensitive() -> None:
    """⛔ The DSN carries a password, so it must not land in `~/.claude/settings.json`.

    `sensitive: true` routes it to the OS keychain or `.credentials.json` instead, and masks it in
    the prompt. Without the flag it is written in clear text to a file users routinely paste into
    issues.
    """
    dsn = _json(MANIFEST)["userConfig"]["serving_dsn"]
    assert dsn.get("sensitive") is True
    assert dsn.get("required") is True


# --------------------------------------------------------------------------------------------
# The reference that spans two files.
# --------------------------------------------------------------------------------------------


def test_hooks_invoke_a_console_script_that_pyproject_declares() -> None:
    """⚠️ `hooks.json` names `recall-hooks`; only `pyproject.toml` makes that name exist.

    Nothing else connects them. Renaming the entry point leaves a plugin that validates, installs,
    and fails on every session start for every user.
    """
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]["scripts"]
    assert HOOK_SCRIPT in scripts, f"{HOOK_SCRIPT} is not declared in [project.scripts]"

    module, _, attribute = scripts[HOOK_SCRIPT].partition(":")
    imported = __import__(module, fromlist=[attribute])
    assert callable(getattr(imported, attribute)), f"{scripts[HOOK_SCRIPT]} is not callable"


def test_hooks_cover_the_three_events_with_the_right_subcommand() -> None:
    """Each event has to pass the subcommand `recall_hooks.main` dispatches on.

    An unrecognised subcommand returns 0, so a typo here is a hook that runs, succeeds, and does
    nothing at all.
    """
    groups = _json(HOOKS)["hooks"]
    assert set(groups) == set(EXPECTED_HOOKS)

    for event, subcommand in EXPECTED_HOOKS.items():
        (entry,) = groups[event]
        (handler,) = entry["hooks"]
        assert handler["command"] == HOOK_SCRIPT
        assert handler["args"] == [subcommand], f"{event} passes {handler['args']}"


@pytest.mark.parametrize("event", sorted(EXPECTED_HOOKS))
def test_the_subcommand_is_one_the_hook_module_dispatches_on(event: str) -> None:
    """Asserted against the source of `recall_hooks.main`, not against a second list.

    `main` returns 0 for anything it does not recognise, so no runtime check can tell a working
    subcommand from a silently ignored one.
    """
    import inspect

    import recall_hooks

    source = inspect.getsource(recall_hooks.main)
    assert f'"{EXPECTED_HOOKS[event]}"' in source


def test_only_session_start_blocks() -> None:
    """⚠️ `PreCompact` exit code 2 BLOCKS compaction, so that hook must stay async.

    A memory tool that can wedge a session whose context window is already full is worse than no
    memory tool. `SessionEnd` is async because it cannot delay termination anyway, and a
    synchronous index there is a promise the client is not obliged to keep.
    """
    groups = _json(HOOKS)["hooks"]
    for event in ("PreCompact", "SessionEnd"):
        (handler,) = groups[event][0]["hooks"]
        assert handler.get("async") is True, f"{event} must not block"

    (start,) = groups["SessionStart"][0]["hooks"]
    assert start.get("async") is not True
    assert isinstance(start.get("timeout"), int), "a blocking hook needs a timeout"


# --------------------------------------------------------------------------------------------
# The version written in three places.
# --------------------------------------------------------------------------------------------


def test_plugin_and_marketplace_versions_track_the_package() -> None:
    entry = _json(MARKETPLACE)["plugins"][0]
    assert _json(MANIFEST)["version"] == recall.__version__
    assert entry["version"] == recall.__version__


def test_the_marketplace_entry_points_at_this_plugin() -> None:
    """A `source` path that does not resolve is an install that fails after the user opts in."""
    entry = _json(MARKETPLACE)["plugins"][0]
    assert entry["name"] == _json(MANIFEST)["name"]
    assert (REPO / entry["source"]).resolve() == PLUGIN.resolve()
    assert MANIFEST.is_file()


# --------------------------------------------------------------------------------------------
# Layout, which the validator does not check.
# --------------------------------------------------------------------------------------------


def test_component_directories_are_at_the_plugin_root() -> None:
    """⚠️ The documented common mistake: only `plugin.json` goes inside `.claude-plugin/`.

    A `skills/` or `hooks/` directory nested there is silently ignored, and the plugin installs
    looking healthy with none of its components loaded.
    """
    nested = MANIFEST.parent
    for component in ("skills", "hooks", "agents", "commands"):
        assert not (nested / component).exists(), f"{component}/ must not be inside .claude-plugin/"

    assert (PLUGIN / "hooks" / "hooks.json").is_file()
    assert (PLUGIN / "skills").is_dir()


def test_every_skill_has_a_description_so_the_model_can_invoke_it() -> None:
    """A skill with no `description` is one Claude never reaches for on its own."""
    skills = list((PLUGIN / "skills").glob("*/SKILL.md"))
    assert skills, "the plugin ships no skills"

    for skill in skills:
        text = skill.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{skill.name} has no frontmatter"
        frontmatter = text.split("---", 2)[1]
        assert re.search(r"^description:\s*\S", frontmatter, re.M), f"{skill.parent.name} has none"


def test_the_session_commands_are_shipped() -> None:
    """The half of the loop the hooks cannot do.

    `SessionEnd` and `PreCompact` index whatever is in the memory store, and nothing makes
    anything be there. `SessionStart` carries no user prompt, so it can announce the corpus but
    cannot search it. Both gaps need a command, because a command is the only surface here that
    has the user's own words to work from.
    """
    for name in ("session-open", "session-close"):
        assert (PLUGIN / "commands" / f"{name}.md").is_file(), f"{name} is not shipped"


def test_every_command_has_a_description_so_it_appears_in_the_menu() -> None:
    """Same failure as a skill with no description: it exists and nobody finds it."""
    commands = sorted((PLUGIN / "commands").glob("*.md"))
    assert commands, "the plugin ships no commands"

    for command in commands:
        text = command.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{command.name} has no frontmatter"
        frontmatter = text.split("---", 2)[1]
        assert re.search(r"^description:\s*\S", frontmatter, re.M), f"{command.stem} has none"


def test_the_commands_only_name_tools_the_server_actually_exposes() -> None:
    """⚠️ A command naming a tool that does not exist fails at the worst possible moment.

    Nothing validates a prompt's references, so `recall_remember` would read perfectly, ship, and
    be discovered only when a user runs the command and the model reaches for a tool that is not
    there. The tool names are read off the server rather than listed here, so a rename on either
    side is caught from whichever side moves.
    """
    import ast
    import inspect

    from recall_mcp import server

    # Read the `@mcp.tool(name=...)` declarations rather than the module namespace: the tools are
    # defined inside a factory function, so they are not module attributes and `dir()` sees none
    # of them. Importing is not enough either, since registering them means constructing a server.
    tree = ast.parse(inspect.getsource(server))
    exposed = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "name"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
        and keyword.value.value.startswith("recall_")
    }
    assert "recall_search" in exposed, "the server surface could not be read"

    for command in sorted((PLUGIN / "commands").glob("*.md")):
        named = set(re.findall(r"`(recall_[a-z_]+)`", command.read_text(encoding="utf-8")))
        unknown = sorted(named - exposed)
        assert not unknown, f"{command.stem} names tools that do not exist: {unknown}"


def test_the_commands_teach_operation_vocabulary_rather_than_goal_vocabulary() -> None:
    """The one instruction in this product whose wording is measured rather than styled.

    Run `agent-ab-skill-001` (54 pairs, 2026-08-23) put the governing-memo rate at 0.674 against
    0.319 for the goal framing. A command that quietly reverts to "search before proposing an
    idea" would undo that without failing anything else.
    Pre-registration: `docs/preregistrations/2026-08-25-instruction-channel-unification.md`.
    """
    text = (PLUGIN / "commands" / "session-open.md").read_text(encoding="utf-8")

    assert "operations" in text
    assert "Do not search for the goal" in text
    assert "proposing an idea" not in text


def test_session_close_teaches_the_validity_keys_the_trust_layer_reads() -> None:
    """Measured 2026-08-19: zero of 152 memos in recall's own store declared any of them.

    A corpus whose memos carry no validity keys is one where the trust layer can only ever return
    `ok`, which is the shipped feature switched off. A close command that collects prose without
    the frontmatter would keep producing exactly that corpus.
    """
    text = (PLUGIN / "commands" / "session-close.md").read_text(encoding="utf-8")

    for key in ("valid_from", "valid_until", "supersedes"):
        assert key in text, f"session-close never mentions {key}"
    assert "do not edit it, and do not delete it" in text.lower()


def test_the_mcp_server_env_keys_are_ones_recall_actually_reads() -> None:
    """A renamed variable would leave the server on its defaults rather than erroring.

    `RECALL_TRUST_MODE` is the one that matters: unset, the server is strict, and a strict server
    refuses every query against the uncalibrated corpus `recall quickstart` builds. The user's
    first search returns nothing and they conclude the product does not work.
    """
    from recall.trust_policy import TrustPolicy

    env = _json(MANIFEST)["mcpServers"]["memory"]["env"]
    assert set(env) == {"RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_TRUST_MODE"}

    assert TrustPolicy.from_env({"RECALL_TRUST_MODE": "development"}).strict is False
    assert TrustPolicy.from_env({}).strict is True


def test_every_plugin_file_is_tracked_by_git() -> None:
    """⛔ The bug this exists for: a gitignore rule silently swallowed the plugin's `.mcp.json`.

    This repository ignores `.mcp.json` outright, because the generated one at the root carries
    bearer tokens and internal host addresses for a different project. That rule is a security
    control, and it matched `plugin/.mcp.json` as well. The file existed on disk, `claude plugin
    validate` passed, every test in this file passed, and it was absent from the commit: users
    would have installed a plugin with no MCP server.

    The fix was to move `mcpServers` inline into the manifest rather than add a
    `!plugin/.mcp.json` exception, because widening a secrets rule so a feature works is the wrong
    trade, and the exception would have covered any future file at that path too.

    Reading from disk is what every other test here does and is exactly what could not see this.
    Ask git instead.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "--", "plugin", ".claude-plugin"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.split()
    on_disk = {
        path.relative_to(REPO).as_posix()
        for path in list(PLUGIN.rglob("*")) + list((REPO / ".claude-plugin").rglob("*"))
        if path.is_file()
    }

    untracked = sorted(on_disk - set(tracked))
    assert not untracked, f"present on disk but not in git, so users never receive them: {untracked}"


def test_the_manifest_carries_the_mcp_server_itself() -> None:
    """Inline, because `plugin/.mcp.json` is unshippable here. See the test above."""
    assert not (PLUGIN / ".mcp.json").exists(), (
        "a plugin/.mcp.json is gitignored by this repository's secrets rule and would not ship"
    )
    assert "memory" in _json(MANIFEST)["mcpServers"]
