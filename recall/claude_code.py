"""Wire this machine's Claude Code into recall, so the next session uses it without being asked.

`recall setup` already scaffolds `CLAUDE.md` and `memory/`, which tells Claude *how* to use the
tools. It does not give Claude the tools. Everything between "the user installed recall" and
"Claude called `recall_search`" is currently a page of hand-edited JSON in
`docs/USING_WITH_CLAUDE.md`, and a step a user performs by hand is a step most users skip.

Two decisions here are load-bearing and were verified against the Claude Code documentation and
the 2.1.220 CLI rather than assumed:

**Local scope, and the reason is the corpus boundary rather than the approval gate.** A recall
server carries one `RECALL_TENANT` and one DSN. Registered at user scope it would load in every
project on the machine, so opening any unrelated checkout would put a server there that answers
about a corpus belonging to somewhere else. This project documents that as the worst failure
available here, precisely because it is not an error: it is a confident, well-formed answer about
the wrong repository. Local scope lives in the same `~/.claude.json` under
``projects[<dir>].mcpServers``, loads only in the project it was added to, and so matches the
boundary the tenant already implies.

Project scope is the one to avoid. It lives in `.mcp.json` inside the repository, which puts a DSN
somewhere it can be committed, and it is the only scope the approval prompt covers: "Claude Code
prompts for approval in interactive sessions before using project-scoped servers from `.mcp.json`
files."

⚠️ **Two things here are documentary rather than measured, and are recorded as such.** That local
scope sits outside the approval gate follows from the gate being about `.mcp.json` specifically,
and from the keys being named ``enabledMcpjsonServers``; no project on this machine carried a
local-scope entry to watch it work.

And what approval gates at all is unsettled. `docs/preregistrations/
2026-08-16-sessionstart-hook-mcp-ordering.md` established (#429, narrowed by #432) that **approval
is not necessary**: a session in a never-approved project received a full recall tool set. The
companion claim, that `resume` versus fresh `startup` is what separates the outcome, was
**withdrawn** by #432 once its two supporting rows turned out to be sessions that died on a 401
with empty tool deltas, which is an uninterpretable null rather than a measurement. So the state of
the art is one positive observation and an open question.

Nothing in this module depends on any of it having an answer, which is the point of choosing the
scope that raises neither question.

⚠️ A local entry is keyed by the project's path, so moving or renaming the project orphans it in
silence. Same shape as the memory store this project has already lost once to a directory rename.

**The hooks are what make it used rather than merely available.** Registering tools makes them
callable; a `SessionStart` hook that injects context makes them present in the first turn without
the user changing any habit. `SessionEnd` closes the loop by indexing what the session learned,
which is the difference between memory that compounds and memory that decays.

This module is the **installer** half only. The hooks it installs run out of `recall_hooks`, a
separate top-level package, because `recall/__init__.py` eagerly imports the calibration, evidence
and lineage modules and `python -m recall.anything` therefore costs about a second before any hook
code runs. That second would be charged to opening Claude, every time. The measurement and the
re-measure command are in `recall_hooks/__init__.py`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from recall.atomic_write import atomic_write_bytes
from recall_hooks import claude_config_home, config_path as hook_config_path, refresh_stats

#: The MCP server name registered with the client. Also the name `claude mcp get` is probed with.
SERVER_NAME = "recall"

#: Hook handlers are identified for upgrade and uninstall by this string appearing in their
#: `args`. JSON has no comments, so the delimited-block trick used for `.env` and `CLAUDE.md` is
#: unavailable; matching on the module we invoke is both stable across versions and impossible to
#: collide with a hand-written hook that is not ours.
HOOK_MODULE = "recall_hooks"

#: A session-start hook runs before the user's first turn, so its cost is felt every single time
#: Claude opens. The documented default timeout is 600 seconds, which for this event is a way to
#: make a broken database look like a hung client.
SESSION_START_TIMEOUT_SECONDS = 15

#: The commands that install the Claude Code plugin, exactly as `plugin/README.md` states them.
#: They are typed into Claude Code itself, not a shell, which is why the wizard prints them
#: rather than running them: there is nothing on this machine they could be executed against.
PLUGIN_INSTALL_LINES = (
    "/plugin marketplace add GiulioDER/RE-call",
    "/plugin install recall@re-call",
)

#: The one skill the plugin ships. The name is the directory name Claude Code loads it by, both
#: in `plugin/skills/` here and under the user's `~/.claude/skills/`.
SKILL_NAME = "check-memory-before-acting"


def client_config_path() -> Path:
    """Where Claude Code keeps `.claude.json`, honouring `CLAUDE_CONFIG_DIR`.

    ⚠️ `recall.wizard.wiring.claude_config_path` returns `Path.home() / ".claude.json"`
    unconditionally, so on a machine where the user has set `CLAUDE_CONFIG_DIR` it names a file the
    client does not read. Registration would report success and write somewhere nothing looks.
    Reported to the wizard session, which owns that function; until it moves, the path is resolved
    here and passed in explicitly rather than left to the default.

    The unset case delegates, so there is still one definition of "the usual place".
    """
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser() / ".claude.json"
    from recall.wizard.wiring import claude_config_path

    return claude_config_path()


def settings_path() -> Path:
    return claude_config_home() / "settings.json"


def _redacted(dsn: str) -> str:
    """A DSN safe to print. The password is the only part worth hiding and the only part shown."""
    if "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    if "@" not in rest:
        return dsn
    userinfo, _, host = rest.rpartition("@")
    user, sep, _password = userinfo.partition(":")
    return f"{scheme}://{user}{':***' if sep else ''}@{host}"


# --------------------------------------------------------------------------------------------
# MCP registration
# --------------------------------------------------------------------------------------------


def _claude_cli() -> str | None:
    """The `claude` executable, or None. `shutil.which` honours PATHEXT, so `.cmd` is found."""
    return shutil.which("claude")


def claude_code_detected() -> bool:
    """Is there a Claude Code on this machine worth wiring up?

    Either signal alone is enough. The CLI can be absent while the desktop app is installed, and
    the config directory exists as soon as the client has run once, which on Windows is the common
    shape: an app that was installed without ever putting `claude` on PATH.
    """
    return _claude_cli() is not None or claude_config_home().is_dir()


def server_env(*, dsn: str, tenant: str, trust_mode: str) -> dict[str, str]:
    """The environment the MCP server is launched with.

    `RECALL_TRUST_MODE` is not optional at install time and leaving it out is the single most
    likely way for a correct installation to look broken. A stdio server started with an explicit
    `env` block does not inherit exported shell variables, and a fresh corpus is uncalibrated and
    bound to no generation, which a strict server correctly refuses with `INDEX_NOT_READY`. The
    user's first search would return nothing and they would conclude the product does not work.
    Set it, and say so in the UI: an uncalibrated corpus is the honest starting state and
    calibration is the upgrade, not a footnote.
    """
    return {
        "RECALL_SERVING_DSN": dsn,
        "RECALL_TENANT": tenant,
        "RECALL_TRUST_MODE": trust_mode,
    }


def register_mcp_server(
    *,
    dsn: str,
    tenant: str = "default",
    trust_mode: str = "development",
    project_root: Path | None = None,
    python_executable: str | None = None,
    print_fn: Callable[..., None] = print,
) -> str:
    """Register the recall MCP server at local scope. Returns a short status for the caller.

    **This delegates. It used to carry its own writer and no longer does.** The mechanism is
    `recall.wizard.wiring.register_local_scope`, which the wizard already owned; this module kept a
    parallel implementation only for as long as that one was unmerged. Two implementations of a
    write into a user's client configuration is exactly the duplication this repository has paid
    for before, so the moment there was one to call, the copy went.

    What stays here is the decision and the reporting, which is all this layer ever had to
    contribute: one logical server for this project's tenant, and a printed record of the keys it
    landed under.

    Local scope, because a recall server carries one `RECALL_TENANT` and one DSN. At user scope it
    would follow the user into every unrelated checkout and answer about a corpus belonging to
    somewhere else, which is this project's documented worst case precisely because it never raises
    an error.
    """
    from recall.wizard.wiring import ServerBlock, register_local_scope

    root = Path(project_root or Path.cwd()).resolve()
    # A one-element tuple rather than a scalar path into the mechanism. `register_local_scope`
    # takes blocks because a wizard run configures one server per servable tenant, and adding a
    # single-server door would be a second shape to keep working for one caller's convenience.
    blocks = (
        ServerBlock(
            name=SERVER_NAME,
            tenant=tenant,
            env=server_env(dsn=dsn, tenant=tenant, trust_mode=trust_mode),
            rationale=(
                "registered by `recall setup` for this project, at local scope so the tenant's "
                "corpus cannot be served into an unrelated checkout"
            ),
        ),
    )

    result = register_local_scope(
        blocks,
        project_root=root,
        config_path=client_config_path(),
        interpreter=python_executable or sys.executable,
    )

    if result.skipped_reason:
        print_fn(f"MCP server '{SERVER_NAME}' not registered: {result.skipped_reason}")
        return "skipped"
    if not result.registered:
        # Names already present that this install refused to overwrite. Repointing a working
        # install at another corpus silently is worse than declining to.
        for name, points_at in result.conflicts:
            print_fn(f"MCP server '{name}' already exists, left pointing at {points_at}.")
        return "already-registered"

    keys = ", ".join(result.project_keys) or str(root)
    print_fn(
        f"Registered MCP server '{SERVER_NAME}' at local scope, DSN {_redacted(dsn)}, "
        f"in {result.config_path} keyed to {keys}"
    )
    return "registered"


# --------------------------------------------------------------------------------------------
# The plugin, and the user-level skill
# --------------------------------------------------------------------------------------------


def plugin_skill_source() -> Path | None:
    """The repository's copy of the skill, or None under an installed wheel.

    The wheel ships `recall`, `recall_hooks` and `recall_mcp` only (pyproject's
    `[tool.hatch.build.targets.wheel]`), so `plugin/` exists on disk for a checkout or an
    editable install and for nothing else. None rather than an exception, because a pip install
    is not a broken state: it is the case where the plugin itself is how the skill arrives, and
    the wizard still has the install lines to print.
    """
    source = Path(__file__).resolve().parent.parent / "plugin" / "skills" / SKILL_NAME / "SKILL.md"
    return source if source.is_file() else None


def user_skill_dir() -> Path:
    """Where Claude Code loads user-level skills from, honouring `CLAUDE_CONFIG_DIR`."""
    return claude_config_home() / "skills"


#: ⛔ The plugin's commands are deliberately NOT installed the way the skill is, and this is a
#: decision rather than an omission.
#:
#: A plugin's commands are namespaced by the plugin (`/recall:session-open`). A user-level command
#: is a bare file in `~/.claude/commands/`, so the same two files would install as `/session-open`
#: and `/session-close` and **silently replace whatever the user already had under those names**.
#: Those are ordinary names: the author's own machine carries unrelated commands at exactly both
#: of them, checked 2026-08-25. A memory tool that overwrites a person's session workflow to
#: advertise itself has done more damage than the feature is worth.
#:
#: Renaming them on the way in (`recall-session-open`) would avoid the collision and cost more
#: than it saves: the docs, the plugin and the README would then name a command that does not
#: exist for half the users. So commands arrive with the plugin or not at all, and the wizard
#: prints `PLUGIN_INSTALL_LINES` for anyone who wants them.
COMMANDS_ARE_PLUGIN_ONLY = True


def install_user_skill(
    source: Path,
    *,
    print_fn: Callable[..., None] = print,
) -> None:
    """Copy the skill into the user's skills directory, saying what actually happened.

    A single file, because the skill IS a single `SKILL.md` today; if it ever grows supporting
    files, `plugin_skill_source` and this copy both have to learn about them, and the test that
    resolves the real repository copy is what will notice.

    User level rather than project level on purpose: the caller offered this as the alternative
    to installing the plugin, and a user-level skill is the only form that follows the user into
    projects where `recall setup` was never run.
    """
    content = source.read_text(encoding="utf-8")
    dest = user_skill_dir() / SKILL_NAME / "SKILL.md"
    if dest.exists() and dest.read_text(encoding="utf-8") == content:
        print_fn(f"The {SKILL_NAME} skill at {dest} is already current, left unchanged.")
        return
    replaced = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(dest, content.encode("utf-8"))
    verb = "Replaced" if replaced else "Installed"
    print_fn(f"{verb} the {SKILL_NAME} skill at {dest}. It loads in every project's sessions.")


# --------------------------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------------------------


def hook_entries(python_executable: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """The two hook groups this installer owns, keyed by event name.

    `args` rather than a single command string: an absolute interpreter path on Windows routinely
    contains a space, and passing it as `command` with the module appended would hand the client a
    string that a shell re-splits in the wrong place.
    """
    python_executable = python_executable or sys.executable
    return {
        "SessionStart": [
            {
                # Valid values: startup, resume, clear, compact, fork. `fork` is excluded: a fork
                # inherits its parent's context, so the digest is already in it.
                #
                # `compact` was excluded here too, on the reasoning that a compaction is
                # mid-conversation so re-injecting would "spend context to say what was already
                # said". That was wrong, and in the direction that matters: a compaction is
                # precisely the event that may have discarded the standing instruction, so after
                # one it has NOT already been said. This is the only moment the client offers to
                # put it back, since `PreCompact` and `PostCompact` support no `additionalContext`
                # of their own. Inference from the documented matcher, not a measurement: what a
                # compaction actually keeps has not been observed here.
                #
                # A matcher containing only letters, digits, `_`, `-`, spaces, `,` and `|` is read
                # as a list of exact strings, NOT as a regular expression. That is why these stay
                # alphabetic: adding a `.` or a `*` would silently move the whole matcher onto the
                # regex path, where it is tested unanchored and would match more events than named.
                "matcher": "startup|resume|clear|compact",
                "hooks": [
                    {
                        "type": "command",
                        "command": python_executable,
                        "args": ["-m", HOOK_MODULE, "session-start"],
                        "timeout": SESSION_START_TIMEOUT_SECONDS,
                        "statusMessage": "Recalling project memory",
                    }
                ],
            }
        ],
        # Compaction is the failure this product was built for: the moment a long session loses
        # the detail behind its conclusions. Anything already written to `memory/` survives it, so
        # this closes the write-to-searchable gap here rather than at `SessionEnd`, which can be
        # hours away and is the wrong side of the turn that most needs the memo.
        "PreCompact": [
            {
                # Valid values: manual, auto. Both, because an auto compaction is the one the user
                # did not ask for and therefore the one whose context loss is unanticipated.
                "matcher": "manual|auto",
                "hooks": [
                    {
                        "type": "command",
                        "command": python_executable,
                        "args": ["-m", HOOK_MODULE, "pre-compact"],
                        # Async, and never blocking. Exit code 2 on this event BLOCKS compaction,
                        # which would let a memory tool wedge a session whose context window is
                        # already full. Async also means a cold embedder cannot delay the
                        # compaction the user is waiting on.
                        "async": True,
                        "statusMessage": "Saving memory before compaction",
                    }
                ],
            }
        ],
        "SessionEnd": [
            {
                "matcher": "clear|resume|logout|prompt_input_exit|other",
                "hooks": [
                    {
                        "type": "command",
                        "command": python_executable,
                        "args": ["-m", HOOK_MODULE, "session-end"],
                        # SessionEnd cannot block termination, so a synchronous index is a promise
                        # the client is not obliged to keep. Async says so honestly, and the next
                        # session's start reconciles anything that was cut short.
                        "async": True,
                        "statusMessage": "Indexing session into recall",
                    }
                ],
            }
        ],
    }


def _is_recall_handler(handler: dict[str, Any]) -> bool:
    if HOOK_MODULE in str(handler.get("command", "")):
        return True
    return any(HOOK_MODULE in str(arg) for arg in handler.get("args", []) or [])


def _strip_recall_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove our handlers from a user's hook list, preserving everything else.

    Handler-level rather than group-level, because a user may have added their own handler to the
    group we wrote. Dropping the whole group would delete their work; dropping our handler and
    leaving an empty group behind would accumulate noise, so empty groups go too.
    """
    kept: list[dict[str, Any]] = []
    for group in groups:
        handlers = [h for h in group.get("hooks", []) if not _is_recall_handler(h)]
        if not handlers:
            continue
        if len(handlers) == len(group.get("hooks", [])):
            kept.append(group)
        else:
            kept.append({**group, "hooks": handlers})
    return kept


def merge_hooks(
    settings: dict[str, Any],
    entries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Merge our hook groups into a settings document, idempotently.

    Pure and side-effect free so the merge can be tested without a filesystem, and so a caller can
    diff the result before committing to it.
    """
    # Deep copy: the input is the caller's live config document, not ours to mutate.
    merged: dict[str, Any] = json.loads(json.dumps(settings))
    hooks: dict[str, Any] = merged.setdefault("hooks", {})
    for event, groups in entries.items():
        hooks[event] = [*_strip_recall_groups(list(hooks.get(event, []))), *groups]
    return merged


def install_hooks(
    *,
    dsn: str,
    tenant: str = "default",
    embedder: str = "fastembed",
    python_executable: str | None = None,
    path: Path | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    """Write the hook config, then merge the hook entries into the client's settings."""
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    _write_hook_config(dsn=dsn, tenant=tenant, embedder=embedder)

    settings: dict[str, Any] = {}
    if target.exists():
        raw = target.read_text(encoding="utf-8")
        settings = json.loads(raw) if raw.strip() else {}
        _backup(target)

    _write_json(target, merge_hooks(settings, hook_entries(python_executable)))
    print_fn(f"Installed SessionStart and SessionEnd hooks in {target}")


def uninstall(
    *,
    path: Path | None = None,
    project_root: Path | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    """Remove exactly what was added, and nothing else.

    An installer that writes into files shared with every project on the machine owes the user a
    removal that is precise rather than approximate.

    The server is removed by editing the client's configuration rather than by shelling out to
    `claude mcp remove`. Two reasons, both measured: the CLI's output cannot be decoded under the
    console codec on Windows and the failure is silent (see the wizard's notes and `#428`), and
    `--scope local` resolves which project it means from the working directory, so a removal run
    from the wrong place would report success and remove nothing.

    `recall.wizard.wiring` owns registration and exports `claude_config_path`, so the file is
    located the same way it was written. It has no inverse operation to call, which is why this one
    is written out rather than delegated.
    """
    target = path or settings_path()
    if target.exists():
        raw = target.read_text(encoding="utf-8")
        settings: dict[str, Any] = json.loads(raw) if raw.strip() else {}
        hooks = settings.get("hooks", {})
        for event in list(hooks):
            remaining = _strip_recall_groups(list(hooks.get(event, [])))
            if remaining:
                hooks[event] = remaining
            else:
                del hooks[event]
        if not hooks:
            settings.pop("hooks", None)
        _backup(target)
        _write_json(target, settings)
        print_fn(f"Removed recall hooks from {target}")

    hook_config_path().unlink(missing_ok=True)

    config_file = client_config_path()
    if not config_file.exists():
        return
    raw = config_file.read_text(encoding="utf-8")
    document: dict[str, Any] = json.loads(raw) if raw.strip() else {}
    root = Path(project_root or Path.cwd()).resolve()
    removed: list[str] = []
    for key, entry in document.get("projects", {}).items():
        servers = entry.get("mcpServers") if isinstance(entry, dict) else None
        if not isinstance(servers, dict) or SERVER_NAME not in servers:
            continue
        # Resolved `Path` equality rather than a string compare, so every spelling of this project
        # that the client happens to hold is cleaned up, and a differently-cased directory on POSIX
        # is left alone.
        try:
            if Path(key).resolve() != root:
                continue
        except (OSError, ValueError):
            continue
        del servers[SERVER_NAME]
        if not servers:
            del entry["mcpServers"]
        removed.append(key)
    if removed:
        _backup(config_file)
        _write_json(config_file, document)
        print_fn(f"Removed MCP server '{SERVER_NAME}' from {len(removed)} key(s): " + ", ".join(removed))


def _write_hook_config(*, dsn: str, tenant: str, embedder: str, table: str = "chunks") -> None:
    path = hook_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"dsn": dsn, "tenant": tenant, "embedder": embedder, "table": table, "chunks": 0}
    _write_json(path, config)
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ACLs do not model the POSIX bits, and a failure here must not abort an install
        # that has otherwise succeeded. The file still sits inside the user's own profile.
        pass
    # Seed the cached count now, so the very first session start has a real number to report
    # rather than staying silent until a session has ended. Failure here is a zero, not an error:
    # `refresh_stats` swallows an unreachable database on purpose.
    refresh_stats(config)


def _backup(path: Path) -> Path:
    """Copy a file the client owns before editing it, once per second at worst."""
    backup = path.with_name(f"{path.name}.recall-backup-{int(time.time())}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _write_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, payload.encode("utf-8"))
