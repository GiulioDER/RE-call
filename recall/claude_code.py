"""Wire this machine's Claude Code into recall, so the next session uses it without being asked.

`recall setup` already scaffolds `CLAUDE.md` and `memory/`, which tells Claude *how* to use the
tools. It does not give Claude the tools. Everything between "the user installed recall" and
"Claude called `recall_search`" is currently a page of hand-edited JSON in
`docs/USING_WITH_CLAUDE.md`, and a step a user performs by hand is a step most users skip.

Two decisions here are load-bearing and were verified against the Claude Code documentation and
the 2.1.220 CLI rather than assumed:

**User scope, never project scope.** A project-scoped server lives in `.mcp.json` and, quoting the
docs, "Claude Code prompts for approval in interactive sessions before using project-scoped servers
from `.mcp.json` files." A user-scoped server has no approval step, is the only scope that loads in
every project rather than one, and keeps the DSN out of the user's repository.

⚠️ **What that approval actually gates is not settled, and nothing here depends on it.** An earlier
version of this docstring asserted that a server stays silently absent until the approval is
recorded under ``projects[<dir>].enabledMcpjsonServers``. `docs/preregistrations/
2026-08-16-sessionstart-hook-mcp-ordering.md` (#429) then measured the opposite on this machine:
two sessions holding both the file and a recorded approval received no recall tools, while a
session in a never-approved project received the full set. What separates every row in that corpus
is `resume` versus fresh `startup`, confounded with date. Since v2.1.196 there is also a workspace
trust gate, because "a cloned repository can't approve its own servers".

So the case for user scope is that it has no approval step to reason about, not that project scope
is known to be blocked by one. It avoids a question this repository has not yet answered rather
than betting on a particular answer.

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
import subprocess
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


def settings_path() -> Path:
    return claude_config_home() / "settings.json"


def user_config_file() -> Path:
    """`.claude.json`, which holds local- and user-scoped MCP servers.

    NOT inside `claude_config_home()`. The default config home is `~/.claude/`, a directory, while
    this file is `~/.claude.json`, its sibling. Writing `~/.claude/.claude.json` creates a file the
    client never reads, so the fallback below would report a successful registration and register
    nothing. That is the failure this function exists to prevent, and it would have landed only on
    machines without `claude` on PATH, which is the Windows case this whole feature is for.
    """
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser() / ".claude.json"
    return Path.home() / ".claude.json"


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


def _run_claude(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    executable = _claude_cli()
    if executable is None:  # pragma: no cover - guarded by the caller
        raise FileNotFoundError("claude")
    return subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


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
    python_executable: str | None = None,
    replace: bool = False,
    print_fn: Callable[..., None] = print,
) -> str:
    """Register the recall MCP server at user scope. Returns a short status for the caller to log.

    Prefers the CLI, because `~/.claude.json` is a large file whose schema the client owns and
    whose other keys are none of our business. Falls back to an atomic merge when `claude` is not
    on PATH, which on Windows is common enough to be the normal case rather than the edge one.
    """
    python_executable = python_executable or sys.executable
    env = server_env(dsn=dsn, tenant=tenant, trust_mode=trust_mode)

    if _claude_cli() is None:
        _write_user_scope_server(env=env, python_executable=python_executable)
        print_fn(
            f"Registered MCP server '{SERVER_NAME}' by writing {user_config_file()} "
            "directly: the `claude` CLI is not on PATH."
        )
        return "written-directly"

    existing = _run_claude(["mcp", "get", SERVER_NAME])
    if existing.returncode == 0:
        if not replace:
            print_fn(
                f"MCP server '{SERVER_NAME}' is already registered, left unchanged. "
                "Re-run with replace=True to point it at this installation."
            )
            return "already-registered"
        # `claude mcp add` refuses a duplicate name at the same scope with "already exists in
        # local config", so a replace is a remove followed by an add rather than an overwrite.
        _run_claude(["mcp", "remove", "--scope", "user", SERVER_NAME])

    args = ["mcp", "add", "--scope", "user", SERVER_NAME]
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    # Everything after `--` is passed to the server untouched, which is what keeps a Windows
    # interpreter path containing spaces from being re-split by the client.
    args.extend(["--", python_executable, "-m", "recall_mcp.server"])

    result = _run_claude(args)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"claude mcp add failed: {message}")

    print_fn(f"Registered MCP server '{SERVER_NAME}' at user scope, DSN {_redacted(dsn)}.")
    return "registered"


def _write_user_scope_server(*, env: dict[str, str], python_executable: str) -> None:
    """Fallback path: user scope lives at the top level of `~/.claude.json` under `mcpServers`.

    Distinct from local scope, which the same file holds under `projects[<dir>].mcpServers` and
    which loads in one project only.
    """
    config_file = user_config_file()
    document: dict[str, Any] = {}
    if config_file.exists():
        raw = config_file.read_text(encoding="utf-8")
        document = json.loads(raw) if raw.strip() else {}
        _backup(config_file)
    servers = document.setdefault("mcpServers", {})
    servers[SERVER_NAME] = {
        "type": "stdio",
        "command": python_executable,
        "args": ["-m", "recall_mcp.server"],
        "env": dict(env),
    }
    _write_json(config_file, document)


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
                # Valid values: startup, resume, clear, compact, fork. `compact` is deliberately
                # excluded, since a compaction is mid-conversation and re-injecting the standing
                # instruction there spends context to say what was already said.
                #
                # A matcher containing only letters, digits, `_`, `-`, spaces, `,` and `|` is read
                # as a list of exact strings, NOT as a regular expression. That is why these stay
                # alphabetic: adding a `.` or a `*` would silently move the whole matcher onto the
                # regex path, where it is tested unanchored and would match more events than named.
                "matcher": "startup|resume|clear",
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


def uninstall(*, path: Path | None = None, print_fn: Callable[..., None] = print) -> None:
    """Remove exactly what was added, and nothing else.

    An installer that writes into a file shared with every project on the machine owes the user a
    removal that is precise rather than approximate.
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

    if _claude_cli() is not None:
        _run_claude(["mcp", "remove", "--scope", "user", SERVER_NAME])
        print_fn(f"Removed MCP server '{SERVER_NAME}'")


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
