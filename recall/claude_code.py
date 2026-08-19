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


def _run_claude(
    args: list[str], *, cwd: Path | None = None, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run the CLI, optionally from a given directory.

    `cwd` is not a convenience. `--scope local` and `mcp get` both resolve which project they mean
    from the working directory, so a call made from wherever the wizard happened to be started
    would register under the wrong project key. That is a silent miss: the command succeeds, the
    entry exists, and it belongs to a directory nobody will open.
    """
    executable = _claude_cli()
    if executable is None:  # pragma: no cover - guarded by the caller
        raise FileNotFoundError("claude")
    return subprocess.run(
        [executable, *args],
        capture_output=True,
        # NOT `text=True`. That decodes with the console's preferred codec, which on Windows is
        # cp1252, and this CLI emits bytes it cannot represent. Measured on this machine:
        # `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f`. The failure is nastier
        # than a raise, because it happens inside `subprocess`'s reader THREAD: the exception
        # never reaches the caller, `stdout` arrives as None, and an error path that reports
        # `result.stderr or result.stdout or ""` reports an empty reason for a real failure.
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        cwd=str(cwd) if cwd is not None else None,
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
    scope: str = "local",
    project_root: Path | None = None,
    python_executable: str | None = None,
    replace: bool = False,
    prefer_cli: bool = False,
    print_fn: Callable[..., None] = print,
) -> str:
    """Register the recall MCP server. Returns a short status for the caller to log.

    **Local scope by default, and the reason is the corpus boundary rather than the approval
    gate.** A recall server carries one `RECALL_TENANT` and one DSN, so at user scope it follows
    the user into every unrelated checkout and answers about a corpus that is not the repository
    they are in. Local scope lives in the same file under `projects[<dir>].mcpServers` and loads
    only in the project it was added to, which is the boundary the tenant already implies.

    **The direct merge is the primary path and `claude mcp add` is the option**, which reverses
    what this function did first. Deferring to the client's own writer is the better instinct and
    it loses here on two measurements. Its output is undecodable under the console codec on
    Windows, the platform this feature exists for, and the failure is silent rather than loud (see
    `_run_claude`). And it writes only the single project key matching its working directory, while
    the client itself keeps several spellings of one directory (see `_project_keys`). Pass
    `prefer_cli=True` to use it anyway.

    ⚠️ A local entry is keyed by the project's path, so moving or renaming the project orphans it
    in silence. The keys written are printed for that reason.
    """
    if scope not in {"local", "user"}:
        raise ValueError(f"scope must be 'local' or 'user', not {scope!r}")
    python_executable = python_executable or sys.executable
    root = Path(project_root or Path.cwd()).resolve()
    env = server_env(dsn=dsn, tenant=tenant, trust_mode=trust_mode)
    where = f"project {root}" if scope == "local" else "every project on this machine"

    if prefer_cli and _claude_cli() is not None:
        return _register_via_cli(
            env=env,
            python_executable=python_executable,
            scope=scope,
            root=root,
            where=where,
            dsn=dsn,
            replace=replace,
            print_fn=print_fn,
        )

    keys = _write_server_entry(
        env=env, python_executable=python_executable, scope=scope, project_root=root
    )
    print_fn(
        f"Registered MCP server '{SERVER_NAME}' for {where}, DSN {_redacted(dsn)}, "
        f"in {user_config_file()} under {len(keys)} key(s): {', '.join(keys)}"
    )
    return "registered"


def _register_via_cli(
    *,
    env: dict[str, str],
    python_executable: str,
    scope: str,
    root: Path,
    where: str,
    dsn: str,
    replace: bool,
    print_fn: Callable[..., None],
) -> str:
    """Opt-in path that lets the client write its own configuration.

    Every call is made FROM the project root, because `--scope local` and `mcp get` both resolve
    which project they mean from the working directory. Called from wherever the wizard was
    started, the entry lands under a directory nobody opens.
    """
    existing = _run_claude(["mcp", "get", SERVER_NAME], cwd=root)
    if existing.returncode == 0:
        if not replace:
            print_fn(
                f"MCP server '{SERVER_NAME}' is already registered, left unchanged. "
                "Re-run with replace=True to point it at this installation."
            )
            return "already-registered"
        # `claude mcp add` refuses a duplicate name at the same scope with "already exists in
        # local config", so a replace is a remove followed by an add rather than an overwrite.
        _run_claude(["mcp", "remove", "--scope", scope, SERVER_NAME], cwd=root)

    args = ["mcp", "add", "--scope", scope, SERVER_NAME]
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    # Everything after `--` is passed to the server untouched, which is what keeps a Windows
    # interpreter path containing spaces from being re-split by the client.
    args.extend(["--", python_executable, "-m", "recall_mcp.server"])

    result = _run_claude(args, cwd=root)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"claude mcp add failed: {message or 'no output was readable'}")

    print_fn(
        f"Registered MCP server '{SERVER_NAME}' at {scope} scope for {where}, "
        f"DSN {_redacted(dsn)}, via the CLI."
    )
    return "registered"


def _project_keys(document: dict[str, Any], project_root: Path) -> list[str]:
    """Every key under `projects` that means this directory, or one new key when none do.

    **The client does not normalise these keys, and one directory routinely has several.**
    Measured on this machine: 313 project keys, 7 directories carrying two spellings each, one
    backslash-separated from a native launch and one forward-slash-separated from Git Bash. An
    entry written under one spelling is invisible to a session launched the other way, with no
    error and no tools, which is indistinguishable from the install having done nothing.

    Re-measure with:

    ```bash
    python -c "import json,os,collections;from pathlib import Path;d=json.load(open(os.path.expanduser('~/.claude.json')));g=collections.defaultdict(list);[g[str(Path(k).resolve()).casefold()].append(k) for k in d.get('projects',{})];print(sum(1 for v in g.values() if len(v)>1))"
    ```

    Found by the wizard session while landing the same change on its side. Writing every matching
    spelling is strictly safer than picking one, because the cost of an extra key is a duplicate
    entry and the cost of a missing one is silence.
    """
    try:
        target = str(project_root.resolve()).casefold()
    except OSError:  # pragma: no cover - an unresolvable root cannot be matched, only invented
        return [str(project_root)]
    matches: list[str] = []
    for key in document.get("projects", {}):
        try:
            if str(Path(key).resolve()).casefold() == target:
                matches.append(key)
        except (OSError, ValueError):
            # A stored key this platform cannot even parse is not this project.
            continue
    return matches or [str(project_root)]


def _write_server_entry(
    *, env: dict[str, str], python_executable: str, scope: str, project_root: Path
) -> list[str]:
    """Write the server into `~/.claude.json`. Returns the project keys written, for reporting.

    This is the PRIMARY path rather than the fallback it started as. Shelling out to `claude mcp
    add` was preferred on the reasoning that the client owns its own schema, and that still holds,
    but the CLI's output is undecodable under the console codec on the platform this feature
    exists for (see `_run_claude`), and it writes only the single project key matching its working
    directory. A direct merge avoids both, and the schema it writes is small enough to pin with a
    test.

    User scope is the top-level `mcpServers`. Local scope is `projects[<dir>].mcpServers`, under
    every spelling of that directory the client already knows.
    """
    config_file = user_config_file()
    document: dict[str, Any] = {}
    if config_file.exists():
        raw = config_file.read_text(encoding="utf-8")
        document = json.loads(raw) if raw.strip() else {}
        _backup(config_file)
    entry = {
        "type": "stdio",
        "command": python_executable,
        "args": ["-m", "recall_mcp.server"],
        "env": dict(env),
    }
    if scope == "user":
        document.setdefault("mcpServers", {})[SERVER_NAME] = entry
        written = ["<user scope>"]
    else:
        keys = _project_keys(document, project_root)
        projects = document.setdefault("projects", {})
        for key in keys:
            projects.setdefault(key, {}).setdefault("mcpServers", {})[SERVER_NAME] = dict(entry)
        written = keys
    _write_json(config_file, document)
    return written


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
