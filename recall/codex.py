"""Install RE-call's user-level Codex integration.

The Claude installer and this module deliberately share the same hook implementation. Codex gets
an adapter only for its payload shape, while the memory reader, front matter contract, thresholds,
and fail-open behaviour remain in :mod:`recall_hooks`.
"""
from __future__ import annotations

import json
import importlib
import os
import shutil
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, cast

from recall.atomic_write import atomic_write_bytes

CODEX_PLUGIN_NAME = "recall"
CODEX_MARKETPLACE_NAME = "re-call"
CODEX_HOOK_MARKER = "recall_hooks.codex"
# SessionEnd is synchronous in Codex. MCP cleanup reads the Windows process table and may close
# more than one transport, so three seconds was too short to make the repo-owned cleanup reliable.
SESSION_END_TIMEOUT_SECONDS = 15


def codex_home() -> Path:
    """Return Codex's config directory, honouring ``CODEX_HOME``."""
    raw = os.environ.get("CODEX_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def codex_code_detected() -> bool:
    """Whether a Codex installation is present on this machine."""
    return shutil.which("codex") is not None or codex_home().is_dir()


def personal_marketplace_root() -> Path:
    """Return the personal marketplace root documented by Codex."""
    return Path.home() / ".agents" / "plugins"


def personal_marketplace_path() -> Path:
    return personal_marketplace_root() / "marketplace.json"


def codex_integration_dir() -> Path:
    return codex_home() / "re-call"


def codex_hook_config_path() -> Path:
    """The shared hook configuration used by Codex's adapter and MCP launcher."""
    return codex_integration_dir() / "recall-hook.json"


def codex_plugin_sources() -> list[Path]:
    """Return the checkout and wheel layouts that can provide the Codex plugin."""
    return [
        Path(__file__).resolve().parent.parent / "codex-plugin",
        Path(__file__).resolve().parent / "_codex_plugin",
    ]


def codex_plugin_source() -> Path | None:
    for root in codex_plugin_sources():
        if (root / ".codex-plugin" / "plugin.json").is_file():
            return root
    return None


def _copy_plugin(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.is_symlink():
        raise ValueError(f"Codex plugin destination must not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


@contextmanager
def _installation_lock(path: Path) -> Iterator[None]:
    """Serialize installer merges without adding a platform-specific dependency."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")

            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _marketplace_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": CODEX_MARKETPLACE_NAME,
            "interface": {"displayName": "RE-call"},
            "plugins": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Codex marketplace {path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Codex marketplace {path} must contain a JSON object")
    plugins = value.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError(f"Codex marketplace {path} has a non-list plugins field")
    return cast(dict[str, Any], value)


def _update_marketplace(path: Path, plugin_path: Path) -> None:
    root = path.parent.resolve()
    relative = Path(os.path.relpath(plugin_path.resolve(), root)).as_posix()
    if relative == "." or relative == ".." or relative.startswith("../"):
        raise ValueError("Codex plugin must live below the personal marketplace root")
    if not relative.startswith("./"):
        relative = f"./{relative}"
    document = _marketplace_document(path)
    plugins = [
        item
        for item in document.get("plugins", [])
        if not (isinstance(item, dict) and item.get("name") == CODEX_PLUGIN_NAME)
    ]
    plugins.append(
        {
            "name": CODEX_PLUGIN_NAME,
            "source": {"source": "local", "path": relative},
            "policy": {
                "installation": "INSTALLED_BY_DEFAULT",
                "authentication": "ON_INSTALL",
            },
            "category": "Productivity",
        }
    )
    document["name"] = document.get("name") or CODEX_MARKETPLACE_NAME
    interface = document.get("interface")
    if not isinstance(interface, dict):
        interface = {}
    interface.setdefault("displayName", "RE-call")
    document["interface"] = interface
    document["plugins"] = plugins
    atomic_write_bytes(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))


def _write_hook_config(
    *,
    dsn: str,
    tenant: str,
    embedder: str,
    table: str,
    write_time: bool,
    prompt_time: bool,
    write_time_connection_mode: str = "relay",
    project_root: Path | None = None,
    memory_root: Path | None = None,
) -> None:
    """The same document the Claude Code installer writes, at the Codex integration path."""
    from recall.claude_code import hook_config_document

    destination = codex_hook_config_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = hook_config_document(
        destination,
        dsn=dsn,
        tenant=tenant,
        embedder=embedder,
        table=table,
        write_time=write_time,
        write_time_connection_mode=write_time_connection_mode,
        project_root=project_root,
        prompt_time=prompt_time,
        memory_root=memory_root,
    )
    atomic_write_bytes(destination, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
    try:
        destination.chmod(0o600)
    except OSError:
        # As in the Claude Code installer: Windows ACLs do not model the POSIX bits.
        pass


def _hook_entries(python_executable: str | None = None) -> dict[str, list[dict[str, Any]]]:
    command = python_executable or sys.executable

    def command_group(
        event: str,
        *,
        matcher: str | None = None,
        async_: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        hook: dict[str, Any] = {
            "type": "command",
            "command": command,
            # `-P`, as the Claude Code installer does: without it `python -m` puts the session's
            # directory first on sys.path, so a checkout holding another `recall_hooks` or
            # `recall` (another branch of this repository, say) would run instead.
            "args": ["-P", "-m", CODEX_HOOK_MARKER, event],
        }
        if async_:
            hook["async"] = True
        if timeout is not None:
            hook["timeout"] = timeout
        group: dict[str, Any] = {"hooks": [hook]}
        if matcher is not None:
            group["matcher"] = matcher
        return group

    return {
        "SessionStart": [
            command_group("session-start", matcher="startup|resume|clear|compact", timeout=15)
        ],
        "PreCompact": [command_group("pre-compact", matcher="manual|auto", async_=True)],
        "UserPromptSubmit": [command_group("user-prompt-submit", timeout=10)],
        "PreToolUse": [
            command_group(
                "pre-tool-use",
                matcher="^(Bash|PowerShell|Write|Edit|MultiEdit|NotebookEdit|BashOutput|apply_patch)$",
                timeout=5,
            )
        ],
        "PostToolUse": [command_group("post-tool-use", matcher="^(Bash|PowerShell)$", timeout=5)],
        "SubagentStart": [command_group("subagent-start", timeout=5)],
        # Codex always runs SessionEnd synchronously, even when async=true is present. Keep the
        # timeout explicit so the generated config does not promise non-blocking teardown.
        "SessionEnd": [command_group("session-end", timeout=SESSION_END_TIMEOUT_SECONDS)],
    }


def _is_codex_handler(handler: dict[str, Any]) -> bool:
    return CODEX_HOOK_MARKER in str(handler.get("command", "")) or any(
        CODEX_HOOK_MARKER in str(arg) for arg in handler.get("args", []) or []
    )


def _merge_hooks(settings: dict[str, Any], entries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    merged = json.loads(json.dumps(settings))
    hooks = merged.setdefault("hooks", {})
    for event, groups in entries.items():
        kept = []
        for group in hooks.get(event, []) or []:
            if not isinstance(group, dict):
                kept.append(group)
                continue
            handlers = [
                handler
                for handler in group.get("hooks", [])
                if not _is_codex_handler(handler)
            ]
            if handlers:
                kept.append({**group, "hooks": handlers})
        hooks[event] = [*kept, *groups]
    return cast(dict[str, Any], merged)


def install_codex_hooks(
    *,
    dsn: str,
    tenant: str = "default",
    embedder: str = "fastembed",
    table: str = "chunks",
    write_time: bool = True,
    prompt_time: bool = True,
    write_time_connection_mode: str = "relay",
    project_root: Path | None = None,
    memory_root: Path | None = None,
    python_executable: str | None = None,
    hooks_path: Path | None = None,
) -> Path:
    """Write the shared hook config and merge RE-call's user-level hooks into Codex's hooks file.

    Returns the hooks file. Unrelated hooks are kept; RE-call's own handlers are replaced.
    """
    root = (project_root or Path.cwd()).resolve()
    target = hooks_path or codex_home() / "hooks.json"
    with _installation_lock(target):
        _write_hook_config(
            dsn=dsn,
            tenant=tenant,
            embedder=embedder,
            table=table,
            write_time=write_time,
            prompt_time=prompt_time,
            write_time_connection_mode=write_time_connection_mode,
            project_root=root,
            memory_root=memory_root,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        settings: dict[str, Any] = {}
        if target.exists():
            raw = target.read_text(encoding="utf-8")
            settings = json.loads(raw) if raw.strip() else {}
        atomic_write_bytes(
            target,
            (
                json.dumps(_merge_hooks(settings, _hook_entries(python_executable)), indent=2) + "\n"
            ).encode("utf-8"),
        )
    return target


def install_codex_integration(
    *,
    dsn: str,
    tenant: str = "default",
    embedder: str = "fastembed",
    table: str = "chunks",
    write_time: bool = True,
    prompt_time: bool = True,
    write_time_connection_mode: str = "relay",
    project_root: Path | None = None,
    python_executable: str | None = None,
    hooks_path: Path | None = None,
    marketplace_path: Path | None = None,
    plugin_destination: Path | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    """Install the plugin, personal marketplace entry, shared config, and Codex hooks.

    `project_root` (default: the current directory) is the project allowed to use the hook
    config, as for Claude Code: the write-time hook and the session-end memory index refuse an
    event outside it, and refuse a config that names none.
    """
    root = (project_root or Path.cwd()).resolve()
    source = codex_plugin_source()
    if source is None:
        raise FileNotFoundError("the RE-call Codex plugin bundle is not present in this install")

    marketplace = marketplace_path or personal_marketplace_path()
    destination = plugin_destination or marketplace.parent / ".codex" / "plugins" / CODEX_PLUGIN_NAME
    with _installation_lock(marketplace):
        _copy_plugin(source, destination)
        marketplace.parent.mkdir(parents=True, exist_ok=True)
        _update_marketplace(marketplace, destination)
    target = install_codex_hooks(
        dsn=dsn,
        tenant=tenant,
        embedder=embedder,
        table=table,
        write_time=write_time,
        prompt_time=prompt_time,
        write_time_connection_mode=write_time_connection_mode,
        project_root=root,
        python_executable=python_executable,
        hooks_path=hooks_path,
    )
    print_fn(
        f"Installed RE-call for Codex: plugin={destination}, marketplace={marketplace}, hooks={target}"
    )
    if write_time:
        # As in the Claude Code installer: one hook config per machine, so say which project it
        # answers in, since installing in another project moves it.
        print_fn(
            f"Write-time memory search is bound to {root}. It is the only project it answers in, "
            "and installing in another one moves it."
        )


def codex_user_skill_dir() -> Path:
    """Where Codex loads a user's own skills from (`$HOME/.agents/skills`)."""
    return Path.home() / ".agents" / "skills"


def codex_skill_sources() -> dict[str, Path]:
    """``{skill name: its SKILL.md}`` from the Codex plugin bundle, or empty when it is absent."""
    source = codex_plugin_source()
    if source is None:
        return {}
    return {
        path.parent.name: path
        for path in sorted((source / "skills").glob("*/SKILL.md"))
        if path.is_file()
    }


def codex_plugin_installed() -> bool:
    """Whether the personal marketplace already offers the RE-call plugin, which carries the skills."""
    try:
        document = _marketplace_document(personal_marketplace_path())
    except ValueError:
        return False
    return any(
        isinstance(item, dict) and item.get("name") == CODEX_PLUGIN_NAME
        for item in document.get("plugins", [])
    )


#: The lines that delimit the servers this installer owns inside a project's `.codex/config.toml`.
#: Everything outside them is the user's and is never rewritten.
MANAGED_BEGIN = "# BEGIN RE-call MCP servers (written by recall setup --lite and recall wizard)"
MANAGED_END = "# END RE-call MCP servers"
#: Codex waits 10 s for a server by default. A Lite server refreshes its store before it answers
#: and a local embedder loads its model first, so the documented remedy for a missing server is
#: written in from the start.
STARTUP_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class ProjectServerRegistration:
    """What was written into a project's `.codex/config.toml`, and what was refused."""

    config_path: Path
    registered: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    skipped_reason: str = ""
    #: Whether the user config marks this project trusted. Codex ignores project config otherwise.
    trusted: bool = False
    #: Set when git does not ignore the file, which holds this machine's store settings.
    unignored: bool = False


def _toml_string(value: str) -> str:
    # Every escape JSON emits (\", \\, \n, \uXXXX and the rest) is valid in a TOML basic string.
    return json.dumps(value)


def _render_servers(servers: dict[str, dict[str, Any]]) -> str:
    lines = [MANAGED_BEGIN]
    for name, entry in servers.items():
        key = f"mcp_servers.{_toml_string(name)}"
        lines += [
            f"[{key}]",
            f"command = {_toml_string(entry['command'])}",
            f"args = [{', '.join(_toml_string(arg) for arg in entry['args'])}]",
            f"cwd = {_toml_string(entry['cwd'])}",
            f"startup_timeout_sec = {STARTUP_TIMEOUT_SEC}",
            "",
            f"[{key}.env]",
            *(f"{_toml_string(k)} = {_toml_string(v)}" for k, v in entry["env"].items()),
            "",
        ]
    lines.append(MANAGED_END)
    return "\n".join(lines) + "\n"


def _without_managed_block(text: str) -> str:
    start = text.find(MANAGED_BEGIN)
    if start < 0:
        return text
    end = text.find(MANAGED_END, start)
    if end < 0:
        raise ValueError("it has an unterminated RE-call block")
    return text[:start] + text[end + len(MANAGED_END):].lstrip("\r\n")


def _git(project_root: Path, *args: str) -> int | None:
    """The exit status of a read-only git query, or None where git cannot answer."""
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True,
            timeout=10,
            check=False,
        ).returncode
    except (OSError, subprocess.SubprocessError):
        return None


def _project_trusted(project_root: Path) -> bool:
    """Whether `CODEX_HOME/config.toml` marks this project, or a folder above it, `trusted`."""
    try:
        document = tomllib.loads((codex_home() / "config.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    projects = document.get("projects")
    if not isinstance(projects, dict):
        return False
    root = project_root.resolve()
    for key, entry in projects.items():
        if not isinstance(entry, dict) or entry.get("trust_level") != "trusted":
            continue
        try:
            # Codex records some Windows paths in their extended form, `\\?\C:\...`.
            trusted = Path(str(key).removeprefix("\\\\?\\")).resolve()
        except (OSError, ValueError):
            continue
        if trusted == root or trusted in root.parents:
            return True
    return False


def register_project_servers(
    servers: dict[str, dict[str, Any]], *, project_root: Path
) -> ProjectServerRegistration:
    """Register MCP servers for this project only, in its `.codex/config.toml`.

    The Codex counterpart of Claude Code's local scope (`recall.wizard.wiring.register_local_scope`):
    a server carries one store and one tenant, so at user level it would answer about this project
    in every other one. `servers` is the `mcpServers` mapping `recall.wizard.wiring.mcp_config`
    renders for Claude Code, so both clients launch the same command with the same environment.

    Codex reads project config only for a trusted project, and the result says whether this one
    is. A server name the user already defines outside RE-call's block is reported and left alone,
    and a file git tracks is refused, since it would commit this machine's store settings.
    """
    root = project_root.resolve()
    target = root / ".codex" / "config.toml"
    if not servers:
        return ProjectServerRegistration(target, skipped_reason="no servers to register")
    if _git(root, "ls-files", "--error-unmatch", "--", ".codex/config.toml") == 0:
        return ProjectServerRegistration(
            target,
            skipped_reason=(
                f"{target} is tracked by git, and RE-call will not write this machine's store "
                "settings into a committed file"
            ),
        )
    if target.is_symlink():
        return ProjectServerRegistration(target, skipped_reason=f"{target} is a symlink; left untouched")
    try:
        user_text = _without_managed_block(target.read_text(encoding="utf-8") if target.exists() else "")
        user_servers = tomllib.loads(user_text).get("mcp_servers", {})
    except (OSError, ValueError) as exc:
        return ProjectServerRegistration(
            target, skipped_reason=f"{target} cannot be read ({exc}); left untouched"
        )
    taken = user_servers if isinstance(user_servers, dict) else {}
    conflicts = tuple(name for name in servers if name in taken)
    mine = {name: entry for name, entry in servers.items() if name not in taken}
    if not mine:
        return ProjectServerRegistration(
            target, conflicts=conflicts, skipped_reason="every server name is already defined in it"
        )
    kept = user_text.rstrip("\r\n")
    text = (kept + "\n\n" if kept else "") + _render_servers(mine)
    tomllib.loads(text)  # never write a file Codex would refuse to parse
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, text.encode("utf-8"))
    return ProjectServerRegistration(
        target,
        registered=tuple(mine),
        conflicts=conflicts,
        trusted=_project_trusted(root),
        unignored=_git(root, "check-ignore", "-q", "--", ".codex/config.toml") == 1,
    )


__all__ = [
    "CODEX_MARKETPLACE_NAME",
    "CODEX_PLUGIN_NAME",
    "codex_code_detected",
    "codex_home",
    "codex_hook_config_path",
    "codex_plugin_source",
    "ProjectServerRegistration",
    "codex_plugin_installed",
    "codex_skill_sources",
    "codex_user_skill_dir",
    "install_codex_hooks",
    "install_codex_integration",
    "register_project_servers",
    "personal_marketplace_path",
]
