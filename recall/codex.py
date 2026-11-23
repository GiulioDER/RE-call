"""Install RE-call's Codex integration: user-level hooks, skills and plugin, and per-project servers.

`register_project_servers` writes a project's `.codex/config.toml` (and may add it to that
project's `.gitignore`); everything else here is user-level, under `CODEX_HOME` and `~/.agents`.

The Claude installer and this module deliberately share the same hook implementation. Codex gets
an adapter only for its payload shape, while the memory reader, front matter contract, thresholds,
and fail-open behaviour remain in :mod:`recall_hooks`.
"""
from __future__ import annotations

import hashlib
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
from collections.abc import Iterator, Mapping
from typing import Any, Callable, cast

from recall.atomic_write import atomic_write_bytes

CODEX_PLUGIN_NAME = "recall"
CODEX_MARKETPLACE_NAME = "re-call"
CODEX_HOOK_MARKER = "recall_hooks.codex"
# SessionEnd is synchronous in Codex, and Codex clamps its timeout to 1 to 3 s whatever this file
# asks for (`SESSION_END_MAX_TIMEOUT_SEC = 3` in codex-rs/hooks/src/events/session_end.rs, there
# since SessionEnd hooks were added on 2026-07-17). The 15 s written here until #1454 was never
# honoured. The hook fits by doing only MCP cleanup and starting the detached memory index.
SESSION_END_TIMEOUT_SECONDS = 3


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
    with lock_path.open("a+b") as handle:
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
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    server_env: Mapping[str, str] | None = None,
) -> None:
    """The same document the Claude Code installer writes, at the Codex integration path.

    Its `server_env` is the environment the Codex launcher (`recall_mcp.codex_server`) gives the
    plugin's server, which the Claude Code installer puts in that client's server registration."""
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
        server_env=server_env,
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
    server_env: Mapping[str, str] | None = None,
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
            server_env=server_env,
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
    marketplace_path: Path | None = None,
    plugin_destination: Path | None = None,
    print_fn: Callable[..., None] = print,
    **hook_options: Any,
) -> None:
    """Install the plugin, personal marketplace entry, shared config, and Codex hooks.

    `hook_options` are `install_codex_hooks`' arguments, passed on unchanged. Its `project_root`
    (default: the current directory) is the project allowed to use the hook config, as for Claude
    Code: the write-time hook and the session-end memory index refuse an event outside it, and
    refuse a config that names none.
    """
    root = (hook_options.get("project_root") or Path.cwd()).resolve()
    source = codex_plugin_source()
    if source is None:
        raise FileNotFoundError("the RE-call Codex plugin bundle is not present in this install")

    marketplace = marketplace_path or personal_marketplace_path()
    destination = plugin_destination or marketplace.parent / ".codex" / "plugins" / CODEX_PLUGIN_NAME
    with _installation_lock(marketplace):
        _copy_plugin(source, destination)
        marketplace.parent.mkdir(parents=True, exist_ok=True)
        _update_marketplace(marketplace, destination)
    target = install_codex_hooks(**{**hook_options, "project_root": root})
    print_fn(
        f"Installed RE-call for Codex: plugin={destination}, marketplace={marketplace}, hooks={target}"
    )
    if hook_options.get("write_time", True):
        # As in the Claude Code installer: one hook config per machine, so say which project it
        # answers in, since installing in another project moves it.
        print_fn(
            f"Write-time memory search is bound to {root}. It is the only project it answers in, "
            "and installing in another one moves it."
        )


def codex_user_skill_dir() -> Path:
    """Where Codex loads a user's own skills from (`$HOME/.agents/skills`).

    Per the Codex skills documentation (https://learn.chatgpt.com/codex/build-skills).
    """
    return Path.home() / ".agents" / "skills"


def codex_skill_sources() -> dict[str, Path]:
    """``{skill name: its SKILL.md}`` from the Codex plugin bundle, or empty when it is absent."""
    from recall.claude_code import skills_under

    source = codex_plugin_source()
    return {} if source is None else skills_under(source / "skills")


def codex_plugin_installed() -> bool:
    """Whether Codex has the RE-call plugin, which carries the skills, enabled.

    The personal marketplace only offers it; Codex records an enabled plugin in `config.toml` as
    `[plugins."<plugin>@<marketplace>"] enabled = true`, and until then none of its skills load.
    The marketplace part is that file's own name, which `_update_marketplace` keeps when the user's
    marketplace already had one.
    """
    try:
        document = _marketplace_document(personal_marketplace_path())
        config = tomllib.loads((codex_home() / "config.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    offered = any(
        isinstance(item, dict) and item.get("name") == CODEX_PLUGIN_NAME
        for item in document.get("plugins", [])
    )
    marketplace = document.get("name") or CODEX_MARKETPLACE_NAME
    plugins = config.get("plugins")
    entry = plugins.get(f"{CODEX_PLUGIN_NAME}@{marketplace}") if isinstance(plugins, dict) else None
    return offered and isinstance(entry, dict) and entry.get("enabled") is True


#: The lines that delimit the servers this installer owns inside a project's `.codex/config.toml`.
#: Everything outside them is the user's: kept where it is, with its line endings, and backed up
#: once before RE-call first rewrites the file.
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
    #: The `.gitignore` this call added `.codex/config.toml` to, when it had to.
    gitignore: Path | None = None
    #: The copy of the file as it stood before this call rewrote it, made once and never replaced.
    backup: Path | None = None
    #: RE-call servers an earlier run on this machine registered here, kept beside this run's.
    kept: tuple[str, ...] = ()
    #: Servers found in RE-call's block that this machine has no record of writing as they stand, removed.
    dropped: tuple[str, ...] = ()
    #: Servers the file already defined that RE-call did not write: trusting the project enables them.
    others: tuple[str, ...] = ()


def _toml_string(value: str) -> str:
    # A TOML basic string. JSON's escapes for the quote, the backslash and the C0 controls are valid
    # TOML; every other character is written as itself, because `ensure_ascii` would write one
    # outside the BMP as a UTF-16 surrogate pair, which TOML refuses. DEL, which JSON leaves raw,
    # TOML requires escaped.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


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


def _split_managed_block(text: str) -> tuple[str, str, str]:
    """`(before, block, after)` around RE-call's block, whose markers must be whole lines."""
    lines = text.splitlines(keepends=True)
    begin = next((i for i, line in enumerate(lines) if line.strip() == MANAGED_BEGIN), None)
    if begin is None:
        return text, "", ""
    end = next((i for i in range(begin + 1, len(lines)) if lines[i].strip() == MANAGED_END), None)
    if end is None:
        raise ValueError("it has an unterminated RE-call block")
    return "".join(lines[:begin]), "".join(lines[begin:end + 1]), "".join(lines[end + 1:])


def _server_digest(entry: object) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _written_servers_path() -> Path:
    """Where this machine records the servers RE-call wrote into each project's block."""
    return codex_home() / "re-call" / "project-servers.json"


def _project_key(root: Path) -> str:
    return os.path.normcase(str(root))


def _written_servers() -> dict[str, Any]:
    try:
        document = json.loads(_written_servers_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def _earlier_servers(block: str, root: Path) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """`(kept, dropped)`: the servers of RE-call's block that a run on this machine wrote, unchanged.

    The block is in the project, so anything can stand between its markers: a server planted in a
    downloaded project, or one the user disabled or changed by hand. Only an entry whose content
    matches what this machine recorded writing is RE-call's, and is rendered again as it was.
    """
    try:
        servers = tomllib.loads(block).get("mcp_servers", {})
    except ValueError:
        return {}, ()
    if not isinstance(servers, dict):
        return {}, ()
    recorded = _written_servers().get(_project_key(root))
    recorded = recorded if isinstance(recorded, dict) else {}
    keep = {name: entry for name, entry in servers.items() if recorded.get(name) == _server_digest(entry)}
    return keep, tuple(name for name in servers if name not in keep)


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


def _git_tracked(project_root: Path, relative_path: str) -> int | None:
    pathspec = f":(icase,literal){relative_path}" if os.name == "nt" else relative_path
    return _git(project_root, "ls-files", "--error-unmatch", "--", pathspec)


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
    and a file git tracks is refused, since it would commit this machine's store settings (the
    env holds a DSN and may hold a provider key). Inside a git work tree, the config and its
    backup must be untracked and effectively ignored.
    The file is written 0600, and a `.codex` that leads outside the project is refused.

    A later run keeps only the servers in RE-call's block that `CODEX_HOME/re-call/project-servers.json`
    records this machine writing, unchanged; anything else there is removed and reported.
    """
    root = project_root.resolve()
    config = ".codex/config.toml"
    target = root / config
    if not servers:
        return ProjectServerRegistration(target, skipped_reason="no servers to register")
    # A linked `.codex` (or config.toml) would land the credentials outside this project, where
    # neither the tracked-file check below nor the `.gitignore` entry applies.
    if os.path.normcase(os.path.realpath(target)) != os.path.normcase(str(target)):
        return ProjectServerRegistration(
            target, skipped_reason=f"{target} leads outside this project through a link; left untouched"
        )
    in_work_tree = any((folder / ".git").exists() for folder in (root, *root.parents))
    tracked = _git_tracked(root, config)
    if tracked == 0:
        return ProjectServerRegistration(
            target,
            skipped_reason=(
                f"{target} is tracked by git, and RE-call will not write this machine's store "
                "settings into a committed file"
            ),
        )
    # 1 is git's answer for "not tracked"; anything else (no git, a timeout, a refused
    # repository) is no answer, and inside a work tree that must not read as "not tracked".
    if in_work_tree and tracked != 1:
        return ProjectServerRegistration(
            target,
            skipped_reason=(
                f"git could not say whether it tracks {target} (git missing, or the repository "
                "refused), so RE-call will not write this machine's store settings there"
            ),
        )

    # Protect both the project read/modify/write and the shared ownership ledger merge.
    ledger = _written_servers_path()
    with _installation_lock(ledger):
        return _register_project_servers_locked(servers, root, target, in_work_tree, ledger)


def _register_project_servers_locked(
    servers: dict[str, dict[str, Any]], root: Path, target: Path, in_work_tree: bool, ledger: Path,
) -> ProjectServerRegistration:
    from recall.wizard.wiring import _write_private_configuration

    config = target.relative_to(root).as_posix()
    try:
        raw = target.read_bytes() if target.exists() else b""
        # `utf-8-sig`: a BOM a Windows editor wrote is not the user's content, and tomllib refuses it.
        original = raw.decode("utf-8-sig")
        before, block, after = _split_managed_block(original.replace("\r\n", "\n"))
        user_servers = tomllib.loads(before + after).get("mcp_servers", {})
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
    # As Claude Code's local scope keeps the servers an earlier install registered, so does this:
    # Lite and the wizard in one project each keep the other's.
    earlier, unrecorded = _earlier_servers(block, root)
    # A name the user also defines outside the block stays theirs: rendering it again would
    # declare it twice, and Codex, like tomllib, refuses the whole file.
    kept = {name: entry for name, entry in earlier.items() if name not in mine and name not in taken}
    dropped = tuple(name for name in unrecorded if name not in mine and name not in taken)
    head, tail = before.rstrip("\n"), after.lstrip("\n")
    text = (
        (head + "\n\n" if head else "")
        + _render_servers({**kept, **mine})
        + ("\n" + tail if tail else "")
    )
    written = tomllib.loads(text)["mcp_servers"]  # never write a file Codex would refuse to parse
    gitignore = None
    backup_name = f"{config}.recall-backup"
    backup_path = root / backup_name
    make_backup = bool(raw) and not backup_path.exists()
    if in_work_tree:
        covered = (config, backup_name)
        # The backup may contain an earlier block's credentials, even if it is absent on disk now.
        if _git_tracked(root, backup_name) != 1:
            return ProjectServerRegistration(
                target, skipped_reason=f"git could not verify that {backup_path} is untracked; left untouched"
            )
        ignore_status = {path: _git(root, "check-ignore", "-q", "--", path) for path in covered}
        for path, status in ignore_status.items():
            if status not in (0, 1):
                return ProjectServerRegistration(
                    target, skipped_reason=f"git could not verify whether it ignores {root / path}; left untouched"
                )
        if 1 in ignore_status.values():
            gitignore = root / ".gitignore"
            if gitignore.is_symlink():
                return ProjectServerRegistration(
                    target, skipped_reason=f"{gitignore} is a symbolic link; left untouched"
                )
            existing = gitignore.read_bytes() if gitignore.exists() else b""
            separator = b"" if not existing or existing.endswith(b"\n") else b"\n"
            with gitignore.open("ab") as handle:
                handle.write(separator + b".codex/config.toml*\n")
            # A rule closer to these files can take precedence over the rule we just added.
            for path in covered:
                if _git(root, "check-ignore", "-q", "--", path) != 0:
                    return ProjectServerRegistration(
                        target, gitignore=gitignore,
                        skipped_reason=f"git could not confirm that it ignores {root / path}; left untouched",
                    )
    backup = None
    target.parent.mkdir(parents=True, exist_ok=True)
    if make_backup:
        _write_private_configuration(backup_path, original)
        backup = backup_path
    # Record before writing the TOML: a record naming a server the file lacks keeps nothing.
    record = _written_servers()
    record[_project_key(root)] = {name: _server_digest(written[name]) for name in (*kept, *mine)}
    _write_private_configuration(ledger, json.dumps(record, indent=2, sort_keys=True) + "\n")
    _write_private_configuration(target, text.replace("\n", "\r\n") if b"\r\n" in raw else text)
    return ProjectServerRegistration(
        target,
        registered=tuple(mine),
        conflicts=conflicts,
        trusted=_project_trusted(root),
        gitignore=gitignore,
        backup=backup,
        kept=tuple(kept),
        dropped=dropped,
        others=tuple(taken),
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
