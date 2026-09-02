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
import sys
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, cast

from recall.atomic_write import atomic_write_bytes

CODEX_PLUGIN_NAME = "recall"
CODEX_MARKETPLACE_NAME = "re-call"
CODEX_HOOK_MARKER = "recall_hooks.codex"


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
) -> None:
    destination = codex_hook_config_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "dsn": dsn,
        "tenant": tenant,
        "embedder": embedder,
        "table": table,
        "write_time": {"enabled": write_time},
        "prompt_time": {"enabled": prompt_time},
    }
    atomic_write_bytes(destination, (json.dumps(document, indent=2) + "\n").encode("utf-8"))


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
            "args": ["-m", CODEX_HOOK_MARKER, event],
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
                matcher="Bash|Write|Edit|MultiEdit|NotebookEdit|BashOutput",
                timeout=5,
            )
        ],
        # Codex always runs SessionEnd synchronously, even when async=true is present. Keep the
        # timeout explicit so the generated config does not promise non-blocking teardown.
        "SessionEnd": [command_group("session-end", timeout=3)],
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


def install_codex_integration(
    *,
    dsn: str,
    tenant: str = "default",
    embedder: str = "fastembed",
    table: str = "chunks",
    write_time: bool = True,
    prompt_time: bool = True,
    python_executable: str | None = None,
    hooks_path: Path | None = None,
    marketplace_path: Path | None = None,
    plugin_destination: Path | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    """Install the plugin, personal marketplace entry, shared config, and Codex hooks."""
    source = codex_plugin_source()
    if source is None:
        raise FileNotFoundError("the RE-call Codex plugin bundle is not present in this install")

    marketplace = marketplace_path or personal_marketplace_path()
    destination = plugin_destination or marketplace.parent / ".codex" / "plugins" / CODEX_PLUGIN_NAME
    target = hooks_path or codex_home() / "hooks.json"
    with _installation_lock(marketplace):
        _copy_plugin(source, destination)
        marketplace.parent.mkdir(parents=True, exist_ok=True)
        _update_marketplace(marketplace, destination)
        _write_hook_config(
            dsn=dsn,
            tenant=tenant,
            embedder=embedder,
            table=table,
            write_time=write_time,
            prompt_time=prompt_time,
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
    print_fn(
        f"Installed RE-call for Codex: plugin={destination}, marketplace={marketplace}, hooks={target}"
    )


__all__ = [
    "CODEX_MARKETPLACE_NAME",
    "CODEX_PLUGIN_NAME",
    "codex_code_detected",
    "codex_home",
    "codex_hook_config_path",
    "codex_plugin_source",
    "install_codex_integration",
    "personal_marketplace_path",
]
