"""The configuration page's data: redacted reports, never raw values.

Every report is produced by `python -m recall_mcp.config_report`, run as its own process: on this
machine, for each RE-call MCP server registered for the project (its `env` and launch command fed
on stdin), and on a remote host over ssh. The redaction therefore happens in the process that
holds the values, and on a remote host a secret never crosses the network. This module never
imports the schema, which also keeps the `recall` library's import contract with `recall_mcp`.

A report that cannot be produced (no interpreter, ssh refused, a malformed answer) raises
`ConfigUnavailable`, which the page shows as a card rather than an error page.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall.errors import RecallError

__all__ = [
    "DEFAULT_REMOTE_COMMAND",
    "ConfigUnavailable",
    "RegisteredServer",
    "local_report",
    "registered_servers",
    "remote_report",
    "server_report",
]

REPORT_MODULE = "recall_mcp.config_report"
DEFAULT_REMOTE_COMMAND = f"python3 -m {REPORT_MODULE} --json"
REPORT_TIMEOUT_SECONDS = 30
REPORT_VERSION = 1


class ConfigUnavailable(RuntimeError, RecallError):
    """A configuration report could not be produced; the message says why, with no value in it."""


@dataclass(frozen=True)
class RegisteredServer:
    """One RE-call MCP server registered for the project, and where it is registered."""

    name: str
    scope: str  # "local", "project" (.mcp.json) or "user"
    entry: Mapping[str, Any]


def _run(argv: list[str], stdin: str | None = None) -> dict[str, Any]:
    try:
        done = subprocess.run(  # noqa: S603  # a fixed module, or ssh with a host the operator named
            argv, input=stdin, capture_output=True, text=True, timeout=REPORT_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigUnavailable(f"{type(exc).__name__}: could not run the report") from exc
    if done.returncode != 0:
        # Only the last line of stderr, and only its start: a traceback could quote an argument.
        last = (done.stderr.strip().splitlines() or ["no output"])[-1][:200]
        raise ConfigUnavailable(f"the report exited with {done.returncode}: {last}")
    try:
        report = json.loads(done.stdout)
    except ValueError as exc:
        raise ConfigUnavailable("the report did not print JSON") from exc
    if not isinstance(report, dict) or report.get("version") != REPORT_VERSION or not isinstance(report.get("entries"), list):
        raise ConfigUnavailable("the report is not in the shape this dashboard reads")
    return report


def local_report() -> dict[str, Any]:
    """The configuration of this machine's environment, as the `recall` CLI here would see it."""
    return _run([sys.executable, "-m", REPORT_MODULE, "--json"])


def server_report(server: RegisteredServer) -> dict[str, Any]:
    """A registered server's `env` and launch command, redacted by the report process."""
    env = server.entry.get("env")
    args = server.entry.get("args")
    argv = [str(server.entry.get("command", ""))] + ([str(a) for a in args] if isinstance(args, list) else [])
    payload = json.dumps({"env": dict(env) if isinstance(env, Mapping) else {}, "argv": argv})
    return _run([sys.executable, "-m", REPORT_MODULE, "--json", "--env-from-stdin"], stdin=payload)


def remote_report(host: str, command: str = DEFAULT_REMOTE_COMMAND) -> dict[str, Any]:
    """The report run on `host` over ssh: redacted there, so no secret value is sent here."""
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ConfigUnavailable(f"{host!r} is not a host name")
    return _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command])


def _is_recall_server(entry: Mapping[str, Any]) -> bool:
    command = " ".join(str(part) for part in [entry.get("command", ""), *(entry.get("args") or [])])
    return "recall_mcp" in command or "recall-mcp" in command or "recall-codex-mcp" in command


def _same_directory(key: str, root: Path) -> bool:
    try:
        return Path(key).resolve() == root
    except OSError:
        return False


def registered_servers(project_root: Path, *, claude_config: Path | None = None) -> list[RegisteredServer]:
    """RE-call MCP servers the client would start in `project_root`: local, project and user scope.

    Read from the client's own files; nothing here is executed. A file that is absent or not JSON
    contributes nothing.
    """
    from recall.wizard.wiring import claude_config_path

    root = project_root.resolve()
    found: list[RegisteredServer] = []

    def add(scope: str, servers: object) -> None:
        if isinstance(servers, Mapping):
            for name, entry in servers.items():
                if isinstance(entry, Mapping) and _is_recall_server(entry):
                    found.append(RegisteredServer(str(name), scope, entry))

    try:
        document = json.loads((claude_config or claude_config_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = {}
    if isinstance(document, Mapping):
        projects = document.get("projects")
        if isinstance(projects, Mapping):
            for key, project in projects.items():
                if isinstance(project, Mapping) and _same_directory(str(key), root):
                    add("local", project.get("mcpServers"))
        add("user", document.get("mcpServers"))
    try:
        mcp_json = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        mcp_json = {}
    if isinstance(mcp_json, Mapping):
        add("project", mcp_json.get("mcpServers"))
    return found
