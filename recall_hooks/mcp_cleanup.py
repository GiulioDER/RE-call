"""Close only the local MCP transports owned by the ending client session."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any

MCP_PATTERN = os.environ.get("RECALL_MCP_PATTERN", "recall_mcp.server")
_MARKER_RE = re.compile(r"(?<!\S)RECALL_MCP_CLIENT=([^\s;&\"']+)")
_LAUNCHER_RE = re.compile(r"(?<!\S)(?:\S*[\\/])?recall-(?:codex-)?mcp(?:\s|$)")


def _is_mcp_transport(command: str) -> bool:
    return MCP_PATTERN in command or bool(_LAUNCHER_RE.search(command))


def _process_table() -> list[tuple[str, str, str]] | None:
    override = os.environ.get("RECALL_MCP_PS_FILE")
    if override:
        try:
            output = Path(override).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    elif os.name == "nt":
        command = [
            "powershell", "-NoProfile", "-Command",
            'Get-CimInstance Win32_Process | ForEach-Object { "$(($_.ProcessId)) $(($_.ParentProcessId)) $(($_.CommandLine))" }',
        ]
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    else:
        try:
            output = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,args="],
                capture_output=True, text=True, timeout=2,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rows


def _descends_from(pid: str, want: str, parents: dict[str, str]) -> bool:
    hops = 0
    while hops < 8:
        if pid == want:
            return True
        parent = parents.get(pid)
        if not parent or parent == pid:
            return False
        pid = parent
        hops += 1
    return False


def _marker_from_config(cwd: str) -> str:
    """Recover a single marker from the generated project MCP config, if present."""
    if not cwd:
        return ""
    current = Path(cwd).expanduser()
    if not current.is_dir():
        current = current.parent
    for directory in (current, *current.parents):
        path = directory / ".mcp.json"
        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        markers: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, str):
                markers.update(match.group(1) for match in _MARKER_RE.finditer(value))
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(document)
        if len(markers) == 1:
            return next(iter(markers))
        if markers:
            return ""
    return ""


def close_own_mcp_transports(
    client_pid: str = "", client_mark: str = "", *, cwd: str = ""
) -> tuple[str, str]:
    """Close this session's transports and leave every unowned transport alone."""
    client_pid = client_pid or os.environ.get("CLAUDE_PID", "")
    client_mark = client_mark or os.environ.get("RECALL_MCP_CLIENT", "")
    client_mark = client_mark or _marker_from_config(cwd or os.getcwd())
    if not client_pid and not client_mark:
        return "skipped", "no client pid or marker; ownership could not be established"

    rows = _process_table()
    if rows is None:
        return "unknown", "could not read the process table; nothing was closed"
    parents = {pid: ppid for pid, ppid, _ in rows}
    self_pid = str(os.getpid())
    ours: list[str] = []
    others = 0
    for pid, _ppid, command in rows:
        if not _is_mcp_transport(command) or _descends_from(pid, self_pid, parents):
            continue
        by_pid = bool(client_pid) and _descends_from(pid, client_pid, parents)
        by_mark = not client_pid and bool(client_mark) and bool(
            _MARKER_RE.search(command) and _MARKER_RE.search(command).group(1) == client_mark
        )
        if by_pid or by_mark:
            ours.append(pid)
        else:
            others += 1

    if not ours:
        return "none", f"no transport of this session's ({others} belong elsewhere)"

    closed: list[str] = []
    for pid in ours:
        try:
            log = os.environ.get("RECALL_MCP_KILL_FILE")
            if log:
                with Path(log).open("a", encoding="utf-8") as handle:
                    handle.write(pid + "\n")
                ok = os.environ.get("RECALL_MCP_KILL_RC", "0") == "0"
            elif os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", pid, "/T", "/F"],
                    capture_output=True, timeout=1.5,
                )
                ok = result.returncode == 0
            else:
                os.kill(int(pid), signal.SIGTERM)
                ok = True
        except (OSError, ValueError, subprocess.SubprocessError):
            ok = False
        if ok:
            closed.append(pid)

    failed = [pid for pid in ours if pid not in closed]
    detail = f"closed {len(closed)} of {len(ours)}; {others} belong elsewhere and were left alone"
    if failed:
        detail += f"; FAILED {', '.join(failed)}"
        return ("partial" if closed else "failed"), detail
    return "closed", detail
