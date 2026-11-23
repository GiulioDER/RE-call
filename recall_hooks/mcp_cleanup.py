"""Close only the local MCP transports owned by the ending client session."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

MCP_PATTERN = "recall_mcp.server"
_MARKER_RE = re.compile(r"(?<!\S)RECALL_MCP_CLIENT=([^\s;&\"']+)")
_SESSION_ID_RE = re.compile(r"(?<!\S)RECALL_MCP_SESSION_ID=([^\s;&\"']+)")
_LAUNCHER_RE = re.compile(r"(?<!\S)(?:\S*[\\/])?recall-(?:codex-)?mcp(?:\s|$)")
_NEWLINES = re.compile(r"[\r\n]+")


def _mcp_pattern() -> str:
    # RECALL_MCP_PATTERN is a test seam like the two below, honoured only while both of them are
    # set: left in a real environment, a short value would widen the kill selector to every
    # descendant of the client whose command line contains it. An empty or blank override would
    # match every command line, so it falls back to the default too.
    if not (os.environ.get("RECALL_MCP_PS_FILE") and os.environ.get("RECALL_MCP_KILL_FILE")):
        return MCP_PATTERN
    override = os.environ.get("RECALL_MCP_PATTERN", "").strip()
    return override or MCP_PATTERN


def _is_mcp_transport(command: str) -> bool:
    return _mcp_pattern() in command or bool(_LAUNCHER_RE.search(command))


def _process_table(deadline: float | None = None) -> list[tuple[str, str, str]] | None:
    override = os.environ.get("RECALL_MCP_PS_FILE")
    output: str | None
    if override:
        try:
            output = Path(override).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    elif os.name == "nt":
        # From the Win32 API, not PowerShell's Get-CimInstance, which took 2.2 to 9 s
        # (2026-11-04) and 1.2 to 2.4 s (2026-11-20) here, against Codex's SessionEnd cap of 3 s
        # (#1454). The native listing took 0.06 to 0.27 s for the same processes.
        try:
            output = _windows_process_rows(deadline)
        except (OSError, ValueError):
            # A refused API call is an unreadable table: report `unknown` and close nothing.
            return None
    else:
        timeout = 2.0 if deadline is None else min(2.0, deadline - time.monotonic())
        if timeout <= 0:
            return None
        try:
            output = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,args="],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    if output is None:
        return None
    return _parse_rows(output)


def _is_pid(token: str) -> bool:
    # ASCII digits with no leading zero: taskkill reads `01234` as pid 1234, so a padded pid would
    # pass the duplicate filter below as a second, unrelated process (SEC-001, #1454 audit).
    return token.isascii() and token.isdigit() and str(int(token)) == token


def _parse_rows(output: str) -> list[tuple[str, str, str]]:
    """`(pid, ppid, command line)` per line of a process table, forged-looking rows dropped."""
    rows = []
    # "\n" only. `str.splitlines` also breaks on \v, \f, \x1c to \x1e, U+0085, U+2028 and U+2029,
    # so a command line holding one would carry a line of its own shaped like a row (SEC-001; the
    # same row through CR/LF was STAKES-007).
    for line in output.split("\n"):
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and _is_pid(parts[0]) and _is_pid(parts[1]):
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    # A pid listed twice means one of its rows is not a process (a forged line, however it got
    # in); neither row can be trusted for a parent chain that decides a kill.
    seen: dict[str, int] = {}
    for pid, _ppid, _command in rows:
        seen[pid] = seen.get(pid, 0) + 1
    return [row for row in rows if seen[row[0]] == 1]


def _windows_process_rows(deadline: float | None = None) -> str | None:
    """`pid ppid command line` per process, one per line, read with the Win32 API; None on failure.

    Every pid and parent pid comes from one Toolhelp snapshot, so the parent chain is consistent.
    A command line comes from `NtQueryInformationProcess(ProcessCommandLineInformation)`; a
    process this user cannot open (another user's, an elevated one) gets an empty one, as WMI
    gave. Each command line is flattened to one line: a multi-line one (a shell heredoc) could
    otherwise carry a line shaped like a row and name an arbitrary pid as the client's.
    """
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
        ]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
            ("Buffer", ctypes.c_void_p),
        ]

    windll = getattr(ctypes, "WinDLL")
    kernel32 = windll("kernel32", use_last_error=True)
    ntdll = windll("ntdll")
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    th32cs_snapprocess, process_query_limited_information, command_line_class = 0x2, 0x1000, 60

    def command_line(pid: int) -> str:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("process table deadline expired")
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.ULONG(0)
            ntdll.NtQueryInformationProcess(handle, command_line_class, None, 0, ctypes.byref(size))
            if not size.value:
                return ""
            buffer = ctypes.create_string_buffer(size.value)
            if ntdll.NtQueryInformationProcess(
                handle, command_line_class, buffer, size, ctypes.byref(size)
            ) != 0:
                return ""
            text = UnicodeString.from_buffer(buffer)
            return ctypes.wstring_at(text.Buffer, text.Length // 2) if text.Buffer else ""
        finally:
            kernel32.CloseHandle(handle)

    snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return None
    pairs: list[tuple[int, int]] = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            pairs.append((entry.th32ProcessID, entry.th32ParentProcessID))
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if not pairs:
        return None
    return _format_rows(pairs, command_line)


def _format_rows(pairs: list[tuple[int, int]], command_line: Callable[[int], str]) -> str:
    """The table text `_parse_rows` reads, one `pid ppid command line` per line.

    Separate from the Win32 calls so the flattening, which keeps a command line on its own row,
    is tested on every platform (CODE-001, CODE-002 of the #1454 audit).
    """
    return "\n".join(f"{pid} {ppid} {_NEWLINES.sub(' ', command_line(pid))}" for pid, ppid in pairs)


def _same_invocation(parent_command: str, own_command: str) -> bool:
    """Whether two command lines run the same arguments, whatever the executable that runs them."""
    parent_args = parent_command.split(None, 1)[1:]
    own_args = own_command.split(None, 1)[1:]
    return bool(own_args) and parent_args == own_args


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


def _identity_from_config(cwd: str) -> tuple[str, str]:
    """Recover one session ID and one client marker from the generated MCP config."""
    if not cwd:
        return "", ""
    current = Path(cwd).expanduser()
    if not current.is_dir():
        current = current.parent
    for directory in (current, *current.parents):
        path = directory / ".mcp.json"
        try:
            # `utf-8-sig`: Windows PowerShell 5.1 writes a BOM, and a BOM'd config is still ours.
            text = path.read_text(encoding="utf-8-sig")
            document: Any = json.loads(text)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, RecursionError):  # ValueError: undecodable text, bad JSON
            # A nearest config that EXISTS but cannot be read (no permission, a directory of that
            # name) or parsed is a boundary like an ambiguous one: walking past it would adopt an
            # ENCLOSING checkout's identity as this session's. Only an absent file walks on.
            return "", ""
        markers: set[str] = set()
        session_ids: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, str):
                markers.update(match.group(1) for match in _MARKER_RE.finditer(value))
                session_ids.update(match.group(1) for match in _SESSION_ID_RE.finditer(value))
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(document)
        # Conflicting session identities cannot fall back to a shared client marker.
        if len(session_ids) > 1:
            return "", ""
        if len(markers) == 1:
            marker = next(iter(markers))
        elif markers:
            marker = ""
        else:
            marker = ""
        if len(session_ids) == 1:
            session_id = next(iter(session_ids))
        else:
            session_id = ""
        # An ambiguous nearest config is a boundary, not an absent identity.
        if markers or session_ids:
            return session_id, marker
    return "", ""


def _marker_from_config(cwd: str) -> str:
    """Recover a single client marker from the generated project MCP config, if present."""
    return _identity_from_config(cwd)[1]


def _session_id_from_config(cwd: str) -> str:
    """Recover a single per-session ID from the generated project MCP config, if present."""
    return _identity_from_config(cwd)[0]


def close_own_mcp_transports(
    client_pid: str = "", client_mark: str = "", *, session_id: str = "", cwd: str = "",
    deadline: float | None = None,
) -> tuple[str, str]:
    """Close this session's transports and leave every unowned transport alone.

    Ownership is decided in this order: descent from the client pid; without a pid, an exact
    RECALL_MCP_SESSION_ID on the transport's command line; without either, an exact
    RECALL_MCP_CLIENT marker. Conflicting session IDs in the nearest `.mcp.json`, or a nearest
    one that cannot be parsed, yield no identity; conflicting markers alone drop only the marker,
    and a single session ID still counts. RECALL_MCP_PATTERN is honoured only with the two test
    seams below. Returns
    (status, detail) with status one of skipped, unknown, none, closed, partial or failed.
    With a monotonic deadline, stop attempting closes when the shared budget is exhausted.

    RECALL_MCP_PS_FILE (a process table read from a file) and RECALL_MCP_KILL_FILE (a log written
    instead of a kill, with RECALL_MCP_KILL_RC as its result) are test seams. They are honoured
    only together, so a fake table never drives a real kill and a fake kill never reports real
    processes as closed; one set alone closes nothing and reports `unknown`.
    """
    if bool(os.environ.get("RECALL_MCP_PS_FILE")) != bool(os.environ.get("RECALL_MCP_KILL_FILE")):
        return "unknown", (
            "RECALL_MCP_PS_FILE and RECALL_MCP_KILL_FILE are test seams and must be set together; "
            "nothing was closed"
        )
    client_pid = client_pid or os.environ.get("CLAUDE_PID", "")
    # Pid 1 (or 0) is what an orphaned hook reports as its parent: every process descends from
    # it, so it would select every session's transports. Such a pid is no identity at all.
    if client_pid and not (client_pid.isdigit() and int(client_pid) > 1):
        client_pid = ""
    session_id = session_id or os.environ.get("RECALL_MCP_SESSION_ID", "")
    client_mark = client_mark or os.environ.get("RECALL_MCP_CLIENT", "")
    config_session_id, config_mark = _identity_from_config(cwd or os.getcwd())
    session_id = session_id or config_session_id
    client_mark = client_mark or config_mark
    if not client_pid and not session_id and not client_mark:
        return "skipped", "no client pid, session ID, or marker; ownership could not be established"

    if deadline is not None and time.monotonic() >= deadline:
        return "skipped", "transport cleanup budget exhausted"
    rows = _process_table(deadline)
    if rows is None:
        return "unknown", "could not read the process table; nothing was closed"
    parents = {pid: ppid for pid, ppid, _ in rows}
    commands = {pid: command for pid, _ppid, command in rows}
    self_pid = str(os.getpid())
    # A venv's Windows `python.exe` is a launcher that runs the real interpreter as its child, so
    # the hook's parent can be that launcher rather than the client (owner decision, 2026-11-04):
    # nothing descends from it, and cleanup used to close nothing. A parent whose arguments are
    # this process's own is the launcher; the client is ITS parent.
    if client_pid and client_pid in commands and _same_invocation(commands[client_pid], commands.get(self_pid, "")):
        client_pid = parents.get(client_pid, "")
        if client_pid and not (client_pid.isdigit() and int(client_pid) > 1):
            client_pid = ""
    ours: list[str] = []
    others = 0
    for pid, _ppid, command in rows:
        if not _is_mcp_transport(command) or _descends_from(pid, self_pid, parents):
            continue
        by_pid = bool(client_pid) and _descends_from(pid, client_pid, parents)
        if by_pid and session_id:
            # One client process can host several sessions: a transport marked with another
            # session's ID is that session's, even though it descends from the same client.
            marked = _SESSION_ID_RE.search(command)
            if marked is not None and marked.group(1) != session_id:
                by_pid = False
        by_mark = False
        if not client_pid and session_id:
            session_match = _SESSION_ID_RE.search(command)
            by_mark = session_match is not None and session_match.group(1) == session_id
        elif not client_pid and client_mark:
            marker_match = _MARKER_RE.search(command)
            by_mark = marker_match is not None and marker_match.group(1) == client_mark
        if by_pid or by_mark:
            ours.append(pid)
        else:
            others += 1

    if not ours:
        return "none", f"no transport of this session's ({others} belong elsewhere)"

    closed: list[str] = []
    for pid in ours:
        timeout = 1.5 if deadline is None else min(1.5, deadline - time.monotonic())
        if timeout <= 0:
            break
        try:
            log = os.environ.get("RECALL_MCP_KILL_FILE")
            if log:
                with Path(log).open("a", encoding="utf-8") as handle:
                    handle.write(pid + "\n")
                ok = os.environ.get("RECALL_MCP_KILL_RC", "0") == "0"
            elif os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", pid, "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
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
