#!/usr/bin/env python3
"""Sweep MCP servers on VPS2 whose transport is gone, and refuse to touch anything else.

Why this needs a marker at all
------------------------------
`session-mcp-close.sh` closes the transports THIS session opened, which stops a session leaking.
It cannot help with what has already leaked: a server whose client vanished still has a live ssh
wrapper on the host, because ssh sets no keepalive, so from the host it is indistinguishable from a
server somebody is querying right now.

Measured on VPS2 on 2026-08-26, and every number here is the corrected one (see the note below on
what the earlier figures counted): **18 live servers holding 14.67 GB**, plus their 18 transport
wrappers holding 0.39 GB, on a 47 GB host that also runs the live trading services. Sixteen of the
eighteen were launched by a DIFFERENT agent on this same workstation, with a command line that
differs from ours only in which checkout it cds into. So "kill what looks like mine" is not
available: it would take down another agent's tools mid-query, which is the same mistake as
removing a container somebody is mid-run against.

⚠️ **Correction, 2026-08-26.** Earlier notes in this repository, including CLAUDE.md, the
/session-close skill and PR #526, say "89 live `recall_mcp.server` processes, 21.5 GB". The
memory figure is right; the COUNT was roughly double, because it counted each server AND its ssh
wrapper: both carry the string `python -m recall_mcp.server`, one as the command it runs and one
inside `--cmd=...`. Measured today the split is exactly 1:1, 18 servers at ~815 MB against 18
wrappers at ~22 MB. Re-measure with `scripts/session-mcp-close.sh report`, which no longer counts
wrappers as servers.

What the marker is, and what it licenses
----------------------------------------
`scripts/session-mcp.sh` now stamps `RECALL_MCP_CLIENT=<host>-<checkout id>` into every server
command it generates. The wrapper process on the host carries it in `ps`, and so does the local
`ssh` transport, because they are two ends of one command line. That gives a POSITIVE test that
needs no guessing and no age heuristic:

    a marked server whose mark names THIS host, and for which no live local transport
    carries that same mark, has no client. Nothing can be querying it.

Everything else is reported and left alone, including:

  * marks naming another machine, which this workstation cannot speak for;
  * unmarked servers from another agent's config, which are not ours to judge;
  * unmarked servers from OUR config, which predate the marker. `--unmarked` sweeps those, but
    only when this machine has zero live unmarked transports of our shape, because until that is
    true one of those servers might be held by a client still running here.

Usage:
    python scripts/session_mcp_sweep.py                 # report only
    python scripts/session_mcp_sweep.py --kill          # close marked orphans
    python scripts/session_mcp_sweep.py --kill --unmarked   # also pre-marker servers of ours
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass

#: The command that IS a server, as opposed to the wrapper whose `--cmd=` merely CONTAINS it.
#
# ⛔ Anchored at the start, and this is not a nicety. The unanchored form matched `/python` inside
# the wrapper's `--cmd=... exec .venv/bin/python -m recall_mcp.server`, so every server was counted
# twice: once as itself and once as its transport. That is exactly the defect this file's header
# corrects in the older notes, and the first draft of this file reproduced it, reporting 36 servers
# where there were 18.
SERVER_RE = re.compile(r"\S*python[0-9.]* -m recall_mcp\.server")
MARK_RE = re.compile(r"RECALL_MCP_CLIENT=(\S+)")
#: Our config cds into the `serving` symlink; the other agent's config on this machine cds into
#: `~/recall-repos` itself. That is the only thing separating the two before the marker existed.
OUR_SHAPE = "/serving"

REMOTE_PROBE = r"""
import re, subprocess, sys
SERVER = re.compile(r"\S*python[0-9.]* -m recall_mcp\.server")
MARK = re.compile(r"RECALL_MCP_CLIENT=(\S+)")
out = subprocess.run(["ps", "-eo", "pid,ppid,rss,etimes,args", "--no-headers"],
                     capture_output=True, text=True).stdout
table, recs = {}, []
for line in out.splitlines():
    parts = line.split(None, 4)
    if len(parts) < 5:
        continue
    table[parts[0]] = parts[4]
    recs.append(parts)
for pid, ppid, rss, et, args in recs:
    if not SERVER.match(args):
        continue
    parent = table.get(ppid, "")
    m = MARK.search(parent)
    shape = "ours" if "/serving" in parent else "other"
    print("\t".join([pid, ppid, rss, et, m.group(1) if m else "-", shape]))
"""


@dataclass(frozen=True)
class Server:
    pid: str
    wrapper_pid: str
    rss_kb: int
    age_s: int
    mark: str  # "-" when the server predates the marker
    shape: str  # "ours" or "other"

    @property
    def gb(self) -> float:
        return self.rss_kb / 1048576


def parse_remote(text: str) -> list[Server]:
    out = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 6:
            continue
        pid, wpid, rss, age, mark, shape = parts
        if not (pid.isdigit() and rss.isdigit() and age.isdigit()):
            continue
        out.append(Server(pid, wpid, int(rss), int(age), mark, shape))
    return out


def local_transports(table: str) -> tuple[set[str], int]:
    """Marks held by live local transports, and how many of OUR shape carry no mark.

    The second number is the whole gate on `--unmarked`: while it is above zero, a client running
    on this machine may be holding one of the unmarked servers, and no age or count can say which.
    """
    marks: set[str] = set()
    unmarked_ours = 0
    for line in table.splitlines():
        if "recall_mcp.server" not in line:
            continue
        m = MARK_RE.search(line)
        if m:
            marks.add(m.group(1))
        elif OUR_SHAPE in line:
            unmarked_ours += 1
    return marks, unmarked_ours


def classify(servers: list[Server], marks: set[str], host: str,
             unmarked_ours_local: int) -> dict[str, list[Server]]:
    """Split the fleet into what may be closed and what may not, and why."""
    buckets: dict[str, list[Server]] = {
        "orphan": [], "held": [], "other_host": [],
        "unmarked_ours": [], "unmarked_other": [],
    }
    for s in servers:
        if s.mark != "-":
            if s.mark in marks:
                buckets["held"].append(s)
            elif not s.mark.startswith(f"{host}-"):
                buckets["other_host"].append(s)
            else:
                buckets["orphan"].append(s)
        elif s.shape == "ours":
            buckets["unmarked_ours"].append(s)
        else:
            buckets["unmarked_other"].append(s)
    # Sweepable only when nothing local could still own one. Recorded on the bucket rather than
    # decided at the call site, so the reason travels with the decision.
    buckets["unmarked_ours_sweepable"] = (
        buckets["unmarked_ours"] if unmarked_ours_local == 0 else []
    )
    return buckets


# --------------------------------------------------------------------------------------------
# Gathering. Both sides have a FILE seam rather than a command seam, for the reason the SessionEnd
# hook states: `shlex.split` of a Windows path eats its backslashes, so a command-shaped seam only
# works on POSIX and these tests run on Windows.


def read_local_table() -> str:
    override = os.environ.get("RECALL_MCP_PS_FILE")
    if override:
        try:
            return open(override, encoding="utf-8", errors="replace").read()
        except OSError:
            return ""
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process | ForEach-Object "
               '{ "$($_.ProcessId) $($_.ParentProcessId) $($_.CommandLine)" }']
    else:
        cmd = ["ps", "-eo", "pid=,ppid=,args="]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read_remote_table(host: str) -> str | None:
    """None means "could not ask", which must never be read as "nothing is running"."""
    override = os.environ.get("RECALL_MCP_REMOTE_FILE")
    if override:
        try:
            return open(override, encoding="utf-8", errors="replace").read()
        except OSError:
            return None
    ssh = shutil.which("ssh")
    if not ssh:
        return None
    try:
        proc = subprocess.run(
            [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "python3", "-"],
            input=REMOTE_PROBE, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def kill_remote(host: str, pids: list[str]) -> bool:
    """Kill the server AND its wrapper. Recorded to a file instead when the seam is set."""
    if not pids:
        return True
    log = os.environ.get("RECALL_MCP_KILL_FILE")
    if log:
        try:
            with open(log, "a", encoding="utf-8") as fh:
                for pid in pids:
                    print(pid, file=fh)
        except OSError:
            return False
        return os.environ.get("RECALL_MCP_KILL_RC", "0") == "0"
    ssh = shutil.which("ssh")
    if not ssh:
        return False
    try:
        proc = subprocess.run(
            [ssh, "-o", "BatchMode=yes", host, "kill " + " ".join(pids)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _line(label: str, servers: list[Server], note: str) -> str:
    gb = sum(s.gb for s in servers)
    return f"  {label:<16} {len(servers):>3} server(s)  {gb:>6.2f} GB   {note}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Close MCP servers on the host whose client is gone.")
    ap.add_argument("--kill", action="store_true", help="actually close the orphans")
    ap.add_argument("--unmarked", action="store_true",
                    help="also close pre-marker servers of ours, if nothing local could hold one")
    ap.add_argument("--host", default=os.environ.get("RECALL_VPS2_HOST", "vps2"))
    ap.add_argument("--client-host", default=os.environ.get("RECALL_MCP_CLIENT_HOST") or
                    socket.gethostname())
    args = ap.parse_args(argv)

    remote = read_remote_table(args.host)
    if remote is None:
        print(f"UNREACHABLE  could not read the process table on {args.host}; nothing was closed.")
        return 2
    servers = parse_remote(remote)
    marks, unmarked_ours_local = local_transports(read_local_table())
    buckets = classify(servers, marks, args.client_host, unmarked_ours_local)

    total_gb = sum(s.gb for s in servers)
    print(f"FLEET  {len(servers)} server(s) on {args.host}, {total_gb:.2f} GB resident")
    print(f"LOCAL  {len(marks)} marked transport(s) live here, "
          f"{unmarked_ours_local} unmarked one(s) of our shape")
    print(_line("orphan", buckets["orphan"], "marked ours, no live transport -> closeable"))
    print(_line("held", buckets["held"], "a live local transport holds these"))
    print(_line("other host", buckets["other_host"], "launched from another machine"))
    print(_line("unmarked ours", buckets["unmarked_ours"],
                "pre-marker; --unmarked closes these once nothing local is unmarked"))
    print(_line("unmarked other", buckets["unmarked_other"],
                "another agent's config; never swept from here"))

    targets = list(buckets["orphan"])
    if args.unmarked:
        if unmarked_ours_local:
            print(f"REFUSED  --unmarked: {unmarked_ours_local} unmarked transport(s) of our shape "
                  "are live here, so one of those servers may be in use. Restart those clients "
                  "(they will come back marked) and re-run.")
        else:
            targets += buckets["unmarked_ours_sweepable"]

    if not targets:
        print("RESULT  nothing is closeable with positive identity.")
        return 0
    freed = sum(s.gb for s in targets)
    if not args.kill:
        print(f"RESULT  {len(targets)} server(s) closeable, {freed:.2f} GB. Re-run with --kill.")
        return 0

    pids: list[str] = []
    for s in targets:
        pids.extend([s.pid, s.wrapper_pid])
    if not kill_remote(args.host, pids):
        print("FAILED   the kill did not run; nothing is known to have closed.")
        return 1
    print(f"CLOSED   {len(targets)} server(s), about {freed:.2f} GB.")
    after = read_remote_table(args.host)
    if after is not None:
        left = parse_remote(after)
        print(f"FLEET    now {len(left)} server(s), {sum(s.gb for s in left):.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
