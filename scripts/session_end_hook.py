#!/usr/bin/env python3
"""SessionEnd hook: release this session's claim and remove only its own container.

The other half of scripts/session_start_hook.py. Opening was automated because
the protocol was correct and never invoked; closing stayed prose and cost the
same thing within hours: a claim released by hand, and a 4.9-hour indexer still
running on a remote host because nothing swept it.

**It records; it does not tidy.** No commit, no push, no staging, no branch
deletion. Uncommitted work at session end is a decision for a person, and a hook
that quietly commits, or quietly discards, is worse than one that leaves the tree
alone and writes down what it saw.

It closes exactly two things, and both are THIS session's: the container carrying
this checkout's label, and the MCP transports whose parent chain reaches this
session's client process. The second was added on 2026-08-26 after measuring the
cost of not doing it: **18 live servers on VPS2 holding 14.67 GB**, on a 47 GB
host that also runs live trading services. Each server lives exactly as long as
its stdio transport, ssh sets no keepalive, and a client that vanishes therefore
leaves ~815 MB running with nothing anywhere reporting it. The same measurement
found transports with an IDENTICAL command line belonging to a different agent
(`codex.exe`), which is why ownership is the parent chain and never the command
line.

(An earlier version of this paragraph said "89 processes, 21.5 GB". The memory
was right and the count was about double: a server and its ssh wrapper both carry
the string `python -m recall_mcp.server`, one as the command it runs and one
inside `--cmd=`, so a matching count reports every server twice.)

Four rules, every one of them written because the first version of this file
broke it and an audit caught it before it could do damage:

1. **Release only on positive identity.** `holder == this session`, both
   non-empty. The first version guarded with `if holder and session_id and
   holder != session_id`, so an empty session id or a claim file with no
   `session=` line fell through and deleted a claim it did not own, reporting
   "released". Unknown ownership is never a licence to delete.
2. **Remove the container only if the claim was ours.** A session refused the
   workspace still runs (SessionStart cannot veto), and `recall.checkout` labels
   the CHECKOUT, not the session, so an unconditional removal deletes the
   legitimate holder's database out from under a running suite.
3. **Release the claim LAST**, after the container is gone.
   `scripts/session-close.sh` already says why: "A claim dropped early would let
   another session move in while this one is still tearing its container down."
   With a per-checkout label, the incoming session's brand-new container matches
   the outgoing session's filter.
4. **Never reclaim a live lock.** Breaking a lock whose holder is running is how
   two sessions both take one worktree.

`reason == "clear"` is skipped entirely: that ends the conversation, not the
session's work, and tearing down a database under a suite running in another
shell is exactly the harm this file exists to prevent.

SessionEnd output reaches no context, so the record goes to
`~/.claude/session-end.log` (override with `RECALL_SESSION_LOG` for tests).

It never raises, and it always writes a row: a silent no-op is indistinguishable
from the hook not being installed.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The import is GUARDED because it is an install-shape dependency, and an
#: unguarded one is a silent no-op waiting to happen. These hooks are installed
#: by copying single files into ~/.claude/hooks/; the start hook needs one file
#: and this one needs two, so copying "the hook" the way you copied the last one
#: leaves the module missing. At module level, outside every handler, that is
#: ModuleNotFoundError, exit 1, and NO log row: indistinguishable from the hook
#: not being installed, which is exactly what the docstring above promises
#: cannot happen. So it degrades to a row that names the missing file instead.
_IMPORT_ERROR = None
try:
    from session_hook_common import (  # noqa: E402
        LAUNCH_FAILED,
        NOT_ATTEMPTED,
        acquire_claim_lock,
        git,
        read_claim,
        release_claim_lock,
        run,
    )
except Exception as exc:  # noqa: BLE001 - pragma: no cover, exercised by a subprocess test
    # NOT just ImportError. A half-copied module raises SyntaxError, which is not
    # an ImportError, and that put the hook straight back to exit 1 with no row.
    # Under a copy-by-hand install a truncated file is as likely as a missing one.
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

#: A hook at session end competes with the process actually exiting, so it gets a
#: tighter budget than the start hook.
TIMEOUT = 10
BUDGET = 15.0
_STARTED = time.monotonic()


def budget_left() -> float:
    return BUDGET - (time.monotonic() - _STARTED)


def _log_path() -> Path:
    override = os.environ.get("RECALL_SESSION_LOG")
    return Path(override) if override else Path.home() / ".claude" / "session-end.log"


def read_payload(timeout: float = 5.0) -> dict:
    """Read the JSON payload from stdin, bounded.

    `json.load(sys.stdin)` is an unbounded read. A hook invoked with no payload,
    or with an inherited console handle, would block until the harness killed it,
    delaying the shutdown this file promises never to delay and leaving no record
    at all because the log write is the last statement.
    """
    box: dict = {}

    def _read():
        try:
            box["raw"] = sys.stdin.read()
        except Exception as exc:  # noqa: BLE001 - a traceback here breaks the promise
            box["raw"] = ""
            box["err"] = type(exc).__name__

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if box.get("err"):
        return {"_stdin": f"unreadable: {box['err']}"}
    raw = box.get("raw")
    if raw is None:
        return {"_stdin": "timed out"}
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {"_stdin": "unparseable"}
    return payload if isinstance(payload, dict) else {"_stdin": "not-an-object"}


def release_claim(git_dir: Path, session_id: str, pid: str) -> str:
    """Release the claim only on positive identity, and only under the lock.

    Read, decide and unlink all happen while holding `claude-session-claim.lock`,
    because a check taken before the lock is a check about the past. The lock is
    never broken from a live holder.
    """
    claim_file = git_dir / "claude-session-claim"
    lock_dir = claim_file.with_name(claim_file.name + ".lock")
    if not claim_file.exists():
        return "no claim to release"
    if not acquire_claim_lock(lock_dir, pid, budget_left):
        return "left in place: another session holds the claim lock"
    try:
        holder = read_claim(claim_file).get("session", "")
        # Positive identity only. Both must be present AND equal; anything else,
        # including "we could not tell whose this is", leaves the file alone.
        if not session_id or not holder or holder != session_id:
            whose = holder[:8] if holder else "unrecorded"
            return f"left in place: not this session's claim (holder={whose})"
        try:
            claim_file.unlink()
        except OSError as exc:
            return f"could not release: {exc}"
        return "released"
    finally:
        release_claim_lock(lock_dir)


def remove_own_container(root: Path) -> tuple[str, str]:
    """Remove containers labelled with THIS checkout. Returns (status, detail).

    Selection is by the `recall.checkout` label that session-db.sh writes, never
    by the `recall-sess-*` name pattern: matching on the name would sweep every
    session container on the machine, including ones another session has a suite
    running against.
    """
    checkout = str(root).replace("\\", "/")
    rc, out, _ = run(["docker", "ps", "-aq", "--filter",
                      f"label=recall.checkout={checkout}"],
                     timeout=TIMEOUT, budget_left=budget_left)
    if rc == NOT_ATTEMPTED:
        return "not-attempted", "time budget exhausted before the docker query"
    if rc == LAUNCH_FAILED:
        return "unknown", "could not run docker; containers were NOT checked"
    if rc != 0:
        return "unknown", f"docker query failed (rc={rc}); containers were NOT checked"
    ids = [x for x in out.splitlines() if x.strip()]
    if not ids:
        return "none", "no container carries this checkout's label"
    # Three buckets, not two. "Never launched" and "timed out" are not the same
    # fact as "docker said no", and on the timeout path the removal may actually
    # have succeeded, so calling it a failure can be wrong in both directions.
    removed, failed, unattempted = [], [], []
    for cid in ids:
        rc, _, _ = run(["docker", "rm", "-f", cid], timeout=20, budget_left=budget_left)
        if rc == 0:
            removed.append(cid[:12])
        elif rc == NOT_ATTEMPTED:
            unattempted.append(cid[:12])
        elif rc == LAUNCH_FAILED:
            # Re-ask before asserting: a timed-out rm may have completed.
            rc2, still, _ = run(["docker", "ps", "-aq", "--filter", f"id={cid}"],
                                timeout=10, budget_left=budget_left)
            (removed if rc2 == 0 and not still.strip() else failed).append(cid[:12])
        else:
            failed.append(cid[:12])
    parts = []
    if removed:
        parts.append("removed " + ", ".join(removed))
    if failed:
        parts.append("FAILED " + ", ".join(failed))
    if unattempted:
        parts.append("NOT ATTEMPTED (budget) " + ", ".join(unattempted))
    detail = "; ".join(parts)
    if failed:
        return ("failed" if not removed else "partial"), detail
    if unattempted:
        return ("not-attempted" if not removed else "partial"), detail
    return "removed", detail


#: What marks a process as an MCP transport. A module path rather than a server
#: name, because that string is what appears in the launch command `.mcp.json`
#: writes: `ssh <host> '... exec python -m recall_mcp.server'`.
MCP_PATTERN = os.environ.get("RECALL_MCP_PATTERN", "recall_mcp.server")

#: Windows has no `ps`, and `wmic` is gone from current builds. One line per
#: process, `pid ppid command line`, which is the shape the POSIX branch emits
#: too. Interpolation rather than Format-List on purpose: the formatter WRAPS at
#: the console width, and every MCP command line is long enough to be wrapped,
#: which turns one process into several unparseable fragments.
_PS_POWERSHELL = (
    "Get-CimInstance Win32_Process | ForEach-Object "
    '{ "$($_.ProcessId) $($_.ParentProcessId) $($_.CommandLine)" }'
)


def _process_table() -> tuple[list[tuple[str, str, str]] | None, str]:
    """Every process as (pid, ppid, command line), plus why there are none.

    Three outcomes, not two, which is the same rule `remove_own_container` states
    above: "we never asked, the budget was gone" is not "we asked and could not
    read it", and neither of them is "nothing was running". Collapsing the first
    two is how a shutdown that ran out of time gets logged as a broken machine.
    """
    # A FILE rather than a command, and that is deliberate. `shlex.split` of a
    # Windows path eats its backslashes, so a command-shaped seam is a seam that
    # only works on POSIX, and this hook's own tests run on Windows.
    override = os.environ.get("RECALL_MCP_PS_FILE")
    if override:
        try:
            out = Path(override).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, "unreadable"
        rows = _parse_table(out)
        return rows, ("" if rows else "unreadable")
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command", _PS_POWERSHELL]
    else:
        cmd = ["ps", "-eo", "pid=,ppid=,args="]
    rc, out, _ = run(cmd, timeout=8, budget_left=budget_left)
    if rc == NOT_ATTEMPTED:
        return None, "not-attempted"
    if rc != 0 or not out:
        return None, "unreadable"
    rows = _parse_table(out)
    return rows, ("" if rows else "unreadable")


def _parse_table(out: str) -> list[tuple[str, str, str]] | None:
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rows or None


def _descends_from(pid: str, want: str, parents: dict) -> bool:
    """Is `want` an ancestor of `pid`, within 8 hops?

    The hop cap is a termination guarantee, not a performance choice: a recycled
    pid can make a parent chain point back into its own descendants, and a walk
    without a cap never returns from that.
    """
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


def _kill_pid(pid: str) -> bool:
    # The test seam: record the pid instead of signalling it. Every subprocess
    # test in this file sets it, because the alternative is a test run that kills
    # the developer's own MCP servers.
    log = os.environ.get("RECALL_MCP_KILL_FILE")
    if log:
        try:
            with open(log, "a", encoding="utf-8") as fh:
                print(pid, file=fh)
        except OSError:
            return False
        return os.environ.get("RECALL_MCP_KILL_RC", "0") == "0"
    if os.name == "nt":
        # `/T` because the transport has children of its own: an ssh configured
        # with a ProxyCommand spawns a second ssh, and killing only the parent
        # leaves that one orphaned and connected.
        rc, _, _ = run(["taskkill", "/PID", pid, "/T", "/F"], timeout=10,
                       budget_left=budget_left)
        return rc == 0
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except (OSError, ValueError):
        return False


def close_own_mcp_transports(client_pid: str) -> tuple[str, str]:
    """Close the MCP transports this session opened. Returns (status, detail).

    ⛔ Ownership is the PARENT CHAIN, never the command line. Measured on this
    machine on 2026-08-26: three live transports with a byte-identical
    `recall_mcp.server` command line were parented to `codex.exe` rather than to
    Claude, so a `pkill -f recall_mcp.server` here, or a pattern sweep on the
    server, would have killed another agent's servers mid-query. Without a client
    pid there is no positive identity, and this returns without killing anything
    rather than guessing, exactly as the container branch does without a claim.

    Killing the local transport is enough: the server is the process ssh owns on
    the far side, and a marked probe on 2026-08-26 measured the remote server
    gone in under 3 seconds, confirmed by pid. Nothing here reaches the host.
    """
    if not client_pid:
        return "skipped", "no client pid; ownership could not be established"
    rows, why = _process_table()
    if rows is None:
        if why == "not-attempted":
            return "not-attempted", ("the time budget was gone before the process table "
                                     "was read; nothing was closed")
        return "unknown", "could not read the process table; nothing was closed"
    parents = {pid: ppid for pid, ppid, _ in rows}
    self_pid = str(os.getpid())
    ours, others = [], 0
    for pid, _ppid, cmd in rows:
        if MCP_PATTERN not in cmd:
            continue
        # Anything this hook itself started. It spawns no transports, so this is
        # belt and braces, and it costs one comparison.
        if _descends_from(pid, self_pid, parents):
            continue
        if _descends_from(pid, client_pid, parents):
            ours.append(pid)
        else:
            others += 1
    if not ours:
        return "none", f"no transport of this session's ({others} belong elsewhere)"
    closed = [pid for pid in ours if _kill_pid(pid)]
    failed = [pid for pid in ours if pid not in closed]
    detail = f"closed {len(closed)} of {len(ours)}; {others} belong elsewhere and were left alone"
    if failed:
        return "partial" if closed else "failed", detail + f"; FAILED {', '.join(failed)}"
    return "closed", detail


def survey(root: Path) -> dict:
    """What is being left behind. Recorded, never acted on.

    Anything not measured is recorded as None, never as an empty string: the log
    is this hook's only output, and "could not tell" must not read as "nothing".
    """
    # EVERY field, including the list. An unmeasured sample left as [] reads as
    # "no dirty files", which is the ambiguity this rule exists to remove.
    out: dict = {"branch": None, "uncommitted": None, "uncommitted_sample": None,
                 "commits_ahead_of_trunk": None}
    rc, dirty = git(root, "status", "--porcelain", timeout=TIMEOUT, budget_left=budget_left)
    if rc == 0:
        files = [x for x in dirty.splitlines() if x.strip()]
        out["uncommitted"] = len(files)
        out["uncommitted_sample"] = files[:5]
    rc, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD",
                     timeout=TIMEOUT, budget_left=budget_left)
    if rc == 0 and branch:
        out["branch"] = branch
    ref = None
    rc, head = git(root, "rev-parse", "--abbrev-ref", "origin/HEAD",
                   timeout=TIMEOUT, budget_left=budget_left)
    if rc == 0 and head and "/" in head:
        ref = head
    else:
        for cand in ("origin/master", "origin/main"):
            rc, _ = git(root, "rev-parse", "--verify", "-q", cand,
                        timeout=TIMEOUT, budget_left=budget_left)
            if rc == 0:
                ref = cand
                break
    if ref:
        rc, n = git(root, "rev-list", "--count", f"{ref}..HEAD",
                    timeout=TIMEOUT, budget_left=budget_left)
        if rc == 0 and n.isdigit():
            out["commits_ahead_of_trunk"] = int(n)
    return out


def write_row(row: dict) -> None:
    log = _log_path()
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists() and log.stat().st_size > 512_000:
            keep = log.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
            tmp = log.with_name(log.name + f".{os.getpid()}.tmp")
            tmp.write_text("\n".join(keep) + "\n", encoding="utf-8", newline="\n")
            os.replace(tmp, log)
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except (OSError, ValueError, TypeError):
        pass


def main() -> int:
    row: dict = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "outcome": "error"}
    surveyed_root = None
    if _IMPORT_ERROR:
        row["outcome"] = "no-common-module"
        row["detail"] = (f"{_IMPORT_ERROR}. session_hook_common.py must be installed "
                         "beside this hook; nothing was released or removed.")
        row["elapsed_s"] = round(time.monotonic() - _STARTED, 2)
        write_row(row)
        return 0
    try:
        payload = read_payload()
        # An unread payload is not a blank one. `reason` is lost with it, so the
        # `clear` skip silently stops protecting anything, and session_id is lost
        # too, so ownership cannot be established. Refuse rather than proceed on
        # defaults: this path ends in `docker rm -f`.
        if payload.get("_stdin"):
            row["outcome"] = "payload-unreadable"
            row["stdin"] = payload["_stdin"]
            row["detail"] = "nothing was released or removed"
            return 0
        # str() rather than a bare slice: a non-string session_id raised
        # TypeError here, escaped, and was swallowed by the top-level handler
        # before any row was written. A silent no-op is indistinguishable from
        # the hook not being installed.
        session_id = str(payload.get("session_id") or "")
        reason = str(payload.get("reason") or "")
        row.update({"session": session_id[:8], "reason": reason or None,
                    "cwd": str(payload.get("cwd") or "")})
        if payload.get("_stdin"):
            row["stdin"] = payload["_stdin"]

        # `clear` ends the conversation, not the work. The same terminal keeps
        # running, and a suite in another shell would lose its database.
        if reason == "clear":
            row["outcome"] = "skipped-clear"
            return 0

        # BEFORE the cwd and git checks, and outside the claim gate, because the
        # transports are the SESSION's rather than the checkout's, and because
        # every early return below would otherwise skip them. That is not
        # hypothetical: measured 2026-08-26, the last three real rows in this
        # log are `not-a-git-repo` with a home-directory cwd, which is where the
        # app puts a session that never opened a repository. Those sessions have
        # MCP servers like any other, and they were the ones leaking.
        #
        # `CLAUDE_PID` is present in a hook the CLIENT spawns, and that is
        # measured rather than assumed: this worktree's claim file, written at
        # session start by the client-spawned SessionStart hook, records
        # `pid=9764`, and pid 9764 is a `claude.exe` whose own parent is the app
        # root 14992. So the client is a per-SESSION process, and a parent chain
        # reaching it separates two sessions of the same app, which a chain
        # reaching the app root would not. If the variable is ever absent the row
        # says `skipped`, naming the reason, rather than falling back to a guess.
        row["mcp"], row["mcp_detail"] = close_own_mcp_transports(
            os.environ.get("CLAUDE_PID", ""))

        cwd = row["cwd"] or os.getcwd()
        row["cwd_effective"] = cwd
        if not Path(cwd).is_dir():
            row["outcome"] = "cwd-unusable"
            row["detail"] = f"cannot enter {cwd!r}; nothing was released or removed"
            return 0
        rc, top = git(cwd, "rev-parse", "--show-toplevel", timeout=TIMEOUT,
                      budget_left=budget_left)
        if rc in (LAUNCH_FAILED, NOT_ATTEMPTED):
            # "git would not run" is not a fact about this directory. Reporting
            # it as one sends whoever reads the log looking for the wrong thing,
            # and this log is the hook's only output.
            row["outcome"] = "git-unavailable"
            row["detail"] = f"git returned {rc}; nothing was released or removed"
            return 0
        if rc != 0 or not top:
            row["outcome"] = "not-a-git-repo"
            return 0

        root = Path(top)
        surveyed_root = root
        row["root"] = str(root)
        rc, git_dir = git(cwd, "rev-parse", "--absolute-git-dir", timeout=TIMEOUT,
                          budget_left=budget_left)
        if rc in (LAUNCH_FAILED, NOT_ATTEMPTED):
            row["outcome"] = "git-unavailable"
            row["detail"] = f"git returned {rc} locating the git dir"
            return 0
        if rc != 0 or not git_dir:
            row["outcome"] = "no-git-dir"
            return 0

        # Ownership decides everything that follows, and it requires POSITIVE
        # identity. An absent claim is not consent: the main checkout never
        # carries one, because session-space.sh refuses that checkout before it
        # would write one, so "no claim means mine" let a session end in the
        # shared checkout and force-remove another session's running database.
        # Proven by an auditor against a real labelled container. The label is
        # per-CHECKOUT, so it cannot distinguish sessions; only the claim can.
        claim_file = Path(git_dir) / "claude-session-claim"
        holder = read_claim(claim_file).get("session", "")
        row["claim_holder"] = holder[:8] if holder else None
        ours = bool(session_id) and bool(holder) and holder == session_id

        if not ours:
            whose = holder[:8] if holder else "unrecorded"
            why = ("no claim on this checkout" if not claim_file.exists()
                   else f"claimed by {whose}")
            row["container"] = "skipped"
            row["container_detail"] = f"ownership not established: {why}"
            row["claim"] = f"left in place: not this session's claim (holder={whose})"
            row["outcome"] = "closed-not-owner"
            return 0

        # Container FIRST, claim LAST, and the claim only if the container is
        # provably gone. Releasing early lets the next session claim the worktree
        # and start a container this filter would then match; releasing after a
        # FAILED teardown strands a container that nothing can later attribute,
        # because session-db.sh only reports orphans whose checkout has vanished.
        status, detail = remove_own_container(root)
        row["container"], row["container_detail"] = status, detail
        if status in ("removed", "none"):
            row["claim"] = release_claim(Path(git_dir), session_id,
                                         os.environ.get("CLAUDE_PID", str(os.getpid())))
            # Only the literal "released" may read as closed. A claim left in
            # place by a stranded lock, or an unlink that failed, leaves this
            # worktree claimed, and a row saying "closed" beside it is the same
            # conflation the container branch was fixed to avoid.
            row["outcome"] = ("closed" if row["claim"] == "released"
                              else "closed-with-claim-not-released")
        else:
            row["claim"] = ("left in place: the container step reported "
                            f"{status!r}, so this checkout still owns something")
            row["outcome"] = f"closed-with-{status}-container"
    except Exception as exc:  # noqa: BLE001 - a shutdown must not be blocked
        row["outcome"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        # Survey last in TIME, so reporting cannot spend the budget the acting
        # steps need, but on EVERY path that knows a root, and outside the
        # try/except above so a failure here cannot rewrite a completed
        # teardown's outcome as "error".
        if surveyed_root is not None:
            try:
                row.update(survey(surveyed_root))
            except Exception as exc:  # noqa: BLE001
                row["survey_error"] = f"{type(exc).__name__}: {exc}"[:120]
        row["elapsed_s"] = round(time.monotonic() - _STARTED, 2)
        write_row(row)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        sys.exit(0)
