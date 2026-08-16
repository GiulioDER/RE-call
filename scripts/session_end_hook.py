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
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_hook_common import (  # noqa: E402
    LAUNCH_FAILED,
    NOT_ATTEMPTED,
    acquire_claim_lock,
    git,
    read_claim,
    release_claim_lock,
    run,
)

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
        except (OSError, ValueError):
            box["raw"] = ""

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
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
    removed, failed = [], []
    for cid in ids:
        rc, _, _ = run(["docker", "rm", "-f", cid], timeout=20, budget_left=budget_left)
        (removed if rc == 0 else failed).append(cid[:12])
    if failed and not removed:
        return "failed", f"could not remove {', '.join(failed)}"
    if failed:
        return "partial", f"removed {', '.join(removed)}; FAILED {', '.join(failed)}"
    return "removed", ", ".join(removed)


def survey(root: Path) -> dict:
    """What is being left behind. Recorded, never acted on.

    Anything not measured is recorded as None, never as an empty string: the log
    is this hook's only output, and "could not tell" must not read as "nothing".
    """
    out: dict = {"branch": None, "uncommitted": None, "uncommitted_sample": [],
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
    try:
        payload = read_payload()
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

        cwd = row["cwd"] or os.getcwd()
        if not Path(cwd).is_dir():
            row["outcome"] = "cwd-unusable"
            row["detail"] = f"cannot enter {cwd!r}; nothing was released or removed"
            return 0
        rc, top = git(cwd, "rev-parse", "--show-toplevel", timeout=TIMEOUT,
                      budget_left=budget_left)
        if rc != 0 or not top:
            row["outcome"] = "not-a-git-repo"
            return 0

        root = Path(top)
        row.update(survey(root))
        rc, git_dir = git(cwd, "rev-parse", "--absolute-git-dir", timeout=TIMEOUT,
                          budget_left=budget_left)
        if rc != 0 or not git_dir:
            row["outcome"] = "no-git-dir"
            return 0

        # Ownership decides everything that follows. A session refused this
        # workspace still runs, and `recall.checkout` labels the checkout rather
        # than the session, so an unconditional teardown would destroy the
        # legitimate holder's database.
        claim_file = Path(git_dir) / "claude-session-claim"
        holder = read_claim(claim_file).get("session", "")
        ours = (not claim_file.exists()) or (bool(session_id) and holder == session_id)
        row["claim_holder"] = holder[:8] if holder else None

        if not ours:
            row["container"] = "skipped"
            row["container_detail"] = f"this checkout is claimed by {holder[:8]}"
            row["claim"] = f"left in place: not this session's claim (holder={holder[:8]})"
            row["outcome"] = "closed-not-owner"
            return 0

        # Container first, claim LAST. Releasing first lets the next session
        # claim the worktree and start a container that this filter would then
        # match and remove.
        status, detail = remove_own_container(root)
        row["container"], row["container_detail"] = status, detail
        row["claim"] = release_claim(Path(git_dir), session_id,
                                     os.environ.get("CLAUDE_PID", str(os.getpid())))
        row["outcome"] = ("closed" if status in ("removed", "none")
                          else f"closed-with-{status}-container")
    except Exception as exc:  # noqa: BLE001 - a shutdown must not be blocked
        row["outcome"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        row["elapsed_s"] = round(time.monotonic() - _STARTED, 2)
        write_row(row)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        sys.exit(0)
