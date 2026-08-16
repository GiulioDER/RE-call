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
