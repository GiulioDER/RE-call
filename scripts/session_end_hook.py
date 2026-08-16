#!/usr/bin/env python3
"""SessionEnd hook: release this session's claim and remove only its own container.

The other half of scripts/session_start_hook.py, and it exists for the same
reason. Opening was automated on 2026-08-16 because the protocol was correct and
never invoked; closing stayed prose and cost the same thing within hours of the
fix landing. A claim had to be released by hand, and a 4.9-hour indexer was still
running on a remote host because nothing had swept it.

**What this deliberately does NOT do.** It does not commit, push, or delete
branches. Uncommitted work at session end is a decision for a person, and a hook
that quietly commits, or quietly discards, is worse than one that leaves the tree
alone and writes down what it saw. It records; it does not tidy.

**The one rule it must never break: touch only what this session owns.**

- A claim is released only when the claim file names THIS session. Another
  session's claim on a worktree we happen to be sitting in is not ours to clear,
  and clearing it invites the collision the claim exists to prevent.
- A container is removed only when its `recall.checkout` label equals this
  checkout, matching `scripts/session-db.sh down`. A container this session did
  not start may have another session's suite mid-run against it, and a 17-minute
  suite killed at minute 12 reads as a code failure rather than as interference.

SessionEnd output is not added to any context (the session is over), so the
record goes to `~/.claude/session-end.log` where the next session, or a person,
can read it.

Install (registered under SessionEnd in ~/.claude/settings.json):

    python "<path to this file>"

It never raises. Every failure path is recorded and swallowed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOG = Path.home() / ".claude" / "session-end.log"

#: A hook at session end competes with the process actually exiting, so it gets a
#: tighter budget than the start hook. Anything slower than this is not worth
#: delaying a shutdown for.
TIMEOUT = 10
BUDGET = 15.0
_STARTED = time.monotonic()


def budget_left() -> float:
    return BUDGET - (time.monotonic() - _STARTED)


def run(args, cwd=None, timeout=TIMEOUT):
    """Run a command, returning (rc, stdout). Never raises, never hangs."""
    if budget_left() <= 0:
        return -1, ""
    try:
        p = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True,
            timeout=min(timeout, max(1.0, budget_left())),
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
        return p.returncode, (p.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def git(cwd, *args):
    env_keep = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        p = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=min(TIMEOUT, max(1.0, budget_left())), env=env_keep,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
        return p.returncode, (p.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def read_claim(claim_file: Path) -> dict:
    try:
        text = claim_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return {}
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def release_claim(git_dir: Path, session_id: str) -> str:
    """Release the claim only if it names this session.

    A claim we do not own is the single thing this hook must not remove: another
    session may be holding this worktree legitimately, and clearing its claim
    hands the worktree to whoever opens it next.
    """
    claim_file = git_dir / "claude-session-claim"
    if not claim_file.exists():
        return "no claim to release"
    holder = read_claim(claim_file).get("session", "")
    if holder and session_id and holder != session_id:
        return f"left in place: claimed by another session ({holder[:8]})"
    try:
        claim_file.unlink()
    except OSError as exc:
        return f"could not release: {exc}"
    # A lock stranded by a killed holder would otherwise outlive the claim and
    # block the next session, which is the failure the start hook already fixed
    # once; the same file must not be left behind here.
    lock = claim_file.with_name(claim_file.name + ".lock")
    if lock.exists():
        try:
            for child in lock.iterdir():
                child.unlink()
            lock.rmdir()
        except OSError:
            pass
    return "released"


def remove_own_container(root: Path) -> str:
    """Remove only a container labelled with THIS checkout.

    The label is the whole safety argument, and it is the same one
    `scripts/session-db.sh` writes. Matching on a name pattern instead would
    sweep every `recall-sess-*` on the machine, including the ones other sessions
    are mid-suite against.
    """
    checkout = str(root).replace("\\", "/")
    rc, out = run([
        "docker", "ps", "-aq",
        "--filter", f"label=recall.checkout={checkout}",
    ])
    if rc != 0:
        return "docker unavailable" if rc == -1 else "docker query failed"
    ids = [x for x in out.splitlines() if x.strip()]
    if not ids:
        return "none started by this checkout"
    removed = []
    for cid in ids:
        rc, _ = run(["docker", "rm", "-f", cid], timeout=20)
        removed.append(f"{cid[:12]}{'' if rc == 0 else ' (FAILED)'}")
    return "removed " + ", ".join(removed)


def survey(root: Path) -> dict:
    """What is being left behind. Recorded, never acted on."""
    rc, dirty = git(root, "status", "--porcelain")
    files = [x for x in dirty.splitlines() if x.strip()] if rc == 0 else []
    rc, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    ahead = ""
    rc2, ref = git(root, "rev-parse", "--abbrev-ref", "origin/HEAD")
    if rc2 == 0 and ref and "/" in ref:
        rc3, n = git(root, "rev-list", "--count", f"{ref}..HEAD")
        if rc3 == 0:
            ahead = n
    return {
        "branch": branch,
        "uncommitted": len(files),
        "uncommitted_sample": files[:5],
        "commits_ahead_of_trunk": ahead,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    row = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": (payload.get("session_id") or "")[:8],
        "reason": payload.get("reason"),
        "cwd": payload.get("cwd"),
    }
    try:
        cwd = payload.get("cwd") or os.getcwd()
        rc, top = (-1, "")
        # A directory we cannot enter is not the same fact as a directory that
        # is not a repository, and only the second is about the workspace. An
        # MSYS-style `/c/...` path arrives as a real string that Windows cannot
        # resolve, and calling that "not a git repo" hides a skipped cleanup
        # behind a plausible-looking outcome.
        if not Path(cwd).is_dir():
            row["outcome"] = "cwd-unusable"
            row["detail"] = f"cannot enter {cwd!r}; nothing was released or removed"
        else:
            rc, top = git(cwd, "rev-parse", "--show-toplevel")
            if rc != 0 or not top:
                row["outcome"] = "not-a-git-repo"
            else:
                root = Path(top)
                rc, git_dir = git(cwd, "rev-parse", "--absolute-git-dir")
                row["claim"] = (
                    release_claim(Path(git_dir), payload.get("session_id") or "")
                    if rc == 0 and git_dir else "could not locate the git dir"
                )
                row["container"] = remove_own_container(root)
                row.update(survey(root))
                row["outcome"] = "closed"
    except Exception as exc:  # noqa: BLE001 - a shutdown must not be blocked
        row["outcome"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]

    row["elapsed_s"] = round(time.monotonic() - _STARTED, 2)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > 512_000:
            keep = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
            tmp = LOG.with_name(LOG.name + f".{os.getpid()}.tmp")
            tmp.write_text("\n".join(keep) + "\n", encoding="utf-8", newline="\n")
            os.replace(tmp, LOG)
        with LOG.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
    except (OSError, ValueError, TypeError):
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        sys.exit(0)
