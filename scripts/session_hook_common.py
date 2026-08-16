#!/usr/bin/env python3
"""Primitives shared by the SessionStart and SessionEnd workspace hooks.

This module exists because duplicating them was a bug, not a style problem. The
SessionEnd hook reimplemented `run()` without the process-tree kill the
SessionStart hook already had, and so reintroduced a defect that had been
measured and fixed a few hours earlier: a 3 second timeout that took 45.2
seconds because killing bash left `sleep` holding the inherited pipe.

Everything here is written around one asymmetry. A guard that wrongly says
"released" or "removed" costs somebody their work; a guard that wrongly refuses
costs a moment. So **cannot-tell always resolves to the cautious answer**: a pid
we cannot check is alive, a claim whose owner we cannot establish is not ours,
and a command we could not run is never reported as one that ran.

Installed alongside the hooks in ~/.claude/hooks/, which Python puts on sys.path
for a script run by absolute path, so `import session_hook_common` resolves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

#: Distinct from any real exit status. "The guard refused" and "the guard never
#: ran" are opposite facts and only one of them is about the workspace.
LAUNCH_FAILED = -1
#: Distinct again: we chose not to start it, which is not the same as trying and
#: failing. Collapsing these is how "budget exhausted" got logged as "docker is
#: not installed here".
NOT_ATTEMPTED = -2


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child AND its descendants.

    `proc.kill()` ends only the direct child, and on Windows `subprocess.run`
    then calls `communicate()` with **no timeout**, so a grandchild holding the
    inherited pipe keeps the call blocked long past the deadline. Measured on
    this machine: a bash script whose body was `sleep 45`, run with `timeout=3`,
    took 45.2 seconds.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=5, check=False)
        else:
            proc.kill()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run(args, cwd=None, timeout=20, env=None, budget_left=None):
    """Run a command. Returns (rc, stdout, stderr). Never raises, always bounded.

    `budget_left` is a callable returning the seconds remaining for the whole
    hook; when it is exhausted the command is NOT started and the result is
    NOT_ATTEMPTED, which callers must not report as a failure of the thing they
    were asking about.
    """
    if budget_left is not None:
        left = budget_left()
        if left <= 0:
            return NOT_ATTEMPTED, "", "hook time budget exhausted"
        timeout = max(1.0, min(timeout, left))
    try:
        proc = subprocess.Popen(
            args, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL,   # a child that prompts must fail, not hang
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return LAUNCH_FAILED, "", "could not launch"
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, (out or "").strip(), (err or "").strip()
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            out, err = "", ""
        return LAUNCH_FAILED, (out or "").strip(), "timed out"
    except (OSError, ValueError):
        kill_tree(proc)
        return LAUNCH_FAILED, "", "communication failed"


def git(cwd, *args, timeout=20, budget_left=None):
    """Git, stdout only. Warnings on stderr must never enter a parsed value."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    rc, out, _ = run(["git", *args], cwd=cwd, timeout=timeout, env=env,
                     budget_left=budget_left)
    return rc, out


def process_alive(pid: str, budget_left=None) -> bool:
    """Is this pid running? "Cannot tell" counts as ALIVE.

    `kill -0` is wrong on this platform and wrong in the dangerous direction:
    CLAUDE_PID is a Windows pid and Git Bash's kill only knows its own process
    table, so it reported a running claude.exe as dead. A false "alive" costs a
    deliberate release; a false "dead" costs somebody's in-flight work.

    An EMPTY pid means no holder was recorded, which is not a claim of death;
    callers decide what to do with that, and none of them may treat it as a
    licence to delete.
    """
    if not pid:
        return False
    # `msys:<n>`, written by session-space.sh when no Windows pid is available.
    # tasklist cannot see that namespace at all, so the only honest answer is
    # "cannot tell", which is ALIVE. Reading it as dead let racers break each
    # other's live locks: four concurrent claims, two winners, on a clean runner.
    if pid.startswith("msys:"):
        return True
    if not pid.isdigit():
        return True                      # unparseable: cannot tell
    if os.name == "nt":
        rc, out, _ = run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         timeout=10, budget_left=budget_left)
        if rc != 0:
            return True                  # cannot tell
        return pid in out
    rc, _, _ = run(["ps", "-p", pid], timeout=10, budget_left=budget_left)
    if rc in (LAUNCH_FAILED, NOT_ATTEMPTED):
        return True                      # cannot tell
    return rc == 0


def read_claim(claim_file: Path) -> dict:
    """Parse a claim file. Unreadable for ANY reason yields {}.

    errors="replace" and the ValueError arm matter: a claim written under a
    legacy code page, or torn by a concurrent write, raises UnicodeDecodeError,
    which is a ValueError and not an OSError.
    """
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


#: Rounds of 50ms to wait for the claim lock, and how many of those to tolerate a
#: lock directory carrying no holder pid before treating it as an orphan.
LOCK_ROUNDS = 20
LOCK_NOPID_ROUNDS = 10


def acquire_claim_lock(lock_dir: Path, pid: str, budget_left=None) -> bool:
    """Take the claim lock, breaking only a lock whose holder is demonstrably gone.

    `os.mkdir` is the atomic primitive. A lock must be reclaimable or it becomes
    the worst failure in the mechanism, and it must NOT be reclaimed from a live
    holder or it stops being a lock at all: removing one held by a running
    process let two sessions both take a worktree.
    """
    for i in range(LOCK_ROUNDS):
        try:
            os.mkdir(lock_dir)
            try:
                (lock_dir / "pid").write_text(pid, encoding="utf-8")
            except OSError:
                pass
            return True
        except FileExistsError:
            owner = ""
            try:
                owner = (lock_dir / "pid").read_text(encoding="utf-8",
                                                     errors="replace").strip()
            except OSError:
                pass
            dead_holder = owner and not process_alive(owner, budget_left)
            orphaned = not owner and i >= LOCK_NOPID_ROUNDS
            if dead_holder or orphaned:
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            time.sleep(0.05)
        except OSError:
            return False
    return False


def release_claim_lock(lock_dir: Path) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)
