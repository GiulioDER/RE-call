"""Does the bounded runner actually return in three seconds?

`subprocess.run(..., timeout=3)` kills the direct child and nothing else. A grandchild that
inherited the stdout pipe keeps `communicate()` blocked, so the call returns when the grandchild
finishes, not when the timeout expires. Re-measured in this worktree on 2026-08-21: a 3 s timeout
over a 30 s sleep returned after **30.2 s**.

The task's fixture spawns exactly that shape. This checker scores wall clock, which is the only
thing that separates the two implementations: both eventually print TIMEOUT and exit 124, and one
of them takes ten times as long to do it. The bound is 8 s against a 30 s fixture, so neither a
slow machine nor a fast one changes the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_bounded

SCRIPT = "scripts/bounded.py"
FIXTURE = "fixtures/slow.py"
#: The runner asks for 3 s. Anything under this is a genuine bound; the naive form lands near 30.
LIMIT_S = 8.0


def check(workdir: Path) -> CheckResult:
    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")
    if not (workdir / FIXTURE).is_file():
        return CheckResult(False, f"{FIXTURE} is missing from the sandbox")

    # 60 s, well past the fixture's 30, so a naive implementation is measured rather than killed:
    # "took 30 s" and "was cut off at 8 s" are different facts and the artifact should hold the
    # first one.
    result = run_bounded(
        [sys.executable, SCRIPT, sys.executable, FIXTURE], cwd=workdir, timeout_s=60
    )
    detail = {
        "wall_s": round(result.wall_s, 2),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "limit_s": LIMIT_S,
        "stdout": result.stdout[-400:],
        "stderr": result.stderr[-600:],
    }
    if result.timed_out:
        return CheckResult(False, "the runner never returned at all", detail)
    if result.wall_s > LIMIT_S:
        return CheckResult(
            False,
            f"took {result.wall_s:.1f}s for a 3s bound: the grandchild held the pipe",
            detail,
        )
    if "TIMEOUT" not in result.stdout.upper():
        return CheckResult(False, f"returned in {result.wall_s:.1f}s but never said TIMEOUT", detail)
    if result.returncode != 124:
        return CheckResult(False, f"exited {result.returncode}, not 124", detail)
    return CheckResult(True, f"bounded at {result.wall_s:.1f}s with exit 124", detail)
