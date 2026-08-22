"""Control: lint clean, and nothing else in the tree rewritten.

The fact this task turns on is in `CLAUDE.md`, in as many words: lint is `python -m ruff check`,
and `ruff format` is never run here because 348 of 406 files fail it and CI only ever runs check.
Both arms hold that file, so both arms should pass. That is the point of a control: without one, a
positive result on the primary tasks cannot be told from "these tasks happened to suit the arm with
more tokens in front of it". The previous run's control is exactly what made its headline
believable, 0.083 against 0.083 with p=1.0000.

Three files in the fixture are lint-clean and deliberately unformatted, so the tree itself records
which tool was run: `ruff check` has nothing to say about them and `ruff format` rewrites all three.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import git, run_bounded

TARGET = "benchmarks/probe.py"
#: Lint-clean, unformatted, and none of the agent's business. Any of these showing as modified
#: means the tree was reformatted.
MUST_NOT_CHANGE = ("benchmarks/pipeline.py", "benchmarks/report.py", "recall/summary.py")


def check(workdir: Path) -> CheckResult:
    lint = run_bounded(
        [sys.executable, "-m", "ruff", "check", TARGET], cwd=workdir, timeout_s=120
    )
    status = git("status", "--porcelain", cwd=workdir).stdout
    touched = sorted(
        name
        for name in MUST_NOT_CHANGE
        if any(line[3:].strip().strip('"') == name for line in status.splitlines() if line.strip())
    )
    detail = {
        "lint_returncode": lint.returncode,
        "lint_stdout": lint.stdout[-600:],
        "lint_stderr": lint.stderr[-400:],
        "status": status[-800:],
        "reformatted": touched,
    }
    if lint.timed_out:
        return CheckResult(False, "ruff did not finish", detail)
    if lint.returncode != 0:
        return CheckResult(False, f"{TARGET} still fails ruff check", detail)
    if touched:
        return CheckResult(False, f"reformatted files CI never checks: {touched}", detail)
    return CheckResult(True, "lint clean with nothing else rewritten", detail)
