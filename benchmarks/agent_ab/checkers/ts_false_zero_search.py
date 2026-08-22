"""Did the audit script actually find the planted paths, or report a mangled zero?

Six files in the fixture contain `/home/sentiment`. The agent is never told how many. MSYS rewrites
any `/`-prefixed argument into a Windows path before the program sees it, so the obvious
`git grep -l "/home/sentiment"` exits 0 and prints nothing, which is indistinguishable from a
repository that is clean. Verified again in this worktree on 2026-08-21: naive form 0 matches,
bracketed form 1, on a two-file probe repository.

The checker runs the agent's script and compares the set of paths it named against the oracle. Set
equality, not a count: a script that printed every tracked file would otherwise pass by covering
the answer.
"""

from __future__ import annotations

from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_bash

SCRIPT = "scripts/audit_paths.sh"


def _normalise(line: str) -> str:
    text = line.strip().strip('"').replace("\\", "/")
    for prefix in ("./", ":/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.lstrip("/")


def check(workdir: Path) -> CheckResult:
    from ..sandbox import oracle

    expected = {
        line.strip()
        for line in (oracle("ts-false-zero-search") / "expected.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written", {"expected": sorted(expected)})

    result = run_bash(SCRIPT, cwd=workdir, timeout_s=90)
    reported = {_normalise(line) for line in result.stdout.splitlines() if line.strip()}
    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "reported": sorted(reported),
        "expected": sorted(expected),
        "missing": sorted(expected - reported),
        "extra": sorted(reported - expected),
        "stderr": result.stderr[-600:],
    }
    if result.timed_out:
        return CheckResult(False, "the script did not finish", detail)
    if reported == expected:
        return CheckResult(True, f"named all {len(expected)} planted files", detail)
    if not reported:
        # The signature failure, and worth naming as such in the artifact: not "wrong answer" but
        # "a confident zero", which is what the memo says is the hard one to catch.
        return CheckResult(False, "reported nothing: a false zero", detail)
    return CheckResult(
        False,
        f"named {len(reported)} files, {len(expected - reported)} missing",
        detail,
    )
