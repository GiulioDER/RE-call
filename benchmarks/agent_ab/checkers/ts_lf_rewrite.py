"""Does the bump script leave a one-line diff, or rewrite the file as CRLF?

The fixture's `.gitattributes` sets `* text=auto eol=lf`, so the working tree is LF. Python's
`Path.write_text` translates every newline to CRLF on Windows (verified 2026-08-21: `b'a\\r\\nb\\r\\n'`
from a plain `write_text("a\\nb\\n")`), so a script that reads, substitutes and writes back turns a
one-line version bump into a whole-file rewrite. Nothing raises, the version really is bumped, and
`git status` shows the file modified forever.

⚠️ **The checker runs the script rather than inspecting its output.** The task asks for a
re-runnable artifact because CI will call it, and an agent that edited the file by hand with the
Edit tool would otherwise pass on a side effect while leaving a broken script behind. So the file
is reset from git first, and the agent's script is the thing under test.
"""

from __future__ import annotations

from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import git, run_python

SCRIPT = "scripts/bump_version.py"
TARGET = "recall/version.py"


def check(workdir: Path) -> CheckResult:
    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")

    # Reset to the fixture state so the script's own behaviour is what is measured, not whatever
    # the session left behind.
    git("checkout", "--", TARGET, cwd=workdir)
    result = run_python(SCRIPT, "0.9.8", cwd=workdir, timeout_s=90)

    target = workdir / TARGET
    raw = target.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    # Bound to a name rather than inlined: a backslash inside an f-string expression is a syntax
    # error before Python 3.12, and this package supports 3.11.
    carriage_returns = raw.count(b"\r")
    numstat = git("diff", "--numstat", "--", TARGET, cwd=workdir).stdout.strip()
    added = deleted = None
    if numstat:
        parts = numstat.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            added, deleted = int(parts[0]), int(parts[1])

    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "numstat": numstat,
        "added": added,
        "deleted": deleted,
        "carriage_returns": carriage_returns,
        "stdout": result.stdout[-400:],
        "stderr": result.stderr[-600:],
    }

    if result.timed_out or result.returncode != 0:
        return CheckResult(False, "the bump script did not run cleanly", detail)
    if '__version__ = "0.9.8"' not in text:
        return CheckResult(False, "the version was not bumped to 0.9.8", detail)
    if carriage_returns:
        return CheckResult(
            False,
            f"rewrote the file with {carriage_returns} carriage returns against eol=lf",
            detail,
        )
    if added is None or deleted is None:
        return CheckResult(False, "no diff against the fixture after running the script", detail)
    # Four, not one: the fixture asks for VERSION_INFO to be kept in step, so an agent that bumps
    # both the string and the tuple is doing the job properly and must not be penalised for it.
    if added > 4 or deleted > 4:
        return CheckResult(False, f"diff is {added}+/{deleted}-, far more than the bump", detail)
    return CheckResult(True, f"bumped in {added}+/{deleted}- with LF preserved", detail)
