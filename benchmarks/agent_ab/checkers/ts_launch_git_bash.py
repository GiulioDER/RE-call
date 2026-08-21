"""Did the hook actually run, or did Windows hand the launcher to WSL?

`subprocess.run(["bash", script])` from Windows Python reaches `C:\\Windows\\System32\\bash.exe`,
the WSL launcher, not Git Bash. `shutil.which("bash")` answers with Git Bash and is irrelevant:
`CreateProcess` searches the application directory, the current directory and System32 **before**
PATH. Verified again in this worktree on 2026-08-21: `which` returned
`C:\\Program Files\\Git\\usr\\bin\\bash.EXE` and the launch died with
``execvpe(/bin/bash) failed: No such file or directory``, exit 1.

The fixture hook exits **3**, a code nothing else in the path produces. WSL's failure exits 1. So
`HOOK_EXIT=3` is proof the hook ran, and it is the only way to get that string. This is the
distinction the memo says is worth the whole memo: a hook that refused and an interpreter that
never started are different facts, and conflating them turned a guard into `WORKSPACE REFUSED` on
every session in every repository.
"""

from __future__ import annotations

from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_python

SCRIPT = "scripts/run_hook.py"


def check(workdir: Path) -> CheckResult:
    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")

    result = run_python(SCRIPT, cwd=workdir, timeout_s=90)
    out = result.stdout
    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout": out[-400:],
        "stderr": result.stderr[-600:],
    }
    if result.timed_out:
        return CheckResult(False, "the runner never returned", detail)
    if "HOOK_EXIT=3" in out:
        return CheckResult(True, "reported the hook's own exit code", detail)
    if "LAUNCH_FAILED" in out:
        return CheckResult(False, "could not launch the interpreter", detail)
    if "HOOK_EXIT=" in out:
        reported = out.split("HOOK_EXIT=", 1)[1].split()[0]
        # Exit 1 here is the tell: that is WSL failing to find /bin/bash, reported as though the
        # hook had spoken.
        return CheckResult(False, f"reported HOOK_EXIT={reported}, not the hook's 3", detail)
    return CheckResult(False, "said neither HOOK_EXIT= nor LAUNCH_FAILED", detail)
