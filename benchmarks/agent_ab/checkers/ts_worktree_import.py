"""Did the probe measure this checkout, or whichever copy of the package happens to be installed?

Run as a script, Python puts the SCRIPT's directory on `sys.path[0]`, so `benchmarks/` lands on the
path and the worktree root does not. `import recall` then falls through to whatever is installed,
which on this machine is one editable install shared across roughly eighteen worktrees, sitting on
whatever branch the main checkout happens to be on. Run as `python -m benchmarks.probe` the current
directory leads and the local package wins.

Verified in this worktree on 2026-08-21 against a decoy on `PYTHONPATH`: the script form printed the
decoy's number, the module form printed the local one, from the same tree in the same shell.

The dangerous shape is the silent one. The failure that produced the memo was loud only by luck, an
`ImportError` for a symbol that existed on one branch alone. A probe whose symbols resolve in both
trees runs happily and reports a number describing a different branch, and nothing in the output
says which `recall` it measured.

The decoy stands in for the shared editable install, and it lives in `oracles/`, never in the
sandbox. An agent who could see it would see the whole problem.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import git_bash, run_bounded

SCRIPT = "scripts/run_probe.sh"
LOCAL_CHUNKS = "1006"
DECOY_CHUNKS = "999999"


def check(workdir: Path) -> CheckResult:
    from ..sandbox import oracle

    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")

    decoy = oracle("ts-worktree-import") / "installed"
    result = run_bounded(
        [str(git_bash()), SCRIPT],
        cwd=workdir,
        timeout_s=120,
        env={"PYTHONPATH": str(decoy) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    out = result.stdout + result.stderr
    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout": result.stdout[-400:],
        "stderr": result.stderr[-600:],
        "decoy_on_path": str(decoy),
    }
    if result.timed_out:
        return CheckResult(False, "the wrapper never returned", detail)
    if DECOY_CHUNKS in out:
        return CheckResult(False, "reported the installed package's number, not this tree's", detail)
    if LOCAL_CHUNKS not in out:
        return CheckResult(False, "did not report a chunk count at all", detail)
    if result.returncode != 0:
        return CheckResult(False, f"printed the right number but exited {result.returncode}", detail)
    return CheckResult(True, "reported this checkout's number with the decoy on the path", detail)
