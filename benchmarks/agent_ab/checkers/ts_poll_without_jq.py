"""Does the poll loop say what it sees, or hold its silence for the whole timeout?

`jq` is not installed on this machine, confirmed again 2026-08-21 with `command -v jq`. A poll loop
written in the documented shape (parse the field, emit on change, exit when it reaches the terminal
state) then parses every iteration to an empty string: the completion test never fires and the loop
emits nothing until its own timeout. Silence reads exactly like "still running", which is what cost
half an hour on a CI watch that had in fact finished green.

The checker drives a real transition: it starts the script against `state=pending`, moves the file
to `running` and then to `done`, and requires the script to say so and exit 0 well inside its own
60 s budget. Both halves matter. Exiting 0 alone would pass a script that printed nothing and
noticed the terminal state by luck; printing alone would pass one that never terminates.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import git_bash, kill_tree

SCRIPT = "scripts/poll_status.sh"
STATUS = "status.json"
#: The script is told to give up at 60 s. A correct one finishes as soon as it reads `done`, which
#: the checker writes at 6 s, so this bound separates "watched the transition" from "ran out".
LIMIT_S = 30.0


def _write_state(path: Path, state: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"] = state
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def check(workdir: Path) -> CheckResult:
    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")
    status = workdir / STATUS
    if not status.is_file():
        return CheckResult(False, f"{STATUS} is missing from the sandbox")

    _write_state(status, "pending")
    process = subprocess.Popen(  # noqa: S603 - argv list, no shell
        [str(git_bash()), SCRIPT],
        cwd=str(workdir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ),
    )
    start = time.monotonic()
    try:
        time.sleep(3.0)
        _write_state(status, "running")
        time.sleep(3.0)
        _write_state(status, "done")
        output, _ = process.communicate(timeout=LIMIT_S)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        # The tree, not the process. A `bash` loop with `sleep` in it leaves a grandchild holding
        # the stdout pipe, so `process.kill()` returns and `communicate()` then blocks for the rest
        # of the sleep. That is the same defect ts-bounded-runner is scoring, arriving here in the
        # checker: it hung this test for 40 s before being fixed.
        kill_tree(process)
        output, _ = process.communicate(timeout=10)
    wall = time.monotonic() - start
    text = (output or "").lower()

    detail = {
        "wall_s": round(wall, 2),
        "returncode": None if timed_out else process.returncode,
        "timed_out": timed_out,
        "saw_pending": "pending" in text,
        "saw_done": "done" in text,
        "output": (output or "")[-800:],
    }
    if timed_out:
        return CheckResult(False, f"did not exit within {LIMIT_S:.0f}s of reaching done", detail)
    if not (output or "").strip():
        # The signature of the memo: not a wrong answer, an empty one.
        return CheckResult(False, "printed nothing at all", detail)
    if "done" not in text:
        return CheckResult(False, "never reported the terminal state", detail)
    if "pending" not in text:
        return CheckResult(False, "never reported the state it started in", detail)
    if process.returncode != 0:
        return CheckResult(False, f"reported the states but exited {process.returncode}", detail)
    return CheckResult(True, f"followed the transition and exited 0 in {wall:.1f}s", detail)
