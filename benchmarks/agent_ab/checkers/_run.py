"""Running an agent's artifact safely, and the two ways doing it naively goes wrong here.

Every checker in this package executes a script the model wrote. That is the point: the endpoint
is whether the work runs, not whether the prose reads well. It also means this module is the one
place where the harness itself can fall into the same holes the benchmark is testing for, and it
fell into both while being written.

**1. A bare `bash` reaches WSL, not Git Bash.** Windows `CreateProcess` searches System32 before
PATH, so `subprocess.run(["bash", ...])` finds `C:\\Windows\\System32\\bash.exe`. Verified again
2026-08-21: `shutil.which("bash")` returns Git Bash and the launch still dies with
``execvpe(/bin/bash) failed``, exit 1. A checker that ate that would score every shell task as
failing, in both arms, and the run would look like the tasks were impossible. `git_bash()` below
resolves an absolute path by walking up from `git` to the Git root, and refuses anything under
System32.

**2. `subprocess.run(timeout=...)` does not bound wall clock.** It kills the direct child; a
grandchild holding the stdout pipe keeps `communicate()` blocked. Measured 2026-08-21 in this
worktree: a 3 s timeout over ``bash -c 'sleep 30'`` returned after **30.2 s**. Several tasks here
deliberately produce artifacts that spawn grandchildren, so a checker built on the plain timeout
would inherit the very defect it is scoring. `run_bounded` kills the whole process tree.

Neither fix is optional and neither is decorative: without the first the harness cannot run a
shell task at all, and without the second one naive submission can hold a checker for the length
of whatever it spawned.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

#: Nothing a checker runs is allowed to take longer than this, whatever it spawns.
DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Completed:
    """What a bounded run produced, including whether it was killed."""

    returncode: int | None
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def git_bash() -> Path:
    """Absolute path to Git Bash, never System32's WSL launcher.

    `shutil.which("bash")` is not usable for this: it answers with Git Bash while the actual
    launch resolves to WSL, so the two disagree and only the launch matters.
    """

    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is not on PATH, so Git Bash cannot be located")
    for parent in Path(git).resolve().parents:
        candidate = parent / "bin" / "bash.exe"
        if candidate.is_file() and "system32" not in str(candidate).lower():
            return candidate
    raise RuntimeError(f"no Git Bash found by walking up from {git}")


def kill_tree(process: subprocess.Popen) -> None:
    """Kill the process and everything it spawned.

    On Windows `taskkill /T` is the only thing that reaches a grandchild; `Popen.kill` reaches the
    direct child alone, which is exactly the hole this module exists to close.
    """

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(  # noqa: S603,S607 - argv list, no shell
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:  # pragma: no cover - the benchmark runs on Windows
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - taskkill /F rarely fails
        process.kill()


def run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> Completed:
    """Run `command`, and return within `timeout_s` whatever it spawned.

    `stdin` is DEVNULL so an artifact that prompts fails instead of hanging, and the returned
    record keeps the wall clock, because several tasks are scored on how long the artifact took
    rather than only on what it printed.
    """

    merged = dict(os.environ)
    if env:
        merged.update(env)
    start = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - argv list, no shell
        list(command),
        cwd=str(cwd),
        env=merged,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(process)
        # After the tree is gone the pipes close, so this second read cannot block for long. It
        # is still bounded, because a checker that hangs here would be the same defect one level
        # up.
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - tree is already dead
            stdout, stderr = "", ""
    wall = time.monotonic() - start
    return Completed(
        returncode=None if timed_out else process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        wall_s=wall,
        timed_out=timed_out,
    )


def run_bash(script: Path | str, *args: str, cwd: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Completed:
    """Run a shell artifact under Git Bash."""

    return run_bounded([str(git_bash()), str(script), *args], cwd=cwd, timeout_s=timeout_s)


def run_python(script: Path | str, *args: str, cwd: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Completed:
    """Run a Python artifact under the interpreter running the harness."""

    return run_bounded([sys.executable, str(script), *args], cwd=cwd, timeout_s=timeout_s)


def git(*args: str, cwd: Path) -> Completed:
    """Run one git command inside a sandbox, with the harness's identity, never the user's."""

    return run_bounded(
        [
            "git",
            "-c",
            "user.email=agent-ab@localhost",
            "-c",
            "user.name=agent-ab",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        timeout_s=60,
    )
