"""What a result artifact must say about the corpus it measured.

No result JSON in this repo records its corpus row count. On 2026-07-27 two concurrent runs
doubled the LOCOMO corpus (11,764 rows against a correct 5,882); every depth came in ~0.05 low
and nothing errored. #103 later verified 5,882 rows in scripts/run_locomo_arms.sh — and the
number went to the runner's stdout, so its own artifacts cannot show it.

A verification that does not reach the artifact protects only the session that ran it.

`git_sha` alone is not enough: every measurement in this project is launched from a worktree
under active edit, so a run against uncommitted code records a clean-looking sha of code that
did not run. `git_dirty` is the flag that tells a later reader whether `git_sha` actually
describes the tree that produced the numbers beside it, or only the tree it was closest to.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

#: Default anchor for the git commands below: this file's own directory, not the caller's.
#: Without it, `git` inherits the calling process's working directory, and a run launched from
#: inside some OTHER git repo would silently report THAT repo's HEAD/state — a
#: wrong-but-plausible answer, which is exactly the "worse than absent" case the docstrings
#: below guard against. Tests override it with an isolated repo to get a real clean/dirty
#: answer without depending on this checkout's own working-tree state at test time.
_THIS_DIR = Path(__file__).resolve().parent


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run a git command, or None on anything that means "no usable answer".

    Shared by `_git_sha` and `_git_dirty` so both degrade identically: git not installed, `cwd`
    not inside a repo, or the process hangs past the timeout all collapse to the same None a
    caller already has to handle.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_sha(cwd: Path | None = None) -> str | None:
    """Short SHA of the tree that produced this result, or None outside a repo.

    Degrades to None rather than raising or inventing: a result file from a tarball is still a
    result, and a wrong sha is worse than an absent one.
    """
    out = _run_git(["rev-parse", "--short", "HEAD"], cwd or _THIS_DIR)
    if out is None:
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _git_dirty(cwd: Path | None = None) -> bool | None:
    """Whether tracked files differ from HEAD — i.e. whether `_git_sha` might be lying.

    None under exactly the conditions `_git_sha` degrades under (git absent, `cwd` not a repo,
    no HEAD yet): "dirty relative to nothing" is not a fact, so the two must degrade together
    rather than one silently claiming a clean run next to an absent sha.

    Checks tracked files only — the same thing `git describe --dirty` checks — via
    `git diff --quiet HEAD`, which is nonzero for staged or unstaged modifications to tracked
    content. An untracked scratch file (a stray log, a downloaded corpus) does not make a
    measured run "dirty" on its own; it isn't part of the tree `git_sha` names.
    """
    resolved = cwd or _THIS_DIR
    if _git_sha(resolved) is None:
        return None
    out = _run_git(["diff", "--quiet", "HEAD"], resolved)
    if out is None or out.returncode not in (0, 1):
        return None
    return out.returncode == 1


def provenance_block(corpus_rows: int, table: str, tenants: list[str]) -> dict[str, Any]:
    """The block every eval result embeds so a later reader can check it.

    `corpus_rows` is the summed tenant-scoped count actually measured, not a configured or
    expected value — an expectation copied into a result proves nothing about the run.

    `git_sha` names the commit closest to the tree that ran; `git_dirty` says whether that tree
    had uncommitted changes on top of it. Both are None together outside a repo — see
    `_git_dirty`.
    """
    return {
        "corpus_rows": corpus_rows,
        "table": table,
        "tenants": sorted(tenants),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
