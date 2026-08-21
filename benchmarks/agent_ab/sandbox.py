"""Restoring the same repository state for both arms, and proving that it was the same.

A paired comparison over real work needs the two arms to start from identical ground. That sounds
like a copy and it is not, because three things can differ without anyone noticing: the initial
commit's identity and timestamp, the line endings git writes into the working tree, and whatever
a previous session left behind in a reused directory.

So each session gets a fresh directory built from a tracked fixture, and the harness records a
**tree digest** for it. `benchmarks/agent_ab/gate.py` refuses a pair whose two arms did not start
from the same digest, exactly as it refuses a pair that cannot prove the treatment was applied. A
difference in starting state would look like a treatment effect and there would be nothing in the
artifact to say otherwise.

A fixture is a directory under `workspaces/<task_id>/`:

    tree/     the committed state; everything here is added and committed
    dirty/    optional, copied OVER the tree after that commit, leaving those files modified

`dirty/` exists for the one task whose whole point is an unclean working tree. Doing it as an
overlay rather than as a script keeps the fixture readable as data.

⛔ **Oracles live in `oracles/`, a sibling of `workspaces/`, and are never copied.** The endpoint
is only meaningful while the input that decides it stays out of the agent's reach; see the module
docstring in `tasksuccess.py`.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .checkers._run import git

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACES = PACKAGE_ROOT / "workspaces"
ORACLES = PACKAGE_ROOT / "oracles"

#: Never copied into a sandbox and never hashed.
EXCLUDED = frozenset({".git", "__pycache__", ".pytest_cache", ".ruff_cache"})


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in EXCLUDED for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def tree_digest(root: Path) -> str:
    """A digest over every tracked-relevant file's path and bytes.

    Bytes, not text: line endings are precisely one of the things a task here is scored on, so a
    digest that normalised them would call two genuinely different starting states identical.
    """

    digest = hashlib.sha256()
    for path in _iter_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    for path in _iter_files(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, so a fixture that deliberately holds CRLF or a bare LF keeps it. `shutil.copy2`
        # would do as well; this is explicit about the thing that matters.
        target.write_bytes(path.read_bytes())


def restore(workspace: str, destination: Path) -> str:
    """Build one sandbox from its fixture and return its tree digest.

    The directory is created fresh. A reused one is refused rather than cleaned, because a stale
    artifact from an earlier session is indistinguishable from work this session did.
    """

    fixture = WORKSPACES / workspace
    if not fixture.is_dir():
        raise FileNotFoundError(f"no fixture for workspace {workspace!r} at {fixture}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"sandbox {destination} already has contents")
    destination.mkdir(parents=True, exist_ok=True)

    _copy_tree(fixture / "tree", destination)

    # A git repository, because several checkers read the diff and because "leave the tree clean"
    # is only a meaningful instruction where there is an index to be clean against. `init.
    # defaultBranch` is pinned so a checker can name the branch, and gpgsign is off because the
    # user's signing key must never be reachable from a sandbox running model-written code.
    git("init", "-q", "-b", "main", ".", cwd=destination)
    git("add", "--", ".", cwd=destination)
    git("commit", "-q", "-m", "fixture", cwd=destination)

    overlay = fixture / "dirty"
    if overlay.is_dir():
        _copy_tree(overlay, destination)

    return tree_digest(destination)


def oracle(task_id: str) -> Path:
    """The held-out directory for one task, which no sandbox ever sees."""

    path = ORACLES / task_id
    if not path.is_dir():
        raise FileNotFoundError(f"no oracle directory for {task_id!r} at {path}")
    return path


def stage_oracle(task_id: str, workdir: Path, name: str) -> Path:
    """Copy one oracle file into a finished sandbox, for a checker to run the artifact against.

    Called only after the session has ended. The file lands under `.oracle/` so a later reader of
    the sandbox can see what the artifact was judged with.
    """

    source = oracle(task_id) / name
    if not source.is_file():
        raise FileNotFoundError(f"no oracle file {name!r} for {task_id!r}")
    target = workdir / ".oracle" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target
