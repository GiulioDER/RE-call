"""Does the autouse fixture leave every test's own scratch space alone?

An autouse fixture must not create anything inside `tmp_path`. `tmp_path` belongs to the test, and
tests assert on all of it: `list(tmp_path.iterdir()) == [memo]`, or a walk that reads each entry as
a file. One extra directory turns both into failures whose messages name uploads or indexing, in
tests about something else entirely. Measured in this repository on 2026-08-18: 2 failed, 5455
passed, and the two had nothing to do with the fixture. Cost, one full suite run.

The obvious fixture points `RECALL_INDEX_ROOT` at `tmp_path / "recall-index-root"` and creates it.
The fixture module deliberately **requires the directory to exist**, so pointing at a path without
creating it is not an escape: the suite the agent can see would go red immediately, in front of
them. That is what makes the two implementations differ by knowledge rather than by luck.

The held-out tests are the ones that enumerate their own `tmp_path`. They are not in the sandbox,
because a test the agent can run is a test the agent iterates against.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_bounded

CONFTEST = "tests/conftest.py"
HELD_OUT = "test_scratch_isolation.py"


def _pytest(cwd: Path) -> tuple[bool, str]:
    result = run_bounded(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        cwd=cwd,
        timeout_s=300,
    )
    return (result.returncode == 0 and not result.timed_out), (result.stdout + result.stderr)[-900:]


def check(workdir: Path) -> CheckResult:
    from ..sandbox import oracle

    conftest = workdir / CONFTEST
    if not conftest.is_file():
        return CheckResult(False, f"{CONFTEST} was not written")

    green, green_output = _pytest(workdir)
    detail: dict[str, object] = {
        "green_as_written": green,
        "green_output": green_output,
        "conftest_mentions_tmp_path_factory": "tmp_path_factory" in conftest.read_text(
            encoding="utf-8", errors="replace"
        ),
        "uploads_left_in_tree": (workdir / "uploads").exists(),
    }
    if not green:
        return CheckResult(False, "the suite the agent could see is not green", detail)
    if (workdir / "uploads").exists():
        return CheckResult(False, "the suite still wrote uploads/ into the tree", detail)

    # The held-out tests go into a sibling copy, so the sandbox stays as the session left it.
    probe_dir = workdir.parent / f"{workdir.name}__isolation"
    if probe_dir.exists():
        shutil.rmtree(probe_dir, ignore_errors=True)
    shutil.copytree(workdir, probe_dir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copyfile(
        oracle("ts-autouse-tmp-path") / HELD_OUT, probe_dir / "tests" / HELD_OUT
    )

    isolated, isolated_output = _pytest(probe_dir)
    detail.update({"held_out_green": isolated, "held_out_output": isolated_output})
    if not isolated:
        return CheckResult(
            False,
            "a test that enumerates its own tmp_path fails: the fixture wrote into it",
            detail,
        )
    return CheckResult(True, "green with the held-out isolation tests in place", detail)
