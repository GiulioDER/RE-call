"""The mutation job's own time budget: a run that outlasts it is stopped and reported, not lost.

Invariants, for `scripts/mutation_changed.sh` with a mutmut run longer than
MUTATION_BUDGET_SECONDS:
- the run is stopped at the budget, the script exits 0 (the job is report-only), and the summary
  says the counts are partial;
- a mutmut run that fails for any other reason still exits 1;
- a run that finishes inside the budget carries no budget note.

Failure mode caught: CI killing the job at its 20-minute timeout, which shows "cancelled" and
reports nothing (7 of 17 runs on 2026-10-02 ended that way), or a budget that swallows a real
mutmut failure.

The scripts run in a throwaway git repository against a fake `mutmut` on PATH, so no mutant is
generated: what is tested is the script's handling of the run, not mutmut. POSIX only, like the
job (bash, timeout(1), nproc).

Red proof, 2026-10-02, on a Linux test host, each failing at its assertion, then restored and green:
- ``test_a_run_over_budget_is_stopped_and_reported`` against the script as it was before the
  budget (origin/master ``7ed70b5d``): the run took the fake's full 8 s, ``assert 8.3 < 6``. And
  against the budgeted script with mutmut's output piped through ``tee`` again: ``assert 8.4 < 6``,
  because the fake's leftover background process kept the pipe open, so the step waited for it.
- ``test_a_failing_run_still_fails_the_script``, mutation: every non-zero status treated as the
  budget (``elif`` branch folded into ``budget_reached=1``): ``assert 0 == 1``.
- ``test_a_run_inside_the_budget_has_no_budget_note``, mutation: ``budget_reached=1`` set
  unconditionally: the note appeared in the summary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("mutation_changed.sh", "mutmut_type_filter.py", "mutation_code_changes.py")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not all(shutil.which(tool) for tool in ("bash", "timeout", "nproc", "git")),
    reason="POSIX only, like the mutation job: needs bash, timeout(1), nproc and git",
)

FAKE_MUTMUT = """#!/usr/bin/env bash
case "$1" in
  run)
    # mutmut stops its workers on SIGINT and exits cleanly; `sleep & wait` lets the trap fire now.
    trap 'echo "Stopping..."; exit 0' INT
    sleep "${FAKE_RUN_SECONDS:-0}" &
    wait $!
    exit "${FAKE_RUN_EXIT:-0}"
    ;;
  results)
    echo "    recall.thing.x_value__mutmut_1: killed"
    if [ "$2" = "--all" ]; then echo "    recall.thing.x_value__mutmut_2: not checked"; fi
    ;;
  show)
    echo "--- a/recall/thing.py"
    ;;
esac
"""


def _repository(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "recall").mkdir()
    (repo / "recall" / "thing.py").write_text("def x_value():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_thing.py").write_text(
        "from recall.thing import x_value\n\n\ndef test_x():\n    assert x_value() == 1\n",
        encoding="utf-8",
    )
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "fixture"], cwd=repo, check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "mutmut"
    fake.write_text(FAKE_MUTMUT, encoding="utf-8")
    fake.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["PYTHON"] = sys.executable
    env["MUTATE_FILES"] = "recall/thing.py"
    env["GITHUB_STEP_SUMMARY"] = str(tmp_path / "summary.md")
    return repo, env


def _run(repo: Path, env: dict[str, str]) -> tuple[int, float, str]:
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "scripts/mutation_changed.sh"], cwd=repo, env=env, capture_output=True, text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started
    summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    return completed.returncode, elapsed, summary


def test_a_run_over_budget_is_stopped_and_reported(tmp_path: Path) -> None:
    repo, env = _repository(tmp_path)
    env["MUTATION_BUDGET_SECONDS"] = "1"
    env["FAKE_RUN_SECONDS"] = "8"

    returncode, elapsed, summary = _run(repo, env)

    assert round(elapsed, 1) < 6
    assert returncode == 0
    assert "Time budget reached" in summary
    assert "not checked 1" in summary


def test_a_failing_run_still_fails_the_script(tmp_path: Path) -> None:
    repo, env = _repository(tmp_path)
    env["MUTATION_BUDGET_SECONDS"] = "30"
    env["FAKE_RUN_EXIT"] = "3"

    returncode, _elapsed, summary = _run(repo, env)

    assert returncode == 1
    assert "Time budget reached" not in summary


def test_a_run_inside_the_budget_has_no_budget_note(tmp_path: Path) -> None:
    repo, env = _repository(tmp_path)
    env["MUTATION_BUDGET_SECONDS"] = "30"

    returncode, _elapsed, summary = _run(repo, env)

    assert returncode == 0
    assert "Time budget reached" not in summary
    assert "killed 1" in summary
