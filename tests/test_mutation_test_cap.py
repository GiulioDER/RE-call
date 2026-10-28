"""The mutation job's per-module test cap: what it drops is counted and explained correctly.

Invariants, for `scripts/mutation_changed.sh` with MAX_TESTS_PER_FILE set:
- a test file kept for one changed module is never counted as dropped because another module's
  cap left it out, whatever characters its name holds, and its name is never run as shell code
  (the name comes from the pull request's own tree);
- when the cap drops a test file, the summary says that a mutant reached by no selected test may
  be covered by it, not only that a survivor may be killed by it: the cap mostly turns lost
  coverage into "no tests" (CCA of #848, DOC-001).

Failure modes caught: `unset "dropped_tests[$t]"` parsing the subscript twice, which ran a
`$(...)` in a test file's name and left a name with `]` counted as not run (CCA of #848,
SEC-001); and a summary that blamed database tests alone for untested mutants.

Same harness as tests/test_mutation_time_budget.py: a throwaway git repository and a fake
`mutmut` on PATH, so nothing is mutated. POSIX only, like the job.

Red proof, 2026-10-28, on the test host, each against `scripts/mutation_changed.sh` as it was
before the fix (the run's pre-fix snapshot) with this file unchanged, failing at its assertion,
then restored and green:
- ``test_a_test_file_kept_for_one_module_is_not_counted_as_dropped``: for the `$(...)` name the
  marker file its name creates existed; for the `]` name the summary reported one importing test
  file not run.
- ``test_the_summary_names_dropped_test_files_as_a_cause_of_untested_mutants``: the dropped-file
  line spoke of survivors only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
  run) echo "Running mutation testing" ;;
  results) echo "    recall.thing.x_value__mutmut_1: killed" ;;
esac
"""

#: Test file names the double parse got wrong: one runs a command if the script evaluates it, one
#: holds a `]`. Kept apart, because a `]` ends the subscript before a `$(...)` after it is read. No
#: spaces: the script's ranking splits on them, so a real payload would avoid them too.
ODD_NAMES = ("test_odd$(touch${IFS}PWNED).py", "test_odd].py")


def _repository(tmp_path: Path, test_files: dict[str, str]) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "recall").mkdir()
    (repo / "recall" / "thing.py").write_text("def x_value():\n    return 1\n", encoding="utf-8")
    (repo / "recall" / "other.py").write_text("def y_value():\n    return 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    for name, body in test_files.items():
        (repo / "tests" / name).write_text(body, encoding="utf-8")
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
    env["MUTATE_FILES"] = "recall/thing.py recall/other.py"
    env["MAX_TESTS_PER_FILE"] = "1"
    env["MUTATION_BUDGET_SECONDS"] = "30"
    env["GITHUB_STEP_SUMMARY"] = str(tmp_path / "summary.md")
    return repo, env


def _summary(repo: Path, env: dict[str, str]) -> str:
    subprocess.run(
        ["bash", "scripts/mutation_changed.sh"], cwd=repo, env=env, capture_output=True, text=True,
        timeout=60,
    )
    return Path(env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("odd_name", ODD_NAMES)
def test_a_test_file_kept_for_one_module_is_not_counted_as_dropped(
    odd_name: str, tmp_path: Path
) -> None:
    # `thing` keeps test_thing.py (named after it) and drops the odd file; `other` keeps the odd
    # file, its only importer. So the odd file runs, and must not be reported as dropped.
    repo, env = _repository(tmp_path, {
        "test_thing.py": "from recall.thing import x_value\n\n\ndef test_x():\n    assert x_value() == 1\n",
        odd_name: (
            "from recall.thing import x_value\nfrom recall.other import y_value\n\n\n"
            "def test_y():\n    assert y_value() == 2 and x_value() == 1\n"
        ),
    })

    summary = _summary(repo, env)

    assert not (repo / "PWNED").exists()
    assert "importing test file(s) not run" not in summary


def test_the_summary_names_dropped_test_files_as_a_cause_of_untested_mutants(tmp_path: Path) -> None:
    repo, env = _repository(tmp_path, {
        "test_thing.py": "from recall.thing import x_value\n\n\ndef test_x():\n    assert x_value() == 1\n",
        "test_more.py": "from recall.thing import x_value\n\n\ndef test_z():\n    assert x_value()\n",
    })
    env["MUTATE_FILES"] = "recall/thing.py"

    summary = _summary(repo, env)

    dropped = [line for line in summary.splitlines() if "importing test file(s) not run" in line]
    assert len(dropped) == 1
    assert "reached by no selected test" in dropped[0]
