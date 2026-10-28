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
  run)
    # The generated configuration is removed when the script exits; keep a copy for the assertions.
    [ -z "${FAKE_CFG_COPY:-}" ] || cp setup.cfg "$FAKE_CFG_COPY"
    echo "Running mutation testing" ;;
  results) echo "    recall.thing.thing_value__mutmut_1: killed" ;;
esac
"""

#: Test file names the double parse got wrong: one runs a command if the script evaluates it, one
#: holds a `]`. Kept apart, because a `]` ends the subscript before a `$(...)` after it is read. No
#: spaces: the script's ranking splits on them, so a real payload would avoid them too.
ODD_NAMES = ("test_odd$(touch${IFS}PWNED).py", "test_odd].py")


def _repository(
    tmp_path: Path, test_files: dict[str, str], modules: tuple[str, ...] = ("thing", "other")
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "recall").mkdir()
    for n, name in enumerate(modules):
        (repo / "recall" / f"{name}.py").write_text(
            f"def {name}_value():\n    return {n}\n", encoding="utf-8"
        )
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
    env["FAKE_CFG_COPY"] = str(tmp_path / "setup.cfg")
    return repo, env


def _selection(env: dict[str, str]) -> list[str]:
    """The test files the run handed to mutmut, read from its generated configuration."""
    lines = Path(env["FAKE_CFG_COPY"]).read_text(encoding="utf-8").splitlines()
    start = lines.index("pytest_add_cli_args_test_selection=") + 1
    selected = []
    for line in lines[start:]:
        if not line.startswith("    "):
            break
        selected.append(line.strip())
    return selected


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
        "test_thing.py": "from recall.thing import thing_value\n\n\ndef test_x():\n    assert thing_value() == 0\n",
        odd_name: (
            "from recall.thing import thing_value\nfrom recall.other import other_value\n\n\n"
            "def test_y():\n    assert other_value() == 1 and thing_value() == 0\n"
        ),
    })

    summary = _summary(repo, env)

    assert not (repo / "PWNED").exists()
    assert "importing test file(s) not run" not in summary


def test_the_summary_names_dropped_test_files_as_a_cause_of_untested_mutants(tmp_path: Path) -> None:
    repo, env = _repository(tmp_path, {
        "test_thing.py": "from recall.thing import thing_value\n\n\ndef test_x():\n    assert thing_value() == 0\n",
        "test_more.py": "from recall.thing import thing_value\n\n\ndef test_z():\n    assert thing_value() == 0\n",
    })
    env["MUTATE_FILES"] = "recall/thing.py"

    summary = _summary(repo, env)

    dropped = [line for line in summary.splitlines() if "importing test file(s) not run" in line]
    assert len(dropped) == 1
    assert "reached by no selected test" in dropped[0]


def test_conftest_and_helpers_do_not_take_a_test_file_slot(tmp_path: Path) -> None:
    """CCA of #848, BUG-001: conftest.py (loaded by pytest anyway) and helper modules hold no
    tests, so they must not compete with test files for the per-module slots.

    Red proof, 2026-10-28, against the script before the deferred fixes: conftest.py, naming the
    module most often, took the second slot and test_more.py was dropped.
    """
    importer = "from recall.thing import thing_value\n"
    repo, env = _repository(tmp_path, {
        "test_thing.py": importer + "\n\ndef test_a():\n    assert thing_value() == 0\n",
        "test_more.py": importer + "\n\ndef test_b():\n    assert thing_value() == 0\n",
        "conftest.py": importer + "thing = thing_value\nthing2 = thing\nthing3 = thing\n",
    }, modules=("thing",))
    env["MUTATE_FILES"] = "recall/thing.py"
    env["MAX_TESTS_PER_FILE"] = "2"

    summary = _summary(repo, env)

    assert _selection(env) == ["tests/test_more.py", "tests/test_thing.py"]
    assert "importing test file(s) not run" not in summary


def test_a_test_file_named_after_another_word_is_not_ranked_as_named(tmp_path: Path) -> None:
    """CCA of #848, CODE-002: "named after the module" means `test_<leaf>.py` or
    `test_<leaf>_*`, and "mentions" counts the leaf as a word, so test_restore_*.py is not ranked
    as the test of `store`, however often "restore" appears in it.

    Red proof, 2026-10-28, against the script before the deferred fixes: test_restore_cli.py
    matched "store" as a substring on both keys and took the one slot from test_store.py.
    """
    restore = "from recall.store import store_value\n" + "restore = 1\n" * 5
    repo, env = _repository(tmp_path, {
        "test_store.py": "from recall.store import store_value\n\n\ndef test_s():\n    assert store_value() == 0\n",
        "test_restore_cli.py": restore + "\n\ndef test_r():\n    assert store_value() == 0\n",
    }, modules=("store",))
    env["MUTATE_FILES"] = "recall/store.py"
    env["MAX_TESTS_PER_FILE"] = "1"

    _summary(repo, env)

    assert _selection(env) == ["tests/test_store.py"]


def test_modules_over_max_files_add_no_tests_and_no_dropped_count(tmp_path: Path) -> None:
    """CCA of #848, CODE-003: the MAX_FILES cut comes before the per-module ranking, so a module
    that is not mutated adds no test file to the selection and none to the dropped count.

    Red proof, 2026-10-28, against the script before the deferred fixes: `other`, cut by
    MAX_FILES=1, still put its kept test in the selection and its two dropped ones in the count.
    """
    other = "from recall.other import other_value\n\n\ndef test_{0}():\n    assert other_value() == 1\n"
    repo, env = _repository(tmp_path, {
        "test_thing.py": "from recall.thing import thing_value\n\n\ndef test_t():\n    assert thing_value() == 0\n",
        "test_other.py": other.format("o"),
        "test_other_more.py": other.format("p"),
        "test_other_extra.py": other.format("q"),
    })
    env["MAX_FILES"] = "1"

    summary = _summary(repo, env)

    assert _selection(env) == ["tests/test_thing.py"]
    assert "importing test file(s) not run" not in summary
    assert "Not run (over MAX_FILES=1): `recall/other.py`" in summary


@pytest.mark.parametrize("value", ["0", "-1", "2x", "1.5"])
def test_a_cap_that_is_not_a_positive_integer_is_refused(value: str, tmp_path: Path) -> None:
    """CCA of #848, ENV-001: a bad MAX_TESTS_PER_FILE used to select no test file and still exit 0.

    Red proof, 2026-10-28, against the script before the deferred fixes: every value exited 0.
    """
    repo, env = _repository(tmp_path, {
        "test_thing.py": "from recall.thing import thing_value\n\n\ndef test_t():\n    assert thing_value() == 0\n",
    })
    env["MAX_TESTS_PER_FILE"] = value

    completed = subprocess.run(
        ["bash", "scripts/mutation_changed.sh"], cwd=repo, env=env, capture_output=True, text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    assert "MAX_TESTS_PER_FILE must be a positive integer" in completed.stderr
