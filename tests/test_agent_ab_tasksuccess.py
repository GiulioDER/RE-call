"""Every checker is watched failing and watched passing, against the real fixtures.

This file is the benchmark's validity evidence, not a smoke test. A task-success endpoint makes two
claims per task, and both are silent when wrong:

1. an answer written **without** the governing fact fails the checker;
2. an answer written **with** it passes.

If (1) is false the task contributes a row that measures nothing. If (2) is false the task looks
impossible in both arms and drags the estimate toward zero. Neither shows up in the run: the numbers
come out looking like a small or absent effect, which is exactly what a null looks like.

These tests are slow on purpose. They stage eight-way concurrency races, run pytest inside
sandboxes, and wait out a 30 second hang, because those are the conditions under which the facts
are true. Fast substitutes would be testing the substitutes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.agent_ab.reference import REFERENCE
from benchmarks.agent_ab.sandbox import WORKSPACES, restore, tree_digest
from benchmarks.agent_ab.tasksuccess import (
    CONTROL_TASKS,
    DROPPED_BEFORE_MEASUREMENT,
    PRIMARY_TASKS,
    TASKS,
    TASKS_BY_ID,
    check_workspace,
)

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- the shape of the set


def test_there_are_eight_primary_tasks_and_two_controls():
    """Eight, because four was the last design's binding weakness.

    A sign test over four distinct tasks bottoms out at p=0.125, so the per-task view could not
    reach significance at any effect size. Over eight it reaches 0.008. Ten were built; two were
    dropped by qualification, which is the mechanism doing its job rather than a shortfall.
    """

    assert len(PRIMARY_TASKS) == 8
    assert len(CONTROL_TASKS) == 2


def test_every_task_names_a_distinct_memo():
    """Two tasks on one memo are one measurement counted twice."""

    memos = [task.governing_memo for task in TASKS if task.governing_memo]
    assert len(memos) == len(set(memos)), sorted(memos)


def test_dropped_tasks_are_recorded_with_their_reason():
    """The discarded candidates stay visible, because "what did you throw away" is the question."""

    assert len(DROPPED_BEFORE_MEASUREMENT) == 5
    for entry in DROPPED_BEFORE_MEASUREMENT:
        assert entry["task_id"] not in TASKS_BY_ID
        assert len(entry["reason"]) > 200


def test_every_task_has_a_fixture_and_a_loadable_checker():
    for task in TASKS:
        assert (WORKSPACES / task.workspace / "tree").is_dir(), task.task_id
        assert callable(task.load_checker()), task.task_id


def test_every_task_has_both_reference_solutions():
    for task in TASKS:
        assert set(REFERENCE[task.task_id]) == {"naive", "informed"}, task.task_id


def test_the_generator_is_a_no_op_against_the_committed_fixtures():
    """A fixture that cannot be regenerated is a directory nobody can audit."""

    before = subprocess.run(  # noqa: S603 - argv list, no shell
        ["git", "status", "--porcelain", "--", "benchmarks/agent_ab/workspaces",
         "benchmarks/agent_ab/oracles"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
    ).stdout
    subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, "scripts/agent_ab_build_workspaces.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
    )
    after = subprocess.run(  # noqa: S603 - argv list, no shell
        ["git", "status", "--porcelain", "--", "benchmarks/agent_ab/workspaces",
         "benchmarks/agent_ab/oracles"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
    ).stdout
    assert after == before, f"regenerating changed the fixtures:\n{after}"


# --------------------------------------------------------------------------- restoration


def test_restoring_twice_gives_the_same_digest(tmp_path):
    """The property the admission gate leans on: both arms start from identical ground."""

    task = PRIMARY_TASKS[0]
    first = restore(task.workspace, tmp_path / "a")
    second = restore(task.workspace, tmp_path / "b")
    assert first == second


def test_the_digest_notices_a_line_ending(tmp_path):
    """Bytes, not text. Line endings are the endpoint of one task, so they cannot be normalised."""

    restore("ts-lf-rewrite", tmp_path / "a")
    before = tree_digest(tmp_path / "a")
    target = tmp_path / "a" / "recall" / "version.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert tree_digest(tmp_path / "a") != before


def test_restore_refuses_a_directory_that_already_has_contents(tmp_path):
    """A reused sandbox makes a previous session's leftovers look like this session's work."""

    destination = tmp_path / "used"
    destination.mkdir()
    (destination / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        restore("ts-lf-rewrite", destination)


def test_no_oracle_file_is_reachable_from_any_sandbox(tmp_path):
    """The endpoint only means anything while the deciding input stays out of the agent's reach."""

    for task in TASKS:
        workdir = tmp_path / task.task_id
        restore(task.workspace, workdir)
        names = {path.name for path in workdir.rglob("*") if path.is_file()}
        assert "poisoned.jsonl" not in names, task.task_id
        assert "test_scratch_isolation.py" not in names, task.task_id
        assert "normalise_broken.py" not in names, task.task_id
        assert "expected.txt" not in names, task.task_id


# --------------------------------------------------------------------------- discrimination
#
# The two that matter. Parametrised over every task so a new one cannot be added without evidence
# that its checker separates the two answers.
#
# ⚠️ **Windows only, and that is a property of the benchmark rather than a gap in the tests.** Half
# these tasks are built on facts that are only true on this machine: MSYS rewriting a /-prefixed
# argument, `Path.write_text` translating newlines to CRLF, `bash` resolving to System32's WSL
# launcher, `taskkill /T` being the only thing that reaches a grandchild. On ubuntu the checkers
# either cannot run (`git_bash()` raises, correctly, rather than falling back to /bin/bash and
# measuring something else) or the naive answer silently PASSES, because the hazard does not exist
# there. Either way the assertion would be about the runner, not about the task.
#
# Skipped rather than made portable, because making them portable would mean weakening the
# checkers until they agree with a platform the benchmark never runs on. The structural and
# restoration tests above DO run everywhere and still catch a missing fixture, an unloadable
# checker, a fixture that cannot be regenerated, or an oracle leaking into a sandbox.
windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason=(
        "the facts under test are properties of this Windows machine (MSYS path conversion, CRLF "
        "translation, System32 bash, taskkill /T); on another platform these assertions would be "
        "about the runner rather than about the task"
    ),
)


@windows_only
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_the_naive_answer_fails_the_checker(task, tmp_path):
    workdir = tmp_path / f"{task.task_id}-naive"
    restore(task.workspace, workdir)
    REFERENCE[task.task_id]["naive"](workdir)
    result = check_workspace(task, workdir)
    assert not result.detail.get("checker_error"), result.detail.get("traceback")
    assert result.passed is False, f"{task.task_id}: naive answer PASSED -> {result.evidence}"


@windows_only
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_the_informed_answer_passes_the_checker(task, tmp_path):
    workdir = tmp_path / f"{task.task_id}-informed"
    restore(task.workspace, workdir)
    REFERENCE[task.task_id]["informed"](workdir)
    result = check_workspace(task, workdir)
    assert not result.detail.get("checker_error"), result.detail.get("traceback")
    assert result.passed is True, f"{task.task_id}: informed answer FAILED -> {result.evidence}"


@windows_only
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_doing_nothing_fails_the_checker(task, tmp_path):
    """A session that produces nothing must score as failure.

    This is the inversion the hazard benchmark could not have. There, every detector fired on the
    presence of a wrong thing, so a reply that committed to nothing avoided all of them and scored
    clean; `traps.answered()` exists to patch that. An executable endpoint gets it right for free,
    and this test is what says so rather than assuming it.
    """

    workdir = tmp_path / f"{task.task_id}-untouched"
    restore(task.workspace, workdir)
    result = check_workspace(task, workdir)
    assert result.passed is False, f"{task.task_id}: an untouched sandbox PASSED"
