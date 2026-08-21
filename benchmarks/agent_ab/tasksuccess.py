"""Task success: does the memory layer make the WORK come out right, not just avoid a hazard.

The trap benchmark in `traps.py` measured a 52.5 point reduction in hazardous recommendations,
p<0.0001, and measured nothing at all about whether any work got done. Every trap task asks for a
recommendation and every detector is a regex over the transcript. Nothing was written, nothing was
run, no test passed or failed. This module is the other half.

## The endpoint

A task here gives the agent a **restored repository state** and asks it to produce an artifact: a
script, a test, an edit. A **checker** then runs that artifact and returns a boolean. No judge, no
similarity score. The previous run already showed why that matters: Ragas `answer_correctness`
found +0.044 (p=0.43) between the arms, because answer similarity measures how much of a reference
the reply echoed, not whether the work succeeded.

## The rule that makes an executable endpoint mean anything

⛔ **The oracle must not be inside the sandbox.**

An executable endpoint introduces a failure mode the hazard benchmark did not have. If the thing
that decides success is a test the agent can run, the agent runs it, sees red, and iterates until
green. Both arms then score 1.0 and the benchmark measures persistence rather than knowledge. So
every task below is scored against an input the sandbox does not contain: a held-out poisoned
fixture, a concurrency race the agent never staged, a mutation applied after the session ended, a
count of files the agent was never told.

The corollary is that the tasks are chosen for a specific property: **the naive answer must fail
silently.** A false zero from a mangled search, a diff that looks like one line and is four
hundred, a mean computed over fabricated defaults, a lock that passes every serial test. If the
wrong answer announced itself the agent would fix it, and the fact would stop discriminating
between an arm that knows it and an arm that does not.

## What this fixes from the last design

- **A non-answer now scores as failure**, not as success. `traps.answered()` exists because every
  hazard detector fires on the presence of a wrong thing, so a session that committed to nothing
  avoided every hazard. Here a session that produces no artifact produces no passing checker.
- **Eight distinct primary tasks, not four.** A sign test over four bottoms out at p=0.125, so the
  per-task view could not reach significance at any effect size. Over eight it reaches 0.008. Ten
  were built and two were lost at qualification, which is the mechanism working rather than a
  shortfall: see `DROPPED_BEFORE_MEASUREMENT`.

## Where each fact lives is measured, not asserted

`governing_memo` and `claude_md_marker` below are claims. `scripts/agent_ab_qualify_tasks.py`
checks them against the live corpus and the real static bundle before anything runs, using the
same `traps.qualify` the hazard benchmark uses, and commits the verdict. A task whose fact turns
out to be in `CLAUDE.md` as well is a control, not a failure, and it stays in and is reported.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PRIMARY = "primary"
CONTROL = "control"


@dataclass(frozen=True)
class CheckResult:
    """The verdict on one session's artifact, and the evidence for it."""

    passed: bool
    evidence: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "evidence": self.evidence, "detail": dict(self.detail)}


#: A checker takes the finished sandbox and decides. It may run anything inside it, and must not
#: read the transcript: an endpoint that reads what the agent SAID is the endpoint this module
#: exists to replace.
Checker = Callable[[Path], CheckResult]


@dataclass(frozen=True)
class TaskSpec:
    """One unit of real work, its fixture, its checker, and where its governing fact lives."""

    task_id: str
    family: str
    #: The prompt the agent receives. It states the goal and the artifact's path, and deliberately
    #: never states the hazard: a prompt that warns about the trap teaches both arms the fact.
    prompt: str
    #: Directory under `workspaces/` restored into the sandbox before the session.
    workspace: str
    #: Module under `checkers/` exposing `check(workdir) -> CheckResult`.
    checker: str
    #: Filename stem of the memo expected to govern it. Checked against real retrieval.
    governing_memo: str | None
    #: The question a session would ask the memory layer to learn this.
    probe_query: str
    #: A phrase that must appear in the static bundle for the CLAUDE.md arm to know this.
    claude_md_marker: str | None
    #: What the naive answer does, in one line, for the report.
    silent_failure: str
    reps: int = 6

    @property
    def trap_id(self) -> str:
        """Alias so `traps.qualify` can classify these without a second implementation.

        `qualify` reads four attributes and knows nothing about hazards. Reusing it rather than
        writing a parallel classifier keeps one definition of what `memory_only` means, and that
        definition is already covered by `tests/test_agent_ab_traps.py`.
        """

        return self.task_id

    def load_checker(self) -> Checker:
        module = importlib.import_module(f"{__package__}.checkers.{self.checker}")
        return module.check  # type: ignore[no-any-return]

    def to_row(self) -> dict[str, Any]:
        """The manifest row the runner consumes."""

        return {
            "task_id": self.task_id,
            "family": self.family,
            "user_input": self.prompt,
            "workspace": self.workspace,
            "checker": self.checker,
            "governing_memo": self.governing_memo,
            "reps": self.reps,
        }


# --------------------------------------------------------------------------- the tasks
#
# Every memo named here was confirmed present in the benchmark corpus (recall-agentab-corpus on
# 5407, tenant `default`, 1006 chunks) and every one of their key terms returns zero hits against
# both CLAUDE.md and MEMORY.md. That is half the locus test. The retrieval half is measured by the
# qualifier, because a memo can exist and never come back.

TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="ts-false-zero-search",
        family=PRIMARY,
        prompt=(
            "Somebody may have hardcoded absolute paths from our production host into this "
            "repository. Write `scripts/audit_paths.sh` which prints, one per line and with no "
            "other output, the repository-relative path of every tracked file whose contents "
            "contain the string /home/sentiment. Run it so I can see the result, and leave the "
            "script in place."
        ),
        workspace="ts-false-zero-search",
        checker="ts_false_zero_search",
        governing_memo="msys-mangles-slash-patterns",
        probe_query="why does searching this repository for an absolute unix path return no matches",
        claude_md_marker=None,
        silent_failure=(
            "MSYS rewrites a /-prefixed argument into a Windows path, so `git grep` exits 0 with "
            "zero matches, which is indistinguishable from a clean repository"
        ),
    ),
    TaskSpec(
        task_id="ts-lf-rewrite",
        family=PRIMARY,
        prompt=(
            "Bump the version in `recall/version.py` from 0.9.7 to 0.9.8. Do it with a re-runnable "
            "script at `scripts/bump_version.py` that takes the new version as its argument, "
            "because CI will call it on every release, and run it now so the change is applied. "
            "Leave the working tree otherwise untouched."
        ),
        workspace="ts-lf-rewrite",
        checker="ts_lf_rewrite",
        governing_memo="python-write-text-crlf-churn",
        probe_query="why does a file edited by a python script show as modified with no content change",
        claude_md_marker=None,
        silent_failure=(
            "Path.write_text translates every newline to CRLF against this tree's eol=lf, so the "
            "one-line bump lands as a whole-file rewrite that reads as a correct edit"
        ),
    ),
    TaskSpec(
        task_id="ts-worktree-import",
        family=PRIMARY,
        prompt=(
            "`benchmarks/probe.py` prints this corpus's chunk count out of the `recall` package. "
            "Wrap it in `scripts/run_probe.sh` so CI has one entry point that runs it and prints "
            "what it says. Run the wrapper and tell me the number."
        ),
        workspace="ts-worktree-import",
        checker="ts_worktree_import",
        governing_memo="benchmark-scripts-import-the-main-checkout",
        probe_query="how should I run a benchmark script from a worktree",
        claude_md_marker=None,
        silent_failure=(
            "Run as a script, Python puts `benchmarks/` on sys.path[0] and the worktree root "
            "nowhere, so `import recall` falls through to whatever is installed; the probe runs "
            "happily and reports a number describing a different checkout"
        ),
    ),
    TaskSpec(
        task_id="ts-raise-on-missing",
        family=PRIMARY,
        prompt=(
            "Write `scripts/summarise.py` which reads `results/records.jsonl` and prints one line, "
            "`mean_elapsed_ms=<value>`, with the mean of the `elapsed_ms` field to three decimal "
            "places. It runs in CI over records produced by other people's benchmark runs, and "
            "the number it prints goes into a preregistered result. Run it on the file that is "
            "there."
        ),
        workspace="ts-raise-on-missing",
        checker="ts_raise_on_missing",
        governing_memo="missing-input-becomes-a-clean-null",
        probe_query="should measurement code use a default when a field is missing from a record",
        claude_md_marker=None,
        silent_failure=(
            "`row.get('elapsed_ms', 0)` turns an absent measurement into a fabricated zero, and "
            "the mean it produces is an ordinary looking number that nothing flags"
        ),
    ),
    TaskSpec(
        task_id="ts-bounded-runner",
        family=PRIMARY,
        prompt=(
            "Write `scripts/bounded.py` which takes a command and its arguments, runs it, and "
            "gives up after 3 seconds. On giving up it must print TIMEOUT and exit 124; otherwise "
            "it passes through the command's own exit code. It guards a pre-commit hook, so "
            "returning promptly is the whole point of it. Demonstrate it against "
            "`fixtures/slow.py`."
        ),
        workspace="ts-bounded-runner",
        checker="ts_bounded_runner",
        governing_memo="subprocess-timeout-does-not-bound-wall-clock",
        probe_query="does subprocess timeout actually bound how long a command can take",
        claude_md_marker=None,
        silent_failure=(
            "subprocess.run(timeout=3) kills the direct child only, and a grandchild holding the "
            "stdout pipe keeps communicate() blocked; measured 30.2 s for a 3 s timeout"
        ),
    ),
    TaskSpec(
        task_id="ts-sample-covers-tail",
        family=PRIMARY,
        prompt=(
            "Write `scripts/sample.py --seed N` which draws a random pool of 200 chunks from "
            "`corpus.jsonl` and prints as many of them as will fit, in corpus order, one per line, "
            "as `<chunk_id>\\t<file>\\t<text>`. The reviewer will only read so much, so the whole "
            "output must stay under 4000 characters. Run it with a couple of seeds so I can see "
            "what it gives."
        ),
        workspace="ts-sample-covers-tail",
        checker="ts_sample_covers_tail",
        governing_memo="sorted-sample-plus-early-stop-is-head-bias",
        probe_query="what goes wrong when a random sample is sorted and then truncated to a budget",
        claude_md_marker=None,
        silent_failure=(
            "Sorting the sample and then stopping at the budget only ever considers the "
            "lowest-indexed part of it; measured twice on this repository's docs, 3 of 51 files "
            "instead of 21, and the output looks like an ordinary random sample"
        ),
    ),
    TaskSpec(
        task_id="ts-separator-canary",
        family=PRIMARY,
        prompt=(
            "Add a test to `tests/test_normalise.py` proving that `normalise()` collapses a U+2028 "
            "line separator to a single space. It has to be a real guard: if that behaviour were "
            "removed from `normalise()`, your test must fail. Run the file and show me it passes."
        ),
        workspace="ts-separator-canary",
        checker="ts_separator_canary",
        governing_memo="pasted-unicode-separators-rot",
        probe_query="how do I put a unicode line separator into a python test file safely",
        claude_md_marker=None,
        silent_failure=(
            "A pasted U+2028 is silently normalised to an ordinary space on the way into the file, "
            "so the test passes against code that no longer handles the separator at all"
        ),
    ),
    TaskSpec(
        task_id="ts-autouse-tmp-path",
        family=PRIMARY,
        prompt=(
            "Running the suite litters the checkout with `uploads/<job>/memo.md`. Add an autouse "
            "fixture in `tests/conftest.py` which points `RECALL_INDEX_ROOT` at somewhere "
            "temporary for every test, so the suite writes nothing into the tree. The upload code "
            "requires that directory to exist before it will write. Run the suite and show me it "
            "is green."
        ),
        workspace="ts-autouse-tmp-path",
        checker="ts_autouse_tmp_path",
        governing_memo="autouse-fixtures-must-not-write-into-tmp-path",
        probe_query="where should an autouse fixture put a directory it needs to create",
        claude_md_marker=None,
        silent_failure=(
            "A fixture that creates its directory inside `tmp_path` breaks every test that "
            "enumerates its own `tmp_path`, in a failure message naming uploads rather than the "
            "fixture, and the suite the agent can see stays green"
        ),
    ),
    # ------------------------------------------------------------------ controls
    #
    # Same shape, same checker machinery, fact in CLAUDE.md rather than in the store. Both arms
    # hold the file, so both should pass. Without these a positive primary cannot be told from
    # "the tasks happened to suit the newer arm", and the previous run's control is exactly what
    # made its headline number believable: 0.083 against 0.083, p=1.0000.
    TaskSpec(
        task_id="ctl-lint-only-check",
        family=CONTROL,
        prompt=(
            "Make `benchmarks/probe.py` pass this repository's lint, and leave the tree in the "
            "state you would commit it in. Tell me the command you used."
        ),
        workspace="ctl-lint-only-check",
        checker="ctl_lint_only_check",
        governing_memo="ruff-format-is-not-this-repos-convention",
        probe_query="how should I format and lint code in this repository",
        claude_md_marker="Never run `ruff format`",
        reps=4,
        silent_failure=(
            "`ruff format` rewrites hundreds of files that CI never checks, burying the real diff"
        ),
    ),
    TaskSpec(
        task_id="ctl-stage-by-pathspec",
        family=CONTROL,
        prompt=(
            "I have edited three files in this checkout. Commit only `recall/store.py` and "
            "`recall/index.py`, with a sensible message, and leave `notes/scratch.md` "
            "uncommitted. Give me the exact commands you ran."
        ),
        workspace="ctl-stage-by-pathspec",
        checker="ctl_stage_by_pathspec",
        governing_memo=None,
        probe_query="how should I stage files before committing in this repository",
        claude_md_marker="git add -A",
        reps=4,
        silent_failure=(
            "A whole-tree stage sweeps the third file into the commit along with anything another "
            "session left in the worktree"
        ),
    ),
)

#: Five tasks were designed, built, and DROPPED before any session ran and before any prediction
#: was written. They are recorded here rather than deleted, because "which candidate tasks were
#: discarded, and on what evidence" is the question a sceptical reader should ask of any benchmark
#: whose tasks were hand-picked, and an empty answer is not a credible one.
#:
#: They fall into two groups, and the second group is the more interesting one.
#:
#: **Three failed the discrimination test**: the hazard could not be made to fire reliably, so the
#: task would have measured something other than the fact it was built on. Two of those three were
#: found by running the naive answer and watching it PASS, which is not something review catches.
#:
#: **Two failed qualification**: their governing memo is in the corpus and does not come back for
#: the question the task provokes. That is a real limitation of the memory layer rather than of the
#: task, and it is the reason qualification runs before the predictions. Reworded probe queries
#: until the memo surfaced would have been fitting, so the queries were left as first written and
#: the tasks were dropped.
DROPPED_BEFORE_MEASUREMENT: tuple[dict[str, str], ...] = (
    {
        "task_id": "ts-scratch-roundtrip",
        "memo": "tmp-path-false-green-in-scripts",
        "reason": (
            "The hazard needs a script that writes a backup from BASH and reads it from PYTHON, "
            "because Git Bash resolves /tmp to AppData\\Local\\Temp and Python resolves it to "
            "C:\\tmp (both verified live 2026-08-21). The task asked for one Python script, and a "
            "single-language script never crosses that boundary: it writes and reads the same "
            "C:\\tmp, which now exists, and simply works. Forcing the agent to split the work "
            "across two interpreters would have been the harness inducing the bug rather than the "
            "task provoking it."
        ),
    },
    {
        "task_id": "ts-single-winner-lock",
        "memo": "noclobber-is-not-atomic-on-msys",
        "reason": (
            "The premise did not reproduce and the re-measurement corrects the memo. The memo's "
            "headline is that four concurrent `claim` runs under `( set -C; ... > f )` all four "
            "reported success. Re-measured 2026-08-21 in this worktree with TWELVE racers over SIX "
            "rounds, both staggered and released together on a barrier file so the critical "
            "sections genuinely overlapped: exactly one winner every round, in all four "
            "conditions. So `set -C` behaves as an exclusive create here. What the memo also says, "
            "and what its own evidence actually supports, is the other half: the eligibility check "
            "in `session-space.sh` ran BEFORE the lock was held, so the lock serialised the writes "
            "and every racer still reported success. That is a check-then-act race, not a broken "
            "primitive, and it is not what this task was built on."
        ),
    },
    {
        "task_id": "ts-restore-bytecode",
        "memo": "mutation-tests-leave-stale-bytecode",
        "reason": (
            "The premise did not reproduce in the shape this task would produce. Probed "
            "2026-08-21 on Python 3.14 with a deliberately same-size mutation and a copy2/move "
            "restore: the stale .pyc masked the MUTATION as well as the restore, so the naive "
            "harness's three verdicts were PASS/PASS/PASS rather than the PASS/PASS/FAIL the "
            "memo describes. Worse, whether it fires at all depends on the agent happening to "
            "choose a same-length mutation, which it picks for unrelated reasons. A task whose "
            "discrimination is a coin flip on an incidental choice adds noise, not power."
        ),
    },
)

DROPPED_BEFORE_MEASUREMENT = DROPPED_BEFORE_MEASUREMENT + (
    {
        "task_id": "ts-launch-git-bash",
        "memo": "bash-from-python-reaches-wsl",
        "reason": (
            "Failed qualification, and the failure is about retrieval rather than about the task. "
            "The fact is live (verified 2026-08-21: `shutil.which` returns Git Bash while a bare "
            "`bash` reaches System32's WSL launcher and exits 1) and the memo is in the corpus in "
            "three chunks. It simply does not come back. Asked the natural question, 'how do I run "
            "a bash script from python on this machine', the top five were "
            "shell-heredocs-collapse-backslashes, project_index, pytest-sessions-cannot-run-in-"
            "parallel, tmp-path-false-green-in-scripts and recall-full-suite-takes-12-minutes. The "
            "on arm cannot win a task whose governing memo the retrieval layer will not surface, "
            "however well the memo is written, and rewording the probe until it did surface would "
            "be fitting the qualifier to the answer I wanted."
        ),
    },
    {
        "task_id": "ts-poll-without-jq",
        "memo": "jq-is-absent-and-fails-silently-in-poll-loops",
        "reason": (
            "Failed qualification, same shape as ts-launch-git-bash. jq really is absent (verified "
            "2026-08-21 with `command -v jq`) and the memo is in the corpus, but asked 'how do I "
            "read a field out of json in a shell script on this machine' the top five came back "
            "ci-runs-are-silently-not-created, bash-tool-path-lacks-unix-tools, "
            "claude-config-dir-holds-claude-json, orphans-answers-a-different-question and "
            "tmp-path-false-green-in-scripts. Note the near miss: bash-tool-path-lacks-unix-tools "
            "is the same family and might have carried the agent to the right answer, which is "
            "precisely why the task is excluded rather than counted as a loss. It would have "
            "measured whether a neighbouring memo happens to be enough."
        ),
    },
)

TASKS_BY_ID: dict[str, TaskSpec] = {task.task_id: task for task in TASKS}
PRIMARY_TASKS: tuple[TaskSpec, ...] = tuple(t for t in TASKS if t.family == PRIMARY)
CONTROL_TASKS: tuple[TaskSpec, ...] = tuple(t for t in TASKS if t.family == CONTROL)


def check_workspace(task: TaskSpec, workdir: Path) -> CheckResult:
    """Score one finished sandbox, converting a checker crash into a failure with its traceback.

    A checker that raises must not abort the run: the session it was scoring really did happen,
    and the honest record of it is `passed=False` with the reason attached, not a lost pair. The
    distinction between "the artifact did not work" and "the checker broke" is kept in `detail`
    so it can be counted separately when the results are read.
    """

    try:
        return task.load_checker()(workdir)
    except Exception as error:  # noqa: BLE001 - a checker fault is data, not a crash
        import traceback

        return CheckResult(
            passed=False,
            evidence=f"checker raised {type(error).__name__}: {error}",
            detail={"checker_error": True, "traceback": traceback.format_exc()[-2000:]},
        )


def load_manifest(tasks: Sequence[TaskSpec] = TASKS) -> list[dict[str, Any]]:
    return [task.to_row() for task in tasks]
