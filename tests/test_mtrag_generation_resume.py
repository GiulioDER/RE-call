"""A row carrying no answer must not make a re-run think that task is finished.

`already_done` counted any row with a `task_id`, whatever it carried. That was correct while every
written row necessarily held an answer, and stopped being correct the moment a run could write
`predictions: [{"text": ""}]`, which is what this module's own generation path did for every
`content_filter` and `tool_calls` completion until the guards in `generate_one` were added.

So the guards fix new runs and leave OLD artifacts stranded: a submission written before them holds
the empty rows, and re-running the identical command skips those task ids, writes nothing, prints
`already_done`, and exits **0**. The file is unsubmittable and the exit code says it is whole. The
`incomplete` gate cannot catch it either, since that reports on tasks that failed IN THIS RUN and
these tasks are never attempted.

⚠️ Fixing the skip alone would corrupt the file instead of repairing it. The output is opened in
APPEND mode and is the run's checkpoint, so regenerating a task whose empty row is still on disk
leaves TWO rows with the same `task_id`, one unscorable and one real, in a format whose checker
expects one row per task and a judge that would score whichever it met first. The stale row has to
go at the same time, which is why `drop_answerless_rows` exists and why `main` calls it before it
opens the file for appending.

That pruning is deliberately scoped to the tasks THIS RUN will regenerate. Under `--limit`, or a
task list that no longer contains a row's id, a row nothing is going to replace is left exactly
where it is: removing it would shrink the artifact without repairing it, which is a different and
worse failure than the one being fixed.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.mtrag import generation as gen
from tests.test_mtrag_generation import _mtrag_root, _task


def _resumable(out, rows: str, prompt: str = gen.DEFAULT_PROMPT) -> None:
    """An output file AND the manifest a resume requires beside it.

    Resuming without one is refused outright, on the grounds that rows of unknown provenance must
    not be mixed with rows from a known prompt. So a test that drives `main` over an existing file
    has to supply the manifest, exactly as the run that wrote those rows would have.
    """
    out.write_text(rows, encoding="utf-8")
    gen.write_run_manifest(out, prompt=prompt)


def _row(task_id: str, answer: object = "an answer", *, carry_predictions: bool = True) -> str:
    row: dict[str, object] = {"task_id": task_id, "input": [], "contexts": []}
    if carry_predictions:
        row["predictions"] = [{"text": answer}] if answer is not None else []
    return json.dumps(row) + "\n"


# --------------------------------------------------------------------------------------------
# 1. What counts as done.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, why",
    [
        pytest.param(_row("t<::>1", ""), "the exact shape the pre-guard path wrote", id="empty"),
        pytest.param(_row("t<::>1", "  \n "), "reaches a judge as nothing", id="whitespace"),
        pytest.param(_row("t<::>1", None), "predictions present but empty", id="no-prediction"),
        pytest.param(
            _row("t<::>1", carry_predictions=False), "no predictions key at all", id="no-key"
        ),
        pytest.param(json.dumps({"task_id": "t<::>1"}) + "\n", "task_id and nothing else", id="bare"),
    ],
)
def test_a_row_carrying_no_answer_is_not_done(tmp_path, line: str, why: str) -> None:
    """Each of these used to count as a completed task purely for having a `task_id`."""
    out = tmp_path / "preds.jsonl"
    out.write_text(line, encoding="utf-8")

    assert gen.already_done(out) == set(), why


def test_a_row_carrying_a_real_answer_is_still_done(tmp_path) -> None:
    """Guards the guard. Generation is the expensive step and resume exists to protect it, so an
    over-eager emptiness check would repay for every answer the run already holds."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("t<::>1", "Busch Stadium."), encoding="utf-8")

    assert gen.already_done(out) == {"t<::>1"}


def test_a_torn_trailing_line_is_still_ignored_rather_than_fatal(tmp_path) -> None:
    """Pre-existing behaviour that must survive: a half-written final line is what an interrupted
    run leaves, and that task is simply regenerated."""
    out = tmp_path / "preds.jsonl"
    out.write_text(
        _row("a<::>1", "hi") + "{ truncated line that never finished\n", encoding="utf-8"
    )

    assert gen.already_done(out) == {"a<::>1"}


# --------------------------------------------------------------------------------------------
# 2. The stale row is removed, not duplicated.
# --------------------------------------------------------------------------------------------


def test_the_answerless_row_is_dropped_so_the_rerun_does_not_duplicate_it(tmp_path) -> None:
    """The output is append-only, so leaving the empty row would put two rows under one
    `task_id`: the format checker expects one per task, and a judge would score whichever it
    reached first."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("keep<::>1", "real") + _row("redo<::>1", ""), encoding="utf-8")

    dropped = gen.drop_answerless_rows(out, {"redo<::>1"})

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert dropped == 1
    assert [r["task_id"] for r in rows] == ["keep<::>1"], "the real answer must survive untouched"


def test_a_row_this_run_will_not_regenerate_is_left_alone(tmp_path) -> None:
    """⚠️ Scope. Under `--limit`, or against a task list that no longer carries an id, pruning a row
    nothing will replace shrinks the artifact without repairing it. Every row removed here is one
    the caller is about to rewrite."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("out-of-scope<::>1", ""), encoding="utf-8")

    dropped = gen.drop_answerless_rows(out, set())

    assert dropped == 0
    assert out.read_text(encoding="utf-8") == _row("out-of-scope<::>1", "")


def test_a_file_with_nothing_to_drop_is_not_rewritten(tmp_path) -> None:
    """A checkpoint is not worth rewriting to change nothing: the window in which the file does not
    exist is pure risk, paid on every ordinary resume."""
    out = tmp_path / "preds.jsonl"
    original = _row("a<::>1", "real") + _row("b<::>1", "also real")
    out.write_text(original, encoding="utf-8")
    before = out.stat().st_mtime_ns

    assert gen.drop_answerless_rows(out, {"a<::>1", "b<::>1"}) == 0
    assert out.read_text(encoding="utf-8") == original
    assert out.stat().st_mtime_ns == before, "an untouched file must not even be re-written"


def test_a_torn_line_survives_the_prune_that_rewrites_the_file_around_it(tmp_path) -> None:
    """The prune rewrites the whole file, so every line it does not explicitly keep is destroyed.
    A half-written final line carries no `task_id`, so it can never become the duplicate this
    function exists to prevent, and `already_done` already treats it as absent. Discarding it here
    would be a second, unrelated change to what an interrupted write leaves behind, smuggled in on
    the back of an unrelated repair."""
    out = tmp_path / "preds.jsonl"
    torn = "{ truncated line that never finished\n"
    out.write_text(_row("keep<::>1", "real") + _row("redo<::>1", "") + torn, encoding="utf-8")

    assert gen.drop_answerless_rows(out, {"redo<::>1"}) == 1

    text = out.read_text(encoding="utf-8")
    assert torn in text, "the prune must not quietly repair a torn line it was not asked about"
    assert "redo<::>1" not in text


def test_pruning_a_file_that_does_not_exist_is_not_an_error(tmp_path) -> None:
    """The first run of any command reaches this with no output file."""
    assert gen.drop_answerless_rows(tmp_path / "absent.jsonl", {"a<::>1"}) == 0


# --------------------------------------------------------------------------------------------
# 3. End to end through `main`, which is where the defect was actually reachable.
# --------------------------------------------------------------------------------------------


def test_a_rerun_repairs_an_artifact_written_before_the_guards(tmp_path, monkeypatch) -> None:
    """The whole defect, end to end. A submission written by the pre-guard path holds an empty row;
    re-running the identical command used to skip it, write nothing and exit 0."""
    tasks = [_task(task_id="a<::>1"), _task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("a<::>1", "already good") + _row("b<::>1", ""))

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "the regenerated answer")

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_id = {r["task_id"]: r["predictions"][0]["text"] for r in rows}
    assert rc == 0
    assert len(rows) == 2, f"one row per task, not a duplicate for the repaired one: {rows}"
    assert by_id["b<::>1"] == "the regenerated answer"
    assert by_id["a<::>1"] == "already good", "a task already answered must not be paid for again"


def test_a_rerun_that_cannot_repair_the_row_does_not_report_success(tmp_path, monkeypatch) -> None:
    """The repair can fail, and then the file is genuinely incomplete. It must say so through the
    exit code, which is the thing that was returning 0 over an unsubmittable file."""
    tasks = [_task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("b<::>1", ""))

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())

    def refuse(*a, **k):
        raise RuntimeError("the provider returned a completion with no text")

    monkeypatch.setattr(gen, "generate_one", refuse)

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert rc == 1
    assert out.read_text(encoding="utf-8") == "", (
        "the unscorable row must be gone rather than left behind as a fake completed task"
    )
