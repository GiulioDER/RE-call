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


def test_the_answer_reader_returns_the_stripped_text_it_promises() -> None:
    """`already_done` and the prune only ever ask this for TRUTH, so the strip on the way out is
    unobservable through them and can rot into a lie in the docstring. Asserted directly, since a
    reader named `submission_answer` returning padded text is the kind of thing a later caller
    would reasonably trust."""
    row = {"task_id": "t<::>1", "predictions": [{"text": "  Busch Stadium.  "}]}

    assert gen.submission_answer(row) == "Busch Stadium."


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
# 2b. The rewrite touches the run's ONLY checkpoint, so it is held to the standard
#     `benchmarks/mtrag/multiquery.py:write_manifest` already sets for a far cheaper file.
# --------------------------------------------------------------------------------------------


def test_an_answer_in_a_later_prediction_entry_is_not_destroyed(tmp_path) -> None:
    """⚠️ The one shape where the prune could delete model output that was PAID FOR. Emptiness was
    judged from `predictions[0]` alone while the whole ROW is what gets deleted, so a row whose
    first entry is blank and whose second carries the answer was read as answerless and removed.

    `submission_row` only ever writes one entry, so this needs a file this module did not write:
    `--contexts-from` already reads a foreign file, and hand-merged or concatenated submissions are
    the realistic source. Deleting is irreversible; regenerating a task is not. The cheap error has
    to be the second one."""
    out = tmp_path / "preds.jsonl"
    row = json.dumps({
        "task_id": "t<::>1", "predictions": [{"text": ""}, {"text": "THE REAL ANSWER"}],
    }) + "\n"
    out.write_text(row, encoding="utf-8")

    assert gen.already_done(out) == {"t<::>1"}, "the row does carry an answer"
    assert gen.drop_answerless_rows(out, {"t<::>1"}) == 0
    assert "THE REAL ANSWER" in out.read_text(encoding="utf-8")


def test_a_prediction_entry_that_is_not_an_object_carries_no_answer(tmp_path) -> None:
    """Stated by `submission_answer`'s docstring and otherwise unpinned."""
    out = tmp_path / "preds.jsonl"
    out.write_text(
        json.dumps({"task_id": "t<::>1", "predictions": ["not an object"]}) + "\n", encoding="utf-8"
    )

    assert gen.already_done(out) == set()


def test_kept_lines_are_preserved_byte_for_byte_including_their_endings(tmp_path) -> None:
    """⚠️ Asserted on BYTES, because `read_text` normalises endings and cannot see this at all.

    `main` appends through a text handle, so on Windows the rows it writes carry CRLF. Reading them
    back in universal-newline mode strips the CR before the line is kept, so the prune silently
    rewrote the whole existing checkpoint to LF and the run then appended CRLF: one artifact, two
    conventions, from a function whose docstring promises the kept lines are untouched. A torn line
    containing a bare CR was split in two by the same mechanism."""
    out = tmp_path / "preds.jsonl"
    keep = _row("keep<::>1", "real").replace("\n", "\r\n").encode("utf-8")
    drop = _row("redo<::>1", "").replace("\n", "\r\n").encode("utf-8")
    out.write_bytes(keep + drop)

    assert gen.drop_answerless_rows(out, {"redo<::>1"}) == 1
    assert out.read_bytes() == keep, "the surviving line must come back exactly as it went in"


def test_a_file_whose_last_line_lacks_a_newline_does_not_glue_the_next_row_onto_it(
    tmp_path,
) -> None:
    """The prune owns every byte of the file it rewrites, so it is the place to guarantee the
    terminator. Without it the next appended row lands on the end of the last kept line, turning
    two rows into one unparseable line: the kept answer and the newly paid-for one are both lost,
    and it does not self-heal, because the glued line then reads as malformed forever."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("redo<::>1", "") + _row("keep<::>1", "real").rstrip("\n"), encoding="utf-8")

    assert gen.drop_answerless_rows(out, {"redo<::>1"}) == 1

    with out.open("a", encoding="utf-8") as handle:
        handle.write(_row("new<::>1", "fresh"))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in rows] == ["keep<::>1", "new<::>1"]


def test_an_ordinary_resume_terminates_the_last_line_even_with_nothing_to_drop(
    tmp_path, monkeypatch
) -> None:
    """⚠️ The gluing is reachable on the path that repairs NOTHING, which is every ordinary resume.

    An earlier revision put the terminator guarantee inside `drop_answerless_rows`, after its
    `if not dropped: return 0`, so a checkpoint whose last line lacks a newline was only ever fixed
    when something else happened to need fixing. A run killed mid-flush leaves exactly that shape,
    and `main` then appends onto the end of the last line: two rows become one unparseable line,
    losing the answer already paid for and the one just paid for, and `check_submission_size` counts
    bytes rather than rows so the run still exits 0.
    """
    tasks = [_task(task_id="a<::>1"), _task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("a<::>1", "already paid for").rstrip("\n"))

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "the new answer")

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in rows] == ["a<::>1", "b<::>1"], (
        "the appended row must not be glued onto an unterminated last line"
    )


@pytest.mark.parametrize(
    "tail, added",
    [
        pytest.param(b"", True, id="no-terminator"),
        pytest.param(b"\r", True, id="bare-cr-is-not-a-terminator"),
        pytest.param(b"\n", False, id="already-terminated"),
        pytest.param(b"\r\n", False, id="crlf-is-terminated"),
    ],
)
def test_the_resume_terminator_accepts_only_a_newline(tmp_path, tail: bytes, added: bool) -> None:
    """`ensure_final_newline` runs on every resume, including the ones that repair nothing, so it
    is the guard that decides whether the next appended row starts its own line. A bare CR passes
    for Python's readers and fails for everything that splits on LF, so accepting it here would
    leave the glued row for exactly the tools that read the submission."""
    out = tmp_path / "preds.jsonl"
    body = _row("a<::>1", "real").rstrip("\n").encode("utf-8")
    out.write_bytes(body + tail)

    assert gen.ensure_final_newline(out) is added
    assert out.read_bytes() == body + tail + (b"\n" if added else b"")


def test_the_resume_terminator_leaves_an_absent_or_empty_file_alone(tmp_path) -> None:
    """The first run of a command has no output, and an empty one has no last line to terminate.
    Appending a newline to either would create a blank first line in the submission."""
    assert gen.ensure_final_newline(tmp_path / "absent.jsonl") is False
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")

    assert gen.ensure_final_newline(empty) is False
    assert empty.read_bytes() == b""


def test_a_last_line_ending_in_a_bare_cr_still_gets_a_newline(tmp_path) -> None:
    """A lone CR terminates a line for Python's own readers, which is why nothing noticed, but not
    for anything that splits on LF: the official `format_checker.py`, `jq`, `pandas`. Accepting it
    as a terminator leaves the next appended row on the same line for every one of those."""
    out = tmp_path / "preds.jsonl"
    out.write_bytes(_row("keep<::>1", "real").rstrip("\n").encode("utf-8") + b"\r"
                    + _row("redo<::>1", "").encode("utf-8"))

    assert gen.drop_answerless_rows(out, {"redo<::>1"}) == 1
    assert out.read_bytes().endswith(b"\n")
    assert len([b for b in out.read_bytes().split(b"\n") if b.strip()]) == 1


def test_a_checkpoint_torn_mid_character_does_not_kill_the_resume(tmp_path, monkeypatch) -> None:
    """⚠️ `main` writes with `ensure_ascii=False`, so every non-ASCII answer is raw multi-byte UTF-8
    on disk and a kill during the per-row flush can split one. Decoding strictly raised
    `UnicodeDecodeError`, which is a `ValueError` and so slips past the `OSError` handler: an
    842-task resume died with a raw traceback before a single task was attempted, from the very
    shape `already_done` documents as tolerated."""
    out = tmp_path / "preds.jsonl"
    whole = json.dumps(
        {"task_id": "a<::>1", "predictions": [{"text": "Café Busch"}]}, ensure_ascii=False
    ).encode("utf-8")
    torn = json.dumps(
        {"task_id": "b<::>1", "predictions": [{"text": "Café"}]}, ensure_ascii=False
    ).encode("utf-8")
    out.write_bytes(whole + b"\n" + torn[: torn.index("é".encode("utf-8")) + 1])

    assert gen.already_done(out) == {"a<::>1"}, "the intact row must still be readable"
    # Only that the READ survives it. A byte-level claim here could not fail, because a return of
    # 0 means nothing was written at all; that claim belongs to the rewrite arm below.
    assert gen.drop_answerless_rows(out, {"a<::>1", "b<::>1"}) == 0


def test_undecodable_bytes_survive_a_rewrite_that_actually_happens(tmp_path) -> None:
    """The arm above proves the torn line is READ without dying; this one proves it is WRITTEN back
    unchanged, which is a separate claim and the one that needs `surrogateescape` on the write. It
    only bites when something else in the file is dropped, since nothing is rewritten otherwise:
    without the matching handler the write raises `UnicodeEncodeError` on the surrogates the read
    produced, so a repair would die on a file it had already decided to change."""
    out = tmp_path / "preds.jsonl"
    torn = json.dumps(
        {"task_id": "c<::>1", "predictions": [{"text": "Café"}]}, ensure_ascii=False
    ).encode("utf-8")
    torn = torn[: torn.index("é".encode("utf-8")) + 1]
    out.write_bytes(_row("redo<::>1", "").encode("utf-8") + torn)

    assert gen.drop_answerless_rows(out, {"redo<::>1"}) == 1
    assert out.read_bytes() == torn + b"\n", (
        "the undecodable tail must come back byte for byte, with only its terminator added"
    )


def test_an_unexpected_failure_in_the_repair_is_not_reported_as_a_clean_refusal(
    tmp_path, monkeypatch
) -> None:
    """⚠️ Pins the NARROWNESS of the handler, not just the path. Widened to `Exception` it would
    report a `NameError` or an `AttributeError` inside the repair as "the predictions file is
    unchanged and no task was attempted", which is a claim it cannot make about a fault it does not
    understand. Those must surface."""
    tasks = [_task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("b<::>1", ""))

    def broken(*a, **k):
        raise ValueError("a defect in the repair itself, not a filesystem or a second run")

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "drop_answerless_rows", broken)

    with pytest.raises(ValueError):
        gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])


def test_an_answer_that_cannot_be_encoded_fails_its_task_and_not_the_run(
    tmp_path, monkeypatch
) -> None:
    """⚠️ Introduced by making the READERS surrogate-tolerant while the writer stayed strict.

    The append handle encodes with the default strict handler, and the write sits OUTSIDE the
    per-task quarantine, so one unencodable answer raised `UnicodeEncodeError` out of `main` and
    abandoned every remaining task. It is not only a model behaviour: `json.loads` accepts an
    escaped lone surrogate, and both `load_generation_tasks` and `load_recall_contexts` parse
    third-party files whose fields `submission_row` copies straight through, so such a row would
    kill every resume of that artifact forever.

    Writing the row is part of doing the task, so failing to write it is a task failure."""
    tasks = [_task(task_id="a<::>1"), _task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    # Counted rather than matched on the task: `build_messages` deliberately never puts the
    # `task_id` in the prompt, so there is nothing in `messages` to key off.
    def answers(*a, **k):
        answers.n += 1
        return "a good answer" if answers.n == 1 else "bad \ud800 answer"

    answers.n = 0
    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", answers)

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rc == 1, "the submission is missing a task, so it is not whole"
    assert [r["task_id"] for r in rows] == ["a<::>1"], "the other task must still be written"
    failures = out.with_suffix(out.suffix + ".failed.jsonl")
    assert "b<::>1" in failures.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "body, expected, why",
    [
        pytest.param(lambda r: r("a<::>1", "x") + r("b<::>1", "y"), 0, "clean", id="clean"),
        pytest.param(
            lambda r: (r("a<::>1", "x") + r("b<::>1", "y")).replace("\n", "\r\n"), 0,
            "CRLF is two rows to every reader", id="crlf",
        ),
        pytest.param(
            lambda r: r("a<::>1", "x").rstrip("\n") + "\r" + r("b<::>1", "y"), 1,
            "⚠️ glued by a bare CR: ONE line to anything that splits on LF, which is the official "
            "format_checker, jq and pandas, and two to Python's universal-newline reader",
            id="glued-by-a-bare-cr",
        ),
        pytest.param(lambda r: r("a<::>1", "x") + "{ half a row", 1, "the torn tail", id="torn"),
        pytest.param(lambda r: r("a<::>1", "x") + "\n\n", 0, "blank lines are not rows", id="blank"),
        pytest.param(
            lambda r: r("a<::>1", "x") + "{ half\n" + r("b<::>1", "y") + "{ other half", 2,
            "the count is how many lines the operator has to delete, so it has to ACCUMULATE: two "
            "kills at different points is what a flaky disk produces, and reporting 1 would stop "
            "them after the first",
            id="two-fragments",
        ),
    ],
)
def test_the_fragment_count_sees_the_lines_the_official_checker_sees(
    tmp_path, body, expected: int, why: str
) -> None:
    """⚠️ This counter is the only thing standing between a corrupt artifact and a 0 exit code, so
    it has to split the file the way the checker does. Reading it with Python's universal-newline
    rules made a CR-glued pair read back as two clean rows, which is precisely the shape that is
    unparseable to the tools that matter, so the run reported success over it.

    `ensure_final_newline` already refuses a bare CR as a terminator for this reason, and a mid-file
    one is permanent: that guard only fixes the end of the file and the prune keeps lines verbatim.
    """
    out = tmp_path / "preds.jsonl"
    out.write_bytes(body(_row).encode("utf-8"))

    assert gen.unparsable_rows(out) == expected, why


def test_the_fragment_count_survives_a_checkpoint_torn_mid_character(tmp_path) -> None:
    """It runs at the END of `main`, after the whole run has been paid for, so a strict decode here
    would throw the fatal traceback of round two in the most expensive place there is."""
    out = tmp_path / "preds.jsonl"
    torn = json.dumps(
        {"task_id": "b<::>1", "predictions": [{"text": "Café"}]}, ensure_ascii=False
    ).encode("utf-8")
    out.write_bytes(_row("a<::>1", "real").encode("utf-8")
                    + torn[: torn.index("é".encode("utf-8")) + 1])

    assert gen.unparsable_rows(out) == 1


def test_each_answer_reaches_the_disk_before_the_next_one_is_paid_for(tmp_path, monkeypatch) -> None:
    """The checkpoint contract, observed from the only vantage point that can see it: INSIDE the
    run. Buffered, the ~8 KiB wrapper holds several rows, so a kill loses every answer still in it,
    all of them already billed, and no assertion made after `main` returns can tell the two apart
    because closing the handle flushes anyway.

    So the stub reads the output as it is called: by the time task two is requested, task one's
    answer must already be on disk."""
    tasks = [_task(task_id="a<::>1"), _task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    seen_on_disk: list[int] = []

    def answer_and_peek(*a, **k):
        seen_on_disk.append(len(out.read_text(encoding="utf-8").splitlines()))
        return "an answer"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", answer_and_peek)

    gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert seen_on_disk == [0, 1], (
        "the previous answer must be durable before the next call is billed, not held in a buffer "
        f"until the file is closed: saw {seen_on_disk}"
    )


def test_a_write_failure_is_not_filed_as_a_task_that_had_no_answer(tmp_path, monkeypatch) -> None:
    """⚠️ The other side of the quarantine line, and the reason the ENCODE is inside it while the
    WRITE is not.

    A failing flush does not mean the row was lost: `BufferedWriter` keeps the unwritten remainder
    and emits it, in order, on the next successful flush. So quarantining an `OSError` would file
    the task in `.failed.jsonl`, leave it out of `written`, and report "N task(s) have no answer",
    while its complete and already paid-for row sits in the submission. Three artifacts lying about
    the same row.

    A full disk is also not a property of one task, so there is nothing for the next task to do
    differently. It stops the run."""
    tasks = [_task(task_id="a<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    real_open = gen.Path.open

    def flaky(self, *a, **k):
        handle = real_open(self, *a, **k)
        if self == out and a and a[0] == "a":
            handle.flush = lambda: (_ for _ in ()).throw(OSError(28, "no space left on device"))
        return handle

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")
    monkeypatch.setattr(gen.Path, "open", flaky)

    with pytest.raises(OSError):
        gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    monkeypatch.undo()
    failures = out.with_suffix(out.suffix + ".failed.jsonl")
    assert not failures.exists() or "a<::>1" not in failures.read_text(encoding="utf-8"), (
        "the task must not be recorded as having no answer when the failure was the disk"
    )


def test_a_checkpoint_holding_an_unparseable_fragment_does_not_report_success(
    tmp_path, monkeypatch, capsys
) -> None:
    """⚠️ Gluing is fixed; the false green it caused is not. A fragment left by a kill mid-write is
    kept verbatim by the prune, by design, so it survives every later resume, and nothing in the
    module parses the file it wrote: `check_submission_size` counts bytes. The run therefore
    reported success over a file the official checker rejects, which is the same contract this
    whole change set exists to hold, stated at the end of `main`: exit code 0 has to mean the
    submission is whole."""
    tasks = [_task(task_id="a<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, '{"task_id": "b<::>1", "input": [], "contexts": [], "predi')

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    reported = [e for e in events if e.get("event") == "unparsable_rows"]
    assert rc == 1, "a submission carrying an unparseable line is not submittable"
    assert reported and reported[0]["count"] == 1, (
        "the exit code says something is wrong; only this event says WHAT, and what to do"
    )
    assert not [e for e in events if e.get("event") == "incomplete"], (
        "no task failed, so reporting '0 of 1 task(s) have no answer' would send the operator "
        "looking for a generation failure that did not happen"
    )
    assert '"predi' in out.read_text(encoding="utf-8"), (
        "the fragment is REPORTED, not deleted: whatever partial row it holds is the operator's "
        "to discard, and silently dropping it is the failure mode this whole file is about"
    )


def test_a_row_appended_during_the_write_is_not_swapped_away(tmp_path, monkeypatch) -> None:
    """The size check closed the read window and left the write window open, and the write window
    is the slow half: it contains the whole file write and the fsync. A row appended there was
    still destroyed by the swap, silently."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("redo<::>1", "") + _row("keep<::>1", "real"), encoding="utf-8")
    real_fsync = gen.os.fsync

    def append_during_the_write(fd):
        if not getattr(append_during_the_write, "done", False):
            append_during_the_write.done = True
            with out.open("a", encoding="utf-8") as handle:
                handle.write(_row("paid<::>1", "an answer a second run just paid for"))
        return real_fsync(fd)

    monkeypatch.setattr(gen.os, "fsync", append_during_the_write)

    with pytest.raises(gen.CheckpointChanged):
        gen.drop_answerless_rows(out, {"redo<::>1"})

    monkeypatch.undo()
    assert "paid<::>1" in out.read_text(encoding="utf-8")


def test_a_row_with_no_task_id_is_never_matched_by_a_scope_entry(tmp_path) -> None:
    """`str(row.get("task_id"))` renders a missing key as the string "None", so a task whose id is
    null, which `load_generation_tasks` permits since it only checks the key is present, put "None"
    in the scope set and made every untagged answerless row in the file a match."""
    out = tmp_path / "preds.jsonl"
    orphan = json.dumps({"predictions": [{"text": ""}], "note": "belongs to no task"}) + "\n"
    out.write_text(orphan, encoding="utf-8")

    assert gen.drop_answerless_rows(out, {str(None)}) == 0
    assert out.read_text(encoding="utf-8") == orphan


def test_the_replacement_is_atomic_rather_than_a_write_over_the_checkpoint(tmp_path, monkeypatch) -> None:
    """⚠️ The docstring claims atomicity and nothing held it to that. An in-place `write_text` over
    the output passes every other arm in this file while turning any interruption into a truncated
    checkpoint: this is the run's only record of answers that cost real money.

    Pinned by making the rename fail. The original must still be intact afterwards, which is only
    true if the new body was staged somewhere else first."""
    out = tmp_path / "preds.jsonl"
    original = _row("keep<::>1", "real") + _row("redo<::>1", "")
    out.write_text(original, encoding="utf-8")

    def refuse(*a, **k):
        raise PermissionError("another process holds the destination open")

    monkeypatch.setattr(gen.os, "replace", refuse)

    with pytest.raises(OSError):
        gen.drop_answerless_rows(out, {"redo<::>1"})

    assert out.read_text(encoding="utf-8") == original, "the checkpoint must survive a failed swap"
    assert list(tmp_path.glob("*.pruned*")) == [], "a failed repair must not leave its scratch file"


def test_the_scratch_file_cannot_collide_with_a_sibling_or_another_run(tmp_path, monkeypatch) -> None:
    """The staging path is the only thing keeping the prune from writing over something else.
    `.failed.jsonl` is a real sibling `main` maintains, and two runs against one `--out` would
    otherwise stage into the same name and promote each other's partial bodies."""
    seen: list[str] = []
    real = gen.os.replace

    def record(src, dst):
        seen.append(str(src))
        return real(src, dst)

    monkeypatch.setattr(gen.os, "replace", record)
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("redo<::>1", ""), encoding="utf-8")

    gen.drop_answerless_rows(out, {"redo<::>1"})

    assert len(seen) == 1
    staged = seen[0]
    assert str(gen.os.getpid()) in staged, "two concurrent runs must not stage into one path"
    # Named explicitly rather than by suffix: a PID-suffixed name cannot END with either sibling's
    # suffix, so an `endswith` pair here could never fail and would only look like a check.
    siblings = {str(out.with_suffix(out.suffix + s)) for s in (".failed.jsonl", ".manifest.json")}
    assert staged not in siblings, "staging must not write over a sibling the run maintains"


def test_a_file_that_grew_while_being_read_is_not_rewritten_over(tmp_path, monkeypatch) -> None:
    """Read, modify, write, with no lock anywhere in `benchmarks/`. A row appended between the read
    and the swap is destroyed by the swap, and it is a row a second run has already PAID for. Two
    runs sharing an `--out` is operator error, but the cost of it here is silently losing answers,
    so the prune refuses rather than races.

    The concurrent appender is simulated at the only seam that sits inside the read loop: growth
    that begins after the first line has been parsed is exactly the window the check has to close.
    """
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("redo<::>1", "") + _row("keep<::>1", "real"), encoding="utf-8")
    real_loads = gen.json.loads

    def append_a_row_mid_read(*a, **k):
        if not getattr(append_a_row_mid_read, "done", False):
            append_a_row_mid_read.done = True
            with out.open("a", encoding="utf-8") as handle:
                handle.write(_row("paid<::>1", "an answer a second run just paid for"))
        return real_loads(*a, **k)

    staged: list[int] = []
    monkeypatch.setattr(gen.json, "loads", append_a_row_mid_read)
    monkeypatch.setattr(gen.os, "fsync", lambda fd: staged.append(fd))

    with pytest.raises(RuntimeError):
        gen.drop_answerless_rows(out, {"redo<::>1"})

    monkeypatch.undo()
    assert "paid<::>1" in out.read_text(encoding="utf-8"), "the concurrent row must survive"
    assert "keep<::>1" in out.read_text(encoding="utf-8")
    # ⚠️ The READ-window check is what this arm pins, and only this assertion can tell it apart
    # from the second check before the swap, which catches the same growth a whole file write
    # later. Both refuse, so the outcome is identical and the scratch file is cleaned up either
    # way; what differs is whether the body was written at all. Refusing early is the behaviour.
    assert staged == [], "the refusal must come before anything is staged, not after"


def test_an_already_duplicated_task_is_repaired_even_though_it_is_not_pending(tmp_path) -> None:
    """The duplicate this function exists to prevent could not be removed once it existed. A file
    holding both an answerless row and a real one under one id reads as done, so the id never
    reached the scope set, the twin stayed, and the run exited 0 over an artifact the official
    checker rejects: the same false green this whole change set out to kill."""
    out = tmp_path / "preds.jsonl"
    out.write_text(_row("t<::>1", "") + _row("t<::>1", "real"), encoding="utf-8")

    assert gen.drop_answerless_rows(out, set()) == 1, (
        "dropping it leaves exactly the good row, so nothing is left unreplaced"
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["predictions"][0]["text"] for r in rows] == ["real"]


# --------------------------------------------------------------------------------------------
# 3. End to end through `main`, which is where the defect was actually reachable.
# --------------------------------------------------------------------------------------------


def test_a_repair_that_cannot_be_written_stops_the_run_instead_of_tracebacking(
    tmp_path, monkeypatch
) -> None:
    """On Windows `os.replace` raises if any process holds the destination open, which a tail, an
    editor or a virus scanner will do. Letting that escape killed an 842-task resume with a raw
    traceback before a single task was attempted. The file is intact at that point, so refusing to
    append beside the stale rows is the right outcome, and it has to be reported as a failure."""
    tasks = [_task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("b<::>1", ""))

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")
    monkeypatch.setattr(gen.os, "replace", lambda *a, **k: (_ for _ in ()).throw(PermissionError("held")))

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert rc == 1, "a run that could not repair its checkpoint must not report success"
    assert out.read_text(encoding="utf-8") == _row("b<::>1", ""), "the checkpoint is untouched"


def test_a_concurrent_run_stops_this_one_by_name_rather_than_by_accident(
    tmp_path, monkeypatch, capsys
) -> None:
    """Pins the handler's NARROWNESS as well as the path. Widening the except to `Exception` left
    every arm green while turning a `NameError` inside the repair into "the output is unchanged",
    and dropping `CheckpointChanged` from the tuple left the concurrent-run case tracebacking."""
    tasks = [_task(task_id="b<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    _resumable(out, _row("b<::>1", ""))
    stale = out.with_suffix(out.suffix + ".failed.jsonl")
    stale.write_text('{"task_id": "from the previous run"}\n', encoding="utf-8")

    # Seamed on `fsync`, which only `drop_answerless_rows` calls. Patching `json.loads` would fire
    # during `already_done` earlier in `main`, so the growth would be inside the size the repair
    # then measures and the check could not see it: the test would pass for the wrong reason.
    real_fsync = gen.os.fsync

    def append_during_the_repair(fd):
        if not getattr(append_during_the_repair, "done", False):
            append_during_the_repair.done = True
            with out.open("a", encoding="utf-8") as handle:
                handle.write(_row("paid<::>1", "a second run's answer"))
        return real_fsync(fd)

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")
    monkeypatch.setattr(gen.os, "fsync", append_during_the_repair)

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])
    monkeypatch.undo()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    reported = [e for e in events if e.get("event") == "repair_failed"]
    assert rc == 1
    assert reported and reported[0]["error_type"] == "CheckpointChanged", (
        "the operator's fix is a separate --out, which only this type says"
    )
    assert "paid<::>1" in out.read_text(encoding="utf-8"), "the other run's answer must survive"
    assert stale.exists(), (
        "a run that refuses to touch the predictions must not have already deleted the previous "
        "run's failure log on its way to refusing"
    )


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
