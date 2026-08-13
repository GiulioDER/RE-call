"""Tasks B and C generation: the properties that cost money or credibility if they slip."""

from __future__ import annotations

import json

import pytest

from benchmarks.llm import CompletionTruncated
from benchmarks.mtrag import generation as gen


#: The literal gold answer, which ships in every task's top-level `targets`. Distinctive on
#: purpose: an assertion on a generic string could pass by luck.
GOLD_ANSWER = "No, the Cardinals play only within the United States."

#: A half-written answer, distinctive enough that finding it in a submission file is proof rather
#: than coincidence. It reads like a real answer, which is the entire problem: a judge scores it.
CUT_OFF_ANSWER = "The Cardinals play their home games at State Farm Stad"


def _task(task_id: str = "conv1<::>2", answerability: str = "UNANSWERABLE") -> dict:
    """A task shaped like the REAL release, including every field that must not leak.

    MTRAG ships the label in **two different shapes**, and this fixture carries both, because a
    guard that only knows one is a guard for half the corpus. Counted, not assumed:

      `mtrag-human/generation_tasks/reference.jsonl` and `RAG.jsonl`, 842 rows each
        `Answerability` (capitalised) TOP-LEVEL, 842/842. `targets` TOP-LEVEL, 842/842.
        Turns carry `metadata`. `enrichments`: 0 of 6684 turns.

      `scripts/evaluation/sample_data/responses-10.jsonl`, 10 rows
        `answerability` (lower case) TOP-LEVEL, and `enrichments.answerability` nested INSIDE
        turns, 39 of them.

    The generation runs read the first pair, so that is the shape that governs our leak risk. The
    second is why the assertions are case-insensitive and why turns here carry `enrichments`: an
    earlier fixture had only the nested lower-case form and the assertions only the capitalised
    top-level name, so each covered what the other tested. Either alone stays green through a real
    leak of the other.

    `targets` is the most dangerous of all of them: `Answerability` reveals whether to abstain,
    `targets` is the answer itself.
    """
    return {
        "task_id": task_id,
        "conversation_id": task_id.split("<::>")[0],
        "Collection": "clapnq",
        "Answerability": [answerability],
        "answerability": [answerability],
        "Question Type": ["Factoid"],
        "Multi-Turn": ["Follow-up"],
        "targets": [{"text": GOLD_ANSWER}],
        "input": [
            {
                "speaker": "user",
                "text": "where do the arizona cardinals play",
                "metadata": {"author_type": "human", "created_at": "2024-01-01T00:00:00Z"},
                "enrichments": {"Question Type": ["Factoid"], "answerability": [answerability]},
            },
            {
                "speaker": "agent",
                "text": "In Glendale, Arizona.",
                "metadata": {"author_type": "human", "created_at": "2024-01-01T00:00:01Z"},
            },
            {
                "speaker": "user",
                "text": "do they play outside the US?",
                "metadata": {"author_type": "human", "created_at": "2024-01-01T00:00:02Z"},
                "enrichments": {"answerability": [answerability]},
            },
        ],
        "contexts": [
            {"document_id": f"d{i}", "score": float(20 - i), "text": f"passage {i}"}
            for i in range(20)
        ],
    }


def _assert_no_withheld_metadata(blob: str) -> None:
    """The leak checks themselves, factored out so the mutation test can exercise THESE.

    If the mutation test re-implemented the assertions it would only prove that a copy of them
    fails, which says nothing about the copy that guards the real run.
    """
    lowered = blob.lower()
    assert "unanswerable" not in lowered, "the abstention label leaked into the prompt"
    assert GOLD_ANSWER not in blob, "the gold answer leaked into the prompt"
    # Case-insensitive on the key names: the release spells it `Answerability`, the official
    # sample data spells it `answerability`, and a guard that knows one shape is a guard for half
    # the corpus. `enrichments` is checked because the sample nests the label inside turns.
    for field in ("answerability", "targets", "question type", "multi-turn", "enrichments"):
        assert field not in lowered, f"withheld field {field!r} leaked into the prompt"


def test_no_withheld_task_metadata_reaches_the_prompt() -> None:
    """⛔ The single most important property in this module.

    A correct "I don't know" on an UNANSWERABLE task scores exactly 1.0 on all three metrics at
    once, so a model that can SEE the label scores far above one that has to infer it. That would
    not be a better system, it would be a leaked answer key, and nothing downstream would show it.
    `targets` is worse still: it is the gold answer, so leaking it would score near-perfectly on
    every task, answerable or not.

    Field names below are the real ones (see `_task`), so this fails if a future edit reads
    `task["Answerability"]` or `task["targets"]` directly.
    """
    task = _task(answerability="UNANSWERABLE")
    contexts = gen.contexts_for(task, None)

    blob = json.dumps(gen.build_messages(task, contexts))

    _assert_no_withheld_metadata(blob)
    # The conversation itself must survive, or this would pass on an empty prompt.
    assert "do they play outside the US?" in blob
    assert "where do the arizona cardinals play" in blob


def test_the_leak_guard_fires_on_the_nested_lowercase_shape(monkeypatch) -> None:
    """The sample-data shape: the label nested inside a turn's `enrichments`, lower case.

    A leak here is the most plausible of all of them, because it needs no new field access at all:
    `conversation_text` reads each turn, and one careless edit that serialises the turn dict
    instead of picking `speaker`/`text` out of it leaks the label with no code that names it.
    """
    task = _task(answerability="UNANSWERABLE")

    # Exactly that careless edit: dump the whole turn rather than the two fields wanted.
    monkeypatch.setattr(
        gen, "conversation_text", lambda t: "\n".join(json.dumps(x) for x in t["input"])
    )
    blob = json.dumps(gen.build_messages(task, gen.contexts_for(task, None)))

    with pytest.raises(AssertionError):
        _assert_no_withheld_metadata(blob)


@pytest.mark.parametrize(
    "field",
    ["Answerability", "answerability", "targets", "Question Type", "Multi-Turn"],
    ids=["answerability_upper", "answerability_lower", "gold_answer", "question_type", "multi_turn"],
)
def test_the_leak_guard_fires_on_a_build_messages_that_leaks(monkeypatch, field: str) -> None:
    """The guard above only means something if it can fail. This is that proof.

    A leak test never shown to fail is a hypothesis, not a guard: it reads as protection while
    being unable to fire. So mutate `conversation_text`, the function a careless "give the model
    more context" edit would most plausibly touch, into one that appends a withheld top-level
    field, and assert the REAL checks reject the result. One case per field, so a guard that
    covers three of four still fails here.
    """
    task = _task(answerability="UNANSWERABLE")
    honest = gen.conversation_text

    monkeypatch.setattr(
        gen, "conversation_text", lambda t: f"{honest(t)}\n{field}: {t[field]}"
    )
    blob = json.dumps(gen.build_messages(task, gen.contexts_for(task, None)))

    with pytest.raises(AssertionError):
        _assert_no_withheld_metadata(blob)


def test_contexts_are_trimmed_to_the_official_maximum() -> None:
    """`format_checker.py` sets MAX_CONTEXTS = 10 and rejects more. 20 in, 10 out."""
    contexts = gen.contexts_for(_task(), None)

    assert len(contexts) == gen.MAX_CONTEXTS
    assert [c["document_id"] for c in contexts] == [f"d{i}" for i in range(10)], (
        "the trim must keep the TOP of the ranking, not an arbitrary slice"
    )


def test_recall_contexts_do_not_silently_fall_back_to_the_benchmarks_own() -> None:
    """A task RE-call never retrieved must come back empty, not borrowed.

    Falling back would report RE-call numbers for turns RE-call never saw, which is the flattering
    direction and would be invisible in the output.
    """
    task = _task(task_id="missing<::>1")

    contexts = gen.contexts_for(task, {"other<::>1": [{"document_id": "x", "score": 1.0}]})

    assert contexts == []


def test_resume_skips_task_ids_already_written(tmp_path) -> None:
    """Generation is the expensive step; a re-run must not pay for answers it already has."""
    out = tmp_path / "preds.jsonl"
    out.write_text(
        json.dumps({"task_id": "a<::>1", "predictions": [{"text": "hi"}]}) + "\n"
        + "{ truncated line that never finished\n",
        encoding="utf-8",
    )

    done = gen.already_done(out)

    assert done == {"a<::>1"}, (
        "a malformed trailing line, which is what an interrupted write leaves, must be ignored "
        "rather than fatal: that task is simply regenerated"
    )


def test_a_permanent_error_is_not_retried(monkeypatch) -> None:
    """A bad key is not a flaky network, and every attempt is billed."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)

    class AuthenticationError(Exception):
        pass

    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            raise AuthenticationError("bad key")

    client = Client()
    with pytest.raises(RuntimeError, match="bad key"):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert client.calls == 1


def test_a_transient_error_is_retried(monkeypatch) -> None:
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)

    class Reply:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient")
            return Reply("  an answer  ")

    client = Client()
    answer = gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert answer == "an answer"
    assert client.calls == 3


#: A provider that never sends the field at all, which is a different shape from one that sends
#: it as null. The guard reads `getattr(..., "finish_reason", None)`, and only an omitted
#: attribute exercises that default; a `None` VALUE reaches the comparison either way.
_ABSENT = object()


class _TruncatingClient:
    """A client whose every completion stops for `length`, counting the requests it is charged for.

    Shaped like the SDK object `generate_one` reads: `client.chat.completions.create(...)` giving
    a response with `choices[0].message.content` and `choices[0].finish_reason`. Pass `_ABSENT` to
    build a choice carrying no `finish_reason` attribute whatsoever.
    """

    def __init__(self, text: str = CUT_OFF_ANSWER, finish_reason: object = "length") -> None:
        self.calls = 0
        self.text = text
        self.finish_reason = finish_reason
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        attrs: dict[str, object] = {"message": type("M", (), {"content": self.text})()}
        if self.finish_reason is not _ABSENT:
            attrs["finish_reason"] = self.finish_reason
        return type("R", (), {"choices": [type("C", (), attrs)()]})()


def test_a_completion_cut_off_by_the_ceiling_is_refused_not_returned(monkeypatch) -> None:
    """`--max-tokens` defaults to 512 here, so truncation is an everyday outcome rather than an
    exotic one, and a truncated answer is the most dangerous shape a failure can take: it is a
    plausible string that a judge scores as if the system produced it. That is a measurement error
    of our own making, and it is indistinguishable from a genuine failure once it is in an
    artifact. `benchmarks/llm.py` has raised on this for its own path since it gained a ceiling;
    this path sent one and never looked at `finish_reason`."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)
    client = _TruncatingClient()

    with pytest.raises(CompletionTruncated, match="max_tokens"):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 512)


def test_truncation_costs_exactly_one_request(monkeypatch) -> None:
    """Every attempt is billed, and the ceiling that caused this one causes the next three too.
    The fix is a bigger `--max-tokens`, never another attempt."""
    slept: list[float] = []
    monkeypatch.setattr(gen.time, "sleep", lambda seconds: slept.append(seconds))
    client = _TruncatingClient()

    with pytest.raises(CompletionTruncated):
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 512)

    assert client.calls == 1, "a guaranteed-to-repeat failure must not burn the retry budget"
    assert slept == [], "and must not pay the backoff either"
    assert gen.GENERATION_ATTEMPTS > 1, "this test is only meaningful while retries exist"


def test_the_operator_is_told_which_knob_to_turn(monkeypatch) -> None:
    """The ceiling is this module's own `--max-tokens`, not the one in `benchmarks.llm`, so the
    message has to name the flag the operator actually has. Being told to raise the wrong constant
    is how a run gets repeated at the same ceiling.

    Driven at 128 rather than the 512 every other test here uses, which is also the argparse
    default: against 512 an assertion on the ceiling cannot tell an interpolated value from a
    hardcoded one, because the two agree everywhere the suite looks."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)
    client = _TruncatingClient()

    with pytest.raises(CompletionTruncated) as caught:
        gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    message = str(caught.value)
    assert "--max-tokens" in message
    assert "128" in message, "the ceiling that was actually sent"
    assert "512" not in message, "and not the default, which a hardcoded message would name"


def test_a_finished_answer_is_still_returned(monkeypatch) -> None:
    """Guards the guard: without this the new check could fire on every call and refuse the whole
    run."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)
    client = _TruncatingClient(text="  a complete answer  ", finish_reason="stop")

    assert gen.generate_one(client, "m", [{"role": "user", "content": "x"}], 512) == (
        "a complete answer"
    )
    assert client.calls == 1


@pytest.mark.parametrize(
    "finish_reason, shape",
    [(None, "sent as null"), (_ABSENT, "not sent at all")],
    ids=["null", "omitted"],
)
def test_a_response_without_finish_reason_is_not_treated_as_truncated(
    monkeypatch, finish_reason: object, shape: str
) -> None:
    """Not every provider returns the field, and absence is not evidence of truncation. Refusing
    on a missing attribute would fail every task on such a provider, which is a worse failure than
    the one being fixed.

    Both shapes, because they reach the guard differently: a null VALUE meets the comparison, an
    omitted ATTRIBUTE meets the `getattr` default. Parametrised after an audit showed the omitted
    case was covered only by an unrelated fixture in the retry test above, so adding a realistic
    `finish_reason` there would have quietly deleted the coverage this test is named for."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)
    client = _TruncatingClient(text="an answer", finish_reason=finish_reason)

    assert gen.generate_one(client, "m", [{"role": "user", "content": "x"}], 512) == "an answer"
    assert client.calls == 1, f"a completion whose stop reason was {shape} is not a retry either"


def test_a_truncated_task_never_reaches_the_submission(tmp_path, monkeypatch) -> None:
    """The property that actually matters, end to end: what lands in the artifact.

    Raising is only half of it. The run-level quarantine catches every per-task failure, so the
    truncated task must come out of `main` recorded as a failure and ABSENT from the submission,
    rather than written as an answer that a judge will score.
    """
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(3)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    client = _TruncatingClient()
    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: client)

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert rc == 1, "an incomplete submission must not report success through the exit code"
    written = out.read_text(encoding="utf-8").strip()
    assert written == "", "a truncated answer must not be written as if it were an answer"
    failures = [
        json.loads(line)
        for line in out.with_suffix(out.suffix + ".failed.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(failures) == 3, "each truncated task must be recorded, not silently dropped"
    assert {f["error_type"] for f in failures} == {"CompletionTruncated"}
    assert all("--max-tokens" in f["error"] for f in failures), (
        "the log is where an operator reads what went wrong, so the remedy has to survive into it"
    )
    assert client.calls == 3, "one billed request per task, not one per task per attempt"


def test_the_submission_row_carries_every_required_field() -> None:
    """`format_checker.py` requires the union across modes: rag_taskc is the strictest."""
    task = _task()
    row = gen.submission_row(task, gen.contexts_for(task, None), "an answer")

    for field in ("task_id", "Collection", "input", "contexts", "predictions"):
        assert field in row, field
    assert row["predictions"] == [{"text": "an answer"}]
    for ctx in row["contexts"]:
        assert isinstance(ctx["document_id"], str)
        assert isinstance(ctx["score"], (int, float))


def test_an_oversized_submission_is_refused(tmp_path) -> None:
    """The 20 MB limit is the official one, and RE-call's own Task A files are 127 MB."""
    big = tmp_path / "big.jsonl"
    big.write_bytes(b"x" * (gen.MAX_SUBMISSION_MB * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="above the official"):
        gen.check_submission_size(big)


def _write_tasks(path, rows) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "mangle, expected",
    [
        (lambda t: t.pop("Collection"), "must be a string"),
        (lambda t: t.update({"Collection": None}), "must be a string"),
        (lambda t: t.update({"input": []}), "no `input` turns"),
    ],
    ids=["collection_missing", "collection_null", "input_empty"],
)
def test_a_task_the_official_checker_would_reject_fails_before_anything_is_spent(
    tmp_path, mangle, expected: str
) -> None:
    """Validation belongs at load time, not at submission time.

    `submission_row` defaults `Collection` and `input` to `None`, which `format_checker.py`
    rejects. Without this gate the rejection surfaces only after ~842 answers have been paid for,
    which is the most expensive possible moment to learn the shape was wrong.
    """
    task = _task()
    mangle(task)
    path = tmp_path / "tasks.jsonl"
    _write_tasks(path, [task])

    with pytest.raises(RuntimeError, match=expected):
        gen.load_generation_tasks(path)


def _mtrag_root(tmp_path, tasks) -> "object":
    root = tmp_path / "mt-rag-benchmark"
    d = root / "mtrag-human" / "generation_tasks"
    d.mkdir(parents=True)
    _write_tasks(d / "reference.jsonl", tasks)
    return root


def test_one_unanswerable_task_does_not_take_the_other_840_down_with_it(
    tmp_path, monkeypatch
) -> None:
    """A per-task failure must be quarantined, not fatal.

    A content-policy refusal on one conversational turn is a property of that task. Aborting would
    be bad enough on its own, but resume is driven by what was WRITTEN, so the failing task is
    first in line on every retry: the run would park at the same place forever and the remaining
    tasks would never be reached.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(4)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())

    def fake(client, model, messages, max_tokens):
        if "c1<::>1" in json.dumps(messages) or fake.n == 1:
            fake.n += 1
            raise RuntimeError("moderation refused this turn")
        fake.n += 1
        return "an answer"

    fake.n = 0
    monkeypatch.setattr(gen, "generate_one", fake)

    rc = gen.main([
        "--mtrag-root", str(root), "--task", "b", "--out", str(out),
    ])

    assert rc == 1, "an incomplete submission must not report success through the exit code"
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 3, "the three answerable tasks must still be written"
    failures = out.with_suffix(out.suffix + ".failed.jsonl")
    assert failures.exists(), "the failed task must be recorded, not silently dropped"
    assert len(failures.read_text(encoding="utf-8").splitlines()) == 1
    assert all("moderation" not in json.dumps(r) for r in written), (
        "a failed task must not be written to the submission as a fabricated answer"
    )


def test_a_malformed_context_is_quarantined_like_any_other_per_task_failure(
    tmp_path, monkeypatch
) -> None:
    """The quarantine must cover ALL per-task work, not just the API call.

    This is the same defect as the test above, reached by the path the first fix did not cover.
    `contexts_for` used to run one line ABOVE the `try`, so a context row that is not a dict raised
    `AttributeError` out of `normalise_context`, sailed past `except RuntimeError`, killed the run
    and logged nothing. The guard read as "no task can end the run" while implementing "no API
    failure can end the run". `--contexts-from` reads a file we do not control, so this is reachable
    with real data, not only in theory.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(4)]
    tasks[1]["contexts"] = ["this is a string, not a context dict"]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert rc == 1
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in written] == ["c0<::>1", "c2<::>1", "c3<::>1"], (
        "the tasks AFTER the malformed one must still be attempted"
    )
    logged = [
        json.loads(line)
        for line in out.with_suffix(out.suffix + ".failed.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [r["task_id"] for r in logged] == ["c1<::>1"]
    assert logged[0]["error_type"] == "AttributeError", (
        "the exception type must be recorded, or a systematic fault reads as 'some failures'"
    )


def test_the_failures_log_describes_this_run_not_every_run(tmp_path, monkeypatch) -> None:
    """A task that failed once and later succeeded must not stay in the log forever.

    Appending across runs makes the log over-report while the `done` event under-reports, and the
    two then disagree about the same run.
    """
    tasks = [_task(task_id="c0<::>1")]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    failures = out.with_suffix(out.suffix + ".failed.jsonl")
    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())

    def fail(*a, **k):
        raise RuntimeError("transient outage")

    monkeypatch.setattr(gen, "generate_one", fail)
    gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])
    assert len(failures.read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")
    gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    # Deleted outright when the retry leaves nothing to report, rather than left as an empty file.
    assert not failures.exists() or failures.read_text(encoding="utf-8").strip() == "", (
        "the retry succeeded, so the stale failure record must be gone"
    )


def test_a_scattered_fault_that_never_trips_the_breaker_still_fails_the_run(
    tmp_path, monkeypatch
) -> None:
    """The false green the consecutive-only breaker cannot see.

    A fault that hits most tasks but never 5 in a row leaves the breaker untouched, so the run
    ends normally with a submission missing most of its answers. `check_submission_size` counts
    bytes, not completeness, so nothing downstream objects either. If the exit code were 0 here,
    every caller that gates on it, which is the normal thing to do, would read success.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(40)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())

    calls = {"n": 0}

    def every_fourth_succeeds(*a, **k):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            return "an answer"
        raise RuntimeError("scattered fault")

    monkeypatch.setattr(gen, "generate_one", every_fourth_succeeds)

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    written = out.read_text(encoding="utf-8").splitlines()
    assert len(written) == 10, "3 of every 4 tasks failed, so only 10 answers exist"
    assert rc == 1, (
        "30 of 40 tasks failed and the breaker never tripped; exit code 0 here is a false green"
    )


def test_build_messages_never_names_a_withheld_field_in_its_source() -> None:
    """A tripwire, NOT the leak defence. Do not treat a green here as sufficient.

    It text-scans the literal source of exactly two functions, so any indirection defeats it: a
    helper that reads `task["targets"]` and is called from `build_messages` passes this cleanly
    (verified). It catches the careless direct edit and nothing subtler.

    The property that actually matters is `_assert_no_withheld_metadata`, which inspects the built
    prompt itself and is driven by the mutation tests above. This exists because that check
    compares literal strings and so cannot see a TRANSFORMED leak (a truncated slice of the gold
    answer, a re-cased label); the two cover different halves and neither is sufficient alone.
    """
    import inspect

    source = inspect.getsource(gen.build_messages) + inspect.getsource(gen.conversation_text)

    for field in ("Answerability", "targets", "Question Type", "Multi-Turn"):
        assert field not in source, (
            f"{field!r} is named in the prompt-building path; it is withheld benchmark metadata "
            f"and must never be readable from there"
        )


def test_a_run_where_nothing_succeeds_stops_instead_of_billing_every_task(
    tmp_path, monkeypatch
) -> None:
    """Quarantine must not become 'burn through 842 tasks with a dead key'.

    A bad key, an unknown model or exhausted credit fails every task identically, and that is
    indistinguishable from bad luck until you count. The circuit breaker is what makes the
    per-task quarantine safe to have.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(40)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    calls = {"n": 0}

    def always_fails(*a, **k):
        calls["n"] += 1
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(gen, "generate_one", always_fails)

    with pytest.raises(RuntimeError, match="in a row"):
        gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)])

    assert calls["n"] == gen.CONSECUTIVE_FAILURE_LIMIT, (
        f"should stop after {gen.CONSECUTIVE_FAILURE_LIMIT} consecutive failures, not attempt all "
        f"{len(tasks)}"
    )


def test_a_context_score_of_zero_and_an_id_of_zero_survive_normalisation() -> None:
    """`or` would swallow both: a real id of `0` and a real score of `0.0` are not absences.

    `bool` is an `int` subclass, so a `True` score must NOT be taken as numeric either, or it
    silently becomes the score `1.0` and reorders the ranking.
    """
    assert gen.normalise_context({"document_id": 0, "score": 0.0, "text": "t"}, 0, 5) == {
        "document_id": "0", "score": 0.0, "text": "t", "title": None,
    }
    assert gen.normalise_context({"id": "d", "score": True, "text": "t"}, 1, 5)["score"] == 4.0, (
        "a boolean is not a score; it must fall through to the rank-derived value"
    )


def test_the_official_prompt_is_the_one_the_baselines_actually_got() -> None:
    """Quoted from the MTRAG paper, arXiv 2501.03468 Appendix D.2 "Model invocation".

    Not paraphrased. Comparing our number against theirs while feeding the model a different
    instruction measures our prompt, and the three differences that mattered were a missing 150
    word limit, a vaguer abstention trigger, and a different passage layout.

    ⛔ NOT the prompt in `conversations/conversations.json`: that one built the conversation
    DATASET with mixtral-8x7b and carries no length limit. Reaching for the first prompt found in
    the repo would be the same mistake one level quieter.
    """
    official = gen.PROMPTS["official"]

    assert "less than 150 words" in official, "the length limit is load-bearing for RB_alg"
    assert '"I do not have specific information"' in official, "the exact abstention string"
    assert "grounded in the provided documents" in official


def test_the_official_layout_follows_appendix_d2_ordering() -> None:
    """Instruction, then PASSAGE 1..M, then the turns. Order is part of the prompt."""
    task = _task()
    contexts = gen.contexts_for(task, None)

    msgs = gen.build_messages(task, contexts, gen.PROMPTS["official"], layout="official")
    user = msgs[1]["content"]

    assert msgs[0]["content"] == gen.PROMPTS["official"]
    assert user.startswith("PASSAGE 1\n"), "passages lead the user message, per D.2"
    assert "PASSAGE 10" in user and "PASSAGE 11" not in user, "trimmed to MAX_CONTEXTS"
    assert user.index("PASSAGE 1\n") < user.index("do they play outside the US?"), (
        "passages precede the conversation"
    )
    # The leak guard must hold for this layout too, not only the original one.
    _assert_no_withheld_metadata(json.dumps(msgs))


def test_resuming_with_a_different_prompt_is_refused(tmp_path, monkeypatch) -> None:
    """Resume dedupes on task_id alone, so a prompt change would silently mix two arms in one file.

    Nothing in a submission row records the prompt, so the mixture would be undetectable
    afterwards — and the prompt is worth 0.07 harmonic mean, which is larger than most of the
    differences the file exists to measure.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(3)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")

    rc = gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out),
                   "--prompt", "abstain"])
    assert rc == 0
    manifest = json.loads(gen.run_manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["prompt"] == "abstain", "the artifact must name its own prompt, on disk"

    with pytest.raises(RuntimeError, match="would mix two prompts"):
        gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out),
                  "--prompt", "official"])


def test_passage_numbering_has_no_gaps_when_a_context_is_empty() -> None:
    """D.2 numbers passages 1..M continuously. Filtering after enumerating leaves holes.

    `normalise_context` deliberately tolerates an empty `text`, so this is reachable with release
    data rather than hypothetical, and it changes the prompt the model actually reads.
    """
    task = _task()
    contexts = [
        {"document_id": "a", "score": 3.0, "text": "first"},
        {"document_id": "b", "score": 2.0, "text": ""},
        {"document_id": "c", "score": 1.0, "text": "third"},
    ]

    user = gen.build_messages(task, contexts, gen.PROMPTS["official"], layout="official")[1]

    assert "PASSAGE 1\nfirst" in user["content"]
    assert "PASSAGE 2\nthird" in user["content"], "numbering must close the gap, not skip to 3"
    assert "PASSAGE 3" not in user["content"]


def test_resume_refuses_when_the_existing_rows_have_UNKNOWN_provenance(tmp_path, monkeypatch) -> None:
    """The hole the first version of this guard left, and the one that mattered.

    It only fired when a manifest recorded a DIFFERENT prompt, so it could not fire for any file
    written before manifests existed — which was every artifact generated up to that point, i.e.
    exactly the population it was added to protect.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(3)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"
    # A file from before manifests: rows present, no manifest beside it.
    out.write_text(json.dumps({"task_id": "c0<::>1", "predictions": [{"text": "hi"}]}) + "\n",
                   encoding="utf-8")
    assert not gen.run_manifest_path(out).exists()

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")

    with pytest.raises(RuntimeError, match="UNKNOWN prompt"):
        gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out),
                  "--prompt", "official"])


def test_a_dry_run_does_not_overwrite_a_real_runs_manifest(tmp_path, monkeypatch) -> None:
    """A preview must not rewrite the provenance of the run it is previewing.

    `tasks` in the manifest is post-`--limit`, so a limited dry run would otherwise stamp a partial
    count onto a manifest describing a full run that actually wrote rows.
    """
    tasks = [_task(task_id=f"c{i}<::>1") for i in range(3)]
    root = _mtrag_root(tmp_path, tasks)
    out = tmp_path / "preds.jsonl"

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", lambda *a, **k: "an answer")
    gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out), "--prompt", "official"])
    before = gen.run_manifest_path(out).read_text(encoding="utf-8")

    gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out),
              "--prompt", "official", "--limit", "1", "--dry-run"])

    assert gen.run_manifest_path(out).read_text(encoding="utf-8") == before
