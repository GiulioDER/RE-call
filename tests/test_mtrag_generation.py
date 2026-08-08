"""Tasks B and C generation: the properties that cost money or credibility if they slip."""

from __future__ import annotations

import json

import pytest

from benchmarks.mtrag import generation as gen


#: The literal gold answer, which ships in every task's top-level `targets`. Distinctive on
#: purpose: an assertion on a generic string could pass by luck.
GOLD_ANSWER = "No, the Cardinals play only within the United States."


def _task(task_id: str = "conv1<::>2", answerability: str = "UNANSWERABLE") -> dict:
    """A task shaped like the REAL release, including every field that must not leak.

    Verified against `mtrag-human/generation_tasks/reference.jsonl` and `RAG.jsonl` (842 rows
    each): `Answerability`, `targets`, `Question Type` and `Multi-Turn` are TOP-LEVEL keys on the
    task, siblings of `input`; turns carry `metadata`, and **zero** turns in either file carry an
    `enrichments` key.

    An earlier version of this fixture invented `enrichments.answerability` nested inside each
    turn. That shape does not exist in the release, so the leak test guarded a path the producer
    never emits while leaving the two real ones unguarded, and it would have stayed green through
    a genuine leak. Fabricating an input the producer does not emit does not test the guard.

    `targets` is the more dangerous of the two: `Answerability` reveals whether to abstain, but
    `targets` is the answer itself.
    """
    return {
        "task_id": task_id,
        "conversation_id": task_id.split("<::>")[0],
        "Collection": "clapnq",
        "Answerability": [answerability],
        "Question Type": ["Factoid"],
        "Multi-Turn": ["Follow-up"],
        "targets": [{"text": GOLD_ANSWER}],
        "input": [
            {
                "speaker": "user",
                "text": "where do the arizona cardinals play",
                "metadata": {"author_type": "human", "created_at": "2024-01-01T00:00:00Z"},
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
    assert "UNANSWERABLE" not in blob, "the abstention label leaked into the prompt"
    assert "Answerability" not in blob
    assert GOLD_ANSWER not in blob, "the gold answer leaked into the prompt"
    assert "targets" not in blob
    assert "Question Type" not in blob
    assert "Multi-Turn" not in blob


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


@pytest.mark.parametrize(
    "field",
    ["Answerability", "targets", "Question Type", "Multi-Turn"],
    ids=["answerability", "gold_answer", "question_type", "multi_turn"],
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
