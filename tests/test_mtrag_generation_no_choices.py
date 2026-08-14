"""A 200 whose `choices` array is empty must name the provider, not a Python operation.

`benchmarks/mtrag/generation.py:generate_one` indexed `response.choices[0]` blind. OpenRouter
answers 200 with an empty `choices` list when the upstream it routed to faults, so that surfaced as

    generation gave up after 4 attempts (IndexError: list index out of range)

which names a list operation rather than the provider, and sends whoever reads `.failed.jsonl`
hunting for a bug in the harness. `benchmarks/llm.py` closed the same hole in PR #307 with
`NoCompletionChoices`; mtrag was left behind because the fix for it was written on a branch that
never merged, and the comment at the read site said so rather than doing anything about it.

⚠️ **TRANSIENT, and the classification is the whole design.** This is a fault on the PROVIDER'S
side of the wire, not a property of the request, so a re-route can serve the same request
correctly. That makes it the opposite of its two neighbours here:

    CompletionTruncated   permanent   our ceiling cut it; the next three attempts cut it too
    EmptyCompletion       permanent   a filter fires on the prompt, which does not change
    NoCompletionChoices   TRANSIENT   the upstream faulted; another attempt can be routed better

So it is deliberately absent from the `except (CompletionTruncated, EmptyCompletion): raise`
tuple, and deliberately absent from `PERMANENT_ERROR_NAMES` — mtrag's loop retries anything not
named there, which is the behaviour wanted. That mirrors `TRANSIENT_ERRORS` in `benchmarks/llm.py`,
where the same exception type is classified the same way for the same reason.

Being wrong in this direction is bounded: `GENERATION_ATTEMPTS` caps it at four calls on one task,
and `CONSECUTIVE_FAILURE_LIMIT` stops the run after five such tasks in a row. Being wrong the other
way fails a task that a second attempt would have answered.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from benchmarks.llm import NoCompletionChoices
from benchmarks.mtrag.generation import (
    GENERATION_ATTEMPTS,
    PERMANENT_ERROR_NAMES,
    generate_one,
)


class _Client:
    """An OpenAI-shaped client that counts BILLED calls and replays one scripted response.

    `choices` is the whole list rather than one choice, because an EMPTY list is the fault under
    test. Counting calls locally is the only way to see the bill: this arrives as a 200, so the
    SDK's own transport retries never fire on it and `generate_one`'s loop is the entire cost.
    """

    def __init__(self, choices: object) -> None:
        self.calls = 0
        self._choices = choices

    @property
    def chat(self) -> "_Client":
        return self

    @property
    def completions(self) -> "_Client":
        return self

    def create(self, **_: Any) -> object:
        self.calls += 1
        # NOT `list(...)`: one arm scripts `choices=None`, the shape the SDK
        # surfaces when the body omits the field, and coercing it here would hide
        # exactly the case that arm exists to cover.
        return types.SimpleNamespace(choices=self._choices)


def _choice(content: str) -> object:
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=content), finish_reason="stop"
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GENERATION_BACKOFF_S` is 2.0 with exponential growth, so four attempts would otherwise
    sleep 14 seconds to prove a count."""
    import benchmarks.mtrag.generation as gen

    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)


def test_an_empty_choices_list_names_the_fault_rather_than_the_indexing() -> None:
    """The defect, pinned. `response.choices[0]` on an empty list is `IndexError: list index out
    of range`, and that is what an operator found in `.failed.jsonl`."""
    client = _Client([])

    with pytest.raises(RuntimeError) as caught:
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    message = str(caught.value)
    assert "IndexError" not in message, "the raw indexing failure must not be what surfaces"
    assert "choices" in message, "the message has to say WHAT the provider returned"


def test_an_empty_choices_list_is_retried() -> None:
    """Transient on purpose: the request is well formed and a healthy upstream can serve it.

    This is also the liveness proof for the classification test below, which asserts a NAME is
    absent from a set — an assertion that would pass just as happily if the guard did not exist.
    """
    client = _Client([])

    with pytest.raises(RuntimeError):
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert client.calls == GENERATION_ATTEMPTS


def test_the_no_choices_error_is_not_classified_permanent() -> None:
    """Pins the decision itself, so flipping it is an edit to a stated choice rather than a silent
    consequence of adding a name to a set."""
    assert NoCompletionChoices.__name__ not in PERMANENT_ERROR_NAMES


def test_the_guard_uses_the_shared_type_rather_than_a_local_one() -> None:
    """One type for this fault across the repo. `benchmarks/llm.py` raises `NoCompletionChoices`
    for the identical wire shape and classifies it TRANSIENT in `TRANSIENT_ERRORS`; a private
    lookalike here would drift from that and mean two answers to one question.

    ⛔ THE FIRST VERSION OF THIS TEST ASSERTED ONLY THE `TRANSIENT_ERRORS` MEMBERSHIP, which is a
    fact about `benchmarks/llm.py` alone and never observed what mtrag raises. Mutation-proven
    inert: defining a private `_LocalNoChoices` in `generation.py` and raising THAT left all five
    tests in this file green, because the lookalike's message also contains "choices", and a
    lookalike is equally absent from `PERMANENT_ERROR_NAMES` so the billed-call count was
    unchanged. The only backstop was ruff's unused-import warning, and `ruff check --fix` deletes
    that backstop automatically.

    So the type is now read off the exception mtrag actually raised. `generate_one` ends with
    `raise RuntimeError(...) from last`, which puts it on `__cause__`.
    """
    from benchmarks.llm import TRANSIENT_ERRORS

    assert NoCompletionChoices in TRANSIENT_ERRORS

    client = _Client([])
    with pytest.raises(RuntimeError) as caught:
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert type(caught.value.__cause__) is NoCompletionChoices, (
        "mtrag must raise the SHARED type, not a private lookalike that drifts from it"
    )


def test_an_absent_choices_field_is_named_as_such() -> None:
    """`not choices` takes the branch for `None` too, and `None` is what the SDK surfaces when the
    body omits `choices` entirely — OpenRouter's documented failure shape, so the realistic case.
    The message said "empty", asserting a shape that had not been observed."""
    client = _Client(None)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError) as caught:
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert "absent" in str(caught.value)
    assert type(caught.value.__cause__) is NoCompletionChoices


def test_an_ordinary_completion_is_unaffected() -> None:
    """Guards the guard: the check sits on the success path, so an over-eager one turns every
    answered task into a failure."""
    client = _Client([_choice("  an answer  ")])

    assert generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128) == (
        "an answer"
    )
    assert client.calls == 1


def test_the_failure_record_names_the_typed_cause_not_just_the_wrapper(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """⛔ END TO END THROUGH `main`, ON THE ARTIFACT A READER OPENS.

    `generate_one` wraps every retried failure in a `RuntimeError`, so `.failed.jsonl`'s
    `error_type` field is the word "RuntimeError" for essentially every generation failure. The
    comment at that write site says the type is recorded "so a systematic fault is legible rather
    than hidden as 'some failures'", and the wrapper defeated that for this entire class: the only
    trace of `NoCompletionChoices` was inside the free-text `error` string, where nothing can count
    or group it.

    Pinned HERE rather than on a return value, because a mutation dropping `cause_type` left every
    other arm in this file green. That is the same gap shape this session hit repeatedly: a test
    that reads what a function returned while the defect lives in the file an operator reads.
    """
    from tests.test_mtrag_generation import _mtrag_root, _task
    import benchmarks.mtrag.generation as gen

    root = _mtrag_root(tmp_path, [_task(task_id="c0<::>1")])
    out = tmp_path / "preds.jsonl"

    def _empty_choices(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("generation gave up after 4 attempts") from NoCompletionChoices(
            "the provider returned a 200 with an empty or absent `choices` list"
        )

    monkeypatch.setattr(gen, "openrouter_client", lambda *a, **k: object())
    monkeypatch.setattr(gen, "generate_one", _empty_choices)

    assert gen.main(["--mtrag-root", str(root), "--task", "b", "--out", str(out)]) == 1

    import json as _json

    logged = [
        _json.loads(line)
        for line in out.with_suffix(out.suffix + ".failed.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert logged[0]["error_type"] == "RuntimeError", "the wrapper is still recorded"
    assert logged[0]["cause_type"] == "NoCompletionChoices", (
        "and the TYPED CAUSE beside it, which is the field a systematic fault can be counted by"
    )

    # ⛔ THE OTHER HALF, which a mutation showed was unpinned: deleting `cause_type` from the
    # `task_failed` event alone left the whole suite green. The event is what an operator WATCHES
    # during a paid 842-task run; `.failed.jsonl` is only read afterwards, so the live half is the
    # one that matters first and it was the one with no test.
    events = [
        _json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{") and '"task_failed"' in line
    ]
    assert events, "the run must announce the failure as it happens, not only in the file"
    assert events[0]["cause_type"] == "NoCompletionChoices"


def test_a_choice_carrying_no_message_names_the_provider_too() -> None:
    """⛔ The guard stopped ONE FIELD SHORT. `choices` was hardened and `choice.message.content`
    still dereferenced blind, so a 200 whose first choice carries `message: null` produced

        AttributeError: 'NoneType' object has no attribute 'content'

    at four billed calls — the exact failure class this file exists to abolish, reached through
    the next field along. `assistant_text` is total, so `message` was the only crash surface left
    on this path.

    A MISSING `message` is a malformed body, so it is the provider's fault and transient. An
    `arrived-but-null` CONTENT is different: text that is not there is not a malformed body, and
    it must keep falling through to the permanent `EmptyCompletion` below. The sentinel is what
    tells those two apart; `getattr(..., None)` could not.
    """
    client = _Client([types.SimpleNamespace(message=None, finish_reason="stop")])

    with pytest.raises(RuntimeError) as caught:
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert "AttributeError" not in str(caught.value)
    assert type(caught.value.__cause__) is NoCompletionChoices


def test_a_choice_whose_content_is_null_is_still_a_permanent_empty_completion() -> None:
    """Guards the guard above. Hardening `message` must not swallow the case where the field IS
    there and carries nothing: that is an empty answer, which is permanent and costs ONE call, not
    a malformed body, which is transient and costs four."""
    from benchmarks.llm import EmptyCompletion

    client = _Client([types.SimpleNamespace(message=types.SimpleNamespace(content=None),
                                            finish_reason="stop")])

    with pytest.raises(EmptyCompletion):
        generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert client.calls == 1, "an empty answer repeats; it must not be retried"


def test_the_give_up_message_prices_the_attempts_actually_spent() -> None:
    """The message interpolated the BUDGET, not the attempts reached, so a cause that breaks early
    was priced at 4x what it cost. With `cause_type` now beside it, the record read
    `cause_type=BadRequestError` next to "after 4 attempts" for a failure that billed once: two
    accurate fields and one wrong one in the same line."""

    class _Permanent(Exception):
        pass

    _Permanent.__name__ = "BadRequestError"  # matched by NAME in PERMANENT_ERROR_NAMES

    calls = {"n": 0}

    class _Failing:
        @property
        def chat(self) -> "_Failing":
            return self

        @property
        def completions(self) -> "_Failing":
            return self

        def create(self, **_: Any) -> object:
            calls["n"] += 1
            raise _Permanent("input is not valid for this model")

    with pytest.raises(RuntimeError) as caught:
        generate_one(_Failing(), "openai/gpt-4o", [{"role": "user", "content": "x"}], 128)

    assert calls["n"] == 1, "a permanent error must break at the first attempt"
    # ⛔ The TOKEN, not a prefix. `"1 attempt" in "1 attempts"` is True, so the obvious assertion
    # was satisfied by the ungrammatical string and the pluralisation conditional could be deleted
    # with the suite green — a guard on the count that could not see the plural.
    assert "after 1 attempt " in str(caught.value), (
        f"priced the budget rather than the spend: {caught.value}"
    )
    assert "1 attempts" not in str(caught.value), "singular count, plural noun"
    assert "4 attempts" not in str(caught.value)
