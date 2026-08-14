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

    def __init__(self, choices: list[object]) -> None:
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
        return types.SimpleNamespace(choices=list(self._choices))


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
    lookalike here would drift from that and mean two answers to one question."""
    from benchmarks.llm import TRANSIENT_ERRORS

    assert NoCompletionChoices in TRANSIENT_ERRORS


def test_an_ordinary_completion_is_unaffected() -> None:
    """Guards the guard: the check sits on the success path, so an over-eager one turns every
    answered task into a failure."""
    client = _Client([_choice("  an answer  ")])

    assert generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], 128) == (
        "an answer"
    )
    assert client.calls == 1
