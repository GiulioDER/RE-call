"""A 200 whose body carries no answer must not become a submission row.

Two shapes of "the call succeeded and there is still nothing to score", both pre-existing and both
independent of the `finish_reason == "length"` guard in
`tests/test_mtrag_generation_truncation.py`:

  1. **`choices[0].message.content` is `None`.** What an OpenAI-compatible provider returns for
     `finish_reason == "content_filter"` and for `"tool_calls"`. `generate_one` returned
     `(None or "").strip()`, so the caller wrote `predictions: [{"text": ""}]` and reset
     `consecutive` to 0: the run counted the task as completed and the submission carried a row
     that no judge can score as anything but a failure by the system under test.
  2. **`choices` is empty.** OpenRouter can answer 200 with `{"choices": []}` when the upstream it
     routed to faults. `response.choices[0]` raised `IndexError`, which is not in
     `PERMANENT_ERROR_NAMES`, so the task paid four attempts and then reported
     `IndexError: list index out of range`, which names neither the provider nor the fault.

⚠️ **The two are classified differently on purpose, and the difference is the whole design.**
`CompletionTruncated` established that a cause which is a property of the REQUEST must not be
retried, because every attempt buys the same failure again. Empty content is that kind of cause: a
content filter fires on the prompt, and the prompt does not change between attempts. An empty
`choices` list is the opposite kind, a fault on the provider's side of the wire that a re-route can
serve correctly, which is exactly what the retry loop is for.

Being wrong costs differently in each direction, and the classification follows the cost rather
than the certainty:

  * Retrying a filter refusal is unbounded in practice. Refusals land on PARTICULAR tasks, so they
    are scattered rather than consecutive, `CONSECUTIVE_FAILURE_LIMIT` never trips, and the run
    pays 4x on each affected task for an outcome that cannot change.
  * Not retrying an upstream fault costs one task, which `load_done_task_ids` lets a re-run pick up
    for the price of the calls that failed. Retrying it and being wrong is capped at 5 consecutive
    tasks by the breaker, or 20 attempts.

So: empty content is PERMANENT, no choices is TRANSIENT.

⚠️ Neither is truncation, and neither may be folded into that guard. `CompletionTruncated` means
the completion hit OUR `max_tokens` ceiling and its message tells the operator to raise
`--max-tokens`. That is right advice for exactly one cause and wrong advice for these two: no
ceiling produces a content filter, and no ceiling produces an empty `choices` array.

Billed `create` calls are counted on a local stub, for the same reason the truncation file gives:
both faults arrive as a **200**, so the OpenAI SDK's transport-level retries never fire on them and
`generate_one`'s own loop is the entire bill.
"""

from __future__ import annotations

import pytest

from benchmarks.llm import CompletionTruncated
from benchmarks.mtrag import generation as gen

#: A ceiling that cannot match anything by luck. Asserting it is ABSENT from these messages is
#: evidence the empty-answer and no-choices paths do not hand back truncation's advice.
_CEILING = 137


def _choice(
    content: str | None, finish_reason: str | None = "stop", *, carry_reason: bool = True
) -> object:
    """One `choices[0]` in the SDK's shape.

    `content=None` is the real wire shape for a filtered completion: the key is present and null,
    not absent. `carry_reason=False` omits `finish_reason` entirely, which is how the fakes in
    `tests/test_mtrag_generation.py` are built.
    """
    attrs: dict[str, object] = {"message": type("M", (), {"content": content})()}
    if carry_reason:
        attrs["finish_reason"] = finish_reason
    return type("C", (), attrs)()


class _Client:
    """An OpenAI-shaped client that counts billed calls and replays a scripted response.

    `raises` takes precedence over `choices`, so one stub serves the response arms and the liveness
    arm. `choices` is the whole list rather than one choice, because an EMPTY list is one of the two
    faults under test here.
    """

    def __init__(
        self, choices: list[object] | None = None, raises: Exception | None = None
    ) -> None:
        self.calls = 0
        self._choices = choices if choices is not None else []
        self._raises = raises
        self.chat = self

    @property
    def completions(self) -> "_Client":
        return self

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return type("R", (), {"choices": list(self._choices)})()


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """The retry arms run at test speed. `GENERATION_BACKOFF_S` is 2.0 with exponential growth, so
    four attempts would otherwise sleep 14 seconds to prove a count."""
    monkeypatch.setattr(gen.time, "sleep", lambda *_: None)


def _generate(client: _Client) -> str:
    return gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING)


# --------------------------------------------------------------------------------------------
# 1. `content` is None: nothing to score, and no attempt can change that.
# --------------------------------------------------------------------------------------------


def test_a_completion_with_no_text_is_never_returned_as_an_answer() -> None:
    """The defect, pinned. `(None or "").strip()` is `""`, and the caller writes whatever it gets:
    `predictions: [{"text": ""}]`, `consecutive = 0`, task counted as done. Nothing downstream of
    `generate_one` looks at the answer again, so this is the only place it can be caught."""
    client = _Client([_choice(None, "content_filter")])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert "content_filter" in str(caught.value), (
        "the operator has to tell a filter refusal from a tool-call stub from a provider bug, and "
        "`finish_reason` is the only field that distinguishes them"
    )


def test_a_completion_with_no_text_costs_exactly_one_request() -> None:
    """A content filter fires on the PROMPT, and the prompt is byte-identical on every attempt.
    Retrying buys the same refusal three more times at the same price."""
    client = _Client([_choice(None, "content_filter")])

    with pytest.raises(RuntimeError):
        _generate(client)

    assert client.calls == 1, "a refusal is deterministic in the request; it must not be retried"


def test_a_completion_of_only_whitespace_is_refused_too() -> None:
    """`.strip()` is applied before the answer is written, so `"  \\n "` reaches the submission as
    `""`. A guard that checked the raw string instead of the stripped one would pass this through
    and write exactly the empty prediction it exists to prevent."""
    client = _Client([_choice("  \n \t ", "stop")])

    with pytest.raises(RuntimeError):
        _generate(client)


def test_the_empty_answer_marker_is_wired_to_the_class_and_not_to_a_stale_string() -> None:
    """`PERMANENT_ERROR_NAMES` classifies by `type(exc).__name__`, so a literal entry would be a
    string coupling to a class that can be renamed out from under it: the guard would keep raising
    correctly while silently restoring the 4x retry, and the cost arm above would not catch it
    because it raises the class and follows the rename."""
    assert gen.EmptyCompletion.__name__ in gen.PERMANENT_ERROR_NAMES


def test_an_empty_answer_is_not_reported_as_truncation() -> None:
    """⚠️ The reason this is a separate error and not a wider `finish_reason` check.
    `CompletionTruncated` means the completion hit OUR ceiling, and its message says to raise
    `--max-tokens`. No value of `--max-tokens` produces a filter refusal, so folding the two
    together would hand the operator advice that cannot work, for a cause it does not name."""
    client = _Client([_choice(None, "content_filter")])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert not isinstance(caught.value, CompletionTruncated)
    assert "max-tokens" not in str(caught.value) and str(_CEILING) not in str(caught.value), (
        "the ceiling is irrelevant to a refusal; naming it sends the operator to the wrong flag"
    )


# --------------------------------------------------------------------------------------------
# 2. `choices` is empty: a fault on the provider's side, legible and retried.
# --------------------------------------------------------------------------------------------


def test_a_response_with_no_choices_names_the_fault() -> None:
    """`response.choices[0]` raised `IndexError`, and the message an operator finally saw was
    `generation gave up after 4 attempts (IndexError: list index out of range)`. That names a
    Python operation, not the provider returning an answerless 200, and it sends whoever reads it
    looking for a bug in the harness."""
    client = _Client([])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    message = str(caught.value)
    assert "IndexError" not in message, "the raw indexing failure must not be what surfaces"
    assert "choices" in message, "the message has to say WHAT the provider returned"


def test_a_response_with_no_choices_is_retried() -> None:
    """Deliberately transient, unlike the arms above. An empty `choices` on a 200 is OpenRouter
    reporting that the upstream it routed to faulted; the request is well-formed and a second
    attempt can be served by a healthy upstream. Classifying it permanent would fail a task that
    would have succeeded, to save three calls that the `CONSECUTIVE_FAILURE_LIMIT` breaker already
    caps at 5 tasks in a row.

    This arm is also the liveness proof for the two `calls == 1` assertions above: it drives the
    same stub and the same loop, and watches the count reach four."""
    client = _Client([])

    with pytest.raises(RuntimeError):
        _generate(client)

    assert client.calls == gen.GENERATION_ATTEMPTS


def test_the_no_choices_error_is_not_classified_permanent() -> None:
    """Pins the decision itself, so flipping it is an edit to a stated choice rather than a silent
    consequence of adding a name to a set."""
    assert gen.NoCompletionChoices.__name__ not in gen.PERMANENT_ERROR_NAMES


# --------------------------------------------------------------------------------------------
# 3. Guards the guards: every ordinary answer still comes back.
# --------------------------------------------------------------------------------------------


def test_an_ordinary_answer_is_still_returned_and_still_stripped() -> None:
    """Both new checks sit on the success path, so an over-eager one converts every completed task
    into a failure. This is the arm that goes red if `not answer` is ever widened."""
    client = _Client([_choice("  an answer  ", "stop")])

    assert _generate(client) == "an answer"
    assert client.calls == 1


def test_an_answer_from_a_response_carrying_no_finish_reason_is_still_returned() -> None:
    """The no-choices check must read the list defensively without making the SHAPE stricter: the
    fakes in `tests/test_mtrag_generation.py` omit `finish_reason` entirely, and so does any
    hand-rolled double."""
    client = _Client([_choice("an answer", carry_reason=False)])

    assert _generate(client) == "an answer"
    assert client.calls == 1
