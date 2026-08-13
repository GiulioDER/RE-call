"""A 200 whose body carries no answer must not become a submission row, and one that carries an
answer in an unfamiliar shape must not be mistaken for one.

Four shapes of "the call succeeded and `generate_one` still went wrong", all pre-existing and all
independent of the `finish_reason == "length"` guard in
`tests/test_mtrag_generation_truncation.py`:

  1. **`choices[0].message.content` is `None`.** What an OpenAI-compatible provider returns for
     `finish_reason == "content_filter"` and for `"tool_calls"`. `generate_one` returned
     `(None or "").strip()`, so the caller wrote `predictions: [{"text": ""}]` and reset
     `consecutive` to 0: the run counted the task as completed and the submission carried a row
     that no judge can score as anything but a failure by the system under test.
  2. **`choices` is empty**, and 3. **`choices[0]` carries no readable `message.content` field**,
     whether because the key is omitted, is null, holds something that is not an object, or holds
     an object with no `content`. OpenRouter can answer 200 with `{"choices": []}` when the
     upstream it routed to faults, and the SDK builds responses leniently rather than validating
     them, so a missing required field arrives as `None` instead of as a parse error.
     `response.choices[0].message.content` raised `IndexError` and `AttributeError` respectively,
     neither in `PERMANENT_ERROR_NAMES`, so each cost four attempts and then reported `list index
     out of range` or `'NoneType' object has no attribute 'content'`: messages naming neither the
     provider nor the fault.
  4. **`content` is a LIST of blocks rather than a string.** The odd one out, because here the
     provider sent a perfectly good answer. `(content or "")` left the list truthy and `.strip()`
     raised `AttributeError: 'list' object has no attribute 'strip'`, so the run paid four attempts
     and blamed the model for a shape the reader did not know how to open.

⚠️ **The refusals are classified differently on purpose, and the difference is the whole design.**
`CompletionTruncated` established that a cause which is a property of the REQUEST must not be
retried, because every attempt buys the same failure again. Empty content is that kind of cause: a
content filter fires on the prompt, and the prompt does not change between attempts. A malformed
response body, whether that is no `choices` or no `message`, is the opposite kind: a fault on the
provider's side of the wire that a re-route can serve correctly, which is what the retry loop is
for. The split is by CAUSE, not by symptom, which is why a `message` that arrived intact carrying
no text sits with the refusals and not with the malformations.

Being wrong costs differently in each direction, and the classification follows the cost rather
than the certainty:

  * Retrying a filter refusal is unbounded in practice. Refusals land on PARTICULAR tasks, so they
    are scattered rather than consecutive, `CONSECUTIVE_FAILURE_LIMIT` never trips, and the run
    pays 4x on each affected task for an outcome that cannot change.
  * Not retrying an upstream fault costs one task, which `already_done` lets a re-run pick up
    for the price of the calls that failed. Retrying it and being wrong is capped at 5 consecutive
    tasks by the breaker, or 20 attempts, because a malformed body is systematic rather than
    per-task.

So: no text is PERMANENT, a malformed body is TRANSIENT, and a block list is an ANSWER.

⚠️ None of them is truncation, and none may be folded into that guard. `CompletionTruncated` means
the completion hit OUR `max_tokens` ceiling and its message tells the operator to raise
`--max-tokens`. That is right advice for exactly one cause and wrong advice for these: no ceiling
produces a content filter, an empty `choices` array, or a block list.

Billed `create` calls are counted on a local stub, for the same reason the truncation file gives:
every one of these arrives as a **200**, so the OpenAI SDK's transport-level retries never fire on
them and `generate_one`'s own loop is the entire bill.
"""

from __future__ import annotations

import pytest

from benchmarks.llm import CompletionTruncated
from benchmarks.mtrag import generation as gen

#: A ceiling that cannot match anything by luck. Asserting it is ABSENT from these messages is
#: evidence the empty-answer and no-choices paths do not hand back truncation's advice.
_CEILING = 137


#: `_choice(message=...)` defaults to building a message object around `content`. `_ABSENT` omits
#: the key entirely, which is a DIFFERENT shape from setting it to `None` and the one that decides
#: whether the reader may use plain attribute access.
_BUILD = object()
_ABSENT = object()


def _choice(
    content: object = None,
    finish_reason: str | None = "stop",
    *,
    carry_reason: bool = True,
    message: object = _BUILD,
) -> object:
    """One `choices[0]` in the SDK's shape.

    `content=None` is the real wire shape for a filtered completion: the key is present and null,
    not absent. `carry_reason=False` omits `finish_reason` entirely, which is how the fakes in
    `tests/test_mtrag_generation.py` are built.

    `message` overrides the whole message object, so the malformed shapes can be built directly:
    `None` is what the SDK constructs from an upstream-error body (it builds responses leniently
    and does not validate that a required field arrived), `_ABSENT` omits the key, and any other
    value stands in for a body whose `message` is not an object at all.

    `content` is typed `object` rather than `str | None` because a gateway may send it as a LIST of
    content blocks, and reading that shape is one of the properties under test.
    """
    attrs: dict[str, object] = {}
    if message is not _ABSENT:
        attrs["message"] = type("M", (), {"content": content})() if message is _BUILD else message
    if carry_reason:
        attrs["finish_reason"] = finish_reason
    return type("C", (), attrs)()


def _text_block(text: object) -> dict[str, object]:
    """One entry of a block-list `content`, in the shape a gateway sends."""
    return {"type": "text", "text": text}


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

    # By NAME through the wrapper, not `isinstance`: `generate_one` re-raises everything as a plain
    # `RuntimeError`, so an `isinstance` check against a subclass passes no matter what was raised.
    assert CompletionTruncated.__name__ not in str(caught.value)
    assert "max-tokens" not in str(caught.value) and str(_CEILING) not in str(caught.value), (
        "the ceiling is irrelevant to a refusal; naming it sends the operator to the wrong flag"
    )


# --------------------------------------------------------------------------------------------
# 2. The response carries no completion to read: a fault on the provider's side of the wire,
#    legible and retried. Two shapes reach this, and they are classified together because the
#    CAUSE is the same one: the body is malformed, rather than the model having produced no text.
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


def test_the_malformed_response_error_is_not_classified_permanent() -> None:
    """Pins the decision itself, so flipping it is an edit to a stated choice rather than a silent
    consequence of adding a name to a set."""
    assert gen.NoCompletionInResponse.__name__ not in gen.PERMANENT_ERROR_NAMES


def test_a_choice_carrying_no_message_is_reported_as_a_provider_fault() -> None:
    """The second shape of the same fault, and the field immediately after `choices`. The SDK
    constructs responses leniently rather than validating them, so an upstream-error body reaches
    this line as a choice whose `message` is `None`. Reading `.content` off it raised
    `AttributeError: 'NoneType' object has no attribute 'content'`, which is not in
    `PERMANENT_ERROR_NAMES`: four billed attempts, reported as a Python type."""
    client = _Client([_choice(message=None, finish_reason="error")])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert "AttributeError" not in str(caught.value), (
        "the raw attribute failure must not be what surfaces"
    )
    assert "message" in str(caught.value), "the message has to say WHICH field was missing"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(_ABSENT, id="key-omitted"),
        pytest.param("the answer", id="a-bare-string-in-place-of-the-object"),
        pytest.param(type("M", (), {})(), id="an-object-with-no-content-field"),
    ],
)
def test_any_choice_without_a_readable_message_content_is_a_provider_fault(message: object) -> None:
    """The `message is None` arm above is one of four ways the field can fail to arrive, and it is
    the only one a `None` check catches. A key omitted entirely, a `message` that is a bare string,
    and an object carrying no `content` field ALL reached `message.content` and raised
    `AttributeError` for four billed attempts.

    ⚠️ This is the second time this fix stopped one field short: `choices` was hardened while
    `message` was left bare, then `message` was hardened while `.content` was left bare. Reading
    the whole path to a sentinel in one step is what makes that impossible to repeat, so these
    arms exist to keep any of the four from being reintroduced separately."""
    client = _Client([_choice(message=message, finish_reason="error")])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert "AttributeError" not in str(caught.value)
    assert gen.NoCompletionInResponse.__name__ in str(caught.value), (
        "a body the reader cannot open is a malformed RESPONSE, not a model that produced no text"
    )
    assert client.calls == gen.GENERATION_ATTEMPTS, "malformed bodies are the retried class"


def test_a_choice_carrying_no_message_is_retried() -> None:
    """⚠️ Grouped with the empty-`choices` arm and NOT with `EmptyCompletion`, which is the one
    judgement call in this file worth arguing with. A missing `message` object is a malformed
    RESPONSE, so it is a fault on the provider's side that a re-route can serve correctly. That is
    a different cause from a `message` that arrived intact carrying no text, which is the model
    having produced nothing and cannot change between attempts. Classifying by cause is what keeps
    `EmptyCompletion` meaning exactly one thing."""
    client = _Client([_choice(message=None, finish_reason="error")])

    with pytest.raises(RuntimeError):
        _generate(client)

    assert client.calls == gen.GENERATION_ATTEMPTS


# --------------------------------------------------------------------------------------------
# 3. `content` as a list of blocks: a WELL FORMED answer in a shape the reader has to know.
# --------------------------------------------------------------------------------------------


def test_content_returned_as_text_blocks_is_read_rather_than_discarded() -> None:
    """A gateway may send `content` as a list of blocks instead of a string. `(content or "")` left
    that list truthy and `.strip()` then raised `AttributeError: 'list' object has no attribute
    'strip'`, so a genuine answer became four billed attempts and a failed task. Worse than the
    cost: the run blames the model for a shape the reader did not know how to open.

    `recall/truth_extraction/_openai_engine.py` already reads this shape, with the note that
    discarding it "would refuse a WELL FORMED reply and blame the model for it"."""
    client = _Client([_choice([_text_block("The Cardinals play at "), _text_block("Busch.")])])

    assert _generate(client) == "The Cardinals play at Busch."
    assert client.calls == 1


def test_a_block_whose_text_is_not_a_string_contributes_nothing_instead_of_crashing() -> None:
    """⚠️ A DELIBERATE divergence from the sibling reader in
    `_text_of` in `recall/truth_extraction/_openai_engine.py`, whose dict branch is
    `block.get("text", "")`
    with no fallback for a key that is PRESENT and null. `"".join` over a `None` raises `TypeError`
    there.

    The list is arbitrary JSON from a gateway this repo does not control, so "has a `text` key" is
    not the same claim as "has text". Every non-string here has to contribute nothing rather than
    reach `"".join`: a null, a number, a block of another type, and a bare string in place of a
    block. The surrounding text still arrives, which is the point of not simply discarding the
    whole list on the first odd entry."""
    client = _Client([_choice([
        _text_block(None), _text_block(42), {"type": "image"}, "loose", _text_block("an answer"),
    ])])

    assert _generate(client) == "an answer"


def test_a_block_list_is_stripped_like_a_plain_string() -> None:
    """The two `content` shapes must produce the SAME answer, or the submission's text depends on
    which shape the gateway happened to send. Blocks are joined before stripping, so padding at the
    seam between two blocks survives on purpose and only the ends are trimmed."""
    client = _Client([_choice([_text_block("  The Cardinals play "), _text_block("at Busch.  ")])])

    assert _generate(client) == "The Cardinals play at Busch."


def test_a_block_list_of_only_whitespace_is_refused_too() -> None:
    """The list-shaped twin of the whitespace arm above, and the one an earlier revision left
    unpinned: dropping `.strip()` from the list branch alone kept every other test green while a
    whitespace-only answer went back to being written to the submission."""
    client = _Client([_choice([_text_block("  \n "), _text_block(" \t ")])])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert gen.EmptyCompletion.__name__ in str(caught.value)
    assert client.calls == 1


def test_a_block_list_carrying_no_text_at_all_is_refused_as_empty() -> None:
    """A list that yields nothing is the same outcome as `content=None`: no answer to score. It
    must land on the permanent refusal rather than being written as `""`, and it must not be
    mistaken for the malformed-response fault, because the response here is well formed."""
    client = _Client([_choice([{"type": "image"}], "tool_calls")])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    # Asserted through the wrapper's `type(last).__name__`, not with `pytest.raises(EmptyCompletion)`:
    # `generate_one` re-raises everything as a plain `RuntimeError`, so matching on the class would
    # be a guard that cannot fail.
    assert gen.EmptyCompletion.__name__ in str(caught.value)
    assert gen.NoCompletionInResponse.__name__ not in str(caught.value)
    assert client.calls == 1


# --------------------------------------------------------------------------------------------
# 4. Guards the guards: every ordinary answer still comes back.
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
