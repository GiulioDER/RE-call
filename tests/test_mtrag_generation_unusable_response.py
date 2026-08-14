"""`generate_one` must not index a response it has not checked.

`benchmarks/llm.py` grew both of these guards (PR #307, #314) and this path did not, which its own
comment says out loud: "the empty-`choices` guard that `benchmarks/llm.py` has is still missing
here". Two shapes reach the read, both as a **200**, so the SDK's transport retries never fire and
`generate_one`'s own four-attempt loop is the entire bill:

  1. **`choices` is empty.** OpenRouter answers this way when the upstream it routed to faults.
     `response.choices[0]` raised `IndexError: list index out of range`, which names a Python
     operation rather than the provider and sends whoever reads it hunting for a harness bug.
  2. **`message` did not arrive as an object.** The SDK builds responses leniently rather than
     validating them, so a missing required field arrives as `None` and a non-conforming one passes
     straight through. `choice.message.content` then raised `AttributeError`.

Neither type is in `PERMANENT_ERROR_NAMES`, so each cost four BILLED attempts and then reported the
Python operation that tripped over the fault instead of the fault.

`NoCompletionChoices` is imported from `benchmarks/llm.py` rather than defined again here. It is
already the repo's name for this, already documented as deliberately TRANSIENT, and already
classified that way for the other client in `TRANSIENT_ERRORS`; a second local type would be the
drift PR #314 removed for `assistant_text`.

⚠️ Transient is the deliberate half. A malformed body is a fault on the provider's side of the
wire, not a property of the request, so a re-route can serve it, which is what the retry loop is
for. That is the opposite classification from `EmptyCompletion` beside it, where a filter fires on
a prompt that is byte-identical on every attempt.
"""

from __future__ import annotations

import pytest

from benchmarks.llm import CompletionTruncated, EmptyCompletion, NoCompletionChoices
from benchmarks.mtrag import generation as gen

_CEILING = 137

#: `_BUILD` makes the ordinary message object around `content`; `_ABSENT` omits the key entirely,
#: which is a DIFFERENT shape from setting it to `None` and the one that decides whether the reader
#: may use plain attribute access.
_BUILD = object()
_ABSENT = object()


def _choice(
    content: object = "an answer", finish_reason: str = "stop", *, message: object = _BUILD
) -> object:
    attrs: dict[str, object] = {"finish_reason": finish_reason}
    if message is not _ABSENT:
        attrs["message"] = type("M", (), {"content": content})() if message is _BUILD else message
    return type("C", (), attrs)()


class _Client:
    """An OpenAI-shaped client that counts billed calls and replays a scripted response.

    `choices` is the whole LIST rather than one choice, because an empty list is one of the two
    faults under test.
    """

    def __init__(self, choices: list[object] | None = None) -> None:
        self.calls = 0
        self._choices = choices if choices is not None else []
        self.chat = self

    @property
    def completions(self) -> "_Client":
        return self

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        return type("R", (), {"choices": list(self._choices)})()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """`GENERATION_BACKOFF_S` is 2.0 with exponential growth, so four attempts would otherwise
    sleep 14 seconds to prove a count."""
    monkeypatch.setattr(gen, "GENERATION_BACKOFF_S", 0.0)


def _generate(client: _Client) -> str:
    return gen.generate_one(client, "openai/gpt-4o", [{"role": "user", "content": "x"}], _CEILING)


def test_a_response_with_no_choices_names_the_provider_and_not_an_index(tmp_path) -> None:
    """The message an operator actually saw was `generation gave up after 4 attempts (IndexError:
    list index out of range)`, which names neither the provider nor the fault."""
    client = _Client([])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert "IndexError" not in str(caught.value)
    assert NoCompletionChoices.__name__ in str(caught.value)
    assert "choices" in str(caught.value), "the message has to say WHAT the provider returned"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(None, id="null"),
        pytest.param(_ABSENT, id="key-omitted"),
        pytest.param("the answer", id="a-bare-string-in-place-of-the-object"),
        pytest.param(type("M", (), {})(), id="a-double-with-no-content-attribute"),
    ],
)
def test_a_choice_without_a_readable_message_content_is_a_provider_fault(message: object) -> None:
    """Four ways the field can fail to arrive, and a `None` check catches one of them. Verified
    against the pinned SDK: a body sending `"message": "the answer"` constructs
    `Choice(message='the answer')`, passes a `None` check, then raises `AttributeError: 'str'
    object has no attribute 'content'`. Reading the whole path to a sentinel in one step is what
    makes it impossible to harden one field at a time and stop short of the next."""
    client = _Client([_choice(message=message)])

    with pytest.raises(RuntimeError) as caught:
        _generate(client)

    assert "AttributeError" not in str(caught.value)
    assert NoCompletionChoices.__name__ in str(caught.value)
    assert "message.content" in str(caught.value), "say WHICH field was missing"


@pytest.mark.parametrize("choices", [pytest.param([], id="no-choices"),
                                     pytest.param([_choice(message=None)], id="no-message")])
def test_a_malformed_body_is_retried_because_a_re_route_can_serve_it(choices: list[object]) -> None:
    """⚠️ The classification, pinned. This is a fault on the provider's side rather than a property
    of the request, so a second attempt can land on a healthy upstream. `EmptyCompletion` beside it
    is the opposite: a filter fires on a prompt that does not change between attempts.

    Also the liveness proof for every `calls == 1` assertion elsewhere: the same stub and the same
    loop reach four here."""
    client = _Client(choices)

    with pytest.raises(RuntimeError):
        _generate(client)

    assert client.calls == gen.GENERATION_ATTEMPTS


def test_a_content_field_that_arrived_carrying_null_is_the_permanent_class() -> None:
    """⚠️ The distinction the sentinel exists for, and the reason it cannot BE `None`.

    `content: null` is the wire shape for `finish_reason == "content_filter"`: the field arrived,
    and the model produced no text. That is `EmptyCompletion`, permanent, one billed call, because
    a filter fires on a prompt that is byte-identical on every attempt. No `content` field at all is
    a malformed body, which is retried. Collapsing the sentinel to `None` reads the first as the
    second and pays four attempts for an outcome that cannot change."""
    client = _Client([_choice(None)])

    with pytest.raises(EmptyCompletion):
        _generate(client)

    assert client.calls == 1, "a refusal is deterministic in the request; it must not be retried"


@pytest.mark.parametrize(
    "message, content",
    [
        pytest.param(_BUILD, None, id="content-null"),
        pytest.param(None, None, id="message-null"),
        pytest.param(_ABSENT, None, id="message-key-omitted"),
        pytest.param("half a sen", None, id="message-is-not-an-object"),
    ],
)
def test_a_truncated_completion_is_truncation_even_when_its_text_is_unreadable(
    message: object, content: object
) -> None:
    """⚠️ This pins ORDER, and order here is a diagnosis rather than layout.

    `finish_reason == "length"` says the ceiling was hit, which is a property of OUR request and
    the most specific thing known about the response, so it wins over whatever shape the body
    turned out to have. Reading the body first sends a `length` completion with an unreadable
    `message` to `NoCompletionChoices`: four billed attempts against a ceiling no retry can move,
    printing "an upstream fault on the provider's side, not a property of the request" while
    interpolating `finish_reason=length` into the same sentence.

    `content-null` is the everyday carrier: a reasoning model that spends the whole ceiling on
    reasoning tokens returns `length` with no text at all. Routing that to `EmptyCompletion` would
    tell the operator no value of `--max-tokens` can change it, which is the opposite of true.

    This arm was dropped when the fix was ported onto master, leaving the order correct in the code
    and unheld by anything, which is how it broke in the first place."""
    client = _Client([_choice(content, "length", message=message)])

    with pytest.raises(CompletionTruncated) as caught:
        _generate(client)

    assert str(_CEILING) in str(caught.value), "the operator's fix is the flag, so name the ceiling"
    assert client.calls == 1, "no attempt count can make an over-long request fit"


def test_the_transient_classification_is_not_quietly_reversed() -> None:
    """Pins the decision itself, so flipping it is an edit to a stated choice rather than a silent
    consequence of adding a name to a set."""
    assert NoCompletionChoices.__name__ not in gen.PERMANENT_ERROR_NAMES


def test_an_ordinary_answer_is_still_returned(tmp_path) -> None:
    """Both guards sit on the success path, so an over-eager one converts every completed task
    into a failure."""
    client = _Client([_choice("  an answer  ")])

    assert _generate(client) == "an answer"
    assert client.calls == 1
