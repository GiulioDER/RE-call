"""The shipped answer provider: the port `reason()` has always found empty.

Properties, one test each:
  1. The provider ships OFF: no `RECALL_REASONING`, no provider.
  2. Every documented truthy and falsey spelling is honoured, case insensitively.
  3. A value that is NEITHER is refused, not read as off. A typo that silently means "off" leaves
     `reason()` abstaining forever with nothing saying why.
  4. The default model matches the benchmark's, so a parity run cannot swap models by accident.
  5. A truncated completion is NOT retried, whatever digits the ceiling happens to contain.
  6. An empty `choices` list IS retried, because that is an upstream routing fault.
  7. `strip_json_fence` removes one enclosing fence and nothing else.
  8. A backtick inside a JSON string VALUE survives fence stripping.
  9. Malformed framing passes through untouched, to fail loudly at the envelope parser.
  10. `provider_metadata()` is constructible before any call has been made.
  11. The system and user messages are passed through unaltered, in that order.
  12. `max_tokens` is always sent, because omitting it bills the reservation.
"""
from __future__ import annotations

import json

import pytest

from recall.answer_provider import (
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_MODEL,
    AnswerTruncated,
    OpenAIAnswerProvider,
    resolve_answer_provider,
    strip_json_fence,
)
from recall.embeddings import _is_transient

FENCE = "`" * 3


class _Fake:
    """A three line chat client, which is what the narrow `ChatClient` protocol buys."""

    def __init__(self, reply: str = '{"answer": "a", "citations": ["c"], '
                                    '"insufficient_evidence": false}') -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append((messages, kwargs))
        return self.reply


def test_the_provider_ships_off() -> None:
    assert resolve_answer_provider({}) is None
    # ...and an API key alone does not turn it on. Enabling generation is a decision.
    assert resolve_answer_provider({"RECALL_REASONING_API_KEY": "k"}) is None


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "   "])
def test_every_falsey_spelling_returns_none(value: str) -> None:
    assert resolve_answer_provider({"RECALL_REASONING": value}) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"])
def test_every_truthy_spelling_is_honoured(value: str) -> None:
    """Reaching the key check proves the gate opened; the key is absent, so it refuses there."""
    with pytest.raises(ValueError, match="RECALL_REASONING_API_KEY"):
        resolve_answer_provider({"RECALL_REASONING": value})


@pytest.mark.parametrize("value", ["maybe", "ture", "enabled", "2", "y", "t"])
def test_a_value_that_is_neither_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="is neither true"):
        resolve_answer_provider({"RECALL_REASONING": value})


def test_the_default_model_matches_the_benchmarks() -> None:
    """A parity arm that swapped the model would report a model delta as a prompt delta.

    This is the belt to `main`'s braces: the runner passes `--model` through explicitly, and this
    pins the default the provider would otherwise fall back to.
    """
    from benchmarks.enterprise_rag import DEFAULT_MODEL

    assert DEFAULT_ANSWER_MODEL == DEFAULT_MODEL


@pytest.mark.parametrize("ceiling", ["429", "4290", "1429", "16384"])
def test_a_truncated_completion_is_never_retried(ceiling: str) -> None:
    """The marker, not the message, decides.

    `_is_transient` substring-matches rendered text as a last resort, so a ceiling spelled `4290`
    contains "429" and would be retried three times at full prompt price for a failure the same
    ceiling guarantees will repeat.
    """
    assert not _is_transient(AnswerTruncated(f"hit its completion ceiling of {ceiling}"))


def test_an_empty_choices_response_is_retried() -> None:
    """The opposite direction: an upstream routing fault is worth resending."""
    assert _is_transient(ConnectionError("openai/gpt-4o returned no choices"))


def test_one_enclosing_fence_is_removed() -> None:
    payload = '{"answer": "a", "citations": [], "insufficient_evidence": true}'
    assert strip_json_fence(f"{FENCE}json\n{payload}\n{FENCE}") == payload
    assert strip_json_fence(f"{FENCE}\n{payload}\n{FENCE}") == payload
    assert strip_json_fence(f"{FENCE}json\r\n{payload}\r\n{FENCE}") == payload
    assert strip_json_fence(payload) == payload


def test_a_backtick_inside_a_string_value_survives() -> None:
    """The `\\Z` anchor forces the body to backtrack to the REAL closing fence.

    A de-framer that stopped at the first interior fence would truncate the JSON and turn a
    correct answer into a parse error.
    """
    payload = json.dumps(
        {"answer": f"use {FENCE}code{FENCE} here", "citations": ["c"],
         "insufficient_evidence": False}
    )
    assert json.loads(strip_json_fence(f"{FENCE}json\n{payload}\n{FENCE}")) == json.loads(payload)


@pytest.mark.parametrize(
    "text",
    [
        "prose before\n" + FENCE + "json\n{}\n" + FENCE,
        FENCE + "json\n{}",
        FENCE + "json\n{}\n" + FENCE + "\n" + FENCE + "json\n{}\n" + FENCE,
    ],
    ids=["prose-first", "unterminated", "two-blocks"],
)
def test_malformed_framing_passes_through_untouched(text: str) -> None:
    """Declined rather than guessed at, so it fails loudly at `parse_answer_envelope`."""
    assert strip_json_fence(text) == text


def test_provider_metadata_before_any_call() -> None:
    meta = OpenAIAnswerProvider(_Fake()).provider_metadata()
    assert meta.model_id == DEFAULT_ANSWER_MODEL
    assert meta.latency_ms is None


def test_the_messages_are_passed_through_unaltered() -> None:
    fake = _Fake()
    OpenAIAnswerProvider(fake)("SYSTEM TEXT", "USER TEXT")
    (messages, kwargs), = fake.calls
    assert messages == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    assert kwargs["max_tokens"] == DEFAULT_ANSWER_MAX_TOKENS
    assert kwargs["temperature"] == 0.0


def test_json_mode_is_off_by_default() -> None:
    """It cannot be on: OpenAI 400s unless the word "json" appears in the messages, and
    `SYSTEM_PROMPT` never says it. Found by a smoke run, which is what smoke runs are for.

    Nothing is lost: `parse_answer_envelope` is stricter than JSON mode anyway.
    """
    from recall.evidence import SYSTEM_PROMPT

    assert "json" not in SYSTEM_PROMPT.lower(), (
        "if the prompt ever says json, revisit request_json_object; until then it must stay off"
    )
    fake = _Fake()
    OpenAIAnswerProvider(fake)("s", "u")
    assert "response_format" not in fake.calls[-1][1]

    OpenAIAnswerProvider(fake, request_json_object=True)("s", "u")
    assert fake.calls[-1][1]["response_format"] == {"type": "json_object"}


def test_max_tokens_is_sent_and_can_be_lowered_but_not_silently_dropped() -> None:
    fake = _Fake()
    OpenAIAnswerProvider(fake, max_tokens=256)("s", "u")
    assert fake.calls[-1][1]["max_tokens"] == 256
    # Explicit None is the only way to omit it, and that is a caller's stated choice.
    OpenAIAnswerProvider(fake, max_tokens=None)("s", "u")
    assert "max_tokens" not in fake.calls[-1][1]
