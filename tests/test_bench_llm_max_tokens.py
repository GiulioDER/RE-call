"""A request must not reserve the model's maximum output, and truncation must never be silent.

Omitting `max_tokens` reserves the model maximum — 65,536 on gpt-5 — and providers check
availability against that RESERVATION rather than against what the call returns. A BEAM run died
mid-arm on `402 ... You requested up to 65536 tokens, but can only afford 64714` while its answers
were measuring ~850 completion tokens. Nothing was wrong with the balance for the work being done;
the request was refused over a ceiling it would never have approached.

The fix introduces its own hazard, which is why the second half of this file exists: a ceiling can
TRUNCATE. A truncated answer is a plausible-looking string that a judge scores as if the system
produced it — a measurement error of our own making, indistinguishable from a real failure once it
is in an artifact. So truncation raises.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from benchmarks.llm import DEFAULT_MAX_TOKENS, CompletionTruncated, OpenRouterLLM


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch, *, content: str = "ok", finish_reason: str | None = "stop"
) -> list[dict[str, Any]]:
    create_calls: list[dict[str, Any]] = []

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            create_calls.append(kwargs)
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=content), finish_reason=finish_reason
            )
            return types.SimpleNamespace(choices=[choice], usage=None)

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return create_calls


def test_a_ceiling_is_sent_by_default() -> None:
    assert OpenRouterLLM(model="m", api_key="k").max_tokens == DEFAULT_MAX_TOKENS


def test_the_default_ceiling_clears_the_longest_completion_ever_observed() -> None:
    """4,214 tokens is the longest answerer completion measured across the BEAM and pilot runs."""
    assert DEFAULT_MAX_TOKENS >= 2 * 4214


def test_the_request_carries_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_openai(monkeypatch)
    OpenRouterLLM(model="m", api_key="k").complete("s", "u")
    assert calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_an_explicit_ceiling_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_openai(monkeypatch)
    OpenRouterLLM(model="m", api_key="k", max_tokens=1234).complete("s", "u")
    assert calls[0]["max_tokens"] == 1234


def test_none_restores_the_old_behaviour_of_sending_no_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An escape hatch, so a caller that genuinely needs the model maximum can still have it."""
    calls = _install_fake_openai(monkeypatch)
    OpenRouterLLM(model="m", api_key="k", max_tokens=None).complete("s", "u")
    assert "max_tokens" not in calls[0]


def test_truncation_raises_instead_of_returning_half_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, content="the answer is cut off ri", finish_reason="length")
    with pytest.raises(CompletionTruncated, match="max_tokens"):
        OpenRouterLLM(model="m", api_key="k").complete("s", "u")


def test_a_normal_completion_is_returned_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the guard: without this the truncation check could fire on every call."""
    _install_fake_openai(monkeypatch, content="a complete answer", finish_reason="stop")
    assert OpenRouterLLM(model="m", api_key="k").complete("s", "u") == "a complete answer"


def test_a_response_without_finish_reason_is_not_treated_as_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not every provider returns the field; absence is not evidence of truncation."""
    _install_fake_openai(monkeypatch, content="fine", finish_reason=None)
    assert OpenRouterLLM(model="m", api_key="k").complete("s", "u") == "fine"


def test_truncation_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying an over-long request pays for the same failure again. The fix is a bigger ceiling."""
    calls = _install_fake_openai(monkeypatch, content="cut", finish_reason="length")
    slept: list[float] = []
    llm = OpenRouterLLM(model="m", api_key="k", max_attempts=4, sleep=slept.append)
    with pytest.raises(CompletionTruncated):
        llm.complete("s", "u")
    assert len(calls) == 1, "truncation must fail fast, not burn the retry budget"
    assert slept == []
