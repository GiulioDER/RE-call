from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from benchmarks.llm import Completer, OpenRouterLLM, _usage_cost_usd


def _identity_completer(system: str, user: str) -> str:
    # a plain function satisfies the injected-LLM seam used everywhere downstream
    return "ok"


def test_completer_is_callable_alias() -> None:
    fn: Completer = _identity_completer
    assert fn("s", "u") == "ok"


def test_openrouter_llm_builds_with_defaults() -> None:
    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    assert llm.model == "openai/gpt-4o-mini"
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.temperature == 0.0


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch, content: str | None, usage: object | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Insert a fake `openai` module into sys.modules for the duration of a test, standing in for
    the real SDK (not installed / not required). Returns (construction_calls, create_calls) —
    both lists are mutated in place by the fake as `OpenAI(...)` and `.chat.completions.create(...)`
    are invoked, so a test can assert on them after calling `complete()`."""
    construction_calls: list[dict[str, Any]] = []
    create_calls: list[dict[str, Any]] = []

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            create_calls.append(kwargs)
            message = types.SimpleNamespace(content=content)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice], usage=usage, model="provider/rev")

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            construction_calls.append(kwargs)
            self.chat = _FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    return construction_calls, create_calls


def test_complete_sends_expected_request_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _construction_calls, create_calls = _install_fake_openai(monkeypatch, content="answer")

    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    result = llm.complete("you are helpful", "what is 2+2?")

    assert result == "answer"
    assert len(create_calls) == 1
    call = create_calls[0]
    assert call["model"] == "openai/gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "what is 2+2?"},
    ]


def test_complete_coalesces_none_content_to_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, content=None)

    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    result = llm.complete("s", "u")

    assert result == ""


def test_openrouter_provider_metadata_records_tokens_revision_latency_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = types.SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_cost_usd=0.003)
    _install_fake_openai(monkeypatch, content="answer", usage=usage)

    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    assert llm.complete("s", "u") == "answer"

    metadata = llm.provider_metadata()
    assert metadata.provider_id == "openrouter"
    assert metadata.model_id == "openai/gpt-4o-mini"
    assert metadata.model_revision == "provider/rev"
    assert metadata.prompt_tokens == 12
    assert metadata.completion_tokens == 8
    assert metadata.total_tokens == 20
    assert metadata.latency_ms is not None
    assert metadata.monetary_cost_usd == 0.003


def test_usage_cost_preserves_zero_usd_from_provider_cost_dict() -> None:
    usage = types.SimpleNamespace(cost={"usd": 0.0, "total_usd": 0.25})

    assert _usage_cost_usd(usage) == 0.0


def test_complete_reuses_client_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    construction_calls, create_calls = _install_fake_openai(monkeypatch, content="hi")

    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    llm.complete("s1", "u1")
    llm.complete("s2", "u2")

    # lazy-singleton: the client is built once and reused, not rebuilt per call
    assert len(construction_calls) == 1
    assert len(create_calls) == 2


class _RateLimited(Exception):
    """Stands in for the SDK's rate-limit error: `status_code` is what `_is_transient` reads."""

    status_code = 429


class _Unauthorized(Exception):
    status_code = 401


def _install_flaky_openai(
    monkeypatch: pytest.MonkeyPatch, failures: list[Exception], content: str
) -> list[str]:
    """Fake `openai` whose `create` raises each of `failures` in turn, then succeeds.

    Returns the list of prompts that actually reached the wire, so a test can count real attempts.
    Nothing here touches the network: the SDK module itself is the fake.
    """
    sent: list[str] = []
    pending = list(failures)

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            sent.append(kwargs["messages"][1]["content"])
            if pending:
                raise pending.pop(0)
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return sent


def test_complete_retries_a_rate_limit_instead_of_killing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One 429 mid-run must not propagate: it would abort `main` before the `.json` is written.

    A full run makes thousands of paid calls back to back, which is precisely the traffic a
    provider throttles. Without a retry, question 900 of 1,986 raising 429 costs the whole
    aggregate — only the incremental sidecar survives. `sleep` is injected so the backoff is
    exercised with no wall-clock cost and no network.
    """
    slept: list[float] = []
    sent = _install_flaky_openai(monkeypatch, [_RateLimited("429 too many requests")], "answer")

    llm = OpenRouterLLM(
        model="openai/gpt-4o-mini", api_key="sk-test", sleep=slept.append, max_attempts=3
    )
    assert llm.complete("s", "u") == "answer"

    assert sent == ["u", "u"]  # one failed attempt, then the retry that succeeded
    assert len(slept) == 1  # and it really did back off between them


def test_complete_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry is bounded: a provider that is down must fail the run, not hammer it forever."""
    slept: list[float] = []
    failures: list[Exception] = [_RateLimited("429") for _ in range(5)]
    sent = _install_flaky_openai(monkeypatch, failures, "answer")

    llm = OpenRouterLLM(
        model="openai/gpt-4o-mini", api_key="sk-test", sleep=slept.append, max_attempts=3
    )
    with pytest.raises(_RateLimited):
        llm.complete("s", "u")

    assert len(sent) == 3  # exactly max_attempts, no more
    assert len(slept) == 2


def test_complete_does_not_retry_a_non_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 is not going to get better. Retrying it burns time and hides the real cause."""
    slept: list[float] = []
    sent = _install_flaky_openai(monkeypatch, [_Unauthorized("401 unauthorized")], "answer")

    llm = OpenRouterLLM(
        model="openai/gpt-4o-mini", api_key="sk-bad", sleep=slept.append, max_attempts=3
    )
    with pytest.raises(_Unauthorized):
        llm.complete("s", "u")

    assert len(sent) == 1
    assert slept == []
