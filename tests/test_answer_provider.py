from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import recall.answer_provider as answer_provider
from recall.answer_provider import OllamaAnswerProvider, OpenRouterAnswerProvider, resolve_answer_provider


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())


def test_ollama_answer_provider_uses_json_and_disables_thinking() -> None:
    client = _Client()
    provider = OllamaAnswerProvider(client, model_id="qwen3:4b")

    assert provider("system", "user") == '{"ok": true}'
    kwargs = client.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"think": False}
    assert provider.provider_metadata().total_tokens == 14
    assert provider.provider_metadata().monetary_cost_usd == 0.0


def test_answer_provider_is_disabled_by_default() -> None:
    assert resolve_answer_provider({}) is None


def test_answer_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="must be 'ollama'.*openrouter"):
        resolve_answer_provider(
            {
                "RECALL_REASONING_ANSWER_ENABLED": "1",
                "RECALL_REASONING_ANSWER_PROVIDER": "unsupported",
            }
        )


def test_openrouter_answer_provider_uses_model_and_api_key() -> None:
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": "openrouter",
            "RECALL_REASONING_ANSWER_MODEL": "deepseek/deepseek-v4-flash",
            "OPENROUTER_API_KEY": "test-key",
        }
    )

    assert isinstance(provider, OpenRouterAnswerProvider)
    assert provider.model_id == "deepseek/deepseek-v4-flash"
    assert provider.client.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert provider.client.api_key == "test-key"


def test_openrouter_answer_provider_posts_json_and_records_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            ).encode()

    def _urlopen(req: object, *, timeout: float) -> _Response:
        seen["url"] = getattr(req, "full_url")
        seen["headers"] = dict(getattr(req, "headers"))
        seen["payload"] = json.loads(getattr(req, "data").decode())
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(answer_provider.request, "urlopen", _urlopen)
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": "openrouter",
            "RECALL_REASONING_ANSWER_MODEL": "deepseek/deepseek-v4-flash",
            "OPENROUTER_API_KEY": "test-key",
        }
    )
    assert provider is not None

    assert provider("system", "user") == '{"answer":"ok"}'
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-type": "application/json",
    }
    assert seen["payload"] == {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "none"},
    }
    assert provider.provider_metadata().total_tokens == 18


def test_native_ollama_client_sends_strict_schema_and_thinking_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "message": {
                        "content": '{"answer":"ok","citations":["c1"],"insufficient_evidence":false}'
                    },
                    "prompt_eval_count": 11,
                    "eval_count": 7,
                }
            ).encode()

    def _urlopen(req: object, *, timeout: float) -> _Response:
        seen["url"] = getattr(req, "full_url")
        seen["payload"] = json.loads(getattr(req, "data").decode())
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(answer_provider.request, "urlopen", _urlopen)
    client = answer_provider._NativeOllamaClient("http://127.0.0.1:11434/v1", timeout=12)
    response = client.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": "user"}],
        max_tokens=128,
        context_tokens=1024,
        thinking=False,
    )

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert payload["options"] == {
        "temperature": 0,
        "num_predict": 128,
        "num_ctx": 1024,
    }
    assert response.usage.total_tokens == 18


def test_ollama_answer_provider_native_path_uses_native_client_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": '{"answer":"ok"}'}, "eval_count": 1}
            ).encode()

    monkeypatch.setattr(answer_provider.request, "urlopen", lambda req, *, timeout: _Response())
    client = answer_provider._NativeOllamaClient("http://127.0.0.1:11434/v1", timeout=12)
    provider = OllamaAnswerProvider(client, model_id="qwen3:4b")

    assert provider("system", "user") == '{"answer":"ok"}'


def test_openai_compatible_provider_omits_openrouter_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": '{"answer":"ok"}'}}]}
            ).encode()

    def _urlopen(req: object, *, timeout: float) -> _Response:
        seen["payload"] = json.loads(getattr(req, "data").decode())
        return _Response()

    monkeypatch.setattr(answer_provider.request, "urlopen", _urlopen)
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": "openai",
            "RECALL_REASONING_ANSWER_MODEL": "local-model",
            "RECALL_REASONING_ANSWER_API_KEY": "test-key",
        }
    )
    assert provider is not None

    assert provider("system", "user") == '{"answer":"ok"}'
    assert "reasoning" not in seen["payload"]


def test_answer_provider_context_tokens_are_configurable() -> None:
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_MODEL": "qwen3:4b",
            "RECALL_REASONING_ANSWER_CONTEXT_TOKENS": "2048",
        }
    )

    assert provider is not None
    assert provider.context_tokens == 2048


def test_non_openrouter_providers_ignore_openrouter_effort_setting() -> None:
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": "ollama",
            "RECALL_REASONING_ANSWER_MODEL": "qwen3:4b",
            "RECALL_REASONING_ANSWER_REASONING_EFFORT": "not-valid-for-ollama",
        }
    )

    assert provider is not None
    assert provider.reasoning_effort == "none"


def _rendered_prompts() -> list[tuple[str, tuple[str, str]]]:
    """Every evidence renderer whose output reaches an answer provider, on a real bundle."""
    from datetime import datetime, timedelta, timezone

    from recall.evidence import (
        build_evidence_bundle,
        render_compact_evidence_prompt,
        render_evidence_prompt,
    )
    from recall.types import (
        Chunk,
        Provenance,
        RetrievalDiagnostics,
        StalenessReport,
        TrustedHit,
        TrustedResult,
        Validity,
    )

    jan = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hit = TrustedHit(
        chunk=Chunk(id="c1", source="source.md", text="The deploy runs on Tuesdays.", metadata={}),
        cosine=0.8,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(source="source.md", file="source.md", ord=0, indexed_at=jan),
        validity=Validity(valid_from=jan, valid_until=None, superseded_by=None),
    )
    result = TrustedResult(
        query="When does the deploy run?",
        hits=[hit],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, None, None, timedelta(days=2)),
        diagnostics=RetrievalDiagnostics("profile-v1", "fast", "g1", 20, False, {}),
        calibration_id="cal-answer-provider-fixture",
        calibration_status="certified",
        trust_state="trusted",
    )
    bundle = build_evidence_bundle(result)
    assert bundle.items, "the fixture must render real evidence, not an empty bundle"
    return [
        ("render_evidence_prompt", render_evidence_prompt(bundle)),
        ("render_compact_evidence_prompt", render_compact_evidence_prompt(bundle)),
    ]


@pytest.mark.parametrize("provider_name", ["openrouter", "openai"])
def test_a_json_object_request_carries_the_word_json_in_its_system_message(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    """OpenAI refuses ``response_format: json_object`` unless a message says "json".

    The provider's HTTP 400 body reads: "'messages' must contain the word 'json' in some form, to
    use 'response_format' of type 'json_object'." OpenAI and Azure upstreams both enforce it, so
    through OpenRouter every ``openai/*`` answer model failed on every call, while DeepSeek and
    Gemini, which do not enforce it, answered normally. That is why the only live measurements
    (all DeepSeek) never saw it. Found 2026-09-29 while measuring answer prompts.

    The test asserts on the payload the provider actually POSTs, built from the real renderers, so
    it holds at the boundary OpenAI checks rather than at the constant.

    Red proof, 2026-09-29: against ``recall.evidence.SYSTEM_PROMPT`` as it stood at ``1f8666df``,
    ending "Return only an object matching the requested answer envelope.", both parametrisations
    failed in the ``"json" in sent_system.casefold()`` assertion with "render_evidence_prompt:
    system message never says JSON" (the loop stops at the first renderer; the compact renderer
    returns the same constant). Green after the sentence became "Return only a JSON object ...".
    """
    seen: list[dict[str, object]] = []

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"answer":"ok"}'}}]}).encode()

    def _urlopen(req: object, *, timeout: float) -> _Response:
        seen.append(json.loads(getattr(req, "data").decode()))
        return _Response()

    monkeypatch.setattr(answer_provider.request, "urlopen", _urlopen)
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": provider_name,
            "RECALL_REASONING_ANSWER_MODEL": "openai/gpt-4o-mini",
            "RECALL_REASONING_ANSWER_API_KEY": "test-key",
        }
    )
    assert provider is not None

    for renderer, (system, user) in _rendered_prompts():
        seen.clear()
        provider(system, user)
        (payload,) = seen
        assert payload["response_format"] == {"type": "json_object"}
        messages = payload["messages"]
        assert isinstance(messages, list)
        sent_system = next(m["content"] for m in messages if m["role"] == "system")
        assert sent_system == system
        assert "json" in sent_system.casefold(), f"{renderer}: system message never says JSON"


def test_the_sdk_client_path_also_carries_the_word_json() -> None:
    """The third client path: an OpenAI SDK style client handed to ``OllamaAnswerProvider``.

    It sends ``response_format: json_object`` too, so the same refusal applies to it. Red proof as
    for the test above: it failed the ``"json"`` assertion on ``render_evidence_prompt`` against
    the constant at ``1f8666df``.
    """
    client = _Client()
    provider = OllamaAnswerProvider(client, model_id="gpt-4o-mini")

    for renderer, (system, user) in _rendered_prompts():
        provider(system, user)
        kwargs = client.chat.completions.kwargs
        assert kwargs is not None
        assert kwargs["response_format"] == {"type": "json_object"}
        sent_system = kwargs["messages"][0]["content"]
        assert sent_system == system
        assert "json" in sent_system.casefold(), f"{renderer}: system message never says JSON"
