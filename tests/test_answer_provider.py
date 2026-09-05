from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import recall.answer_provider as answer_provider
from recall.answer_provider import OllamaAnswerProvider, resolve_answer_provider


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
    with pytest.raises(ValueError, match="must be 'ollama'"):
        resolve_answer_provider(
            {
                "RECALL_REASONING_ANSWER_ENABLED": "1",
                "RECALL_REASONING_ANSWER_PROVIDER": "openrouter",
            }
        )


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
        thinking=False,
    )

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 128}
    assert response.usage.total_tokens == 18
