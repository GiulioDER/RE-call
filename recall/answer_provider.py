"""Optional local answer provider for the reasoning evidence boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import hashlib
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from urllib import request
from urllib.parse import urlparse

from recall.provider_metadata import ProviderMetadata

ANSWER_PROMPT_DIGEST = hashlib.sha256(b"recall-answer-provider-v1").hexdigest()


class OllamaAnswerProvider:
    """Call Ollama's native chat endpoint and return raw answer JSON."""

    provider_id = "recall.reasoning.answer.ollama"

    def __init__(
        self,
        client: Any,
        *,
        model_id: str,
        revision: str = "unpinned",
        max_tokens: int = 512,
        context_tokens: int = 1024,
        thinking: bool = False,
        cost_per_1k_tokens: float | None = 0.0,
    ) -> None:
        if not model_id.strip():
            raise ValueError("answer model id must be non-empty")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("answer max tokens must be between 1 and 4096")
        if context_tokens < 512 or context_tokens > 32768:
            raise ValueError("answer context tokens must be between 512 and 32768")
        self.client = client
        self.model_id = model_id
        self.revision = revision
        self.max_tokens = max_tokens
        self.context_tokens = context_tokens
        self.thinking = thinking
        self.cost_per_1k_tokens = cost_per_1k_tokens
        initial_metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=model_id,
            model_revision=revision,
            prompt_digest=ANSWER_PROMPT_DIGEST,
        )
        self._last_metadata = initial_metadata
        self._metadata_local = threading.local()
        self._metadata_local.value = initial_metadata

    def __call__(self, system: str, user: str) -> str:
        started = time.perf_counter()
        response: object | None = None
        try:
            if isinstance(self.client, _NativeOllamaClient):
                response = self.client.chat(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_tokens,
                    context_tokens=self.context_tokens,
                    thinking=self.thinking,
                )
            elif isinstance(self.client, _OpenRouterClient):
                response = self.client.chat(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"think": self.thinking},
                )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError("answer provider returned no choices")
            content = getattr(getattr(choices[0], "message", None), "content", None)
            if not isinstance(content, str) or not content.strip():
                raise ValueError("answer provider returned empty content")
            return content
        finally:
            self._record_metadata(response, started)

    def _record_metadata(self, response: object | None, started: float) -> None:
        usage = getattr(response, "usage", None) if response is not None else None
        prompt = _usage_int(usage, "prompt_tokens")
        completion = _usage_int(usage, "completion_tokens")
        total = _usage_int(usage, "total_tokens")
        if (
            (total is None or total == 0)
            and prompt is not None
            and completion is not None
            and prompt + completion > 0
        ):
            total = prompt + completion
        monetary_cost = None
        if self.cost_per_1k_tokens is not None and total is not None:
            monetary_cost = round(total * self.cost_per_1k_tokens / 1000.0, 8)
        metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_revision=self.revision,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            monetary_cost_usd=monetary_cost,
            prompt_digest=ANSWER_PROMPT_DIGEST,
        )
        self._last_metadata = metadata
        self._metadata_local.value = metadata

    def provider_metadata(self) -> ProviderMetadata:
        return getattr(self._metadata_local, "value", self._last_metadata)


class OpenRouterAnswerProvider(OllamaAnswerProvider):
    """Call an OpenAI-compatible OpenRouter endpoint and return raw answer JSON."""

    provider_id = "recall.reasoning.answer.openrouter"


def resolve_answer_provider(
    env: Mapping[str, str] | None = None,
) -> OllamaAnswerProvider | OpenRouterAnswerProvider | None:
    """Resolve the explicitly enabled answer provider, otherwise return ``None``."""

    source = env if env is not None else os.environ
    enabled = source.get("RECALL_REASONING_ANSWER_ENABLED", "0").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError("RECALL_REASONING_ANSWER_ENABLED must be an explicit boolean")
    provider = source.get("RECALL_REASONING_ANSWER_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "openrouter", "openai"}:
        raise ValueError(
            "RECALL_REASONING_ANSWER_PROVIDER must be 'ollama', 'openrouter', or 'openai'"
        )
    model = source.get("RECALL_REASONING_ANSWER_MODEL", "").strip()
    if not model:
        raise ValueError(
            "RECALL_REASONING_ANSWER_MODEL is required when the answer provider is enabled"
        )
    default_base_url = (
        "http://127.0.0.1:11434/v1" if provider == "ollama" else "https://openrouter.ai/api/v1"
    )
    base_url = source.get("RECALL_REASONING_ANSWER_BASE_URL", default_base_url).strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("RECALL_REASONING_ANSWER_BASE_URL must be an absolute http(s) URL")
    try:
        timeout = float(source.get("RECALL_REASONING_ANSWER_TIMEOUT", "60"))
    except ValueError:
        raise ValueError("RECALL_REASONING_ANSWER_TIMEOUT must be finite and positive") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RECALL_REASONING_ANSWER_TIMEOUT must be finite and positive")
    try:
        max_tokens = int(source.get("RECALL_REASONING_ANSWER_MAX_TOKENS", "512"))
    except ValueError:
        raise ValueError("RECALL_REASONING_ANSWER_MAX_TOKENS must be an integer") from None
    try:
        context_tokens = int(source.get("RECALL_REASONING_ANSWER_CONTEXT_TOKENS", "1024"))
    except ValueError:
        raise ValueError(
            "RECALL_REASONING_ANSWER_CONTEXT_TOKENS must be an integer"
        ) from None
    cost_raw = source.get("RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS", "").strip()
    cost_per_1k_tokens: float | None = None
    if cost_raw:
        try:
            cost_per_1k_tokens = float(cost_raw)
        except ValueError:
            raise ValueError(
                "RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS must be a finite non-negative number"
            ) from None
        if not math.isfinite(cost_per_1k_tokens) or cost_per_1k_tokens < 0:
            raise ValueError(
                "RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS must be a finite non-negative number"
            )
    revision = source.get("RECALL_REASONING_ANSWER_REVISION", "unpinned")
    if provider == "ollama":
        client = _NativeOllamaClient(base_url, timeout=timeout)
        return OllamaAnswerProvider(
            client,
            model_id=model,
            revision=revision,
            max_tokens=max_tokens,
            context_tokens=context_tokens,
            thinking=source.get("RECALL_REASONING_ANSWER_THINKING", "0").lower()
            in {"1", "true", "yes", "on"},
        )
    api_key = source.get("RECALL_REASONING_ANSWER_API_KEY", "").strip()
    if not api_key and provider == "openrouter":
        api_key = source.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        api_key = source.get("RECALL_REASONING_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "RECALL_REASONING_ANSWER_API_KEY or OPENROUTER_API_KEY is required for OpenRouter"
        )
    client = _OpenRouterClient(base_url, api_key=api_key, timeout=timeout)
    return OpenRouterAnswerProvider(
        client,
        model_id=model,
        revision=revision,
        max_tokens=max_tokens,
        context_tokens=context_tokens,
        cost_per_1k_tokens=cost_per_1k_tokens,
    )


def _usage_int(usage: object | None, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and value >= 0:
        return int(value)
    return None


class _NativeOllamaClient:
    """Small stdlib client for Ollama's native API.

    The native API is used because it exposes Qwen's ``think`` switch directly.
    This keeps answer generation bounded by ``num_predict`` instead of spending
    the whole output budget on hidden reasoning tokens.
    """

    def __init__(self, base_url: str, *, timeout: float) -> None:
        parsed = urlparse(base_url.rstrip("/"))
        path = parsed.path.removesuffix("/v1")
        self.endpoint = parsed._replace(path=f"{path}/api/chat").geturl()
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        context_tokens: int,
        thinking: bool,
    ) -> object:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": thinking,
            "format": {
                "type": "object",
                "properties": {
                    "answer": {"type": ["string", "null"]},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "insufficient_evidence": {"type": "boolean"},
                },
                "required": ["answer", "citations", "insufficient_evidence"],
                "additionalProperties": False,
            },
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
                "num_ctx": context_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw.get("message") or {}
        usage = SimpleNamespace(
            prompt_tokens=raw.get("prompt_eval_count"),
            completion_tokens=raw.get("eval_count"),
            total_tokens=(raw.get("prompt_eval_count") or 0)
            + (raw.get("eval_count") or 0),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content=message.get("content")),
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class _OpenRouterClient:
    """Small stdlib client for OpenRouter's OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, *, api_key: str, timeout: float) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> object:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        choices = raw.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        usage = raw.get("usage") or {}
        usage_object = SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=message.get("content")))
        return SimpleNamespace(choices=[choice], usage=usage_object)


__all__ = ["OllamaAnswerProvider", "OpenRouterAnswerProvider", "resolve_answer_provider"]
