"""Optional local answer provider for the reasoning evidence boundary."""

from __future__ import annotations

from collections.abc import Mapping
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
        client: object,
        *,
        model_id: str,
        revision: str = "unpinned",
        max_tokens: int = 512,
        thinking: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError("answer model id must be non-empty")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("answer max tokens must be between 1 and 4096")
        self.client = client
        self.model_id = model_id
        self.revision = revision
        self.max_tokens = max_tokens
        self.thinking = thinking
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
                    thinking=self.thinking,
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
        metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_revision=self.revision,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            monetary_cost_usd=0.0,
            prompt_digest=ANSWER_PROMPT_DIGEST,
        )
        self._last_metadata = metadata
        self._metadata_local.value = metadata

    def provider_metadata(self) -> ProviderMetadata:
        return getattr(self._metadata_local, "value", self._last_metadata)


def resolve_answer_provider(
    env: Mapping[str, str] | None = None,
) -> OllamaAnswerProvider | None:
    """Resolve the explicitly enabled local answer provider, otherwise return ``None``."""

    source = env if env is not None else os.environ
    enabled = source.get("RECALL_REASONING_ANSWER_ENABLED", "0").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError("RECALL_REASONING_ANSWER_ENABLED must be an explicit boolean")
    provider = source.get("RECALL_REASONING_ANSWER_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise ValueError("RECALL_REASONING_ANSWER_PROVIDER must be 'ollama'")
    model = source.get("RECALL_REASONING_ANSWER_MODEL", "qwen3:4b").strip()
    base_url = source.get(
        "RECALL_REASONING_ANSWER_BASE_URL", "http://127.0.0.1:11434/v1"
    ).strip()
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
    client = _NativeOllamaClient(base_url, timeout=timeout)
    return OllamaAnswerProvider(
        client,
        model_id=model,
        revision=source.get("RECALL_REASONING_ANSWER_REVISION", "unpinned"),
        max_tokens=max_tokens,
        thinking=source.get("RECALL_REASONING_ANSWER_THINKING", "0").lower()
        in {"1", "true", "yes", "on"},
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
            "options": {"temperature": 0, "num_predict": max_tokens},
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


__all__ = ["OllamaAnswerProvider", "resolve_answer_provider"]
