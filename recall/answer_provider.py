"""Optional answer providers for the reasoning evidence boundary.

Two backends: a local Ollama adapter, and any OpenAI-compatible `/chat/completions`
endpoint, defaulting to OpenRouter. Both are off unless explicitly enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
import time
from types import SimpleNamespace
from typing import Protocol
from urllib import request
from urllib.parse import urlparse

from recall.provider_metadata import ProviderMetadata

ANSWER_PROMPT_DIGEST = hashlib.sha256(b"recall-answer-provider-v1").hexdigest()


class _AnswerClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        thinking: bool,
    ) -> object: ...


class OllamaAnswerProvider:
    """Call Ollama's native chat endpoint and return raw answer JSON."""

    provider_id = "recall.reasoning.answer.ollama"

    def __init__(
        self,
        client: _AnswerClient,
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
        self._last_metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=model_id,
            model_revision=revision,
            prompt_digest=ANSWER_PROMPT_DIGEST,
        )

    def __call__(self, system: str, user: str) -> str:
        started = time.perf_counter()
        response: object | None = None
        try:
            response = self.client.chat(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                thinking=self.thinking,
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
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        self._last_metadata = ProviderMetadata(
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

    def provider_metadata(self) -> ProviderMetadata:
        return self._last_metadata


class OpenAICompatibleAnswerProvider:
    """Call an OpenAI compatible `/chat/completions` endpoint and return raw answer JSON.

    Exists because :class:`OllamaAnswerProvider` cannot reach a hosted endpoint whatever its
    base URL is set to: `_NativeOllamaClient` rewrites the path to ``<base>/api/chat``, sends
    Ollama's ``think``/``options`` payload, and attaches no ``Authorization`` header. Pointing
    it at OpenRouter produces an unauthenticated POST to a path that does not exist.

    The client is injected in tests and imported lazily by :func:`resolve_answer_provider`, so
    the core package stays usable without the optional OpenAI dependency. This mirrors
    :class:`recall.reasoning_expansion.OpenAIExpansionProvider`, which is the tested precedent
    for an OpenRouter-backed port in this package.
    """

    provider_id = "recall.reasoning.answer.openai"

    def __init__(
        self,
        client: object,
        *,
        model_id: str,
        revision: str = "unpinned",
        max_tokens: int = 512,
        cost_per_1k_tokens: float | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("answer model id must be non-empty")
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("answer max tokens must be between 1 and 4096")
        if cost_per_1k_tokens is not None and (
            not math.isfinite(cost_per_1k_tokens) or cost_per_1k_tokens < 0
        ):
            raise ValueError("cost_per_1k_tokens must be finite and non-negative")
        if cost_per_1k_tokens is not None:
            # Same normalisation as the resolver: `-0.0 < 0` is False, so a signed zero passes
            # the guard above and reaches `monetary_cost_usd` as a negative-signed money value.
            # Applied HERE as well because this constructor is the library-level invariant for a
            # caller who bypasses `resolve_answer_provider` -- the same argument the max-tokens
            # bound makes, and leaving it one-sided was the asymmetry the architect gate named.
            cost_per_1k_tokens += 0.0
        self.client = client
        self.model_id = model_id
        self.revision = revision
        self.max_tokens = max_tokens
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self._last_metadata = ProviderMetadata(
            provider_id=self.provider_id,
            model_id=model_id,
            model_revision=revision,
            prompt_digest=ANSWER_PROMPT_DIGEST,
        )

    def __call__(self, system: str, user: str) -> str:
        started = time.perf_counter()
        response: object | None = None
        try:
            # No `reasoning_effort`: it is an OpenAI-specific parameter and a non-OpenAI model
            # behind an OpenAI-compatible gateway can reject an unknown field outright. The
            # expansion provider sends it because its own resolver validates the value against
            # a fixed set; an answer wants determinism, which `temperature=0` already gives.
            #
            # `json_object` rather than a strict `json_schema`: schema mode is not universal
            # across the models an OpenAI-compatible gateway fronts, and the envelope is
            # validated downstream by `recall.reasoning` either way. The Ollama adapter can
            # afford a strict schema because it talks to exactly one implementation.
            create = self.client.chat.completions.create  # type: ignore[attr-defined]
            response = create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
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
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        elif (
            total is not None
            and prompt is not None
            and completion is not None
            and total != prompt + completion
        ):
            # A THIRD-PARTY usage object can disagree with itself honestly: a gateway may count
            # reasoning or cached-prefill tokens in the total and not in the parts, and
            # `_usage_int` floors each of the three independently. `ProviderMetadata` REFUSES
            # that triple, and this runs in a `finally`, so the raise replaced an answer that
            # had already been returned AND paid for. Keep the total, which is what billing
            # follows, and drop the parts rather than inventing a reconciliation nobody
            # measured. This provider is the first here to accept a usage object it did not
            # compute; `_NativeOllamaClient` derives its own total, which is why the Ollama
            # adapter never had to face this.
            prompt = completion = None
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        # Null, never 0.0, when no price was configured. `OllamaAnswerProvider` records 0.0
        # because local inference genuinely costs nothing; a hosted call does cost something,
        # and a 0.0 here would be a false monetary CLAIM rather than a missing measurement.
        # `docs/REASONING_OPERATIONS.md` rejects benchmark cost claims on missing cost, which
        # only works if missing is recorded as missing.
        cost = (
            total * self.cost_per_1k_tokens / 1000
            if total is not None and self.cost_per_1k_tokens is not None
            else None
        )
        try:
            self._last_metadata = ProviderMetadata(
                provider_id=self.provider_id,
                model_id=self.model_id,
                model_revision=self.revision,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                latency_ms=latency_ms,
                monetary_cost_usd=cost,
                prompt_digest=ANSWER_PROMPT_DIGEST,
            )
        except ValueError:
            # Metadata RECORDS a call; it must never gate one. The fallback also CLEARS the
            # previous call's numbers, which is the half that would otherwise bite silently:
            # leaving `_last_metadata` untouched re-reports the earlier call's tokens and cost
            # against this response, double-counting real money.
            self._last_metadata = ProviderMetadata(
                provider_id=self.provider_id,
                model_id=self.model_id,
                model_revision=self.revision,
                latency_ms=latency_ms,
                prompt_digest=ANSWER_PROMPT_DIGEST,
            )

    def provider_metadata(self) -> ProviderMetadata:
        return self._last_metadata


AnswerProvider = OllamaAnswerProvider | OpenAICompatibleAnswerProvider

#: Default endpoint per backend. Ollama's is a loopback address and OpenRouter's is a hosted
#: one, so a single shared default would silently point one backend at the other's endpoint.
_DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "openai": "https://openrouter.ai/api/v1",
}


def resolve_answer_provider(env: Mapping[str, str] | None = None) -> AnswerProvider | None:
    """Resolve the explicitly enabled answer provider, otherwise return ``None``.

    Every value is validated BEFORE the optional ``openai`` import, so an installation without
    that extra still gets the configuration error it actually has rather than an import error
    standing in for it. That ordering is not cosmetic: it is the exact defect that turned the
    `floor` CI job red on PR #366, where the resolver imported first and a bad timeout surfaced
    as a missing dependency.
    """

    source = env if env is not None else os.environ
    enabled = source.get("RECALL_REASONING_ANSWER_ENABLED", "0").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError("RECALL_REASONING_ANSWER_ENABLED must be an explicit boolean")
    # Defaults to ollama, which is the only backend that existed before 2026-08-31, so an
    # operator who already enabled the provider keeps the behaviour they configured.
    backend = source.get("RECALL_REASONING_ANSWER_PROVIDER", "").strip().lower() or "ollama"
    if backend not in _DEFAULT_BASE_URLS:
        raise ValueError(
            "RECALL_REASONING_ANSWER_PROVIDER must be 'ollama' or 'openai', not "
            f"{backend!r}"
        )
    # Required rather than defaulted, matching the expansion resolver: a silent model default
    # means enabling the provider quietly selects a model nobody chose.
    model = source.get("RECALL_REASONING_ANSWER_MODEL", "").strip()
    if not model:
        raise ValueError(
            "RECALL_REASONING_ANSWER_MODEL is required when the answer provider is enabled"
        )
    # The legacy bare names are read as a TRIO (key, base URL, timeout), matching
    # `resolve_expansion_provider`. Taking the legacy KEY while ignoring the legacy BASE URL was
    # the dangerous half-measure: a pre-0.11 config naming a private gateway kept its key and
    # silently acquired the OpenRouter default, so the credential AND the retrieved evidence went
    # to a third party the operator never named. Read both or neither.
    #
    # Ollama keeps its loopback default and does not read the bare base URL: the bare family is
    # the hosted OpenAI-compatible arm's, and pointing the Ollama transport at it would rebuild
    # the exact `<base>/api/chat` mismatch documented below.
    legacy_base_url = (
        source.get("RECALL_REASONING_BASE_URL", "").strip() if backend == "openai" else ""
    )
    base_url = (
        source.get("RECALL_REASONING_ANSWER_BASE_URL", "").strip()
        or legacy_base_url
        or _DEFAULT_BASE_URLS[backend]
    )
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("RECALL_REASONING_ANSWER_BASE_URL must be an absolute http(s) URL")
    raw_timeout = source.get("RECALL_REASONING_ANSWER_TIMEOUT", "").strip() or (
        source.get("RECALL_REASONING_TIMEOUT", "").strip() if backend == "openai" else ""
    )
    if not raw_timeout:
        # Empty means unset: .env templates ship the key valueless, matching recall/profiles.py.
        raw_timeout = "60"
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError("RECALL_REASONING_ANSWER_TIMEOUT must be a number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RECALL_REASONING_ANSWER_TIMEOUT must be finite and positive")
    raw_max_tokens = source.get("RECALL_REASONING_ANSWER_MAX_TOKENS", "").strip()
    if not raw_max_tokens:
        raw_max_tokens = "512"
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError as exc:
        raise ValueError("RECALL_REASONING_ANSWER_MAX_TOKENS must be an integer") from exc
    # Range-checked HERE, not only in the constructors, for the two reasons the rest of this
    # resolver already observes: the constructors run AFTER the optional `openai` import, so a
    # floor install reported an out-of-range value as a missing dependency (the exact PR #366
    # ordering defect this function's docstring claims to have eliminated); and their message
    # names no variable, which every other numeric check here is careful to do. The constructor
    # bounds stay as the library-level invariant for a caller who bypasses this resolver.
    if max_tokens < 1 or max_tokens > 4096:
        raise ValueError("RECALL_REASONING_ANSWER_MAX_TOKENS must be between 1 and 4096")
    thinking_raw = source.get("RECALL_REASONING_ANSWER_THINKING", "0").strip().lower()
    if thinking_raw in {"", "0", "false", "no", "off"}:
        thinking = False
    elif thinking_raw in {"1", "true", "yes", "on"}:
        thinking = True
    else:
        raise ValueError("RECALL_REASONING_ANSWER_THINKING must be an explicit boolean")
    # `.strip() or` for the same "empty means unset" reason as the timeout and max-tokens
    # above: .env templates ship keys valueless. Without it a valueless line recorded an
    # EMPTY model_revision, which `validate_cost_claim` then refuses as an unsupported cost
    # claim -- on the one backend that actually has a cost.
    revision = source.get("RECALL_REASONING_ANSWER_REVISION", "").strip() or "unpinned"

    if backend == "ollama":
        # Refused rather than ignored, symmetrically with the THINKING refusal below. Returning
        # here without reading these left an operator who set a price or a key believing they
        # had configured something, which is the failure mode this whole change exists to
        # remove. `_ANSWER_KEY` is excluded from the check on purpose: it is legitimately
        # present in a shared .env for the other reasoning arm.
        for openai_only in ("RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS",):
            if source.get(openai_only, "").strip():
                raise ValueError(
                    f"{openai_only} is an openai-backend setting and cannot be used with "
                    "RECALL_REASONING_ANSWER_PROVIDER=ollama, where inference is local and "
                    "the recorded cost is always 0.0"
                )
        return OllamaAnswerProvider(
            _NativeOllamaClient(base_url, timeout=timeout),
            model_id=model,
            revision=revision,
            max_tokens=max_tokens,
            thinking=thinking,
        )

    if thinking:
        # Refused rather than ignored. `think` is an Ollama request field with no OpenAI
        # equivalent, so accepting it here would leave an operator believing they enabled
        # something, which is the failure mode this whole change exists to remove.
        raise ValueError(
            "RECALL_REASONING_ANSWER_THINKING is an Ollama-only setting and cannot be used "
            "with RECALL_REASONING_ANSWER_PROVIDER=openai"
        )
    # The bare RECALL_REASONING_API_KEY is a LEGACY spelling, accepted so a hand-written or
    # pre-0.11 .env keeps working, and matching how `resolve_expansion_provider` treats the
    # same name.
    #
    # It is NOT what `recall setup` writes. An earlier version of this comment said it was,
    # and that was wrong: `recall/setup.py` returns only the `RECALL_REASONING_EXPANSION_*`
    # spellings, and says why in its own comment -- the bare pair is SHARED between reasoning
    # arms, so writing it would leak one arm's key into another. A wizard-configured install
    # therefore has NO key this resolver can find, and must set
    # RECALL_REASONING_ANSWER_API_KEY explicitly.
    key = (
        source.get("RECALL_REASONING_ANSWER_API_KEY", "").strip()
        or source.get("RECALL_REASONING_API_KEY", "").strip()
    )
    if not key:
        raise ValueError(
            "RECALL_REASONING_ANSWER_API_KEY (or the legacy RECALL_REASONING_API_KEY) is "
            "required when RECALL_REASONING_ANSWER_PROVIDER=openai"
        )
    # The shared validation above accepts `http` because the OLLAMA default is loopback. This
    # backend is the first to attach an `Authorization: Bearer` header to that URL, so a
    # copy-pasted or downgraded `http://` endpoint would put the operator's key on the wire in
    # cleartext. Loopback stays allowed: it is how a local gateway or a test double is reached,
    # and it never leaves the machine.
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" and host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "RECALL_REASONING_ANSWER_BASE_URL must use https for a non-loopback host when "
            "RECALL_REASONING_ANSWER_PROVIDER=openai, because the API key is sent with every "
            f"request (got {parsed.scheme!r} for host {host!r})"
        )
    raw_cost = source.get("RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS", "").strip()
    cost: float | None = None
    if raw_cost:
        # Wrapped rather than a bare `float()`. `resolve_expansion_provider` leaves its
        # equivalent unwrapped, so a typo there surfaces as "could not convert string to
        # float" with no variable named, which is the one thing every other numeric in this
        # resolver is careful to do.
        try:
            cost = float(raw_cost)
        except ValueError as exc:
            raise ValueError(
                "RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS must be a number"
            ) from exc
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                "RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS must be finite and non-negative"
            )
        # `-0.0 < 0` is False, so a signed zero passes every guard here AND in ProviderMetadata,
        # then serializes into the artifact as `-0.0`: a negative-signed money value. Normalise
        # rather than reject, since an operator writing -0.0 plainly means free.
        cost += 0.0
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ValueError(
            "RECALL_REASONING_ANSWER_PROVIDER=openai needs the openai extra: "
            'pip install "recall-rag[openai]"'
        ) from exc
    return OpenAICompatibleAnswerProvider(
        OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout),
        model_id=model,
        revision=revision,
        max_tokens=max_tokens,
        cost_per_1k_tokens=cost,
    )


def _usage_int(usage: object | None, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class _NativeOllamaClient:
    """Small stdlib client for Ollama's native API."""

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
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw.get("message") or {}
        usage = SimpleNamespace(
            prompt_tokens=raw.get("prompt_eval_count"),
            completion_tokens=raw.get("eval_count"),
            total_tokens=(raw.get("prompt_eval_count") or 0) + (raw.get("eval_count") or 0),
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=message.get("content")))
        return SimpleNamespace(choices=[choice], usage=usage)


__all__ = [
    "AnswerProvider",
    "OllamaAnswerProvider",
    "OpenAICompatibleAnswerProvider",
    "resolve_answer_provider",
]
