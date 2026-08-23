"""Optional, presentation-only translation for RE-call responses.

The translation service is deliberately kept outside the retrieval and evidence paths.  A
localized value is an aid for a human or client UI.  It is never substituted for canonical
corpus text, provenance, or the evidence prompt that a generator is allowed to use.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from http.client import IncompleteRead
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from recall.errors import RecallError
from recall._env import truthy

if TYPE_CHECKING:
    from recall_mcp.service import EvidenceResult, SearchResult

_LOCALE_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_BATCH = 32
_DEFAULT_MAX_TEXT_CHARS = 20_000
_DEFAULT_MAX_RESPONSE_BYTES = 2_000_000


class TranslationError(RuntimeError, RecallError):
    """A provider failed or returned a response that cannot be trusted."""


class TranslationProvider(Protocol):
    """Small provider port so retrieval remains independent of any translation vendor."""

    name: str

    def translate_batch(self, texts: Sequence[str], *, locale: str) -> list[str]:
        """Translate texts in order, or raise ``TranslationError``."""


class NullTranslationProvider:
    """Provider used when localization is not explicitly enabled."""

    name = "disabled"

    def translate_batch(self, texts: Sequence[str], *, locale: str) -> list[str]:
        del texts, locale
        raise TranslationError("translation is disabled")


def normalize_locale(locale: str | None) -> str | None:
    """Return a safe locale token, treating the explicit original value as no-op."""

    if locale is None:
        return None
    value = locale.strip()
    if value.lower() in {"", "original", "none"}:
        return None
    if not _LOCALE_RE.fullmatch(value):
        raise ValueError("locale must contain only letters, numbers, and underscores")
    return value


def _read_bool(env: dict[str, str], name: str) -> bool:
    return truthy(env.get(name))


def _read_positive_float(
    env: dict[str, str], name: str, default: float, maximum: float
) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0 or not math.isfinite(value) or value > maximum:
        raise ValueError(f"{name} must be a positive number no greater than {maximum:g}")
    return value


def _read_positive_int(env: dict[str, str], name: str, default: int, maximum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _validate_endpoint(endpoint: str, *, allow_http: bool) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password:
        raise ValueError("translation endpoint must not contain credentials")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and (parsed.hostname in _LOOPBACK_HOSTS or allow_http):
        pass
    else:
        raise ValueError(
            "translation endpoint must use HTTPS; HTTP is allowed only for loopback or with "
            "RECALL_TRANSLATION_ALLOW_HTTP=1"
        )
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("translation endpoint must be an absolute URL without query or fragment")
    return endpoint


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler)


class TranslateJsProvider:
    """Client for the documented xnx3/translate JSON endpoint.

    The upstream project also ships a browser DOM translator with dynamic script loading and
    runtime evaluation.  RE-call uses only its documented text HTTP endpoint, which keeps that
    browser execution model out of the server process.
    """

    name = "xnx3-translate"

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_batch: int = _DEFAULT_MAX_BATCH,
        max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        allow_http: bool = False,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint, allow_http=allow_http)
        self.timeout_seconds = timeout_seconds
        self.max_batch = max_batch
        self.max_text_chars = max_text_chars
        self.max_response_bytes = max_response_bytes

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TranslationProvider:
        values = env if env is not None else dict(os.environ)
        if not _read_bool(values, "RECALL_TRANSLATION_ENABLED"):
            return NullTranslationProvider()
        endpoint = values.get("RECALL_TRANSLATION_ENDPOINT", "").strip()
        if not endpoint:
            raise ValueError(
                "RECALL_TRANSLATION_ENDPOINT is required when translation is enabled; "
                "use a self-hosted or explicitly approved HTTPS endpoint"
            )
        return cls(
            endpoint,
            timeout_seconds=_read_positive_float(
                values,
                "RECALL_TRANSLATION_TIMEOUT_SECONDS",
                _DEFAULT_TIMEOUT_SECONDS,
                _MAX_TIMEOUT_SECONDS,
            ),
            max_batch=_read_positive_int(
                values, "RECALL_TRANSLATION_MAX_BATCH", _DEFAULT_MAX_BATCH, 128
            ),
            max_text_chars=_read_positive_int(
                values, "RECALL_TRANSLATION_MAX_TEXT_CHARS", _DEFAULT_MAX_TEXT_CHARS, 200_000
            ),
            max_response_bytes=_read_positive_int(
                values,
                "RECALL_TRANSLATION_MAX_RESPONSE_BYTES",
                _DEFAULT_MAX_RESPONSE_BYTES,
                20_000_000,
            ),
            allow_http=_read_bool(values, "RECALL_TRANSLATION_ALLOW_HTTP"),
        )

    def translate_batch(self, texts: Sequence[str], *, locale: str) -> list[str]:
        normalized = normalize_locale(locale)
        if normalized is None:
            return list(texts)
        if len(texts) > self.max_batch:
            raise TranslationError("translation batch exceeds the configured limit")
        if any(not isinstance(text, str) or len(text) > self.max_text_chars for text in texts):
            raise TranslationError("translation text exceeds the configured limit")
        if not texts:
            return []

        body = urlencode({"to": normalized, "text": json.dumps(list(texts), ensure_ascii=False)})
        request = Request(
            self.endpoint,
            data=body.encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except (IncompleteRead, OSError, TimeoutError, ValueError) as exc:
            raise TranslationError("translation provider request failed") from exc
        if len(raw) > self.max_response_bytes:
            raise TranslationError("translation provider response exceeded the configured limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranslationError("translation provider returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("result") != 1:
            raise TranslationError("translation provider returned an unsuccessful result")
        translated = payload.get("text")
        if (
            not isinstance(translated, list)
            or len(translated) != len(texts)
            or not all(isinstance(item, str) for item in translated)
        ):
            raise TranslationError("translation provider returned an invalid text list")
        return translated


def provider_from_env(env: dict[str, str] | None = None) -> TranslationProvider:
    return TranslateJsProvider.from_env(env)


def translate_for_display(
    texts: Sequence[str], locale: str, provider: TranslationProvider
) -> tuple[list[str], bool, str | None]:
    try:
        batch_size = getattr(provider, "max_batch", _DEFAULT_MAX_BATCH)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise TranslationError("translation provider has an invalid batch limit")
        values: list[str] = []
        for start in range(0, len(texts), batch_size):
            values.extend(provider.translate_batch(texts[start : start + batch_size], locale=locale))
    except TranslationError:
        return list(texts), False, "Translation is unavailable; canonical values are unchanged."
    return values, True, None


def _localization_metadata(
    locale: str, provider: TranslationProvider, translated: bool, warning: str | None
) -> dict[str, object]:
    return {
        "locale": locale,
        "provider": provider.name,
        "translated": translated,
        "fallback": not translated,
        "warning": warning,
        "canonical_unchanged": True,
    }


def render_search_response(
    result: SearchResult, locale: str | None, provider: TranslationProvider
) -> str:
    """Render canonical search JSON, optionally adding localized display values."""

    normalized = normalize_locale(locale)
    if normalized is None:
        return result.model_dump_json(indent=2)
    texts = [hit.text for hit in result.hits]
    if result.advice:
        texts.append(result.advice)
    values, translated, warning = translate_for_display(texts, normalized, provider)
    hit_count = len(result.hits)
    payload = json.loads(result.model_dump_json(indent=2))
    payload["localized"] = {
        **_localization_metadata(normalized, provider, translated, warning),
        "hits": [
            {"chunk_id": hit.chunk_id, "text": value}
            for hit, value in zip(result.hits, values[:hit_count], strict=True)
        ],
        "advice": values[-1] if result.advice else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_evidence_response(
    result: EvidenceResult, locale: str | None, provider: TranslationProvider
) -> str:
    """Render canonical evidence JSON plus localized display text.

    The exact ``system_prompt`` and ``user_message`` stay canonical.  They contain citation
    identifiers and data boundaries that must not be rewritten by a presentation provider.
    """

    normalized = normalize_locale(locale)
    if normalized is None:
        return result.model_dump_json(indent=2)
    texts = [item.text for item in result.items]
    if result.advice:
        texts.append(result.advice)
    values, translated, warning = translate_for_display(texts, normalized, provider)
    item_count = len(result.items)
    payload = json.loads(result.model_dump_json(indent=2))
    payload["localized"] = {
        **_localization_metadata(normalized, provider, translated, warning),
        "items": [
            {"chunk_id": item.chunk_id, "text": value}
            for item, value in zip(result.items, values[:item_count], strict=True)
        ],
        "advice": values[-1] if result.advice else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
