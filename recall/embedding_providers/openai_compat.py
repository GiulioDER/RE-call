"""The embedder for any OpenAI-compatible ``/embeddings`` endpoint.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping
from urllib.parse import urlsplit

from recall.embedding_core import (
    EmbeddingProfile,
    _check_declared_width,
    batched_embed,
    retry_with_backoff,
)

_OPENAI_COMPAT_REMOTE_HOSTS = frozenset({"api.openai.com", "openrouter.ai"})


_OPENAI_COMPAT_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


_OPENAI_COMPAT_REMOTE_PATHS = {
    "api.openai.com": "/v1",
    "openrouter.ai": "/api/v1",
}


_OPENAI_COMPAT_MAX_BASE_URL_LENGTH = 2_048


_OPENAI_COMPAT_MAX_PATH_LENGTH = 256


_OPENAI_COMPAT_MAX_PATH_SEGMENTS = 8


class OpenAICompatEmbedder:
    """OpenAI-compatible cloud embeddings via any ``base_url`` (OpenRouter, OpenAI, Azure, vLLM).

    Defaults to OpenRouter so the exact ``openai/text-embedding-3-small`` model is reachable on the
    same key the benchmark already uses for its generator and judge — no separate OpenAI billing,
    which is the whole reason this backend exists. The request/response shape is OpenAI's
    ``/v1/embeddings``, so the stock ``openai`` SDK works unchanged against OpenRouter's endpoint.

    ``dimensions`` is optional rather than implicit: ``text-embedding-3-small`` returns its native
    1536-wide vector when the field is omitted, while newer OpenRouter embedding models such as
    Gemini Embedding 2 expose selectable output widths.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 128,
        max_retries: int = 3,
        dimensions: int | None = None,
        name_prefix: str = "openai",
        identity: EmbeddingProfile | None = None,
    ) -> None:
        """Build an OpenAI-compatible client, optionally under a registered profile's identity.

        See `VoyageEmbedder.__init__` for why ``identity`` matters; the reasoning is identical.
        Note that ``base_url`` and ``dimensions`` are NOT derivable from the identity and must be
        passed alongside it: the same model name answers at different widths depending on the
        ``dimensions`` field, so a registered profile has to supply both and `_check_declared_width`
        then holds them to it.
        """
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty URL")
        candidate_base_url = base_url.strip()
        if len(candidate_base_url) > _OPENAI_COMPAT_MAX_BASE_URL_LENGTH:
            raise ValueError("base_url exceeds the maximum supported length")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in candidate_base_url):
            raise ValueError("base_url must not contain control characters")
        try:
            parsed_base_url = urlsplit(candidate_base_url)
            # Read for its side effect: `.port` raises ValueError on a malformed or out of range port.
            _ = parsed_base_url.port
        except ValueError as exc:
            raise ValueError("base_url must be a valid absolute HTTP(S) URL") from exc
        normalized_base_url = parsed_base_url.geturl().rstrip("/")
        url_components = (
            parsed_base_url.scheme,
            parsed_base_url.netloc,
            parsed_base_url.path,
            parsed_base_url.query,
            parsed_base_url.fragment,
        )
        if any(
            any(ord(char) < 0x20 or ord(char) == 0x7F for char in component)
            for component in url_components
        ):
            raise ValueError("base_url URL components must not contain control characters")
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.hostname
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
            or parsed_base_url.query
            or parsed_base_url.fragment
            or any(char.isspace() for char in parsed_base_url.netloc)
            or any(char.isspace() for char in parsed_base_url.path)
        ):
            raise ValueError(
                "base_url must be an absolute HTTP(S) URL without credentials, query, or fragment"
            )
        hostname = parsed_base_url.hostname.lower()
        raw_path = parsed_base_url.path or "/"
        bounded_raw_path = raw_path.rstrip("/") or "/"
        normalized_path = posixpath.normpath(bounded_raw_path)
        if bounded_raw_path != normalized_path:
            raise ValueError("base_url path must not contain dot segments or duplicate separators")
        path_segments = tuple(segment for segment in normalized_path.split("/") if segment)
        if (
            len(raw_path) > _OPENAI_COMPAT_MAX_PATH_LENGTH
            or len(normalized_path) > _OPENAI_COMPAT_MAX_PATH_LENGTH
            or len(path_segments) > _OPENAI_COMPAT_MAX_PATH_SEGMENTS
        ):
            raise ValueError("base_url path exceeds the maximum supported length or depth")
        is_approved_remote = (
            parsed_base_url.scheme == "https" and hostname in _OPENAI_COMPAT_REMOTE_HOSTS
        )
        is_approved_local = (
            parsed_base_url.scheme in {"http", "https"}
            and hostname in _OPENAI_COMPAT_LOCAL_HOSTS
        )
        if not (is_approved_remote or is_approved_local):
            raise ValueError("base_url hostname is not an approved OpenAI-compatible endpoint")
        if is_approved_remote and normalized_path != _OPENAI_COMPAT_REMOTE_PATHS[hostname]:
            raise ValueError("base_url must use the canonical path for the approved remote endpoint")
        if hostname in _OPENAI_COMPAT_REMOTE_HOSTS and parsed_base_url.port not in {None, 443}:
            raise ValueError("approved remote OpenAI-compatible endpoints must use port 443")
        safe_base_url = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}{normalized_path}"
        provider_key_env = {
            "https://openrouter.ai/api/v1": "OPENROUTER_API_KEY",
            "https://api.openai.com/v1": "OPENAI_API_KEY",
        }.get(normalized_base_url)
        if api_key is None and provider_key_env is None:
            raise RuntimeError(
                "OpenAICompatEmbedder requires an explicit api_key for an unrecognized base_url"
            )
        key: str | None
        if api_key is not None:
            key = api_key
        else:
            assert provider_key_env is not None
            key = os.environ.get(provider_key_env)
        if not key:
            raise RuntimeError(
                f"OpenAICompatEmbedder needs {provider_key_env} for {base_url!r}, or an explicit api_key"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise ImportError(
                'OpenAICompatEmbedder requires: pip install "recall-rag[openai]"'
            ) from exc
        # `max_retries=0` because `retry_with_backoff` in `_embed_one_batch` owns the retry
        # policy. The SDK default is 2 retries, which does not replace ours but multiplies with
        # it: one 429 costs 3 x 3 = 9 requests rather than the 3 the policy asks for, and the
        # outer FULL-jitter backoff (which exists so a fleet does not remarch onto the provider
        # in lockstep) ends up wrapping an inner loop whose own jitter is only `1 - 0.25 *
        # random()` on its doubling schedule — a 25% smear, not a spread across the interval, so
        # the fleet it is meant to separate stays largely in step. This is the
        # corpus indexing path, so the multiplication lands batch after batch on a provider that
        # has just said it is overloaded.
        self._client = OpenAI(api_key=key, base_url=safe_base_url, max_retries=0)
        self._model = identity.model_name if identity is not None else model
        self._name = f"{name_prefix}:{self._model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        if dimensions is not None and dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        # Probe the width once, the same way the other cloud embedder does, so a store can be built
        # at the matching ``dim`` before the first real batch is embedded.
        self._dim = len(self._embed_one_batch(["probe"])[0])
        _check_declared_width(identity, self._dim, f"{base_url} model {self._model!r}")
        self._profile = identity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile | None:
        """The registered identity, or None for a legacy construction."""
        return self._profile

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        request: dict[str, object] = {
            "model": self._model,
            "input": batch,
            "encoding_format": "float",
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions
        result = retry_with_backoff(
            lambda: self._client.embeddings.create(**request),
            attempts=self._max_retries,
        )
        return [[float(x) for x in item.embedding] for item in result.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch — the same
        contract as ``VoyageEmbedder.embed``, so this is a drop-in cloud embedder on the RE-call
        arm."""
        return batched_embed(texts, self._embed_one_batch, batch_size=self._batch_size)


def _optional_dimensions(source: Mapping[str, str]) -> int | None:
    raw = source.get("RECALL_EMBED_DIMENSIONS", "").strip()
    if not raw:
        return None
    try:
        dimensions = int(raw)
    except ValueError as exc:
        raise ValueError("RECALL_EMBED_DIMENSIONS must be a positive integer") from exc
    if dimensions < 1:
        raise ValueError("RECALL_EMBED_DIMENSIONS must be a positive integer")
    return dimensions
