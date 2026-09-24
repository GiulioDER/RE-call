"""Keep a Voyage Context request that is too large from failing an Add on every retry.

`VoyageContextualizedEmbedder` bounds a request by 60,000 characters as a proxy for the model's
32K-token window, and it never splits a single chunk. Token-dense text (base64, minified code,
CJK written without spaces) passes that proxy and Voyage refuses it with HTTP 400. The error is
not transient, so it surfaced as a 503 that AML retried 32 times with the identical payload, each
attempt failing the same way.

This guard changes nothing for a request Voyage accepts: the first call is exactly the call the
wrapped embedder would make. Only after a 400 does it re-plan the SAME texts under a byte budget
and send each part on its own. UTF-8 bytes bound the token count, because every token of a byte
level or byte-fallback tokenizer covers at least one byte, and a chunk longer than the budget has
only its embedding input cut; the stored text is untouched. The cost is weaker document context
across part boundaries, paid only by requests that could not be embedded at all before.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from recall.embeddings import Embedder


log = logging.getLogger("recall_aml")
# 30,000 bytes stays under the 32K-token window with room for per-chunk special tokens.
CONTEXT_FALLBACK_BYTES = 30_000


def _is_request_rejection(exc: Exception) -> bool:
    return getattr(exc, "http_status", None) == 400


def utf8_prefix(text: str, limit: int) -> str:
    """The longest prefix of ``text`` whose UTF-8 encoding fits in ``limit`` bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def byte_bounded_parts(group: list[str], limit: int) -> list[list[str]]:
    """Split one ordered document into parts of at most ``limit`` UTF-8 bytes each."""
    parts: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in group:
        piece = utf8_prefix(text, limit)
        size = len(piece.encode("utf-8"))
        if current and used + size > limit:
            parts.append(current)
            current = []
            used = 0
        current.append(piece)
        used += size
    if current:
        parts.append(current)
    return parts


class ContextOverflowGuard:
    """Wrap a Voyage Context embedder with a retry that fits a refused request."""

    def __init__(self, inner: Embedder, *, limit_bytes: int = CONTEXT_FALLBACK_BYTES) -> None:
        if limit_bytes < 1:
            raise ValueError("the fallback byte budget must be positive")
        self._inner = inner
        self._limit = limit_bytes

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def profile(self) -> Any:
        return getattr(self._inner, "profile", None)

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        grouped = cast(Any, self._inner).embed_document_groups
        try:
            return cast(list[list[list[float]]], grouped(groups))
        except Exception as exc:  # BROAD-CATCH: only a provider 400 is re-planned below
            if not _is_request_rejection(exc):
                raise
            log.warning(
                "context_request_refitted",
                extra={
                    "error_class": type(exc).__name__,
                    "group_count": len(groups),
                    "chunk_count": sum(len(group) for group in groups),
                    "limit_bytes": self._limit,
                },
            )
        output: list[list[list[float]]] = []
        for group in groups:
            vectors: list[list[float]] = []
            for part in byte_bounded_parts(group, self._limit):
                vectors.extend(cast(list[list[list[float]]], grouped([part]))[0])
            output.append(vectors)
        return output

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_document_groups([texts])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        query = cast(Any, self._inner).embed_query
        try:
            return cast(list[float], query(text))
        except Exception as exc:  # BROAD-CATCH: only a provider 400 is re-planned below
            if not _is_request_rejection(exc):
                raise
            log.warning(
                "context_query_refitted",
                extra={"error_class": type(exc).__name__, "limit_bytes": self._limit},
            )
        return cast(list[float], query(utf8_prefix(text, self._limit)))
