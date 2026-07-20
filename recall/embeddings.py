from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception worth retrying?

    Covers rate-limit (429), server (5xx) and network/timeout errors WITHOUT importing any
    provider-specific exception type (voyageai is an optional dependency). Checks a numeric
    ``status_code``/``status`` attribute first, then falls back to matching well-known markers
    in the exception text. A non-transient error (e.g. 401 auth) returns False so it fails fast.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "429", " 500", " 502", " 503", " 504", "rate limit", "too many requests",
        "timeout", "timed out", "temporarily", "connection", "reset by peer", "unavailable",
    )
    return any(m in text for m in markers)


def retry_with_backoff(
    fn: Callable[[], object],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    is_transient: Callable[[Exception], bool] = _is_transient,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call ``fn()`` with exponential backoff, retrying only transient failures.

    Re-raises immediately for a non-transient error, and re-raises the last error after
    ``attempts`` tries. ``sleep`` is injectable so tests can exercise the retry path without
    real delays. Delay for retry i is ``min(max_delay, base_delay * 2**i)``.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if i == attempts - 1 or not is_transient(exc):
                raise
            sleep(min(max_delay, base_delay * (2 ** i)))
    assert last is not None  # unreachable: loop either returns or raises
    raise last


def _batches(seq: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def batched_embed(
    texts: list[str],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int = 128,
    max_batch_chars: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in provider-safe batches, concatenating results in input order.

    ``embed_batch`` embeds a single batch. Batches are cut on ``batch_size`` (count) and, when
    ``max_batch_chars`` is set, also on a cumulative character budget — a guard against a batch
    that is few in count but huge in tokens. A single text over the char budget still goes out
    alone (never dropped). Order is preserved: batch results are appended in sequence.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    out: list[list[float]] = []
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (
            len(batch) >= batch_size
            or (max_batch_chars is not None and chars + len(t) > max_batch_chars)
        ):
            out.extend(embed_batch(batch))
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        out.extend(embed_batch(batch))
    return out


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors. Implementations must be deterministic and
    order-preserving: `embed(texts)` returns one vector per input text, in input order,
    each of length `dim`. `name` identifies the backend (used in logging / evals).
    """

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests and offline demos.

    Hashes whitespace tokens into a fixed-width bag-of-words vector, then
    L2-normalizes. Not semantic, but stable and fast — good enough to exercise
    plumbing and to keep the test suite offline.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install recall[fastembed]`."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "FastEmbedEmbedder requires the fastembed extra: pip install recall[fastembed]"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self._name = model_name
        self._dim = len(next(iter(self._model.embed(["probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install recall[voyage]` and VOYAGE_API_KEY."""

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        batch_size: int = 128,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VoyageEmbedder needs VOYAGE_API_KEY (env) or an explicit api_key")
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("VoyageEmbedder requires: pip install recall[voyage]") from exc
        self._client = voyageai.Client(api_key=key)
        self._model = model
        self._name = f"voyage:{model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._dim = len(self._client.embed(["probe"], model=model).embeddings[0])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch.

        A single request sending every chunk at once will exceed the API's per-request limit on
        a real corpus and has no tolerance for a transient 429/5xx; batching + retry make bulk
        indexing survivable. Results are concatenated in input order (see ``batched_embed``).
        """
        def _embed_batch(batch: list[str]) -> list[list[float]]:
            result = retry_with_backoff(
                lambda: self._client.embed(batch, model=self._model),
                attempts=self._max_retries,
            )
            return [[float(x) for x in v] for v in result.embeddings]

        return batched_embed(texts, _embed_batch, batch_size=self._batch_size)
