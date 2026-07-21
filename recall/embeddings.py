from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable


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


#: Texts per Voyage request. The API caps batch size (and total tokens) per call, so a corpus
#: cannot be embedded in one shot — indexing a few hundred memos would simply error.
VOYAGE_BATCH_SIZE = 128
VOYAGE_MAX_RETRIES = 4

#: Error kinds worth retrying. Matched on the exception's class name + message so the SDK's
#: exception hierarchy does not have to be imported here (it is an optional extra), and so a
#: permanent error — a bad API key, a malformed request — fails fast instead of sleeping
#: through four pointless attempts.
_RETRYABLE = ("ratelimit", "rate limit", "servererror", "server error", "timeout",
              "connection", "temporarily", "503", "502", "429")


def _is_retryable(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RETRYABLE)


def _embed_batched(
    client: object,
    model: str,
    texts: list[str],
    batch_size: int = VOYAGE_BATCH_SIZE,
    max_retries: int = VOYAGE_MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """Embed `texts` through `client` in provider-sized batches, retrying transient failures.

    Order is preserved: batches are concatenated in input order, which the Embedder protocol
    requires (chunk i must line up with vector i). Backoff is exponential (1s, 2s, 4s, …);
    a non-transient error is re-raised on the first attempt.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1 (it counts attempts, not extra tries)")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(max_retries):
            try:
                result = client.embed(batch, model=model)  # type: ignore[attr-defined]
                break
            except Exception as exc:
                if attempt == max_retries - 1 or not _is_retryable(exc):
                    raise
                sleep(2.0**attempt)
        vecs = list(result.embeddings)
        if len(vecs) != len(batch):
            # Positional pairing is the whole contract (chunk i ↔ vector i). A short batch
            # would shift every later chunk onto its neighbour's vector — silently, since the
            # only downstream check is the TOTAL count, which a compensating batch satisfies.
            raise RuntimeError(
                f"embedder returned {len(vecs)} embeddings for {len(batch)} texts — refusing "
                f"to index misaligned vectors"
            )
        out.extend([float(x) for x in v] for v in vecs)
    return out


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install recall[voyage]` and VOYAGE_API_KEY."""

    def __init__(self, model: str = "voyage-3", api_key: str | None = None) -> None:
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
        self._dim = len(self._client.embed(["probe"], model=model).embeddings[0])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _embed_batched(self._client, self._model, texts)
