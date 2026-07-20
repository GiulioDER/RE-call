from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from recall.embeddings import Embedder
from recall.rerank import Reranker
from recall.types import ScoredChunk

_T = TypeVar("_T")


@dataclass
class TimingStats:
    """Accumulated wall-clock latency for one wrapped operation.

    Cost is reported as latency + call count, NOT a modeled dollar figure: provider pricing
    lives outside this codebase and drifts, whereas wall time and call count are measured facts.
    For a cloud embedder, ``calls`` (batches) × batch size is the lever on the actual bill.
    """

    calls: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0

    def record(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        self.last_ms = elapsed_ms

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


def timed_call(stats: TimingStats, fn: Callable[[], _T]) -> _T:
    """Run ``fn()``, recording its wall time into ``stats``; returns the result unchanged.

    Records even when ``fn`` raises (the call still cost time), then lets the exception propagate.
    """
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        stats.record((time.perf_counter() - t0) * 1000.0)


class TimedEmbedder:
    """Embedder wrapper that records ``embed`` latency without changing the interface.

    Additive by construction: it satisfies the Embedder protocol (``dim``/``name``/``embed``), so
    it drops in wherever an embedder is expected and the timing is read off ``.stats`` afterward.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.stats = TimingStats()

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return timed_call(self.stats, lambda: self._inner.embed(texts))


class TimedReranker:
    """Reranker wrapper that records ``rerank`` latency without changing the interface."""

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self.stats = TimingStats()

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return timed_call(self.stats, lambda: self._inner.rerank(query, hits))
