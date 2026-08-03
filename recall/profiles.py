"""Process-level retrieval profiles and bounded admission control."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Literal

RetrievalProfileName = Literal["legacy", "fast", "quality"]


@dataclass(frozen=True)
class RetrievalProfile:
    name: RetrievalProfileName
    candidate_k: int
    returned_k: int
    reranker: bool
    latency_budget_ms: int
    max_concurrency: int = 4
    queue_capacity: int = 16
    inference_threads: int | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_k, "candidate_k"),
            (self.returned_k, "returned_k"),
            (self.latency_budget_ms, "latency_budget_ms"),
            (self.max_concurrency, "max_concurrency"),
            (self.queue_capacity, "queue_capacity"),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")


FAST_PROFILE = RetrievalProfile("fast", 20, 5, False, 250)
QUALITY_PROFILE = RetrievalProfile("quality", 20, 5, True, 1500)
LEGACY_PROFILE = RetrievalProfile("legacy", 20, 5, False, 0x7FFFFFFF)


def resolve_retrieval_profile(env: dict[str, str] | None = None) -> RetrievalProfile:
    """Resolve a fixed process profile while preserving the legacy rerank switch."""
    values = os.environ if env is None else env
    selected = values.get("RECALL_RETRIEVAL_PROFILE", "").strip().lower()
    legacy_rerank = values.get("RECALL_RERANK", "").strip().lower()
    if selected not in {"", "fast", "quality"}:
        raise ValueError("RECALL_RETRIEVAL_PROFILE must be 'fast' or 'quality'")
    if selected and legacy_rerank:
        enabled = legacy_rerank in {"1", "true", "yes", "on"}
        if enabled != (selected == "quality"):
            raise ValueError(
                "RECALL_RETRIEVAL_PROFILE conflicts with the legacy RECALL_RERANK setting"
            )
    base = QUALITY_PROFILE if selected == "quality" else FAST_PROFILE if selected == "fast" else LEGACY_PROFILE

    def _positive(name: str, default: int) -> int:
        raw = values.get(name)
        if raw is None:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if parsed < 1:
            raise ValueError(f"{name} must be positive")
        return parsed

    return RetrievalProfile(
        name=base.name,
        candidate_k=base.candidate_k,
        returned_k=base.returned_k,
        reranker=base.reranker,
        latency_budget_ms=base.latency_budget_ms,
        max_concurrency=_positive("RECALL_SEARCH_CONCURRENCY", base.max_concurrency),
        queue_capacity=_positive("RECALL_SEARCH_QUEUE", base.queue_capacity),
        inference_threads=(
            _positive("RECALL_RERANK_THREADS", 1) if base.reranker else None
        ),
    )


class RetrievalOverloaded(RuntimeError):
    """The process has no safe capacity to begin another retrieval."""


class RetrievalAdmission:
    """A bounded process-local queue acquired before query embedding begins."""

    def __init__(self, profile: RetrievalProfile) -> None:
        self._slots = threading.BoundedSemaphore(
            profile.max_concurrency + profile.queue_capacity
        )
        self._running = threading.BoundedSemaphore(profile.max_concurrency)

    def __enter__(self) -> None:
        if not self._slots.acquire(blocking=False):
            raise RetrievalOverloaded("retrieval queue is full")
        self._running.acquire()

    def __exit__(self, *exc: object) -> None:
        self._running.release()
        self._slots.release()
