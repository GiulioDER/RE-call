"""Measured serving SLOs and alert thresholds for retrieval profiles.

These are deployment policy values derived from the supplementary quality load test. They are
separate from ``RetrievalProfile.latency_budget_ms``: that field bounds admission wait, while
these values describe client-visible served latency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalSLO:
    """Operational SLO and alert policy for one process-scoped profile."""

    profile: str
    p95_ms: int
    p99_ms: int
    max_offered_concurrency: int
    max_rss_bytes: int
    max_error_rate: float
    alert_window_minutes: int = 5


# Measured on 2026-09-07 with the pinned local bge-small embedder and ms-marco reranker:
# offered concurrency 4 was inside both ceilings; 8 exceeded p95 and 12 widened the breach.
# 1.25 GiB leaves headroom above the measured 988.3 MiB peak for normal process variance.
QUALITY_RETRIEVAL_SLO = RetrievalSLO(
    profile="quality",
    p95_ms=2_000,
    p99_ms=2_200,
    max_offered_concurrency=4,
    max_rss_bytes=1_280 * 1024 * 1024,
    max_error_rate=0.01,
)


__all__ = ["QUALITY_RETRIEVAL_SLO", "RetrievalSLO"]
