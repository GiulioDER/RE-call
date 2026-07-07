from __future__ import annotations

from datetime import datetime, timedelta

from recall.types import StalenessReport

#: Default cosine-similarity floor: if every candidate scores below this, the corpus is
#: treated as lacking a relevant answer. Single source of truth for the gap heuristic.
DEFAULT_GAP_THRESHOLD = 0.50


def gap_warning(scores: list[float], threshold: float = DEFAULT_GAP_THRESHOLD) -> bool:
    """True when the corpus probably lacks a relevant answer.

    Fires when every candidate similarity score is below `threshold`
    (default 0.50 cosine). An empty candidate set is also a gap.
    """
    if not scores:
        return True
    return max(scores) < threshold


def staleness(
    newest_indexed_at: datetime | None,
    now: datetime,
    max_age: timedelta,
) -> StalenessReport:
    """Report whether the most recent indexing is older than `max_age`."""
    if newest_indexed_at is None:
        return StalenessReport(stale=True, newest_indexed_at=None, age=None, max_age=max_age)
    age = now - newest_indexed_at
    return StalenessReport(
        stale=age > max_age,
        newest_indexed_at=newest_indexed_at,
        age=age,
        max_age=max_age,
    )
