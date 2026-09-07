"""Bounded retry policy for Aurora failover without replaying unsafe writes."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from recall.observability import METRICS, get_logger

_log = get_logger("ops.db_retry")
_T = TypeVar("_T")


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {
        "OperationalError",
        "InterfaceError",
        "AdminShutdown",
        "ConnectionFailure",
        "SerializationFailure",
        "DeadlockDetected",
        "ReadOnlySqlTransaction",
    }


def run_read(operation: Callable[[], _T], *, attempts: int = 3, backoff_seconds: float = 0.1) -> _T:
    """Retry only a caller supplied read operation after a transient connection failure."""
    if attempts < 1 or backoff_seconds < 0:
        raise ValueError("attempts must be positive and backoff_seconds must be nonnegative")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient(exc) or attempt + 1 >= attempts:
                raise
            METRICS.increment("recall_db_retry_total", operation="read")
            _log.warning("retrying transient read failure", extra={"error_type": type(exc).__name__})
            time.sleep(backoff_seconds * (2**attempt))
    raise AssertionError("unreachable")


def run_write(
    operation: Callable[[], _T],
    *,
    idempotency_key: str | None,
    attempts: int = 2,
    backoff_seconds: float = 0.1,
) -> _T:
    """Retry a write only when the caller proves it is idempotent."""
    if not idempotency_key:
        raise ValueError("idempotency_key is required before retrying a write")
    return run_read(operation, attempts=attempts, backoff_seconds=backoff_seconds)
