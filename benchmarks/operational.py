"""Operational measurements kept separate from retrieval quality claims."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import Any, Callable


@dataclass(frozen=True)
class OperationalMeasurement:
    """Startup and availability measurements kept separate from retrieval quality claims."""

    lexical_ready_ms: float | None = None
    semantic_ready_ms: float | None = None
    snapshot_load_ms: float | None = None
    cold_start_ms: float | None = None
    warm_start_ms: float | None = None
    atomic_cutover_ok: bool | None = None
    recovery_ok: bool | None = None
    recovery_ms: float | None = None
    rss_bytes: int | None = None
    memory_usage_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_family": "operational",
            "retrieval_quality_claim": False,
            "metrics": {
                "lexical_ready_ms": self.lexical_ready_ms,
                "semantic_ready_ms": self.semantic_ready_ms,
                "snapshot_load_ms": self.snapshot_load_ms,
                "cold_start_ms": self.cold_start_ms,
                "warm_start_ms": self.warm_start_ms,
                "atomic_cutover_ok": self.atomic_cutover_ok,
                "recovery_ok": self.recovery_ok,
                "recovery_ms": self.recovery_ms,
                "rss_bytes": self.rss_bytes,
                "memory_usage_bytes": self.memory_usage_bytes
                if self.memory_usage_bytes is not None
                else self.rss_bytes,
            },
        }


def _rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1024 if value < 10_000_000 else 1)
    except (ImportError, AttributeError, OSError):
        return None


def measure_callable(callback: Callable[[], Any]) -> tuple[Any, float]:
    """Run one callback and return its value plus elapsed milliseconds."""

    started = time.perf_counter()
    result = callback()
    return result, round((time.perf_counter() - started) * 1000.0, 3)


def measure_snapshot_and_warmup(
    snapshot_loader: Callable[[], Any],
    warmup: Callable[[], Any],
) -> OperationalMeasurement:
    """Measure startup paths without producing a retrieval quality result."""
    del_result, cold_ms = measure_callable(snapshot_loader)
    del del_result
    gc.collect()
    _warm_result, warm_ms = measure_callable(warmup)
    return OperationalMeasurement(
        snapshot_load_ms=cold_ms,
        cold_start_ms=cold_ms,
        warm_start_ms=warm_ms,
        rss_bytes=_rss_bytes(),
    )


def measure_staged_indexing(
    lexical_ready: Callable[[], Any],
    semantic_ready: Callable[[], Any],
    snapshot_loader: Callable[[], Any],
    warmup: Callable[[], Any],
    *,
    atomic_cutover: Callable[[], bool] | None = None,
    recovery: Callable[[], bool] | None = None,
) -> OperationalMeasurement:
    """Measure staged readiness, snapshot startup, cutover, and recovery independently.

    Callbacks should perform only operational work. This helper never receives a quality score and
    its result is explicitly marked as an operational claim family by ``as_dict``.
    """
    _lexical_result, lexical_ms = measure_callable(lexical_ready)
    _semantic_result, semantic_ms = measure_callable(semantic_ready)
    _snapshot_result, snapshot_ms = measure_callable(snapshot_loader)
    del _lexical_result, _semantic_result, _snapshot_result
    gc.collect()
    _warm_result, warm_ms = measure_callable(warmup)
    del _warm_result
    cutover_ok = atomic_cutover() if atomic_cutover is not None else None
    recovery_ms: float | None = None
    if recovery is not None:
        recovery_result, recovery_ms = measure_callable(recovery)
        recovery_ok = bool(recovery_result)
    else:
        recovery_ok = None
    rss_bytes = _rss_bytes()
    return OperationalMeasurement(
        lexical_ready_ms=lexical_ms,
        semantic_ready_ms=semantic_ms,
        snapshot_load_ms=snapshot_ms,
        cold_start_ms=snapshot_ms,
        warm_start_ms=warm_ms,
        atomic_cutover_ok=cutover_ok,
        recovery_ok=recovery_ok,
        recovery_ms=recovery_ms,
        rss_bytes=rss_bytes,
        memory_usage_bytes=rss_bytes,
    )


def attach_operational_metrics(
    artifact: dict[str, Any], measurement: OperationalMeasurement
) -> dict[str, Any]:
    """Attach operational data under its own claim family without touching quality aggregates."""
    result = dict(artifact)
    result["operational_metrics"] = measurement.as_dict()
    return result


def run_operational_benchmark(
    lexical_ready: Callable[[], Any],
    semantic_ready: Callable[[], Any],
    snapshot_loader: Callable[[], Any],
    warmup: Callable[[], Any],
    *,
    atomic_cutover: Callable[[], bool] | None = None,
    recovery: Callable[[], bool] | None = None,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the dedicated operational benchmark and return a publishable separate artifact."""
    measurement = measure_staged_indexing(
        lexical_ready,
        semantic_ready,
        snapshot_loader,
        warmup,
        atomic_cutover=atomic_cutover,
        recovery=recovery,
    )
    return {
        "benchmark": "recall-operational-v1",
        "configuration": dict(configuration or {}),
        "operational_metrics": measurement.as_dict(),
    }


__all__ = [
    "OperationalMeasurement",
    "attach_operational_metrics",
    "measure_callable",
    "measure_snapshot_and_warmup",
    "measure_staged_indexing",
    "run_operational_benchmark",
]
