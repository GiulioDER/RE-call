"""Logging and metrics — the difference between "it works" and "someone can operate it".

Every diagnostic in this library used to be a `print()` to stderr. That is unusable in a service:
it cannot be levelled, filtered, routed, correlated or turned off, and under systemd it lands in
a journal beside everything else. Worse, nothing was *counted*, so the questions an operator
actually asks — how often does it abstain? what fraction of hits are superseded? is p99 latency
drifting? — had no answer short of reading logs by hand.

Two deliberately small pieces:

- **`get_logger`** — a standard `logging.Logger` under the `recall` namespace. The library never
  configures handlers itself (that is the application's job, and a library that calls
  `basicConfig` hijacks the host's logging); `configure_logging` is opt-in for the CLI and the
  MCP server, and can emit JSON for log shipping.
- **`METRICS`** — an in-process counter/histogram registry with no dependencies. Not a
  replacement for Prometheus: it is the source those exporters read from, and it makes the
  numbers available in-process (the MCP server surfaces them) without forcing a scrape endpoint
  or a client library on someone who only wants a CLI.

Metrics are deliberately cheap and bounded: counters are ints, and each histogram keeps a capped
ring of recent samples so a long-running process cannot grow without limit.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any

LOGGER_NAME = "recall"
#: Samples kept per histogram. Enough for a stable p99, small enough to be free.
HISTOGRAM_CAPACITY = 1024


def get_logger(name: str | None = None) -> logging.Logger:
    """A logger under the `recall` namespace. Handlers are the application's business."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, with any structured fields passed via `extra=`."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None, stream=None) -> None:
    """Attach ONE handler to the `recall` logger. Opt-in, for entry points only.

    Reads `RECALL_LOG_LEVEL` (default INFO) and `RECALL_LOG_FORMAT` (`text` or `json`).

    Writes to **stderr** by default and sets `propagate = False`. Both matter for the MCP server:
    stdout carries JSON-RPC, so a log line written there corrupts the protocol, and propagation
    would let a root handler installed by the host re-emit the same record onto stdout.
    """
    import sys

    logger = get_logger()
    level = (level or os.environ.get("RECALL_LOG_LEVEL") or "INFO").upper()
    fmt = (fmt or os.environ.get("RECALL_LOG_FORMAT") or "text").lower()
    logger.setLevel(level)
    for handler in list(logger.handlers):  # idempotent: re-configuring must not double-log
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        _JsonFormatter() if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False


class Metrics:
    """Thread-safe counters and histograms.

    Thread-safe because the MCP server runs tool bodies in a worker-thread pool, so every
    increment here is genuinely concurrent; `+=` on a dict entry is not atomic under a
    read-modify-write.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, deque[float]] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        inner = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{inner}}}"

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            samples = self._histograms.get(key)
            if samples is None:
                samples = self._histograms[key] = deque(maxlen=HISTOGRAM_CAPACITY)
            samples.append(value)

    @contextmanager
    def timer(self, name: str, **labels: str):
        """Record wall time in ms, INCLUDING when the body raises.

        A timer that only records on success hides exactly the slow path worth finding: the one
        that times out.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)

    def snapshot(self) -> dict[str, Any]:
        """Current values: counters as ints, histograms summarised as count/p50/p95/p99."""
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: sorted(v) for k, v in self._histograms.items()}
        summary: dict[str, Any] = {}
        for key, samples in histograms.items():
            if not samples:
                continue
            summary[key] = {
                "count": len(samples),
                "p50": _percentile(samples, 0.50),
                "p95": _percentile(samples, 0.95),
                "p99": _percentile(samples, 0.99),
            }
        return {"counters": counters, "histograms": summary}

    def reset(self) -> None:
        """Drop all state. For tests — a process should not need this."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return float("nan")
    idx = min(len(sorted_samples) - 1, int(q * len(sorted_samples)))
    return round(sorted_samples[idx], 3)


#: Process-wide registry. A module-level singleton because the alternative — threading a registry
#: through every constructor — is the reason libraries end up with no metrics at all.
METRICS = Metrics()
