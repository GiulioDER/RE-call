"""Cross-cutting safety limits shared by storage and transport adapters."""

from __future__ import annotations


# Replay responses are deliberately bounded so an idempotency receipt cannot become an
# unbounded database or Redis payload. Keep the limit in one place so durable and cached replay
# paths cannot drift.
MAX_IDEMPOTENCY_RESULT_BYTES = 512 * 1024
IDEMPOTENCY_RECEIPT_RETENTION_SECONDS = 48 * 60 * 60

# Defaults of the Redis rate limiter, shared by its constructor, its environment builder and the
# documented settings schema so the three cannot disagree.
DEFAULT_REDIS_TIMEOUT_SECONDS = 0.25
DEFAULT_READ_FALLBACK_BUDGET = 3.0
DEFAULT_REDIS_MAX_CONNECTIONS = 32
DEFAULT_RATE_LIMIT_KEY_PREFIX = "recall:rate"
