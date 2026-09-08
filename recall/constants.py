"""Cross-cutting safety limits shared by storage and transport adapters."""

from __future__ import annotations


# Replay responses are deliberately bounded so an idempotency receipt cannot become an
# unbounded database or Redis payload. Keep the limit in one place so durable and cached replay
# paths cannot drift.
MAX_IDEMPOTENCY_RESULT_BYTES = 512 * 1024
IDEMPOTENCY_RECEIPT_RETENTION_SECONDS = 48 * 60 * 60
