"""Idempotency receipts on a lite store, with `PgVectorStore`'s semantics.

A completed mutation's response is kept for `IDEMPOTENCY_RECEIPT_RETENTION_SECONDS`, under the
same key hash PostgreSQL uses, so a retried request replays the identical bytes and a key reused
for another operation or other arguments is refused rather than run again.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from recall.constants import IDEMPOTENCY_RECEIPT_RETENTION_SECONDS, MAX_IDEMPOTENCY_RESULT_BYTES
from recall.errors import IdempotencyConflict
from recall.store import PgVectorStore

if TYPE_CHECKING:
    from recall.lite.store import LiteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_idempotency_receipts (
    tenant_id TEXT NOT NULL,
    idempotency_key_hash TEXT NOT NULL,
    operation TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    result TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, idempotency_key_hash)
);
CREATE INDEX IF NOT EXISTS recall_idempotency_receipts_expiry_idx
    ON recall_idempotency_receipts (tenant_id, expires_at);
"""


def _stamp(moment: datetime) -> str:
    """Fixed width, so text order is time order."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _ensure(store: LiteStore) -> None:
    # Outside `_write()`: `executescript` commits any open transaction first.
    with store._lock:
        store._conn.executescript(_SCHEMA)


def get_receipt(
    store: LiteStore,
    idempotency_key: str,
    *,
    operation: str = "legacy",
    request_fingerprint: str | None = None,
    now: datetime | None = None,
) -> str | None:
    """The stored response for `idempotency_key`, None when absent or expired."""
    key_hash = PgVectorStore._operation_receipt_event_id(idempotency_key, operation)
    _ensure(store)
    rows = store._metadata_values(
        "SELECT result, operation, request_fingerprint FROM recall_idempotency_receipts "
        "WHERE tenant_id = ? AND idempotency_key_hash = ? AND expires_at > ?",
        (store.tenant, key_hash, _stamp(now or datetime.now(UTC))),
    )
    if not rows or rows[0][0] is None:
        return None
    result, stored_operation, stored_fingerprint = rows[0]
    if operation != "legacy" and stored_operation != operation:
        raise IdempotencyConflict()
    if request_fingerprint is not None and stored_fingerprint != request_fingerprint:
        raise IdempotencyConflict()
    return str(result)


def record_receipt(
    store: LiteStore,
    idempotency_key: str,
    result: str,
    *,
    operation: str = "legacy",
    request_fingerprint: str | None = None,
    now: datetime | None = None,
) -> None:
    """Persist a completed response; the first writer of a key wins until it expires."""
    if request_fingerprint is None:
        request_fingerprint = ""
    if len(result.encode("utf-8")) > MAX_IDEMPOTENCY_RESULT_BYTES:
        raise ValueError("idempotent mutation result exceeds the 512 KiB replay limit")
    key_hash = PgVectorStore._operation_receipt_event_id(idempotency_key, operation)
    moment = now or datetime.now(UTC)
    _ensure(store)
    with store._write():
        store._conn.execute(
            "DELETE FROM recall_idempotency_receipts WHERE tenant_id = ? AND expires_at <= ?",
            (store.tenant, _stamp(moment)),
        )
        store._conn.execute(
            "INSERT OR IGNORE INTO recall_idempotency_receipts "
            "(tenant_id, idempotency_key_hash, operation, request_fingerprint, result, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                store.tenant, key_hash, operation, request_fingerprint, result,
                _stamp(moment + timedelta(seconds=IDEMPOTENCY_RECEIPT_RETENTION_SECONDS)),
            ),
        )
