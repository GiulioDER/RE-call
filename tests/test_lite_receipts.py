"""Idempotency receipts and chunk lookup on a lite store, with `PgVectorStore`'s semantics.

Invariants and the failure each one catches:
- R1 a recorded result replays byte for byte; an unknown key is None.
- R2 the same key with another operation or other arguments raises IdempotencyConflict, so a
  client reusing a request id is told rather than served someone else's answer.
- R3 a receipt expires after the retention window; before that the first writer wins.
- R4 a result over 512 KiB is refused before any write.
- R5 `chunk_by_id` returns the chunk without its embedding, None when absent.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from recall.constants import IDEMPOTENCY_RECEIPT_RETENTION_SECONDS, MAX_IDEMPOTENCY_RESULT_BYTES
from recall.errors import IdempotencyConflict
from recall.lite import LiteStore
from recall.types import Chunk

T0 = datetime(2026, 10, 12, 12, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "memory.db", dim=4)


def test_r1_a_recorded_result_replays(store: LiteStore) -> None:
    """Red: `get_receipt` returning None always fails the replay assertion."""
    assert store.get_operation_receipt("k", operation="op", request_fingerprint="f") is None
    store.record_operation_receipt("k", '{"a": 1}', operation="op", request_fingerprint="f")
    assert store.get_operation_receipt("k", operation="op", request_fingerprint="f") == '{"a": 1}'


def test_r2_a_reused_key_conflicts(store: LiteStore) -> None:
    """Red: `get_receipt` without the fingerprint comparison raises nothing on the first check."""
    store.record_operation_receipt("k", "x", operation="op", request_fingerprint="f")
    with pytest.raises(IdempotencyConflict):
        store.get_operation_receipt("k", operation="op", request_fingerprint="other")
    with pytest.raises(IdempotencyConflict):
        store.get_operation_receipt("k", operation="other-op", request_fingerprint="f")


def test_r3_expiry_and_first_writer_wins(store: LiteStore) -> None:
    """Red: a receipt stored without the retention offset (expires_at = now) is gone at once;
    a plain INSERT instead of INSERT OR IGNORE raises on the second write."""
    store.record_operation_receipt("k", "first", operation="op", request_fingerprint="f", now=T0)
    store.record_operation_receipt("k", "second", operation="op", request_fingerprint="f", now=T0)
    assert store.get_operation_receipt("k", operation="op", request_fingerprint="f", now=T0) == "first"
    late = T0 + timedelta(seconds=IDEMPOTENCY_RECEIPT_RETENTION_SECONDS + 1)
    assert store.get_operation_receipt("k", operation="op", request_fingerprint="f", now=late) is None


def test_r4_an_oversized_result_is_refused(store: LiteStore) -> None:
    """Red: `record_receipt` without the size check raises nothing."""
    with pytest.raises(ValueError, match="512 KiB"):
        store.record_operation_receipt("k", "x" * (MAX_IDEMPOTENCY_RESULT_BYTES + 1), operation="op")


def test_r5_chunk_by_id(store: LiteStore) -> None:
    """Red: `chunk_by_id` querying by source instead of id returns None for the stored id."""
    store.upsert(
        [Chunk(id="c1", source="/m/a.md", text="hello", metadata={"file": "a.md"})],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    chunk = store.chunk_by_id("c1")
    assert chunk is not None and chunk.text == "hello" and chunk.metadata["file"] == "a.md"
    assert store.chunk_by_id("missing") is None
    with pytest.raises(ValueError):
        store.chunk_by_id("")
