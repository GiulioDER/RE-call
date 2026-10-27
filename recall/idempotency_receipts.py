"""The idempotency receipt rules every store applies, in one place.

`PgVectorStore` and the lite store keep receipts in different databases, but a client must not be
able to tell them apart: the same key hashes to the same identity, the same reuse of a key for
another operation or other arguments is refused, and the same response size is the replay limit.
Both stores call these helpers, so the rules cannot drift between the backends (CCA CODE-006 on
PR 912, where the lite store reached into a private `PgVectorStore` method and repeated the rest).
"""

from __future__ import annotations

from recall.constants import MAX_IDEMPOTENCY_RESULT_BYTES
from recall.errors import IdempotencyConflict
from recall.lineage import canonical_sha256

RESULT_TOO_LARGE_MESSAGE = "idempotent mutation result exceeds the 512 KiB replay limit"


def receipt_key_hash(idempotency_key: str, operation: str = "legacy") -> str:
    """Derive one bounded, nonrevealing audit identity from the request key.

    ``operation`` remains an accepted compatibility argument for callers that used the old
    operation-scoped identity. New receipts deliberately ignore it so one idempotency key
    cannot be reused for a different mutation operation.
    """
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise ValueError("idempotency_key must be a non-empty str")
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty str")
    return "idem_" + canonical_sha256({"idempotency_key": idempotency_key})


def check_receipt_matches(
    stored_operation: object,
    stored_fingerprint: object,
    *,
    operation: str,
    request_fingerprint: str | None,
) -> None:
    """Refuse a stored receipt that answers another operation or other arguments."""
    if operation != "legacy" and stored_operation != operation:
        raise IdempotencyConflict()
    if request_fingerprint is not None and stored_fingerprint != request_fingerprint:
        raise IdempotencyConflict()


def check_result_size(result: str) -> None:
    """Refuse a response too large to replay."""
    if len(result.encode("utf-8")) > MAX_IDEMPOTENCY_RESULT_BYTES:
        raise ValueError(RESULT_TOO_LARGE_MESSAGE)
