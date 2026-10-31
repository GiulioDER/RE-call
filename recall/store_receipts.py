"""`PgVectorStore`'s audit events and idempotency receipts.

A mixin of `recall.store.PgVectorStore` (architecture step S7d); it owns no state of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from recall.constants import IDEMPOTENCY_RECEIPT_RETENTION_SECONDS
from recall.db_constants import (  # noqa: F401  # re-exported: public names of this module
    _PG_MAX_INT,
    DEFAULT_SCHEMA_LOCK_TIMEOUT_MS,
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    TENANT_GUC,
    _schema_lock_timeout_ms,
)
from recall.idempotency_receipts import (
    ANY_OPERATION,
    check_receipt_matches,
    check_result_size,
    receipt_key_hash,
)
from recall.lineage import canonical_sha256
from recall.observability import get_logger
from recall.supersession import (  # noqa: F401  # re-exported: public names of this module
    SUPERSEDES_TARGET_ROWS_SQL,
    EdgeCandidates,
    SupersessionGraph,
    _basename,
    _resolve_rows,
    resolve_supersession,
    resolve_supersession_candidates,
    resolve_supersession_graph,
    supersession_rows_sql,
)
from recall.types import Chunk

_log = get_logger("store")  # the store's own logger name, which tests filter on

_T = TypeVar("_T")


class ReceiptsAuditMixin:
    if TYPE_CHECKING:  # what this mixin uses of the store it is mixed into
        _tenant: str
        _table: str
        _supersession_cache: tuple | None
        _supersession_scans: int
        _dependency_projection_cache: tuple | None

        def _with_retry(self, op: Callable[["psycopg.Connection"], _T]) -> _T: ...

        def _borrowed(self) -> "AbstractContextManager[psycopg.Connection]": ...

        def _upsert_in(
            self, conn: "psycopg.Connection", chunks: list[Chunk], embeddings: list[list[float]]
        ) -> None: ...

        def _scrub_sparse_rows(
            self, conn: "psycopg.Connection", chunk_table: str, chunk_ids: list[str]
        ) -> int: ...

        @property
        def record_scope(self) -> str: ...

        @property
        def generation_id(self) -> str: ...

    def append_audit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        generation_id: str | None = None,
        source_uri: str | None = None,
        actor: str = "serving",
        event_id: str | None = None,
    ) -> str:
        """Append one event to this tenant's audit ledger (``recall_audit_events``).

        The write side (generation and calibration lifecycle) has appended to that table since
        migration 0008 through its own repositories; this is the read side's pen, used by
        `recall.decision_ledger` to witness search decisions. It runs on the store's own
        tenant-scoped connections, so the row satisfies the table's RLS policy exactly when
        every other statement here does. Append is the ONLY verb on this surface — no update or
        delete exists, because a ledger that can be edited in place is a draft, not a record.

        ``ON CONFLICT DO NOTHING`` makes the append idempotent PER EVENT ID, which is not a
        loophole in append-only (nothing is ever updated) but the honest answer to the
        indeterminate-commit window: `_with_retry` re-runs the INSERT when the connection dies
        after the server committed but before the client heard, and the pre-generated id means
        the collision can only be this call's own earlier success. Without it, that window
        reported a failure for a row that exists, so the failure counter disagreed with the
        table. A caller passing an explicit ``event_id`` twice gets one row and two successes,
        by the same rule.

        The write inherits the store's single reconnect-retry, so under a degraded connection
        it can cost the caller up to two attempts of latency before failing — a deliberate
        trade for a best-effort caller that shares the search path's connection anyway.

        Raises on failure (a schema predating 0008, a connection dead after the retry): whether
        the write is load-bearing is the CALLER's decision, and `DecisionLedger` decides it is
        not by catching here.
        """
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty str")
        event_id = event_id or f"evt_{uuid4().hex}"

        def _op(conn: "psycopg.Connection") -> None:
            conn.execute(
                "INSERT INTO recall_audit_events "
                "(tenant_id, event_id, event_type, actor, generation_id, source_uri, payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, event_id) DO NOTHING",
                (
                    self._tenant,
                    event_id,
                    event_type,
                    actor,
                    generation_id,
                    source_uri,
                    Jsonb(dict(payload)),
                ),
            )

        self._with_retry(_op)
        return event_id

    def recorded_memo_journals(self, sources: Sequence[str]) -> set[str]:
        """The `sources` a forget recorded deleting as assistant memos (`forget_memo_journal`)."""
        if not sources:
            return set()

        def _op(conn: "psycopg.Connection") -> set[str]:
            rows = conn.execute(
                "SELECT DISTINCT source_uri FROM recall_audit_events WHERE tenant_id = %s "
                "AND event_type = 'forget_memo_journal' AND source_uri = ANY(%s)",
                (self._tenant, list(sources)),
            ).fetchall()
            return {str(source) for (source,) in rows if source}

        return self._with_retry(_op)

    def forgotten_source_aliases(self, sources: Sequence[str]) -> set[str]:
        """Recover card names recorded before these exact sources were forgotten."""
        if not sources:
            return set()

        def _op(conn: "psycopg.Connection") -> set[str]:
            rows = conn.execute(
                "SELECT payload FROM recall_audit_events WHERE tenant_id = %s "
                "AND event_type = 'forget_source_aliases' AND source_uri = ANY(%s)",
                (self._tenant, list(sources)),
            ).fetchall()
            return {name for (payload,) in rows for name in payload.get("aliases", ())
                    if isinstance(name, str)}

        return self._with_retry(_op)

    @staticmethod
    def _operation_receipt_event_id(idempotency_key: str, operation: str = ANY_OPERATION) -> str:
        """Derive one bounded, nonrevealing audit identity from the request key.

        Kept for existing callers; the rule is `recall.idempotency_receipts.receipt_key_hash`,
        which the lite store uses too.
        """
        return receipt_key_hash(idempotency_key, operation)

    def get_operation_receipt(
        self,
        idempotency_key: str,
        *,
        operation: str = ANY_OPERATION,
        request_fingerprint: str | None = None,
    ) -> str | None:
        """Return the durable response for an idempotent operation, when one exists."""
        event_id = self._operation_receipt_event_id(idempotency_key, operation)

        def _op(conn: "psycopg.Connection") -> str | None:
            row = conn.execute(
                "SELECT result, operation, request_fingerprint "
                "FROM recall_idempotency_receipts "
                "WHERE tenant_id = %s AND idempotency_key_hash = %s "
                "AND expires_at > clock_timestamp()",
                (self._tenant, event_id),
            ).fetchone()
            if not row or row[0] is None:
                return None
            check_receipt_matches(
                row[1], row[2], operation=operation, request_fingerprint=request_fingerprint
            )
            return str(row[0])

        return self._with_retry(_op)

    @contextmanager
    def operation_lock(self, key: str) -> "Iterator[None]":
        """Serialize one named operation across processes using a PostgreSQL advisory lock."""
        if not isinstance(key, str) or not key:
            raise ValueError("operation lock key must be a non-empty string")
        digest = canonical_sha256({"tenant": self._tenant, "operation_key": key})
        lock_id = int(digest[:16], 16)
        if lock_id >= 2**63:
            lock_id -= 2**64
        with self._borrowed() as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
            try:
                yield
            finally:
                row = conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,)).fetchone()
                if not row or row[0] is not True:
                    _log.error("PostgreSQL reported a hosted operation lock was not held")

    def record_operation_receipt(
        self,
        idempotency_key: str,
        result: str,
        *,
        operation: str = ANY_OPERATION,
        request_fingerprint: str | None = None,
    ) -> None:
        """Persist a completed mutation response before any cache response is attempted."""
        if request_fingerprint is None:
            request_fingerprint = ""
        check_result_size(result)
        key_hash = self._operation_receipt_event_id(idempotency_key, operation)

        def _op(conn: "psycopg.Connection") -> None:
            conn.execute(
                "DELETE FROM recall_idempotency_receipts "
                "WHERE tenant_id = %s AND expires_at <= clock_timestamp()",
                (self._tenant,),
            )
            conn.execute(
                "INSERT INTO recall_idempotency_receipts "
                "(tenant_id, idempotency_key_hash, operation, request_fingerprint, result, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, clock_timestamp() + (%s * interval '1 second')) "
                "ON CONFLICT (tenant_id, idempotency_key_hash) DO NOTHING",
                (
                    self._tenant,
                    key_hash,
                    operation,
                    request_fingerprint,
                    result,
                    IDEMPOTENCY_RECEIPT_RETENTION_SECONDS,
                ),
            )

        self._with_retry(_op)
