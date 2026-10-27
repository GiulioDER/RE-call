"""Storage constants shared by the store, the schema migrator and the installers.

A leaf: it imports nothing from the storage layer, so `recall.schema` and the installers can
name the default table and tenant without loading psycopg. `recall.store` re-exports every
name.
"""

from __future__ import annotations

import os

from recall.observability import get_logger

_log = get_logger("store")

#: Tenant assigned to rows written before tenancy existed, and the default for a
#: single-tenant deployment — so an upgrade changes nothing for an existing install.
DEFAULT_TENANT = "default"
#: Default table name. Named rather than repeated as a literal so a caller that needs to pass it
#: explicitly (see `recall_mcp.stores`) cannot drift from the constructor's default.
DEFAULT_TABLE = "chunks"
#: Postgres session variable the row-level-security policy reads. A custom GUC (it must contain
#: a dot) set per connection, so the policy compares against the connection's own tenant.
TENANT_GUC = "recall.tenant_id"

#: How long schema DDL may WAIT FOR A LOCK before giving up (ms). Not a bound on the work — the
#: HNSW build is deliberately unbounded, see `ensure_schema` — only on queueing. Short on purpose:
#: waiting on a lock is never progress, the DDL is idempotent and retried on the next open, and a
#: fast failure is diagnosable where an indefinite stall is not. Overridable because it can refuse
#: where the previous code waited: `0` restores the old unbounded wait.
DEFAULT_SCHEMA_LOCK_TIMEOUT_MS = 5000
#: PostgreSQL stores `lock_timeout` as a signed 32-bit int; anything larger is a parameter error.
_PG_MAX_INT = 2147483647


def _schema_lock_timeout_ms() -> int:
    """`RECALL_SCHEMA_LOCK_TIMEOUT_MS`, non-negative; anything malformed falls back to default.

    Read per call rather than at import so a long-lived process can be retuned without a reload —
    the same convention `RECALL_MAX_PRUNE_FRACTION` and `RECALL_INDEX_MAX_FILES` follow.
    """
    raw = os.environ.get("RECALL_SCHEMA_LOCK_TIMEOUT_MS")
    if raw is None:
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    try:
        value = int(raw)
    except ValueError:
        _log.warning("ignoring malformed RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r", raw)
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    if value < 0:
        _log.warning("ignoring negative RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r", raw)
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    if value > _PG_MAX_INT:
        # Above PostgreSQL's integer range `SET lock_timeout` raises InvalidParameterValue, which
        # would make ensure_schema fail outright — a knob for loosening a bound must not be able
        # to break the thing it loosens. Clamped, not rejected: the intent ("effectively never")
        # is unambiguous, and 24.8 days of lock wait is indistinguishable from it.
        _log.warning(
            "clamping RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r to %d (PostgreSQL integer range)",
            raw,
            _PG_MAX_INT,
        )
        return _PG_MAX_INT
    return value
