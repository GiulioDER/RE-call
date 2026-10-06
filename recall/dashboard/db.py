"""The corpus database, read-only, for the dashboard's tenant, retrieval and system pages.

Every connection is opened read-only (`default_transaction_read_only=on`) with a statement
timeout, on top of whatever the role itself allows; the intended role is a SELECT-only login such
as `recall_dashboard_ro`. Nothing here writes, and no query interpolates a caller's string into
SQL: tenants and ids are bound parameters.

Searches appear only where the decision ledger is on (`RECALL_DECISION_LEDGER=1` on the server):
each one is a `search_decision` or `search_refusal` row in `recall_audit_events`, carrying the
query, the outcome and each hit's source and verdict, never chunk text.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

SEARCH_EVENTS = ("search_decision", "search_refusal")
STATEMENT_TIMEOUT_MS = 15_000


class DatabaseUnavailable(RuntimeError):
    """The dashboard cannot reach the corpus database; the message says what to check."""


@dataclass(frozen=True)
class DashboardDB:
    dsn: str

    @contextmanager
    def connect(self) -> Iterator[Any]:
        import psycopg

        try:
            connection = psycopg.connect(
                self.dsn,
                autocommit=True,
                options=f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}",
            )
        except psycopg.Error as exc:
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise DatabaseUnavailable(first) from exc
        try:
            yield connection
        finally:
            connection.close()


def _embedder(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        return {}
    embedder = identity.get("embedder") if "embedder" in identity else identity
    if not isinstance(embedder, dict):
        return {}
    return {k: embedder.get(k) for k in ("provider", "model", "dimension", "profile_id", "verified")}


def availability(db: DashboardDB) -> dict[str, Any]:
    """Whether the database answers, its version, and the schema level it is at."""
    with db.connect() as c:
        version = c.execute("select current_setting('server_version')").fetchone()[0]
        role = c.execute("select current_user").fetchone()[0]
        read_only = c.execute("select current_setting('default_transaction_read_only')").fetchone()[0]
        migrations = c.execute(
            "select target_table, max(version) from recall_schema_migrations where state = 'applied' "
            "group by target_table order by target_table"
        ).fetchall()
    return {"server_version": version, "role": role, "read_only": read_only == "on", "migrations": migrations}


def tenants(db: DashboardDB) -> list[dict[str, Any]]:
    """Every tenant with its active generation, embedder, size and calibration."""
    with db.connect() as c:
        rows = c.execute(
            "select t.tenant_id, t.active_generation_id, t.updated_at, g.state, g.activated_at, g.pipeline_identity "
            "from recall_tenant_state t left join recall_generations g "
            "on g.tenant_id = t.tenant_id and g.generation_id = t.active_generation_id order by t.tenant_id"
        ).fetchall()
        out = []
        for tenant, generation, updated, state, activated, identity in rows:
            chunks = sources = 0
            calibration: dict[str, Any] | None = None
            if generation:
                chunks, sources = c.execute(
                    "select count(*), count(distinct source_uri) from recall_chunks_v1 where tenant_id = %s and generation_id = %s",
                    (tenant, generation),
                ).fetchone()
                found = c.execute(
                    "select certified, threshold, separability, published_at, lifecycle_state from recall_calibrations "
                    "where tenant_id = %s and generation_id = %s order by published_at desc nulls last, created_at desc limit 1",
                    (tenant, generation),
                ).fetchone()
                if found:
                    calibration = dict(zip(("certified", "threshold", "separability", "published_at", "state"), found, strict=True))
            out.append(
                {
                    "tenant": tenant,
                    "generation": generation,
                    "state": state,
                    "activated_at": activated,
                    "updated_at": updated,
                    "embedder": _embedder(identity),
                    "chunks": chunks,
                    "sources": sources,
                    "calibration": calibration,
                }
            )
    return out


def generations(db: DashboardDB, tenant: str, limit: int = 15) -> list[dict[str, Any]]:
    with db.connect() as c:
        rows = c.execute(
            "select generation_id, state, created_at, activated_at, retired_at, failure_reason, created_by "
            "from recall_generations where tenant_id = %s order by created_at desc limit %s",
            (tenant, limit),
        ).fetchall()
    keys = ("generation", "state", "created_at", "activated_at", "retired_at", "failure_reason", "created_by")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def lifecycle_events(db: DashboardDB, tenant: str, limit: int = 100) -> list[dict[str, Any]]:
    """Index builds, promotions, calibrations, forgets: everything but searches, newest first."""
    with db.connect() as c:
        rows = c.execute(
            "select event_type, actor, generation_id, source_uri, created_at from recall_audit_events "
            "where tenant_id = %s and event_type <> all(%s) order by created_at desc limit %s",
            (tenant, list(SEARCH_EVENTS), limit),
        ).fetchall()
    keys = ("event", "actor", "generation", "source", "created_at")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def _summarise(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    hits = payload.get("hits") if isinstance(payload.get("hits"), list) else []
    return {
        "query": str(payload.get("query", "")),
        "outcome": payload.get("outcome") or ("abstained" if payload.get("abstained") else "answered"),
        "abstained": bool(payload.get("abstained")),
        "reason": payload.get("reason"),
        "trust_state": payload.get("trust_state"),
        "failure_code": payload.get("failure_code"),
        "verdict_counts": payload.get("verdict_counts") if isinstance(payload.get("verdict_counts"), dict) else {},
        "hits": [
            {
                "source": h.get("source"),
                "verdict": h.get("verdict"),
                "confidence": h.get("confidence"),
                "cosine": h.get("cosine"),
                "superseded_by": h.get("superseded_by"),
                "ord": h.get("ord"),
            }
            for h in hits
            if isinstance(h, dict)
        ],
        "k": payload.get("k"),
        "stage_ms": (payload.get("diagnostics") or {}).get("stage_ms") if isinstance(payload.get("diagnostics"), dict) else None,
    }


def searches(db: DashboardDB, tenant: str, limit: int = 100, contains: str = "") -> list[dict[str, Any]]:
    """Recorded searches, newest first, optionally only those whose query contains `contains`."""
    with db.connect() as c:
        rows = c.execute(
            "select event_id, event_type, actor, generation_id, payload, created_at from recall_audit_events "
            "where tenant_id = %s and event_type = any(%s) and (%s = '' or payload->>'query' ilike %s) "
            "order by created_at desc limit %s",
            (tenant, list(SEARCH_EVENTS), contains, f"%{contains}%", limit),
        ).fetchall()
    out = []
    for event_id, event_type, actor, generation, payload, created in rows:
        out.append({"event_id": event_id, "event": event_type, "actor": actor, "generation": generation,
                    "created_at": created, **_summarise(payload)})
    return out


def search(db: DashboardDB, tenant: str, event_id: str) -> dict[str, Any] | None:
    with db.connect() as c:
        row = c.execute(
            "select event_id, event_type, actor, generation_id, payload, created_at from recall_audit_events "
            "where tenant_id = %s and event_id = %s and event_type = any(%s)",
            (tenant, event_id, list(SEARCH_EVENTS)),
        ).fetchone()
    if row is None:
        return None
    event_id, event_type, actor, generation, payload, created = row
    return {"event_id": event_id, "event": event_type, "actor": actor, "generation": generation, "created_at": created,
            **_summarise(payload)}


def top_sources(db: DashboardDB, tenant: str, generation: str, limit: int = 40) -> list[dict[str, Any]]:
    with db.connect() as c:
        rows = c.execute(
            "select source_uri, count(*), max(indexed_at) from recall_chunks_v1 where tenant_id = %s and generation_id = %s "
            "group by source_uri order by max(indexed_at) desc limit %s",
            (tenant, generation, limit),
        ).fetchall()
    return [{"source": s, "chunks": n, "indexed_at": t} for s, n, t in rows]
