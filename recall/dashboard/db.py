"""The corpus database, read-only, for the dashboard's tenant, retrieval, system and review pages.

Every connection is opened read-only (`default_transaction_read_only=on`) with a statement
timeout, on top of whatever the role itself allows; the intended role is a SELECT-only login such
as `recall_dashboard_ro`. Nothing here writes, and no query interpolates a caller's string into
SQL: tenants and ids are bound parameters.

Searches appear only where the decision ledger is on (`RECALL_DECISION_LEDGER=1` on the server):
each one is a `search_decision` or `search_refusal` row in `recall_audit_events`, carrying the
query, the outcome and each hit's source and verdict, never chunk text.

A failure while a statement runs (a statement timeout, a missing table, a refused grant) is raised
as `DatabaseUnavailable` too, so every page that degrades on an unreachable database degrades the
same way on a failed query, rather than dropping the request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from recall.errors import RecallError
from recall.stale_reports import STALE_REPORT_EVENT, STALE_REPORT_FIELDS

SEARCH_EVENTS = ("search_decision", "search_refusal")
STATEMENT_TIMEOUT_MS = 15_000


class DatabaseUnavailable(RuntimeError, RecallError):
    """The dashboard cannot reach the corpus database; the message says what to check."""


@dataclass(frozen=True)
class DashboardDB:
    dsn: str

    @property
    def lite(self) -> bool:
        """A `sqlite:///` store: every query below is answered by `recall.dashboard.lite_db`."""
        from recall.lite import is_lite_dsn

        return is_lite_dsn(self.dsn)

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
        except psycopg.Error as exc:
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise DatabaseUnavailable(f"{type(exc).__name__}: {first}") from exc
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.availability(db.dsn)
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.tenants(db.dsn)
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.generations(db.dsn, tenant, limit)
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.lifecycle_events(db.dsn, tenant, limit)
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.searches(db.dsn, tenant, limit, contains)
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
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.search(db.dsn, tenant, event_id)
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


USE_REPORT_EVENT = "use_report"

# One row per (search, memory): a memory with three chunks in one result was retrieved once, at
# the rank of its best chunk. The rank is the hit's place in the recorded list, from 1.
_RETRIEVALS = """
with hit as (
    select e.event_id, e.created_at, h.value->>'source' as source, h.ordinality as rank,
           h.value->>'verdict' as verdict
    from recall_audit_events e, jsonb_array_elements(e.payload->'hits') with ordinality h
    where e.tenant_id = %s and e.event_type = 'search_decision'
), per_search as (
    select event_id, source, min(rank) as rank, max(created_at) as at,
           bool_or(verdict = 'superseded') as superseded, bool_or(verdict = 'low_confidence') as low,
           bool_or(verdict = 'ok') as ok
    from hit where source is not null group by event_id, source
)
select source, count(*), count(*) filter (where rank = 1), max(at),
       count(*) filter (where ok), count(*) filter (where superseded), count(*) filter (where low)
from per_search group by source
"""

_REPORTED = """
select m.value #>> '{}' as source, %s::text as role, e.payload->>'effect' as effect,
       e.payload->>'task_succeeded' as succeeded, count(*), max(e.created_at)
from recall_audit_events e, jsonb_array_elements(e.payload->(%s::text)) m
where e.tenant_id = %s and e.event_type = 'use_report'
group by 1, 2, 3, 4
"""


def control(db: DashboardDB, tenant: str) -> dict[str, Any]:
    """Per memory: how often search returned it, and what agents reported it did for their task.

    Retrieval comes from the decision ledger; "used", "wrong" and "helped" come only from
    `use_report` rows, which agents write with `recall_report_use`. The summary says how many
    searches there were beside how many reports, so a reader can see how much is testimony.
    """
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.control(db.dsn, tenant)
    memos: dict[str, dict[str, Any]] = {}

    def memo(source: str) -> dict[str, Any]:
        return memos.setdefault(source, {
            "source": source, "retrieved": 0, "first": 0, "last_retrieved": None, "ok": 0, "superseded": 0,
            "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
            "succeeded": 0, "failed": 0, "last_reported": None,
        })

    with db.connect() as c:
        for source, n, first, last, ok, superseded, low in c.execute(_RETRIEVALS, (tenant,)).fetchall():
            row = memo(source)
            row.update(retrieved=n, first=first, last_retrieved=last, ok=ok, superseded=superseded, low_confidence=low)
        for role in ("used", "wrong"):
            for source, _, effect, succeeded, n, last in c.execute(_REPORTED, (role, role, tenant)).fetchall():
                row = memo(source)
                row[role] += n
                if role == "used" and effect in ("helped", "no_difference"):
                    row[effect] += n
                if role == "wrong" and effect == "misled":
                    row["misled"] += n
                if role == "used" and succeeded in ("true", "false"):
                    row["succeeded" if succeeded == "true" else "failed"] += n
                if last and (row["last_reported"] is None or last > row["last_reported"]):
                    row["last_reported"] = last
        searches, answered = c.execute(
            "select count(*), count(*) filter (where event_type = 'search_decision') from recall_audit_events "
            "where tenant_id = %s and event_type = any(%s)", (tenant, list(SEARCH_EVENTS)),
        ).fetchone()
        reports, helped, no_difference, misled, succeeded, failed = c.execute(
            "select count(*), count(*) filter (where payload->>'effect' = 'helped'), "
            "count(*) filter (where payload->>'effect' = 'no_difference'), count(*) filter (where payload->>'effect' = 'misled'), "
            "count(*) filter (where payload->>'task_succeeded' = 'true'), count(*) filter (where payload->>'task_succeeded' = 'false') "
            "from recall_audit_events where tenant_id = %s and event_type = 'use_report'", (tenant,),
        ).fetchone()
    summary = {
        "searches": searches, "answered": answered, "reports": reports, "helped": helped,
        "no_difference": no_difference, "misled": misled, "succeeded": succeeded, "failed": failed,
        "memos_retrieved": sum(1 for m in memos.values() if m["retrieved"]),
    }
    ordered = sorted(memos.values(), key=lambda m: (-m["retrieved"], -m["used"], m["source"]))
    return {"summary": summary, "memos": ordered}


def use_reports(db: DashboardDB, tenant: str, limit: int = 50, source: str = "") -> list[dict[str, Any]]:
    """Agent reports, newest first; with `source`, only those naming that memory as used or wrong."""
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.use_reports(db.dsn, tenant, limit, source)
    with db.connect() as c:
        rows = c.execute(
            "select event_id, actor, payload, created_at from recall_audit_events "
            "where tenant_id = %s and event_type = 'use_report' "
            "and (%s::text = '' or payload->'used' ? %s::text or payload->'wrong' ? %s::text) "
            "order by created_at desc limit %s",
            (tenant, source, source, source, limit),
        ).fetchall()
    out = []
    for event_id, actor, payload, created in rows:
        payload = payload if isinstance(payload, dict) else {}
        out.append({
            "event_id": event_id, "actor": actor, "created_at": created,
            "task": str(payload.get("task", "")), "effect": payload.get("effect"),
            "used": [str(s) for s in payload.get("used") or []], "wrong": [str(s) for s in payload.get("wrong") or []],
            "task_succeeded": payload.get("task_succeeded"), "query": payload.get("query"), "note": payload.get("note"),
        })
    return out


def top_sources(db: DashboardDB, tenant: str, generation: str, limit: int = 40) -> list[dict[str, Any]]:
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.top_sources(db.dsn, tenant, generation, limit)
    with db.connect() as c:
        rows = c.execute(
            "select source_uri, count(*), max(indexed_at) from recall_chunks_v1 where tenant_id = %s and generation_id = %s "
            "group by source_uri order by max(indexed_at) desc limit %s",
            (tenant, generation, limit),
        ).fetchall()
    return [{"source": s, "chunks": n, "indexed_at": t} for s, n, t in rows]


#: The newest rows the review queue reads. One row per claim per tenant per day reaches the table,
#: and the rows are append-only, so claims already decided stay in it; when the read reaches this
#: bound the queue says so, because an older pending report is then not shown.
MAX_STALE_REPORTS = 2000


def stale_row(event_id: Any, tenant: Any, payload: Any, created: Any) -> dict[str, Any]:
    """One `stale_report` row as the review queue reads it: the payload's fields, and where it came from."""
    payload = payload if isinstance(payload, dict) else {}
    return {"event_id": event_id, "tenant": tenant, "created_at": created, **{key: payload.get(key) for key in STALE_REPORT_FIELDS}}


def stale_reports(db: DashboardDB, tenants: tuple[str, ...], limit: int = MAX_STALE_REPORTS) -> list[dict[str, Any]]:
    """Agent stale reports from the named tenants, newest first.

    Only the tenants whose memos this dashboard's folder holds: a report filed in any other tenant
    is not this folder's to review, whatever names it carries. The tenant filter also lets the
    `(tenant_id, event_type, created_at)` index serve the read.
    """
    if not tenants:
        return []
    if db.lite:
        from recall.dashboard import lite_db

        return lite_db.stale_reports(db.dsn, tenants, limit)
    with db.connect() as c:
        rows = c.execute(
            "select event_id, tenant_id, payload, created_at from recall_audit_events "
            "where tenant_id = any(%s) and event_type = %s order by created_at desc limit %s",
            (list(tenants), STALE_REPORT_EVENT, limit),
        ).fetchall()
    return [stale_row(*row) for row in rows]
