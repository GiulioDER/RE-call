"""The dashboard's database pages over a lite store: one SQLite file, opened read-only.

Every function here answers what its namesake in `recall.dashboard.db` answers, in the same shape,
so the pages render a lite store with no branch of their own. The file holds one tenant and one
corpus, so "tenants" is one row and "generations" is the corpus as it is now; a tenant this file
does not hold gets nothing, as an unknown tenant does in Postgres.

The aggregations the Postgres side does in SQL over `jsonb` (retrievals per memo, agent reports per
memo) are done here in Python over the decoded payloads: a local file holds one person's memory,
so its ledger is small, and keeping the rules in one readable place beats a second SQL dialect.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from recall.dashboard.db import SEARCH_EVENTS, USE_REPORT_EVENT, DatabaseUnavailable, _summarise, stale_row
from recall.lite import LITE_DSN_PREFIX
from recall.stale_reports import STALE_REPORT_EVENT


def store_path(dsn: str) -> Path:
    return Path(dsn[len(LITE_DSN_PREFIX):])


@contextmanager
def connect(dsn: str) -> Iterator[sqlite3.Connection]:
    path = store_path(dsn)
    if not path.exists():
        raise DatabaseUnavailable(f"no lite store at {path} yet: `recall index <memory folder>` creates it")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"{type(exc).__name__}: {exc}") from exc
    try:
        yield connection
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()


def _when(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _payload(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _tenant(c: sqlite3.Connection) -> str:
    row = c.execute("SELECT value FROM meta WHERE key = 'tenant'").fetchone()
    return str(row[0]) if row else ""


def _events(c: sqlite3.Connection, tenant: str, types: tuple[str, ...]) -> list[tuple[Any, ...]]:
    """`(event_id, event_type, actor, generation_id, source_uri, payload, created_at)`, newest first."""
    if tenant != _tenant(c):
        return []
    return c.execute(
        "SELECT event_id, event_type, actor, generation_id, source_uri, payload, created_at FROM audit_events "
        "WHERE event_type IN (SELECT value FROM json_each(?)) ORDER BY created_at DESC, rowid DESC",
        (json.dumps(list(types)),),
    ).fetchall()


def file_tenant(dsn: str) -> str:
    """The tenant the file holds, for the dashboard's default."""
    with connect(dsn) as c:
        return _tenant(c)


def availability(dsn: str) -> dict[str, Any]:
    with connect(dsn) as c:
        version = c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return {
        "engine": "SQLite",
        "server_version": sqlite3.sqlite_version,
        "role": str(store_path(dsn)),
        "read_only": True,
        "migrations": [("lite store", version[0] if version else "?")],
    }


def tenants(dsn: str) -> list[dict[str, Any]]:
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite.inspect import inspect_lite_file

    found = inspect_lite_file(store_path(dsn))
    if found.error is not None:
        raise DatabaseUnavailable(found.error)
    if not found.exists:
        raise DatabaseUnavailable(f"no lite store at {found.path} yet")
    calibration = None
    if found.calibration is not CalibrationStatus.MISSING:
        calibration = {
            "certified": found.calibration is CalibrationStatus.CERTIFIED,
            "threshold": found.threshold,
            "separability": found.separability,
            "published_at": _when(found.calibrated_at),
            "state": found.calibration.value,
        }
    with connect(dsn) as c:
        newest = c.execute("SELECT max(indexed_at) FROM chunks").fetchone()[0]
    return [
        {
            "tenant": found.tenant,
            "generation": found.generation_id if found.chunks else None,
            "state": "active" if found.chunks else None,
            "activated_at": _when(newest),
            "updated_at": _when(newest),
            "embedder": {"provider": None, "model": found.embedder_model, "dimension": found.dim, "profile_id": None, "verified": None},
            "chunks": found.chunks,
            "sources": found.sources,
            "calibration": calibration,
        }
    ]


def generations(dsn: str, tenant: str, limit: int = 15) -> list[dict[str, Any]]:
    """A lite store keeps no history of builds: its one corpus is the current one."""
    every = [t for t in tenants(dsn) if t["tenant"] == tenant and t["generation"]]
    with connect(dsn) as c:
        first = c.execute("SELECT min(first_indexed_at) FROM chunks").fetchone()[0]
    return [
        {"generation": t["generation"], "state": "active", "created_at": _when(first), "activated_at": t["activated_at"],
         "retired_at": None, "failure_reason": None, "created_by": "recall-lite"}
        for t in every
    ][:limit]


def lifecycle_events(dsn: str, tenant: str, limit: int = 100) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return []
        rows = c.execute(
            "SELECT event_type, actor, generation_id, source_uri, created_at FROM audit_events "
            "WHERE event_type NOT IN (SELECT value FROM json_each(?)) ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (json.dumps(list(SEARCH_EVENTS)), limit),
        ).fetchall()
    return [{"event": e, "actor": a, "generation": g, "source": s, "created_at": _when(t)} for e, a, g, s, t in rows]


def _search_row(row: tuple[Any, ...]) -> dict[str, Any]:
    event_id, event_type, actor, generation, _source, payload, created = row
    return {"event_id": event_id, "event": event_type, "actor": actor, "generation": generation,
            "created_at": _when(created), **_summarise(_payload(payload))}


def searches(dsn: str, tenant: str, limit: int = 100, contains: str = "") -> list[dict[str, Any]]:
    with connect(dsn) as c:
        rows = _events(c, tenant, SEARCH_EVENTS)
    needle = contains.casefold()
    found = [_search_row(r) for r in rows]
    return [s for s in found if not needle or needle in s["query"].casefold()][:limit]


def search(dsn: str, tenant: str, event_id: str) -> dict[str, Any] | None:
    with connect(dsn) as c:
        rows = _events(c, tenant, SEARCH_EVENTS)
    return next((_search_row(r) for r in rows if r[0] == event_id), None)


def control(dsn: str, tenant: str) -> dict[str, Any]:
    """`recall.dashboard.db.control`, the same rules: one retrieval per (search, memory), at its best rank."""
    memos: dict[str, dict[str, Any]] = {}

    def memo(source: str) -> dict[str, Any]:
        return memos.setdefault(source, {
            "source": source, "retrieved": 0, "first": 0, "last_retrieved": None, "ok": 0, "superseded": 0,
            "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
            "succeeded": 0, "failed": 0, "last_reported": None,
        })

    with connect(dsn) as c:
        search_rows = _events(c, tenant, SEARCH_EVENTS)
        report_rows = _events(c, tenant, (USE_REPORT_EVENT,))
    for _id, event_type, _a, _g, _s, raw, created in search_rows:
        if event_type != "search_decision":
            continue
        at = _when(created)
        hits = _payload(raw).get("hits")
        best: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(hits if isinstance(hits, list) else [], start=1):
            if not isinstance(hit, dict) or hit.get("source") is None:
                continue
            seen = best.setdefault(str(hit["source"]), {"rank": rank, "verdicts": set()})
            seen["rank"] = min(seen["rank"], rank)
            seen["verdicts"].add(hit.get("verdict"))
        for source, seen in best.items():
            row = memo(source)
            row["retrieved"] += 1
            row["first"] += seen["rank"] == 1
            row["ok"] += "ok" in seen["verdicts"]
            row["superseded"] += "superseded" in seen["verdicts"]
            row["low_confidence"] += "low_confidence" in seen["verdicts"]
            if at and (row["last_retrieved"] is None or at > row["last_retrieved"]):
                row["last_retrieved"] = at
    summary = {"searches": len(search_rows), "answered": sum(r[1] == "search_decision" for r in search_rows),
               "reports": len(report_rows), "helped": 0, "no_difference": 0, "misled": 0, "succeeded": 0, "failed": 0}
    for *_ignored, raw, created in report_rows:
        payload, at = _payload(raw), _when(created)
        effect, succeeded = payload.get("effect"), payload.get("task_succeeded")
        if effect in ("helped", "no_difference", "misled"):
            summary[effect] += 1
        if succeeded is True or succeeded is False:
            summary["succeeded" if succeeded else "failed"] += 1
        for role in ("used", "wrong"):
            for source in payload.get(role) or []:
                row = memo(str(source))
                row[role] += 1
                if role == "used" and effect in ("helped", "no_difference"):
                    row[effect] += 1
                if role == "wrong" and effect == "misled":
                    row["misled"] += 1
                if role == "used" and (succeeded is True or succeeded is False):
                    row["succeeded" if succeeded else "failed"] += 1
                if at and (row["last_reported"] is None or at > row["last_reported"]):
                    row["last_reported"] = at
    summary["memos_retrieved"] = sum(1 for m in memos.values() if m["retrieved"])
    ordered = sorted(memos.values(), key=lambda m: (-m["retrieved"], -m["used"], m["source"]))
    return {"summary": summary, "memos": ordered}


def use_reports(dsn: str, tenant: str, limit: int = 50, source: str = "") -> list[dict[str, Any]]:
    with connect(dsn) as c:
        rows = _events(c, tenant, (USE_REPORT_EVENT,))
    out = []
    for event_id, _type, actor, _g, _s, raw, created in rows:
        payload = _payload(raw)
        used = [str(s) for s in payload.get("used") or []]
        wrong = [str(s) for s in payload.get("wrong") or []]
        if source and source not in used and source not in wrong:
            continue
        out.append({
            "event_id": event_id, "actor": actor, "created_at": _when(created),
            "task": str(payload.get("task", "")), "effect": payload.get("effect"), "used": used, "wrong": wrong,
            "task_succeeded": payload.get("task_succeeded"), "query": payload.get("query"), "note": payload.get("note"),
        })
    return out[:limit]


def top_sources(dsn: str, tenant: str, generation: str, limit: int = 40) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return []
        rows = c.execute(
            "SELECT source, count(*), max(indexed_at) FROM chunks GROUP BY source ORDER BY max(indexed_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"source": s, "chunks": n, "indexed_at": _when(t)} for s, n, t in rows]


def stale_reports(dsn: str, tenants: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        tenant = _tenant(c)
        rows = _events(c, tenant, (STALE_REPORT_EVENT,)) if tenant in tenants else []
    return [stale_row(r[0], tenant, _payload(r[5]), _when(r[6])) for r in rows[:limit]]
