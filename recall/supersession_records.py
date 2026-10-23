"""Where a removed memo's `SupersessionRecord` is written and read, on PostgreSQL and SQLite.

The rule is in `recall.supersession`: forgetting a memo keeps retired what it retired, withdrawing
one gives it back and splices its chain. This module is the storage half. A record is one row of
the tenant's append-only audit ledger (`recall_audit_events` on PostgreSQL, `audit_events` in a
lite file), typed `FORGOTTEN_RECORD` or `WITHDRAWN_RECORD`, keyed by the memo's absolute source
and scoped by its payload to the chunk table it was removed from. Every delete primitive writes it
in the transaction that deletes the chunks, so a memo is never gone without its record or the
reverse, and every caller of those primitives (`recall_forget`, `recall forget`, the indexer's
prune, a migration replay, `recall generation forget`, the lite memo tools) writes it alike.

Names and dates only: the payload is the memo's scan rows (`supersession_rows_sql`), never its
text.
"""
# The SQL identifiers here are fixed by the calling store, never a request value; values are bound.
# ruff: noqa: S608
from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

from recall.supersession import (
    FORGOTTEN_RECORD,
    SUPERSESSION_RECORD_TYPES,
    WITHDRAWN_RECORD,
    SupersessionRecord,
    SupersessionRow,
    latest_records,
    records_from_rows,
    supersession_rows_sql,
)

#: Who wrote a record, as the audit ledger's `actor` names it.
FORGET_ACTOR = "recall_forget"
WITHDRAW_ACTOR = "recall_withdraw"

_PG_EVENTS = (
    "SELECT event_type, source_uri, payload FROM recall_audit_events "
    "WHERE tenant_id = %s AND event_type = ANY(%s) {extra}ORDER BY created_at, event_id"
)
#: How many records a tenant holds: part of a cache key, because a withdrawal of a memo that is
#: already gone writes a record and deletes nothing, so no chunk-side fingerprint moves.
PG_RECORD_COUNT_SQL = (
    "SELECT count(*) FROM recall_audit_events WHERE tenant_id = %s AND event_type = ANY(%s)"
)


def record_kind(withdraw: bool) -> str:
    return WITHDRAWN_RECORD if withdraw else FORGOTTEN_RECORD


def pg_records(conn: Any, tenant: str, scope: str, sources: Sequence[str] | None = None) -> list[SupersessionRecord]:
    """The latest record per source for `scope`, optionally only for `sources`."""
    if sources is not None and not sources:
        return []
    params: list[object] = [tenant, list(SUPERSESSION_RECORD_TYPES)]
    extra = ""
    if sources is not None:
        extra = "AND source_uri = ANY(%s) "
        params.append(list(sources))
    rows = conn.execute(_PG_EVENTS.format(extra=extra), params).fetchall()
    return list(latest_records(rows, scope).values())


def pg_record_count(conn: Any, tenant: str) -> int:
    row = conn.execute(PG_RECORD_COUNT_SQL, (tenant, list(SUPERSESSION_RECORD_TYPES))).fetchone()
    return int(row[0]) if row else 0


def pending_records(
    rows: Sequence[SupersessionRow],
    sources: Sequence[str],
    *,
    withdraw: bool,
    standing: Sequence[SupersessionRecord] = (),
) -> list[SupersessionRecord]:
    """The records removing `sources` owes, given their scan `rows` and the latest record each
    already has (`standing`).

    - a memo whose rows declare a claim: a record of its claims;
    - a memo that claims nothing now but has an older record (it was removed once, re-indexed
      without the claim, and is removed again): a record of no claims, so the older one stops
      standing for a claim the memo no longer made;
    - a withdrawal of a memo with no rows left that a record stands in for (forgotten or pruned
      earlier): the withdrawal takes over that record's claims.
    """
    kind = record_kind(withdraw)
    claimed = {record.source: record for record in records_from_rows(kind, rows)}
    present: dict[str, str] = {}
    for row in rows:
        file, source = row[0], row[3] if len(row) > 3 else None
        if isinstance(file, str) and isinstance(source, str):
            present.setdefault(source, file)
    prior = {record.source: record for record in standing}
    records: list[SupersessionRecord] = []
    for source in sorted(set(sources)):
        if source in claimed:
            records.append(claimed[source])
        elif source in present:
            if source in prior:
                records.append(SupersessionRecord(kind, present[source], source, ()))
        elif withdraw and source in prior:
            records.append(replace(prior[source], kind=WITHDRAWN_RECORD))
    return records


def pg_record_removals(
    conn: Any,
    *,
    tenant: str,
    table: str,
    source_column: str,
    scope: str,
    params: Sequence[object],
    sources: Sequence[str],
    withdraw: bool,
    record_scope: str | None = None,
) -> int:
    """Append, inside the caller's transaction, the records removing `sources` from `table` owes.

    `scope` and `params` select the rows about to be deleted (the alias is ``c``); the caller
    deletes them after this, in the same transaction. Returns how many records were appended.
    """
    from psycopg.types.json import Jsonb  # here, so a lite store never loads psycopg for this

    rows = conn.execute(
        supersession_rows_sql(table, source_column=source_column, scope=scope), tuple(params)
    ).fetchall()
    target = record_scope or table
    standing = pg_records(conn, tenant, target, sources)
    records = pending_records(rows, sources, withdraw=withdraw, standing=standing)
    actor = WITHDRAW_ACTOR if withdraw else FORGET_ACTOR
    for record in records:
        conn.execute(
            "INSERT INTO recall_audit_events "
            "(tenant_id, event_id, event_type, actor, generation_id, source_uri, payload) "
            "VALUES (%s, %s, %s, %s, NULL, %s, %s)",
            (tenant, f"evt_{uuid4().hex}", record.kind, actor, record.source, Jsonb(record.payload(target))),
        )
    return len(records)


def records_matching(
    records: Sequence[SupersessionRecord], identifiers: Sequence[str]
) -> dict[str, list[str]]:
    """`{identifier: [source, ...]}` for identifiers naming a record's `file` or its `source`,
    the rule `sources_for_identifiers` applies to indexed rows."""
    requested = set(identifiers)
    resolved: dict[str, list[str]] = {}
    for record in records:
        for ident in {record.file, record.source} & requested:
            bucket = resolved.setdefault(ident, [])
            if record.source not in bucket:
                bucket.append(record.source)
    return resolved


def sqlite_records(conn: sqlite3.Connection, scope: str, sources: Sequence[str] | None = None) -> list[SupersessionRecord]:
    """`pg_records` for a lite file's `audit_events`."""
    if sources is not None and not sources:
        return []
    sql = (
        "SELECT event_type, source_uri, payload FROM audit_events "
        "WHERE event_type IN (SELECT value FROM json_each(?))"
    )
    params: list[object] = [json.dumps(list(SUPERSESSION_RECORD_TYPES))]
    if sources is not None:
        sql += " AND source_uri IN (SELECT value FROM json_each(?))"
        params.append(json.dumps(list(sources)))
    rows = conn.execute(sql + " ORDER BY created_at, rowid", params).fetchall()
    return list(latest_records(((kind, source, json.loads(payload)) for kind, source, payload in rows), scope).values())


def sqlite_record_removals(
    conn: sqlite3.Connection,
    rows: Sequence[SupersessionRow],
    *,
    scope: str,
    sources: Sequence[str],
    withdraw: bool,
    created_at: str,
) -> int:
    """`pg_record_removals` for a lite file, given the scan rows of the sources being removed."""
    standing = sqlite_records(conn, scope, sources)
    records = pending_records(rows, sources, withdraw=withdraw, standing=standing)
    actor = WITHDRAW_ACTOR if withdraw else FORGET_ACTOR
    for record in records:
        conn.execute(
            "INSERT INTO audit_events (event_id, event_type, actor, generation_id, source_uri, payload, created_at) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?)",
            (f"evt_{uuid4().hex}", record.kind, actor, record.source,
             json.dumps(record.payload(scope), default=str), created_at),
        )
    return len(records)


__all__ = [
    "FORGET_ACTOR",
    "PG_RECORD_COUNT_SQL",
    "WITHDRAW_ACTOR",
    "pending_records",
    "pg_record_count",
    "pg_record_removals",
    "pg_records",
    "record_kind",
    "records_matching",
    "sqlite_record_removals",
    "sqlite_records",
]
