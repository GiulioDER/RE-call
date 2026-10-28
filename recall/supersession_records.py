"""Where a removed memo's `SupersessionRecord` is written and read, on PostgreSQL and SQLite.

The rule is in `recall.supersession`: forgetting a memo keeps retired what it retired, withdrawing
one gives it back and splices its chain. This module is the storage half. A record is one row of
the tenant's append-only audit ledger (`recall_audit_events` on PostgreSQL, `audit_events` in a
lite file), typed `FORGOTTEN_RECORD` or `WITHDRAWN_RECORD`, keyed by the memo's absolute source
and scoped by its payload to the chunk table it was removed from. Every delete primitive writes it
in the transaction that deletes the chunks, so a memo is never gone without its record or the
reverse, and every caller of those primitives (`recall_forget`, `recall forget`, the indexer's
prune, a migration replay, `GenerationManager.forget` on a generation tenant, the lite memo
tools) writes it alike.

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

# Literal types also let generic plans use the partial latest-record index.
_RECORD_TYPES_SQL = "('supersession_forgotten', 'supersession_withdrawn')"
#: A bounded, transactionally maintained cache key. Record-only withdrawals must invalidate
#: readers too, even though no chunk-side fingerprint moves. Missing tenant rows start at zero.
PG_RECORD_REVISION_SQL = (
    "SELECT COALESCE((SELECT revision FROM recall_supersession_revisions WHERE tenant_id = %s), 0)"
)


def record_kind(withdraw: bool) -> str:
    return WITHDRAWN_RECORD if withdraw else FORGOTTEN_RECORD


def pg_records(conn: Any, tenant: str, scope: str, sources: Sequence[str] | None = None) -> list[SupersessionRecord]:
    """Latest usable records, in first-usable-source order, without transferring their history.

    Rank IDs before joining winning payloads. An older truthy noniterable `claims` still raises
    through the ordinary parser: a later usable event must not hide a corrupt ledger. The bad
    event and winners share one SQL snapshot, including under READ COMMITTED.
    """
    if sources is not None and not sources:
        return []
    params: list[object] = [tenant, scope]
    extra = ""
    if sources is not None:
        extra = "AND source_uri = ANY(%s) "
        params.append(list(sources))
    params.append(tenant)
    rows = conn.execute(
        f"""
        WITH eligible AS (
            SELECT event_id, source_uri, created_at,
                   (jsonb_typeof(payload->'claims') = 'number' AND payload->'claims' <> '0'::jsonb)
                   OR payload->'claims' = 'true'::jsonb AS bad_claims
            FROM recall_audit_events
            WHERE tenant_id = %s AND event_type IN {_RECORD_TYPES_SQL}
              AND source_uri <> '' AND jsonb_typeof(payload) = 'object'
              AND jsonb_typeof(payload->'scope') = 'string'
              AND jsonb_typeof(payload->'file') = 'string'
              AND payload->>'scope' = %s {extra}
        ), first_bad AS (
            SELECT event_id FROM eligible WHERE bad_claims
            ORDER BY created_at, event_id LIMIT 1
        ), ranked AS (
            SELECT event_id,
                   row_number() OVER (
                       PARTITION BY source_uri ORDER BY created_at DESC, event_id DESC
                   ) AS winner_rank,
                   first_value(created_at) OVER (
                       PARTITION BY source_uri ORDER BY created_at, event_id
                   ) AS first_at,
                   first_value(event_id) OVER (
                       PARTITION BY source_uri ORDER BY created_at, event_id
                   ) AS first_id
            FROM eligible
        ), selected AS (
            SELECT event_id, first_at, first_id FROM ranked
            WHERE winner_rank = 1 AND NOT EXISTS (SELECT 1 FROM first_bad)
            UNION ALL
            SELECT event_id, NULL::timestamptz, NULL::text FROM first_bad
        )
        SELECT e.event_type, e.source_uri, e.payload
        FROM selected s JOIN recall_audit_events e ON e.event_id = s.event_id AND e.tenant_id = %s
        ORDER BY s.first_at, s.first_id
        """, params,
    ).fetchall()
    return list(latest_records(rows, scope).values())


def pg_record_revision(conn: Any, tenant: str) -> int:
    row = conn.execute(PG_RECORD_REVISION_SQL, (tenant,)).fetchone()
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
    rows: Sequence[SupersessionRow] | None = None,
) -> int:
    """Append, inside the caller's transaction, the records removing `sources` from `table` owes.

    `scope` and `params` select the rows about to be deleted (the alias is ``c``); the caller
    deletes them after this, in the same transaction. Returns how many records were appended.
    """
    from psycopg.types.json import Jsonb  # here, so a lite store never loads psycopg for this

    if rows is None:
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


def pg_delete_removals(
    conn: Any, tenant: str, table: str, sources: list[str], *, withdraw: bool, quote_table: bool = False,
) -> list[str]:
    """Delete legacy rows and retain claims from exactly the versions DELETE removed.

    The materialized DELETE result supplies both IDs for sparse cleanup and the ordinary SQL
    supersession scan. No earlier SELECT can race with an upsert before deletion. The caller
    owns the transaction, so a record-append failure also rolls the deletion back. `quote_table`
    preserves the generation manager's quoted legacy-table contract; ordinary store callers
    retain their existing unquoted identifier semantics. The record scope remains `table`.
    """
    from psycopg import sql

    scan = supersession_rows_sql("removed", source_column="source", scope="true")
    delete = (
        "WITH removed AS (DELETE FROM {} WHERE tenant_id = %s AND source = ANY(%s) "
        "RETURNING id, metadata, first_indexed_at, indexed_at, source) "
    )
    select = (
        "SELECT id, NULL::text, NULL::text, NULL::timestamptz, NULL::text, NULL::text, NULL::text "
        "FROM removed UNION ALL SELECT NULL, scanned.* FROM (" + scan + ") AS scanned"
    )
    query = (
        sql.SQL(delete).format(sql.Identifier(table)) + sql.SQL(select)
        if quote_table else delete.format(table) + select
    )
    result = conn.execute(query, (tenant, sources)).fetchall()
    ids = [str(row[0]) for row in result if row[0] is not None]
    rows = [tuple(row[1:]) for row in result if row[0] is None]
    pg_record_removals(
        conn, tenant=tenant, table=table, source_column="source", scope="true", params=(),
        sources=sources, withdraw=withdraw, rows=rows,
    )
    return ids


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


def erasure_subjects(
    rows: Sequence[Sequence[Any]], records: Sequence[SupersessionRecord] = ()
) -> dict[str, set[str]]:
    """`{source: {name, ...}}` from `(source, file)` rows and the records standing for sources:
    the names a source security policy decides each source by.

    A row is authorized by its root-relative `file`, or by its `source` when it has none (the
    prefix arm of `recall.scope.Scope.sql`), and a record by the same rule. One source can carry
    several `file` values (indexed from two roots, or across generations), so every one counts.
    """
    subjects: dict[str, set[str]] = {}
    for source, file in [*rows, *((record.source, record.file) for record in records)]:
        name = file if isinstance(file, str) and file else source
        subjects.setdefault(str(source), set()).add(str(name))
    return subjects


def sqlite_records(conn: sqlite3.Connection, scope: str, sources: Sequence[str] | None = None) -> list[SupersessionRecord]:
    """`pg_records` for a lite file, preserving timestamp/rowid order and JSON parser refusals.

    Invalid JSON raises even outside `scope` or with an empty source, as the old reader decoded
    every type/source-matching payload before checking its envelope. Select that error row first
    in the same snapshot; guard JSON field expressions so SQLite cannot evaluate them on it.
    Legacy nonstandard tokens or duplicate outer keys use Python's original decoder and stream
    the ordered history. Canonical serializer output takes the bounded payload-transfer path.
    """
    if sources is not None and not sources:
        return []
    extra = ""
    params: list[object] = []
    if sources is not None:
        extra = "AND source_uri IN (SELECT value FROM json_each(?)) "
        params.append(json.dumps(list(sources)))
    params.append(scope)
    if sources is not None:
        params.append(json.dumps(list(sources)))
    safe = "(CASE WHEN json_valid(payload) THEN payload ELSE '{}' END)"
    rows = conn.execute(
        f"""
        WITH first_bad_json AS (
            SELECT rowid AS event_rowid FROM audit_events
            WHERE event_type IN {_RECORD_TYPES_SQL} {extra} AND (
                NOT json_valid(payload) OR EXISTS (
                    SELECT 1 FROM json_each({safe})
                    WHERE key IN ('scope', 'file', 'claims') GROUP BY key HAVING count(*) > 1
                )
            )
            ORDER BY created_at, rowid LIMIT 1
        ), eligible AS (
            SELECT rowid AS event_rowid, source_uri, created_at,
                   json_type({safe}, '$.claims') = 'true'
                   OR (json_type({safe}, '$.claims') IN ('integer', 'real')
                       AND json_extract({safe}, '$.claims') <> 0) AS bad_claims
            FROM audit_events
            WHERE event_type IN {_RECORD_TYPES_SQL} AND json_valid(payload)
              AND source_uri <> '' AND json_type({safe}) = 'object'
              AND json_type({safe}, '$.scope') = 'text'
              AND json_type({safe}, '$.file') = 'text'
              AND json_extract({safe}, '$.scope') = ? {extra}
        ), first_bad_claims AS (
            SELECT event_rowid FROM eligible WHERE bad_claims
            ORDER BY created_at, event_rowid LIMIT 1
        ), ranked AS (
            SELECT event_rowid,
                   row_number() OVER (
                       PARTITION BY source_uri ORDER BY created_at DESC, event_rowid DESC
                   ) AS winner_rank,
                   first_value(created_at) OVER (
                       PARTITION BY source_uri ORDER BY created_at, event_rowid
                   ) AS first_at,
                   first_value(event_rowid) OVER (
                       PARTITION BY source_uri ORDER BY created_at, event_rowid
                   ) AS first_rowid
            FROM eligible
        ), selected AS (
            SELECT event_rowid, first_at, first_rowid, 0 AS legacy_json FROM ranked
            WHERE winner_rank = 1 AND NOT EXISTS (SELECT 1 FROM first_bad_json)
              AND NOT EXISTS (SELECT 1 FROM first_bad_claims)
            UNION ALL
            SELECT event_rowid, NULL, NULL, 1 FROM first_bad_json
            UNION ALL
            SELECT event_rowid, NULL, NULL, 0 FROM first_bad_claims
            WHERE NOT EXISTS (SELECT 1 FROM first_bad_json)
        )
        SELECT e.event_type, e.source_uri, e.payload, s.legacy_json
        FROM selected s JOIN audit_events e ON e.rowid = s.event_rowid
        ORDER BY s.first_at, s.first_rowid
        """, params,
    ).fetchall()
    events: list[tuple[object, object, object]] = []
    for kind, source, payload, legacy_json in rows:
        decoded = json.loads(payload)  # A truly corrupt payload still raises, before any fallback.
        if legacy_json:
            # Python accepts nonstandard tokens and reads duplicate keys differently from
            # SQLite. Preserve that legacy decoder/order, streaming the exceptional history.
            legacy_params = [json.dumps(list(sources))] if sources is not None else []
            cursor = conn.execute(
                "SELECT event_type, source_uri, payload FROM audit_events "
                f"WHERE event_type IN {_RECORD_TYPES_SQL} {extra} ORDER BY created_at, rowid",
                legacy_params,
            )
            return list(latest_records(
                ((kind, source, json.loads(payload)) for kind, source, payload in cursor), scope
            ).values())
        events.append((kind, source, decoded))
    return list(latest_records(events, scope).values())


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
    "PG_RECORD_REVISION_SQL",
    "WITHDRAW_ACTOR",
    "erasure_subjects",
    "pending_records",
    "pg_delete_removals",
    "pg_record_revision",
    "pg_record_removals",
    "pg_records",
    "record_kind",
    "records_matching",
    "sqlite_record_removals",
    "sqlite_records",
]
