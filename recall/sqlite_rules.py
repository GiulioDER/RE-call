"""Check retrieved SQLite field notes before rendering them as evidence."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

from recall.evidence import EvidenceBundle, EvidenceItem

_FIELD = re.compile(r"([\w][\w ()-]*)\.([\w][\w ()-]*)")
_FORMAT = re.compile(r"\b(?:YYMMDD|YYYY-MM-DD)\b", re.IGNORECASE)
_DECLARATION = re.compile(
    r"^[ \t]*(?:format:[ \t]*|in the form[ \t]+)([^\r\n]+)\r?$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SnapshotRuleCheck:
    chunk_id: str
    status: str
    observed_format: str | None = None


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _field_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not conn.execute(
        "SELECT 1 FROM pragma_table_list WHERE schema='main' AND type='table' AND name=?", (table,)
    ).fetchone():
        return False
    return conn.execute(
        "SELECT 1 FROM pragma_table_xinfo(?, 'main') WHERE name=? AND hidden=0", (table, column)
    ).fetchone() is not None


def _date_format(conn: sqlite3.Connection, table: str, column: str) -> str | None:
    field = _identifier(column)
    digits = "[0-9]"
    iso = f"{digits * 4}-{digits * 2}-{digits * 2}"
    count, iso_count, compact_count = conn.execute(
        f"SELECT count({field}), "  # noqa: S608, IDs are schema-checked and quoted, patterns are bound.
        f"sum(typeof({field})='text' AND {field} GLOB ? AND date({field}, '+0 days')={field}), "
        f"sum(typeof({field}) IN ('text', 'integer') AND CAST({field} AS TEXT) GLOB ?) "
        f"FROM main.{_identifier(table)}",
        (iso, digits * 6),
    ).fetchone()
    if count and count == iso_count:
        return "YYYY-MM-DD"
    if count and count == compact_count:
        return "YYMMDD"
    return None


def _check(
    conn: sqlite3.Connection, item: EvidenceItem
) -> tuple[EvidenceItem | None, SnapshotRuleCheck]:
    header, _, body = item.text.partition("\n")
    field = _FIELD.fullmatch(header.strip())
    if field is None or not body.strip():
        return None, SnapshotRuleCheck(item.chunk_id, "not_a_field_note")
    table, column = field.groups()
    if not _field_exists(conn, table, column):
        return None, SnapshotRuleCheck(item.chunk_id, "field_missing")
    formats = _FORMAT.findall(body)
    declarations = list(_DECLARATION.finditer(body))
    if not formats and not declarations:
        return item, SnapshotRuleCheck(item.chunk_id, "field_present")
    if len(formats) != 1 or len(declarations) != 1:
        return None, SnapshotRuleCheck(item.chunk_id, "unsupported_format_claim")
    declaration = declarations[0]
    claimed = declaration[1].strip().removesuffix(".").upper()
    if _FORMAT.fullmatch(claimed) is None:
        return None, SnapshotRuleCheck(item.chunk_id, "unsupported_format_claim")
    observed = _date_format(conn, table, column)
    if observed is None:
        return None, SnapshotRuleCheck(item.chunk_id, "unverified_format")
    if observed == claimed:
        return item, SnapshotRuleCheck(item.chunk_id, "format_verified", observed)
    start, end = declaration.span()
    text = f"{header}\n{body[:start]}Format observed in this SQLite snapshot: {observed}{body[end:]}"
    return replace(item, text=text), SnapshotRuleCheck(item.chunk_id, "format_corrected", observed)


def verify_sqlite_rules(
    bundle: EvidenceBundle, snapshot: Path
) -> tuple[EvidenceBundle, tuple[SnapshotRuleCheck, ...]]:
    """Prepare eligible field-note evidence from a single read-only SQLite transaction."""
    if bundle.decision != "answer":
        return bundle, ()
    items: list[EvidenceItem] = []
    checks: list[SnapshotRuleCheck] = []
    deadline = monotonic() + 2.0
    with closing(sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.set_progress_handler(lambda: int(monotonic() >= deadline), 1_000)
        conn.execute("BEGIN")
        for item in bundle.items:
            checked, audit = _check(conn, item)
            checks.append(audit)
            if checked is not None:
                items.append(checked)
    prepared = replace(bundle, items=tuple(items), cards=())
    if not items:
        prepared = replace(
            prepared, decision="abstain", reason_code="no_verified_database_rules",
            decision_state="no_supporting_evidence",
        )
    return prepared, tuple(checks)
