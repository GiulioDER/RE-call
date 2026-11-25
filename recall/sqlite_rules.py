"""Check retrieved SQLite field notes before rendering them as evidence."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

from recall.evidence import EvidenceBundle, EvidenceItem

_FIELD = re.compile(r"([\w][\w ()-]*)\.([\w][\w ()-]*)")
_FORMAT = re.compile(r"\b(?:YYMMDD|YYYY-MM-DD)\b", re.IGNORECASE)
# Any date notation counts as a format claim, so one in an unsupported notation is omitted rather
# than passed through as an unchecked description. Matched after NFKC, so fullwidth letters count.
_DATE_PATTERN = re.compile(
    r"%[YymdjbBe]"
    r"|\b[YMD]{1,4}(?:[-/._][YMD]{1,4}){1,2}\b"
    r"|\b(?:[YMD]{4,8}|DMY|MDY|YMD)\b"
    r"|\b(?:day|month|year)[-/](?:day|month|year)\b",
    re.IGNORECASE,
)
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
    # sqlite_master and pragma_table_info exist on every SQLite the project supports (floor 3.25);
    # pragma_table_list needs 3.37. table_info omits generated columns.
    if not conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name=? "
        "AND sql NOT LIKE 'CREATE VIRTUAL TABLE%'", (table,)
    ).fetchone():
        return False
    return conn.execute(
        "SELECT 1 FROM pragma_table_info(?, 'main') WHERE name=?", (table, column)
    ).fetchone() is not None


def _date_format(conn: sqlite3.Connection, table: str, column: str, deadline: float) -> str | None:
    field = _identifier(column)
    digits = "[0-9]"
    iso = f"{digits * 4}-{digits * 2}-{digits * 2}"
    compact = f"CAST({field} AS TEXT)"
    as_iso = f"'20'||substr({compact},1,2)||'-'||substr({compact},3,2)||'-'||substr({compact},5,2)"
    expired: list[bool] = []

    def interrupt() -> int:
        if monotonic() >= deadline:
            expired.append(True)
            return 1
        return 0

    conn.set_progress_handler(interrupt, 1_000)
    try:
        count, iso_count, compact_count = conn.execute(
            f"SELECT count({field}), "  # noqa: S608, IDs are schema-checked and quoted, patterns are bound.
            f"sum(typeof({field})='text' AND {field} GLOB ? AND date({field}, '+0 days')={field}), "
            f"sum(typeof({field}) IN ('text', 'integer') AND {compact} GLOB ? "
            f"AND date({as_iso}, '+0 days')={as_iso}) "
            f"FROM main.{_identifier(table)}",
            (iso, digits * 6),
        ).fetchone()
    except sqlite3.OperationalError:
        if expired:
            return None  # The verification budget ran out: this note stays unverified and is omitted.
        raise
    finally:
        conn.set_progress_handler(None, 1_000)
    if count and count == iso_count:
        return "YYYY-MM-DD"
    if count and count == compact_count:
        return "YYMMDD"
    return None


def _check(
    conn: sqlite3.Connection, item: EvidenceItem, deadline: float
) -> tuple[EvidenceItem | None, SnapshotRuleCheck]:
    header, _, body = item.text.partition("\n")
    field = _FIELD.fullmatch(header.strip())
    if field is None or not body.strip():
        return None, SnapshotRuleCheck(item.chunk_id, "not_a_field_note")
    table, column = field.groups()
    if not _field_exists(conn, table, column):
        return None, SnapshotRuleCheck(item.chunk_id, "field_missing")
    formats = _DATE_PATTERN.findall(unicodedata.normalize("NFKC", body))
    declarations = list(_DECLARATION.finditer(body))
    if not formats and not declarations:
        return item, SnapshotRuleCheck(item.chunk_id, "field_present")
    if len(formats) != 1 or len(declarations) != 1:
        return None, SnapshotRuleCheck(item.chunk_id, "unsupported_format_claim")
    declaration = declarations[0]
    claimed = declaration[1].strip().removesuffix(".").upper()
    if _FORMAT.fullmatch(claimed) is None:
        return None, SnapshotRuleCheck(item.chunk_id, "unsupported_format_claim")
    observed = _date_format(conn, table, column, deadline)
    if observed is None:
        return None, SnapshotRuleCheck(item.chunk_id, "unverified_format")
    if observed == claimed:
        checked = replace(item, snapshot_format=observed)
        return checked, SnapshotRuleCheck(item.chunk_id, "format_verified", observed)
    start, end = declaration.span()
    text = f"{header}\n{body[:start]}{body[end:]}"
    corrected = replace(item, text=text, snapshot_format=observed)
    return corrected, SnapshotRuleCheck(item.chunk_id, "format_corrected", observed)


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
        conn.execute("BEGIN")
        for item in bundle.items:
            checked, audit = _check(conn, item, deadline)
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
