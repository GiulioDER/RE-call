"""Check retrieved SQLite field notes before rendering them as evidence."""

from __future__ import annotations

import re
import sqlite3
import string
import unicodedata
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Literal

from recall.evidence import EvidenceBundle, EvidenceItem

_ISO = "YYYY-MM-DD"
_COMPACT = "YYMMDD"
_FIELD = re.compile(r"([\w][\w ()-]*)\.([\w][\w ()-]*)")
_FORMAT = re.compile(rf"\b(?:{_COMPACT}|{re.escape(_ISO)})\b", re.IGNORECASE)
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
_BUDGET_SECONDS = 2.0
_BUSY_TIMEOUT_SECONDS = 0.25
_PROGRESS_STEPS = 1_000
# SQLite folds identifier case for ASCII letters only.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

#: What each status means is documented once, in docs/API.md ("Snapshot-checked SQLite rules").
SnapshotRuleStatus = Literal[
    "not_a_field_note",
    "field_missing",
    "field_present",
    "unsupported_format_claim",
    "unverified_format",
    "format_verified",
    "format_corrected",
]


@dataclass(frozen=True)
class SnapshotRuleCheck:
    chunk_id: str
    status: SnapshotRuleStatus
    observed_format: str | None = None


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fold(name: str) -> str:
    """SQLite resolves table and column names without regard to ASCII case."""
    return name.translate(_ASCII_LOWER)


def _field_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    # sqlite_master and pragma_table_info exist on every SQLite the project supports (floor 3.25);
    # pragma_table_list needs 3.37. table_info omits generated columns. SQLite resolves
    # identifiers without regard to ASCII case, so the lookup does too.
    if not conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name=? COLLATE NOCASE "
        "AND sql NOT LIKE 'CREATE VIRTUAL TABLE%'", (table,)
    ).fetchone():
        return False
    return conn.execute(
        "SELECT 1 FROM pragma_table_info(?, 'main') WHERE name=? COLLATE NOCASE", (table, column)
    ).fetchone() is not None


def _date_format(conn: sqlite3.Connection, table: str, column: str, deadline: float) -> str | None:
    field = _identifier(column)
    digits = "[0-9]"
    iso = f"{digits * 4}-{digits * 2}-{digits * 2}"
    # An INTEGER loses a leading zero (050701 is stored as 50701), so a five-digit one is padded
    # back to six. Shorter ones are not: 000101 cannot be told from an MMDD value 101.
    compact = (
        f"CASE WHEN typeof({field})='integer' AND {field} BETWEEN 10000 AND 99999 "
        f"THEN printf('%06d', {field}) ELSE CAST({field} AS TEXT) END"
    )
    as_iso = f"'20'||substr({compact},1,2)||'-'||substr({compact},3,2)||'-'||substr({compact},5,2)"
    expired: list[bool] = []
    try:
        with _snapshot_deadline(conn, deadline, expired):
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
    if count and count == iso_count:
        return _ISO
    if count and count == compact_count:
        return _COMPACT
    return None


def _check(
    conn: sqlite3.Connection,
    item: EvidenceItem,
    deadline: float,
    observed_formats: dict[tuple[str, str], str | None],
) -> tuple[EvidenceItem | None, SnapshotRuleCheck]:
    # Only an observation made in this transaction may reach the prompt as `snapshot_format`.
    item = replace(item, snapshot_format=None)
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
    # One read transaction serves every note, so a column scanned once answers for all of them.
    key = (_fold(table), _fold(column))
    if key not in observed_formats:
        observed_formats[key] = _date_format(conn, table, column, deadline)
    observed = observed_formats[key]
    if observed is None:
        return None, SnapshotRuleCheck(item.chunk_id, "unverified_format")
    if observed == claimed:
        checked = replace(item, snapshot_format=observed)
        return checked, SnapshotRuleCheck(item.chunk_id, "format_verified", observed)
    start, end = declaration.span()
    text = f"{header}\n{body[:start]}{body[end:]}"
    corrected = replace(item, text=text, snapshot_format=observed)
    return corrected, SnapshotRuleCheck(item.chunk_id, "format_corrected", observed)


@contextmanager
def _open_snapshot(snapshot: Path) -> Iterator[sqlite3.Connection]:
    uri = snapshot.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=_BUSY_TIMEOUT_SECONDS)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        yield conn


@contextmanager
def _snapshot_deadline(
    conn: sqlite3.Connection, deadline: float | None = None, expired: list[bool] | None = None
) -> Iterator[None]:
    """Interrupt SQLite work on `conn` at `deadline`, by default `_BUDGET_SECONDS` from now.

    An interruption is appended to `expired`, so a caller can tell it from another error.
    """
    end = monotonic() + _BUDGET_SECONDS if deadline is None else deadline

    def interrupt() -> int:
        if monotonic() < end:
            return 0
        if expired is not None:
            expired.append(True)
        return 1

    conn.set_progress_handler(interrupt, _PROGRESS_STEPS)
    try:
        yield
    finally:
        conn.set_progress_handler(None, 0)


def _verify_rules(
    conn: sqlite3.Connection, bundle: EvidenceBundle
) -> tuple[EvidenceBundle, tuple[SnapshotRuleCheck, ...]]:
    items: list[EvidenceItem] = []
    checks: list[SnapshotRuleCheck] = []
    observed_formats: dict[tuple[str, str], str | None] = {}
    deadline = monotonic() + _BUDGET_SECONDS
    for item in bundle.items:
        checked, audit = _check(conn, item, deadline, observed_formats)
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


def verify_sqlite_rules(
    bundle: EvidenceBundle, snapshot: Path
) -> tuple[EvidenceBundle, tuple[SnapshotRuleCheck, ...]]:
    """Prepare eligible field-note evidence from a single read-only SQLite transaction.

    A bundle that does not answer is returned unchanged, and the snapshot is not opened. Otherwise
    each item gets a `SnapshotRuleStatus` (docs/API.md says what each means), the cards are
    cleared, and a bundle left with no item abstains with
    ``reason_code="no_verified_database_rules"``. Column scans share a two-second budget; a scan
    past it leaves its note unverified rather than raising.

    Raises:
        sqlite3.Error: the snapshot cannot be opened or read, including when a writer holds it
            locked for longer than the busy timeout.
    """
    if bundle.decision != "answer":
        return bundle, ()
    with _open_snapshot(snapshot) as conn:
        return _verify_rules(conn, bundle)
