"""Supersession reported by an agent at use time, queued for a person to accept or reject.

An agent that reads two memos for one task is often the first to notice that one replaces the
other. This module records that observation and nothing else: it never writes frontmatter, never
changes a verdict, and never removes a memo from search. A report becomes a corpus edit only when
a named person accepts it through the same review path as every other proposal
(`recall rewrite apply`, or the dashboard), and a rejected report is recorded in the corpus's
rejection ledger, so it does not come back from either source.

Two rules carry the safety, and both are checked when the report is written, not when it is read:

* **Both quotes must be verbatim** in the memo they are said to come from (at least
  `MIN_QUOTE_CHARS` after whitespace normalisation). A report the reviewer cannot check against
  the text is refused rather than queued.
* **Both memos must be files inside the corpus root**, the same confinement `recall_index`
  applies. A report naming a path outside the root, the sidecar directory, or a missing file is
  refused.

One row per claim. The key is `rewrite.claim_key("supersedes", stale, replacing)`, the identity the
rejection ledger already uses, so a claim a person rejected is refused here by construction, and a
second agent reporting the same pair raises its count rather than adding a duplicate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recall.errors import RecallError
from recall.frontmatter import encodable_name
from recall.rewrite import RejectionLedger, RewriteRefused, claim_key, default_ledger_path

MIN_QUOTE_CHARS = 20
#: A bound on what one corpus can accumulate unreviewed. A queue nobody reviews is a signal to a
#: person, not something an agent should be able to grow without limit.
DEFAULT_MAX_PENDING = 500
STATUSES = ("pending", "accepted", "rejected")

_SIDECAR_DIR = ".recall"
_QUEUE_NAME = "reports.sqlite3"


class StaleReportRefused(ValueError, RecallError):
    """A stale report that will not be queued, with the reason a caller can act on."""


@dataclass(frozen=True)
class StaleReport:
    """One queued claim that `replacing_source` supersedes `stale_source`."""

    claim_key: str
    stale_source: str
    replacing_source: str
    stale_quote: str
    current_quote: str
    client: str
    task: str | None
    first_reported_at: str
    last_reported_at: str
    report_count: int
    status: str
    reviewer_id: str | None
    review_note: str | None
    reviewed_at: str | None


def default_queue_path(root: Path) -> Path:
    """`<root>/.recall/reports.sqlite3`, beside the rejection ledger."""
    return root / _SIDECAR_DIR / _QUEUE_NAME


def normalise(text: str) -> str:
    return " ".join(text.split())


def grounded(quote: object, text: str) -> bool:
    """A quote counts only if it is at least MIN_QUOTE_CHARS long and verbatim in the text."""
    if not isinstance(quote, str):
        return False
    wanted = normalise(quote)
    return len(wanted) >= MIN_QUOTE_CHARS and wanted in normalise(text)


def resolve_memo(root: Path, name: str) -> tuple[str, Path]:
    """The corpus name and path of a memo the report names, or a refusal.

    The name is relative to the corpus root, as `corpus_proposals` names memos. Absolute paths,
    anything escaping the root, the sidecar directory and missing files are refused.
    """
    if not isinstance(name, str) or not name.strip():
        raise StaleReportRefused("a memo name is required")
    root = root.resolve()
    candidate = Path(name.strip())
    if candidate.is_absolute():
        raise StaleReportRefused(f"{name!r} must be a path relative to the corpus root")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise StaleReportRefused(f"{name!r} is outside the corpus root")
    relative = path.relative_to(root)
    if relative.parts and relative.parts[0] == _SIDECAR_DIR:
        raise StaleReportRefused(f"{name!r} is inside the {_SIDECAR_DIR} sidecar, not a memo")
    if not path.is_file():
        raise StaleReportRefused(f"{name!r} is not a file in the corpus")
    return encodable_name(relative.as_posix()), path


class StaleReportQueue:
    """The durable queue of agent reports for one corpus."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StaleReportRefused(f"report queue directory for {self._path} could not be created: {exc}") from exc
        try:
            connection = sqlite3.connect(str(self._path), timeout=30.0)
        except sqlite3.Error as exc:
            raise StaleReportRefused(f"report queue at {self._path} could not be opened: {exc}") from exc
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS stale_reports ("
                "claim_key TEXT PRIMARY KEY, stale_source TEXT NOT NULL, "
                "replacing_source TEXT NOT NULL, stale_quote TEXT NOT NULL, "
                "current_quote TEXT NOT NULL, client TEXT NOT NULL, task TEXT, "
                "first_reported_at TEXT NOT NULL, last_reported_at TEXT NOT NULL, "
                "report_count INTEGER NOT NULL, status TEXT NOT NULL, reviewer_id TEXT, "
                "review_note TEXT, reviewed_at TEXT)"
            )
            connection.commit()
        except BaseException:
            connection.close()
            raise
        self._conn = connection

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StaleReportQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def submit(
        self,
        root: Path,
        *,
        stale_source: str,
        replacing_source: str,
        stale_quote: str,
        current_quote: str,
        client: str,
        task: str | None,
        reported_at: datetime,
        max_pending: int = DEFAULT_MAX_PENDING,
        ledger_path: Path | None = None,
    ) -> StaleReport:
        """Queue one report after checking it, or refuse it with the reason."""
        stale_name, stale_path = resolve_memo(root, stale_source)
        replacing_name, replacing_path = resolve_memo(root, replacing_source)
        if stale_path == replacing_path:
            raise StaleReportRefused("a memo cannot supersede itself")
        if not grounded(stale_quote, stale_path.read_text(encoding="utf-8", errors="replace")):
            raise StaleReportRefused(
                f"stale_quote is not verbatim in {stale_name} (at least {MIN_QUOTE_CHARS} characters, copied exactly)"
            )
        if not grounded(current_quote, replacing_path.read_text(encoding="utf-8", errors="replace")):
            raise StaleReportRefused(
                f"current_quote is not verbatim in {replacing_name} (at least {MIN_QUOTE_CHARS} characters, copied exactly)"
            )
        key = claim_key("supersedes", stale_name, replacing_name)
        ledger = default_ledger_path(root.resolve()) if ledger_path is None else ledger_path
        if ledger.exists():
            try:
                with RejectionLedger(ledger) as rejections:
                    if rejections.is_rejected(key):
                        raise StaleReportRefused(
                            f"a person already rejected the claim that {replacing_name} supersedes {stale_name}"
                        )
            except RewriteRefused as exc:
                raise StaleReportRefused(str(exc)) from exc
        stamp = reported_at.isoformat()
        existing = self.get(key)
        if existing is not None and existing.status != "pending":
            raise StaleReportRefused(f"this claim was already {existing.status} by {existing.reviewer_id}")
        if existing is None and self.count("pending") >= max_pending:
            raise StaleReportRefused(
                f"the review queue already holds {max_pending} pending reports; a person has to review them first"
            )
        # One statement, so two agents reporting the same pair at once cannot race a read against
        # an insert: the second becomes a count, never a primary-key error. A claim closed in
        # between is left untouched by the WHERE and refused just below.
        self._conn.execute(
            "INSERT INTO stale_reports (claim_key, stale_source, replacing_source, stale_quote, "
            "current_quote, client, task, first_reported_at, last_reported_at, report_count, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'pending') "
            "ON CONFLICT(claim_key) DO UPDATE SET report_count = report_count + 1, "
            "last_reported_at = excluded.last_reported_at WHERE status = 'pending'",
            (key, stale_name, replacing_name, stale_quote, current_quote, client, task, stamp, stamp),
        )
        self._conn.commit()
        stored = self.get(key)
        assert stored is not None
        if stored.status != "pending":
            raise StaleReportRefused(f"this claim was already {stored.status} by {stored.reviewer_id}")
        return stored

    def get(self, key: str) -> StaleReport | None:
        row = self._conn.execute(
            "SELECT claim_key, stale_source, replacing_source, stale_quote, current_quote, client, task, "
            "first_reported_at, last_reported_at, report_count, status, reviewer_id, review_note, reviewed_at "
            "FROM stale_reports WHERE claim_key = ?",
            (key,),
        ).fetchone()
        return StaleReport(*row) if row else None

    def list(self, status: str = "pending") -> list[StaleReport]:
        if status not in STATUSES:
            raise StaleReportRefused(f"unknown status {status!r}")
        rows = self._conn.execute(
            "SELECT claim_key, stale_source, replacing_source, stale_quote, current_quote, client, task, "
            "first_reported_at, last_reported_at, report_count, status, reviewer_id, review_note, reviewed_at "
            "FROM stale_reports WHERE status = ? ORDER BY first_reported_at",
            (status,),
        ).fetchall()
        return [StaleReport(*row) for row in rows]

    def count(self, status: str) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM stale_reports WHERE status = ?", (status,)).fetchone()[0])

    def mark_reviewed(self, key: str, *, status: str, reviewer_id: str, note: str, reviewed_at: datetime) -> StaleReport:
        """Close a pending report as accepted or rejected, by a named person with a note."""
        if status not in ("accepted", "rejected"):
            raise StaleReportRefused(f"a review ends as accepted or rejected, not {status!r}")
        if not reviewer_id.strip() or not note.strip():
            raise StaleReportRefused("a review needs a named reviewer and a note")
        current = self.get(key)
        if current is None:
            raise StaleReportRefused(f"no report {key!r}")
        if current.status != "pending":
            raise StaleReportRefused(f"report {key!r} is already {current.status}")
        self._conn.execute(
            "UPDATE stale_reports SET status = ?, reviewer_id = ?, review_note = ?, reviewed_at = ? "
            "WHERE claim_key = ? AND status = 'pending'",
            (status, reviewer_id, note, reviewed_at.isoformat(), key),
        )
        self._conn.commit()
        return self.get(key)  # type: ignore[return-value]

