"""What happened to the memory recently, from the records the dashboard can read without a database.

Five sources, each already written by something else, read here and never written:

* edits a person made on the dashboard (`<root>/.recall/edits.sqlite3`);
* review decisions on agent reports, accepted or rejected (`<root>/.recall/reports.sqlite3`);
* rejections recorded for claims no agent reported, such as arbiter proposals
  (`<root>/.recall/rejections.sqlite3`);
* agent stale reports as they arrived (the same reports file);
* the memos most recently changed on disk, by file modification time.

The database half of the activity (generations built and promoted, calibrations, forgets) is
merged in by the server when a read-only database is connected; `instant` orders both halves.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from recall.dashboard.edit import history
from recall.dashboard.graph import _memo_files
from recall.rewrite import default_ledger_path
from recall.stale_reports import default_queue_path

RECENT_FILES = 30


@dataclass(frozen=True)
class Event:
    at: str
    kind: str
    title: str
    detail: str
    who: str
    memo: str | None = None


def _rows(path: Path, query: str) -> list[tuple]:
    if not path.exists():
        return []
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10.0)
    try:
        return connection.execute(query).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def recent_activity(root: Path, limit: int = 200) -> list[Event]:
    """Newest first, across every file-backed source."""
    root = root.resolve()
    events: list[Event] = []

    for record in history(root, limit=limit):
        kind = "undo" if record.summary and record.summary[0].startswith("undo of edit") else "edit"
        events.append(Event(record.edited_at, kind, "; ".join(record.summary), record.note, record.editor, record.name))

    reported: set[str] = set()
    for row in _rows(
        default_queue_path(root),
        "SELECT claim_key, stale_source, replacing_source, client, first_reported_at, report_count, "
        "status, reviewer_id, review_note, reviewed_at FROM stale_reports",
    ):
        key, stale, replacing, client, first, count, status, reviewer, note, reviewed = row
        reported.add(key)
        events.append(
            Event(first, "report", f"{replacing} replaces {stale}?", f"reported {count} time(s)", client, replacing)
        )
        if status in ("accepted", "rejected") and reviewed:
            verb = "accepted" if status == "accepted" else "rejected"
            events.append(Event(reviewed, verb, f"{replacing} replaces {stale}: {verb}", note or "", reviewer or "", replacing))

    for key, reviewer, reason, rejected_at in _rows(
        default_ledger_path(root), "SELECT claim_key, reviewer_id, reason, rejected_at FROM rejected_claims"
    ):
        if key not in reported:
            events.append(Event(rejected_at, "rejected", "a proposed claim was rejected", reason, reviewer))

    files = sorted(_memo_files(root), key=lambda p: p.stat().st_mtime, reverse=True)[:RECENT_FILES]
    for path in files:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        name = path.relative_to(root).as_posix()
        events.append(Event(stamp, "changed", name, "file changed on disk", "", name))

    events.sort(key=lambda event: instant(event.at), reverse=True)
    return events[:limit]


def instant(stamp: str) -> datetime:
    """Every source writes ISO 8601; a naive stamp is read as UTC so the sort is total."""
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
