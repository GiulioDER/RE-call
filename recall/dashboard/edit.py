"""A person's edits to a memo's declared values, from the dashboard only.

Four values can be changed, the ones the trust layer acts on: `supersedes` references (added or
withdrawn), `valid_from`, `valid_until`, and the `status` in the derived block. Memo prose is not
editable here; that stays in the person's own editor.

This is a person's tool, and it is shaped so it cannot become an agent's:

* it is reachable only from `recall dashboard` (this machine, a per-launch session, a form token);
  the MCP surface has no edit tool, and this module is not imported by `recall_mcp`;
* every edit names an editor and a reason, is shown as an exact diff first, and is refused when the
  memo changed after the diff was shown;
* every edit is recorded with the bytes before and after in `<root>/.recall/edits.sqlite3`, so it
  can be undone, and an undo is refused when the memo was changed again since.

The writers are the byte-preserving ones in `recall.frontmatter` and `recall.rewrite`, so a BOM,
line endings and every line nobody asked to change survive an edit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from recall.atomic_write import atomic_write_bytes
from recall.dashboard.review import _DECISION_LOCK, ReviewRefused
from recall.document import parse_document
from recall.frontmatter import (
    SupersedesDeclaresNothing,
    add_supersedes_target,
    encodable_name,
    remove_supersedes_target,
    set_frontmatter_scalar,
    supersedes_key,
    supersedes_targets,
    validity_bounds,
)
from recall.rewrite import RewriteRefused, _derived_value, set_derived_status
from recall.stale_reports import StaleReportRefused, resolve_memo
from recall.truth_extraction.types import STATUS_VOCABULARY

#: A sentinel for "leave this value as it is", distinct from None, which clears it.
KEEP = object()

_LEDGER = ".recall/edits.sqlite3"


class EditRefused(ReviewRefused):
    """An edit that was not made, with the reason to show the person."""


@dataclass(frozen=True)
class MemoValues:
    name: str
    supersedes: tuple[str, ...]
    valid_from: str | None
    valid_until: str | None
    status: str | None


@dataclass(frozen=True)
class EditPlan:
    name: str
    before: bytes
    after: bytes
    summary: tuple[str, ...]
    diff: tuple[str, ...]

    @property
    def before_sha(self) -> str:
        return hashlib.sha256(self.before).hexdigest()


@dataclass(frozen=True)
class EditRecord:
    edit_id: int
    name: str
    editor: str
    note: str
    summary: tuple[str, ...]
    edited_at: str
    undone_at: str | None = None
    undone_by: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


def _memo(root: Path, name: str) -> tuple[str, Path]:
    try:
        return resolve_memo(root.resolve(), name)
    except StaleReportRefused as exc:
        raise EditRefused(str(exc)) from exc


def current_values(root: Path, name: str) -> MemoValues:
    corpus_name, path = _memo(root, name)
    raw = path.read_bytes()
    meta = parse_document(raw.decode("utf-8-sig", errors="replace")).meta
    return MemoValues(
        name=corpus_name,
        supersedes=supersedes_targets(meta.get("supersedes")),
        valid_from=str(meta["valid_from"]) if meta.get("valid_from") else None,
        valid_until=str(meta["valid_until"]) if meta.get("valid_until") else None,
        status=_derived_value(raw, "status"),
    )


def _corpus_names(root: Path) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for path in root.rglob("*.md"):
        relative = path.relative_to(root)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            names.setdefault(supersedes_key(path.name), []).append(encodable_name(relative.as_posix()))
    return names


def _check_date(key: str, value: str) -> str:
    value = value.strip()
    try:
        validity_bounds({key: value})
    except ValueError as exc:
        raise EditRefused(f"{key} {value!r} is not a date RE-call can read") from exc
    return value


def plan_edit(
    root: Path,
    name: str,
    *,
    add_supersedes: tuple[str, ...] = (),
    remove_supersedes: tuple[str, ...] = (),
    valid_from: object = KEEP,
    valid_until: object = KEEP,
    status: object = KEEP,
) -> EditPlan:
    """The exact bytes an edit would write, or a refusal. Writes nothing."""
    root = root.resolve()
    corpus_name, path = _memo(root, name)
    before = path.read_bytes()
    after = before
    summary: list[str] = []
    names = _corpus_names(root)

    for target in remove_supersedes:
        changed = remove_supersedes_target(after, target)
        if changed is None:
            raise EditRefused(f"{corpus_name} does not declare that it supersedes {target}")
        after = changed
        summary.append(f"no longer supersedes {target}")

    for target in add_supersedes:
        target = target.strip()
        if not target:
            continue
        matches = names.get(supersedes_key(target), [])
        if not matches:
            raise EditRefused(f"{target} is not a memo in this corpus")
        if len(matches) > 1:
            raise EditRefused(f"{target} matches {len(matches)} memos ({', '.join(matches)}); name it so it is unique")
        if supersedes_key(matches[0]) == supersedes_key(corpus_name):
            raise EditRefused("a memo cannot supersede itself")
        try:
            changed = add_supersedes_target(after, matches[0])
        except (SupersedesDeclaresNothing, ValueError) as exc:
            raise EditRefused(str(exc)) from exc
        if changed is None:
            raise EditRefused(f"{corpus_name} already supersedes {matches[0]}")
        after = changed
        summary.append(f"supersedes {matches[0]}")

    dates: dict[str, str | None] = {}
    for key, value in (("valid_from", valid_from), ("valid_until", valid_until)):
        if value is KEEP:
            continue
        if value is not None and not isinstance(value, str):
            raise EditRefused(f"{key} must be a date or empty")
        checked = _check_date(key, value) if value else None
        dates[key] = checked
        try:
            changed = set_frontmatter_scalar(after, key, checked)
        except ValueError as exc:
            raise EditRefused(str(exc)) from exc
        if changed is not None:
            after = changed
            summary.append(f"{key} {checked}" if checked else f"{key} cleared")
    meta = parse_document(after.decode("utf-8-sig", errors="replace")).meta
    try:
        start, end = validity_bounds(meta)
    except ValueError as exc:
        raise EditRefused(str(exc)) from exc
    if start is not None and end is not None and end < start:
        raise EditRefused("valid_until is before valid_from")

    if status is not KEEP:
        if status is not None and status not in STATUS_VOCABULARY:
            raise EditRefused(f"status must be one of {', '.join(STATUS_VOCABULARY)}")
        try:
            changed = set_derived_status(after, status if isinstance(status, str) else None)
        except RewriteRefused as exc:
            raise EditRefused(str(exc)) from exc
        if changed is not None:
            after = changed
            summary.append(f"status {status}" if status else "status cleared")

    if after == before:
        raise EditRefused("nothing to change")
    diff = tuple(
        difflib.unified_diff(
            before.decode("utf-8", "replace").splitlines(),
            after.decode("utf-8", "replace").splitlines(),
            fromfile=f"{corpus_name} (now)",
            tofile=f"{corpus_name} (after)",
            n=1,
            lineterm="",
        )
    )
    return EditPlan(name=corpus_name, before=before, after=after, summary=tuple(summary), diff=diff)


class EditLedger:
    """Every edit made from the dashboard, with the bytes needed to undo it."""

    def __init__(self, root: Path) -> None:
        path = root / _LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=30.0)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edits ("
            "edit_id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT NOT NULL, editor TEXT NOT NULL, "
            "note TEXT NOT NULL, summary TEXT NOT NULL, before BLOB NOT NULL, after BLOB NOT NULL, "
            "before_sha TEXT NOT NULL, after_sha TEXT NOT NULL, edited_at TEXT NOT NULL, "
            "undone_at TEXT, undone_by TEXT)"
        )
        self._conn.commit()

    def __enter__(self) -> EditLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self._conn.close()

    def record(self, plan: EditPlan, *, editor: str, note: str, at: datetime) -> int:
        cursor = self._conn.execute(
            "INSERT INTO edits (file, editor, note, summary, before, after, before_sha, after_sha, edited_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.name, editor, note, json.dumps(list(plan.summary)), plan.before, plan.after,
                plan.before_sha, hashlib.sha256(plan.after).hexdigest(), at.isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def history(self, name: str | None = None, limit: int = 50) -> list[EditRecord]:
        query = "SELECT edit_id, file, editor, note, summary, edited_at, undone_at, undone_by FROM edits"
        args: tuple[object, ...] = ()
        if name is not None:
            query += " WHERE file = ?"
            args = (name,)
        rows = self._conn.execute(query + " ORDER BY edit_id DESC LIMIT ?", (*args, limit)).fetchall()
        return [
            EditRecord(edit_id=r[0], name=r[1], editor=r[2], note=r[3], summary=tuple(json.loads(r[4])),
                       edited_at=r[5], undone_at=r[6], undone_by=r[7])
            for r in rows
        ]

    def bytes_of(self, edit_id: int) -> tuple[str, bytes, bytes, str | None] | None:
        row = self._conn.execute(
            "SELECT file, before, after, undone_at FROM edits WHERE edit_id = ?", (edit_id,)
        ).fetchone()
        return (row[0], bytes(row[1]), bytes(row[2]), row[3]) if row else None

    def mark_undone(self, edit_id: int, *, editor: str, at: datetime) -> None:
        self._conn.execute(
            "UPDATE edits SET undone_at = ?, undone_by = ? WHERE edit_id = ? AND undone_at IS NULL",
            (at.isoformat(), editor, edit_id),
        )
        self._conn.commit()


def _require_person(editor: str, note: str) -> None:
    if not editor.strip() or not note.strip():
        raise EditRefused("an edit needs your name and a note")


def apply_edit(root: Path, name: str, *, editor: str, note: str, shown_sha: str, now: datetime, **changes: object) -> EditRecord:
    """Make the edit if the memo is still the one the person saw, and record it."""
    _require_person(editor, note)
    root = root.resolve()
    with _DECISION_LOCK:
        plan = plan_edit(root, name, **changes)  # type: ignore[arg-type]
        if plan.before_sha != shown_sha:
            raise EditRefused(f"{plan.name} changed since you reviewed the edit; review it again")
        _corpus_name, path = _memo(root, plan.name)
        atomic_write_bytes(path, plan.after)
        with EditLedger(root) as ledger:
            edit_id = ledger.record(plan, editor=editor, note=note, at=now)
    return EditRecord(edit_id=edit_id, name=plan.name, editor=editor, note=note, summary=plan.summary, edited_at=now.isoformat())


def undo_edit(root: Path, edit_id: int, *, editor: str, note: str, now: datetime) -> str:
    """Put back the bytes from before `edit_id`, if the memo is still exactly what that edit wrote."""
    _require_person(editor, note)
    root = root.resolve()
    with _DECISION_LOCK, EditLedger(root) as ledger:
        found = ledger.bytes_of(edit_id)
        if found is None:
            raise EditRefused(f"no edit {edit_id}")
        name, before, after, undone_at = found
        if undone_at is not None:
            raise EditRefused(f"edit {edit_id} was already undone")
        _corpus_name, path = _memo(root, name)
        if path.read_bytes() != after:
            raise EditRefused(f"{name} changed after edit {edit_id}; undo it by hand or edit it again")
        atomic_write_bytes(path, before)
        ledger.mark_undone(edit_id, editor=editor, at=now)
        ledger.record(
            EditPlan(name=name, before=after, after=before, summary=(f"undo of edit {edit_id}: {note}",), diff=()),
            editor=editor, note=note, at=now,
        )
    return name


def history(root: Path, name: str | None = None, limit: int = 50) -> list[EditRecord]:
    path = root.resolve() / _LEDGER
    if not path.exists():
        return []
    with EditLedger(root.resolve()) as ledger:
        return ledger.history(name, limit)
