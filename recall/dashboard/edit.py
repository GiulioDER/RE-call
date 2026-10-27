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
  can be undone, and an undo is refused when the memo was changed again since. Recovery bytes
  are committed before replacement; reopening the ledger reconciles interrupted edits.

The writers are the byte-preserving ones in `recall.frontmatter` and `recall.rewrite`, so a BOM,
line endings and every line nobody asked to change survive an edit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from recall.atomic_write import atomic_write_bytes
from recall.dashboard.review import _DECISION_LOCK, ReviewRefused, memo_files
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
from recall.lint import (
    corpus_frontmatter,
    describe_cycle,
    supersession_cycle_closed_by,
    supersession_graph_of,
)
from recall.rewrite import RewriteRefused, _derived_value, set_derived_status
from recall.state_dir import make_parent
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
    root = root.resolve()
    try:
        _corpus_name, path = resolve_memo(root, name)
    except StaleReportRefused as exc:
        raise EditRefused(str(exc)) from exc
    listed = _as_listed(root, path)
    return encodable_name(listed.relative_to(root).as_posix()), listed


def _as_listed(root: Path, path: Path) -> Path:
    """`path` spelled as its directories list it.

    On a filesystem that ignores case but keeps it (macOS by default), ``Foo.md`` opens
    ``foo.md`` and `Path.resolve` keeps the spelling it was given. Keyed by that spelling, the
    memo being edited is a different name from the corpus entry that is the same file, so the
    self-supersession and cycle checks compare it with nothing.
    """
    listed = root
    for part in path.relative_to(root).parts:
        try:
            entries = os.listdir(listed)
        except OSError:
            # A directory that can be passed through but not listed: keep the spelling given,
            # which `resolve_memo` has already confirmed opens a file in the corpus.
            return path
        if part not in entries:
            same = [
                entry
                for entry in entries
                if entry.casefold() == part.casefold()
                and os.path.samefile(listed / entry, listed / part)
            ]
            if len(same) == 1:
                part = same[0]
        listed = listed / part
    return listed


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
    """The memos an edit may name, by key: `memo_files`, which skips hidden directories."""
    names: dict[str, list[str]] = {}
    for path in memo_files(root):
        names.setdefault(supersedes_key(path.name), []).append(
            encodable_name(path.relative_to(root).as_posix())
        )
    return names


def _refuse_cycles(root: Path, path: Path, after: bytes, added: list[str]) -> None:
    """Refuse an edit whose new `supersedes:` references would close a cycle.

    Judged on the corpus as it would stand after the edit, this memo's removals included, so the
    preview refuses what `recall lint` would report as `supersession-cycle` after the write. A
    cycle makes search abstain on every memo in it. The corpus is the one lint reads
    (`corpus_frontmatter`), hidden directories included, not the narrower list of memos an edit
    may name: a cycle through an Obsidian `.trash/` memo is still one lint reports.
    """
    if not added:
        return
    documents: list[tuple[str, Mapping[str, object]]] = [
        (other.name, meta) for other, meta in corpus_frontmatter(root) if other != path
    ]
    after_meta = dict(parse_document(after.decode("utf-8-sig", errors="replace")).meta)
    documents.append((path.name, after_meta))
    graph = supersession_graph_of(documents)
    for target in added:
        chain = supersession_cycle_closed_by(graph, path.name, target)
        if chain is not None:
            raise EditRefused(describe_cycle(path.name, target, chain))


def _restored_targets(before: bytes, after: bytes) -> list[str]:
    """The `supersedes:` references `before` declares and `after` does not: what an undo adds."""

    def declared(raw: bytes) -> tuple[str, ...]:
        meta = parse_document(raw.decode("utf-8-sig", errors="replace")).meta
        return supersedes_targets(meta.get("supersedes"))

    present = {supersedes_key(target) for target in declared(after)}
    return [target for target in declared(before) if supersedes_key(target) not in present]


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
    # Listed only when a reference is added: a date or status edit names no other memo.
    names = _corpus_names(root) if add_supersedes else {}
    added: list[str] = []

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
        added.append(matches[0])
        summary.append(f"supersedes {matches[0]}")
    _refuse_cycles(root, path, after, added)

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
        # Refuses an empty window too (`valid_from` after `valid_until`).
        validity_bounds(meta)
    except ValueError as exc:
        raise EditRefused(str(exc)) from exc

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
        make_parent(path)
        # Recovery must distinguish a dead writer from one between its durable prepare
        # and replacement. A separate database holds a process-shared reservation across
        # commits on the history database, and SQLite releases it after process death.
        self._lock_conn = sqlite3.connect(str(path.with_name("edits-lock.sqlite3")), timeout=30.0)
        connection: sqlite3.Connection | None = None
        self._root = root
        self._defer_commit = False
        try:
            self._lock_conn.execute("BEGIN IMMEDIATE")
            connection = sqlite3.connect(str(path), timeout=30.0)
            self._conn = connection
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS edits ("
                "edit_id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT NOT NULL, editor TEXT NOT NULL, "
                "note TEXT NOT NULL, summary TEXT NOT NULL, before BLOB NOT NULL, after BLOB NOT NULL, "
                "before_sha TEXT NOT NULL, after_sha TEXT NOT NULL, edited_at TEXT NOT NULL, "
                "undone_at TEXT, undone_by TEXT)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS edit_recovery ("
                "operation_id TEXT PRIMARY KEY, file TEXT NOT NULL UNIQUE, "
                "before BLOB NOT NULL, after BLOB NOT NULL, summary TEXT NOT NULL, "
                "editor TEXT NOT NULL, note TEXT NOT NULL, edited_at TEXT NOT NULL, undo_id INTEGER)"
            )
            self._conn.commit()
            self._recover()
        except BaseException:
            try:
                if connection is not None:
                    connection.close()
            finally:
                self._lock_conn.close()
            raise

    def __enter__(self) -> EditLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._conn.close()
        finally:
            self._lock_conn.close()

    def record(self, plan: EditPlan, *, editor: str, note: str, at: datetime) -> int:
        cursor = self._conn.execute(
            "INSERT INTO edits (file, editor, note, summary, before, after, before_sha, after_sha, edited_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.name, editor, note, json.dumps(list(plan.summary)), plan.before, plan.after,
                plan.before_sha, hashlib.sha256(plan.after).hexdigest(), at.isoformat(),
            ),
        )
        if not self._defer_commit:
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
        if not self._defer_commit:
            self._conn.commit()

    def _pending(self, operation_id: str) -> tuple[str, bytes, bytes, str, str, str, str, int | None] | None:
        row = self._conn.execute(
            "SELECT file, before, after, summary, editor, note, edited_at, undo_id "
            "FROM edit_recovery WHERE operation_id = ?", (operation_id,),
        ).fetchone()
        if row is None:
            return None
        return (
            str(row[0]), bytes(row[1]), bytes(row[2]), str(row[3]), str(row[4]), str(row[5]),
            str(row[6]), int(row[7]) if row[7] is not None else None,
        )

    def _complete(self, operation_id: str) -> int:
        """Publish the edit, its optional undo mark, and journal removal as one transaction."""
        row = self._pending(operation_id)
        if row is None:
            raise EditRefused("the edit recovery record is missing")
        name, before, after, summary, editor, note, edited_at, undo_id = row
        plan = EditPlan(name, bytes(before), bytes(after), tuple(json.loads(summary)), ())
        at = datetime.fromisoformat(edited_at)
        edit_id: int | None = None
        self._conn.execute("BEGIN IMMEDIATE")
        self._defer_commit = True
        try:
            if undo_id is not None:
                self.mark_undone(undo_id, editor=editor, at=at)
            edit_id = self.record(plan, editor=editor, note=note, at=at)
            self._conn.execute("DELETE FROM edit_recovery WHERE operation_id = ?", (operation_id,))
            self._conn.commit()
        except (sqlite3.Error, OSError):
            # A commit may reach disk before reporting an error. The journal deletion and
            # both history changes committed together, so its absence confirms completion.
            try:
                self._conn.rollback()
                pending = self._pending(operation_id)
            except (sqlite3.Error, OSError) as exc:
                raise EditRefused(
                    "the edit's commit could not be checked; reopen its ledger to recover the durable outcome"
                ) from exc
            if edit_id is not None and pending is None:
                return edit_id
            raise
        finally:
            self._defer_commit = False
        assert edit_id is not None
        return edit_id

    def _discard(self, operation_id: str) -> None:
        self._conn.execute("DELETE FROM edit_recovery WHERE operation_id = ?", (operation_id,))
        self._conn.commit()

    def _recover(self) -> None:
        """Reconcile an interrupted replacement using durable bytes, never guessing past an edit."""
        operations = self._conn.execute("SELECT operation_id FROM edit_recovery").fetchall()
        for (operation_id,) in operations:
            row = self._pending(operation_id)
            if row is None:
                continue
            name, before, after, *_ = row
            _, path = _memo(self._root, name)
            current = path.read_bytes()
            if current == bytes(before):
                self._discard(operation_id)
            elif current == bytes(after):
                self._complete(operation_id)
            else:
                raise EditRefused(
                    f"{name} changed during an interrupted edit; recovery bytes are retained in {_LEDGER}"
                )

    def replace(
        self, plan: EditPlan, *, editor: str, note: str, at: datetime, undo_id: int | None = None,
    ) -> int:
        """Durably prepare recovery before replacement; publish history only after it succeeds."""
        operation_id = os.urandom(16).hex()
        _, path = _memo(self._root, plan.name)
        if path.read_bytes() != plan.before:
            raise EditRefused(f"{plan.name} changed since you reviewed the edit; review it again")
        try:
            self._conn.execute(
                "INSERT INTO edit_recovery "
                "(operation_id, file, before, after, summary, editor, note, edited_at, undo_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (operation_id, plan.name, plan.before, plan.after, json.dumps(list(plan.summary)),
                 editor, note, at.isoformat(), undo_id),
            )
            self._conn.commit()
            atomic_write_bytes(path, plan.after)
            return self._complete(operation_id)
        except (sqlite3.Error, OSError) as exc:
            try:
                self._conn.rollback()
                current = path.read_bytes()
                if current == plan.after:
                    atomic_write_bytes(path, plan.before)
                elif current != plan.before:
                    raise EditRefused("the memo changed again; automatic restoration was refused")
                self._discard(operation_id)
            except (sqlite3.Error, OSError, EditRefused) as recovery_exc:
                raise EditRefused(
                    f"{plan.name}: edit storage failed; recovery bytes are retained in {_LEDGER}: {recovery_exc}"
                ) from exc
            raise EditRefused(f"{plan.name}: edit storage failed; the memo was not changed: {exc}") from exc


def _require_person(editor: str, note: str) -> None:
    if not editor.strip() or not note.strip():
        raise EditRefused("an edit needs your name and a note")


def apply_edit(root: Path, name: str, *, editor: str, note: str, shown_sha: str, now: datetime, **changes: object) -> EditRecord:
    """Make the edit if the memo is still the one the person saw, and record it."""
    _require_person(editor, note)
    root = root.resolve()
    with _DECISION_LOCK:
        try:
            with EditLedger(root) as ledger:
                plan = plan_edit(root, name, **changes)  # type: ignore[arg-type]
                if plan.before_sha != shown_sha:
                    raise EditRefused(f"{plan.name} changed since you reviewed the edit; review it again")
                edit_id = ledger.replace(plan, editor=editor, note=note, at=now)
        except (sqlite3.Error, OSError) as exc:
            raise EditRefused(f"{name}: edit storage is unavailable: {exc}") from exc
    return EditRecord(edit_id=edit_id, name=plan.name, editor=editor, note=note, summary=plan.summary, edited_at=now.isoformat())


def undo_edit(root: Path, edit_id: int, *, editor: str, note: str, now: datetime) -> str:
    """Put back the bytes from before `edit_id`, if the memo is still exactly what that edit wrote."""
    _require_person(editor, note)
    root = root.resolve()
    try:
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
            # An undo can put back an edge that an edit removed, and another memo may have declared
            # the reverse since: the same cycle `plan_edit` refuses, reached the other way round.
            _refuse_cycles(root, path, before, _restored_targets(before, after))
            ledger.replace(
                EditPlan(name=name, before=after, after=before, summary=(f"undo of edit {edit_id}: {note}",), diff=()),
                editor=editor, note=note, at=now, undo_id=edit_id,
            )
    except (sqlite3.Error, OSError) as exc:
        raise EditRefused(f"edit {edit_id}: undo storage is unavailable: {exc}") from exc
    return name


def history(root: Path, name: str | None = None, limit: int = 50) -> list[EditRecord]:
    path = root.resolve() / _LEDGER
    if not path.exists():
        return []
    try:
        with _DECISION_LOCK, EditLedger(root.resolve()) as ledger:
            return ledger.history(name, limit)
    except (sqlite3.Error, OSError) as exc:
        raise EditRefused(f"edit history storage is unavailable: {exc}") from exc
