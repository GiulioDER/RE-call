"""Recoverable, explicitly authored memo writes. Call while holding the store's write lane.

The existing authorship file also journals request identities, hashes and reserved memo ids.
It never journals memo text. A retry can finish a prepared write, but cannot resurrect a
written memo that was subsequently removed or any explicitly forgotten request memo.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from recall.atomic_write import _flush_directory, atomic_write_bytes
from recall.document import parse_document
from recall.frontmatter import supersedes_key, supersedes_targets

_ID = re.compile(r"[a-z]+-\d{4,}")
_REVISION = re.compile(r"[0-9a-f]{64}")


def revision(path: Path) -> str:
    """Match Markdown content_hash: strip the UTF-8 BOM/NULs and normalize newlines."""
    return _indexed_revision(path.read_bytes())


def _indexed_revision(data: bytes) -> str:
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "authors": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("authors"), dict):
        raise ValueError("invalid memo authorship ledger")
    if not isinstance(value.get("requests", {}), dict):
        raise ValueError("invalid memo request journal")
    return value


def _save(path: Path, ledger: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(ledger, sort_keys=True, indent=1) + "\n").encode())


def cancel_requests(ledger_path: Path, memo_id: str) -> None:
    """Persist explicit deletion intent before unlinking, including prepared publications."""
    ledger = load_ledger(ledger_path)
    changed = False
    for entry in ledger.get("requests", {}).values():
        if isinstance(entry, dict) and entry.get("memo_id") == memo_id and entry.get("state") != "cancelled":
            entry["state"] = "cancelled"
            changed = True
    if changed:
        _save(ledger_path, ledger)


def _conflict(code: str, reason: str) -> dict[str, Any]:
    return {"refused": reason, "conflict": code}


def _target_conflict(folder: Path, target: str | None, expected: str | None) -> dict[str, Any] | None:
    if target is None:
        return _conflict("revision_without_target", "expected_revision requires supersedes") if expected else None
    path = folder / f"{target}.md"
    if not _ID.fullmatch(target) or path.is_symlink() or not path.is_file():
        return _conflict("target_missing", "the replacement target is not an existing memo")
    if expected is not None and revision(path) != expected:
        return _conflict("revision_mismatch", "the replacement target changed since it was read")
    for other in folder.glob("*.md"):
        if other.is_symlink():
            continue
        meta = parse_document(other.read_text(encoding="utf-8")).meta
        if any(supersedes_key(ref) == target for ref in supersedes_targets(meta.get("supersedes"))):
            return _conflict("stale_target", "the replacement target already has a successor")
    return None


def _create(path: Path, body: bytes) -> None:
    """Publish complete bytes exclusively. A human-created destination is never replaced."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".recall-memo-", suffix=".tmp")
    os.close(fd)
    staged = Path(name)
    try:
        atomic_write_bytes(staged, body)
        os.link(staged, path)
        _flush_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)


def write_memo(
    folder: Path, ledger_path: Path, body: str, *, kind: str, author: str,
    supersedes: str | None, request_id: str | None, expected_revision: str | None,
    include_receipt: bool,
) -> dict[str, Any]:
    """Preserve legacy replies; guarded callers receive a revision and recoverable write receipt."""
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()
                                   or len(request_id) > 128):
        return {"refused": "request_id must be a nonempty string of at most 128 characters"}
    if expected_revision is not None and not _REVISION.fullmatch(expected_revision):
        return {"refused": "expected_revision must be a lowercase SHA-256 digest"}
    data = body.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    identity = hashlib.sha256(json.dumps(
        [digest, kind, author, supersedes, expected_revision], separators=(",", ":"),
    ).encode()).hexdigest()
    ledger = load_ledger(ledger_path)
    authors = ledger["authors"]
    requests = ledger.get("requests", {})
    entry = requests.get(request_id) if request_id is not None else None
    replayed = entry is not None
    if entry is not None:
        if not isinstance(entry, dict) or entry.get("identity") != identity:
            return _conflict("request_mismatch", "request_id was already used for different content or preconditions")
        memo_id = entry.get("memo_id")
        if (not isinstance(memo_id, str) or not _ID.fullmatch(memo_id)
                or authors.get(memo_id) != author or entry.get("state") not in {"prepared", "written", "cancelled"}):
            raise ValueError("invalid memo request reservation")
        if entry["state"] == "cancelled":
            return _conflict("memo_missing", "the request's memo was explicitly forgotten; retry cannot recreate it")
        path = folder / f"{memo_id}.md"
        if path.is_symlink() or (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            return _conflict("memo_changed", "the request's memo changed; retry cannot overwrite it")
        if not path.exists() and entry["state"] == "written":
            return _conflict("memo_missing", "the request's memo was removed; retry cannot recreate it")
        if not path.exists():
            conflict = _target_conflict(folder, supersedes, expected_revision)
            if conflict:
                return conflict
            _create(path, data)
    else:
        conflict = _target_conflict(folder, supersedes, expected_revision)
        if conflict:
            return conflict
        folder.mkdir(parents=True, exist_ok=True)
        stems = [p.stem for p in folder.glob("*.md")]
        numbers = [int(name.rsplit("-", 1)[1]) for name in (*authors, *stems) if _ID.fullmatch(name)]
        memo_id = f"{kind}-{max(numbers, default=0) + 1:04d}"
        path = folder / f"{memo_id}.md"
        if request_id is not None:
            entry = {"identity": identity, "memo_id": memo_id, "state": "prepared"}
            requests[request_id] = entry
            ledger["requests"] = requests
            authors[memo_id] = author
            _save(ledger_path, ledger)  # durable intent before the memo becomes visible
        _create(path, data)
    authors[memo_id] = author
    if entry is not None:
        entry["state"] = "written"
    _save(ledger_path, ledger)
    result: dict[str, Any] = {"memo_id": memo_id}
    if include_receipt or request_id is not None or expected_revision is not None:
        result["receipt"] = {"revision": _indexed_revision(data), "supersedes": supersedes,
                             "write_state": "written", "replayed": replayed}
    return result
