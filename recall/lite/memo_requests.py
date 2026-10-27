"""Recoverable, explicitly authored memo writes. Call while holding the store's write lane.

The existing authorship file also journals request identities, hashes and reserved memo ids.
It never journals memo text. A retry can finish a prepared write, but cannot resurrect a
written memo that was subsequently removed or any explicitly forgotten request memo.

The hashes are unsalted SHA-256 over the memo bytes, so while a memo exists they say nothing its
file does not; once it is forgotten, a guessable template and a short value (a four-digit code)
would make them an enumeration oracle. `cancel_requests` therefore drops every hash derived from a
forgotten memo's bytes: its own requests' identities and content hashes, and the identity of each
correction that named it as its target, whose revision precondition hashed it.

A reservation is not authorship: `authors` names only memos whose file this module created, so a
file somebody else put in the folder under a reserved name is never claimed. The journal keeps
every prepared entry, and a completed or cancelled entry for `REQUEST_RETENTION_SECONDS` and at
most `REQUEST_RETENTION_COUNT` of them; a request id older than that is a new request.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from recall.atomic_write import _flush_directory, atomic_write_bytes
from recall.document import parse_document
from recall.frontmatter import supersedes_key, supersedes_targets

MEMO_ID = re.compile(r"^[a-z]+-\d{4,}$")
_REVISION = re.compile(r"[0-9a-f]{64}")

#: Completed and cancelled request entries older than this are pruned at the next write.
REQUEST_RETENTION_SECONDS = 30 * 24 * 3600
#: At most this many completed and cancelled entries are kept, the newest first.
REQUEST_RETENTION_COUNT = 1000

# What os.link raises where the filesystem has no hard links (FAT32 and exFAT, many SMB shares,
# some FUSE mounts): EPERM on Linux, EINVAL or ENOTSUP elsewhere, and on Windows the winerrors
# ERROR_INVALID_FUNCTION and ERROR_NOT_SUPPORTED.
_LINK_UNSUPPORTED_ERRNOS = frozenset(
    code for code in (errno.EPERM, errno.EINVAL, errno.ENOSYS, getattr(errno, "ENOTSUP", None),
                      getattr(errno, "EOPNOTSUPP", None)) if code is not None
)
_LINK_UNSUPPORTED_WINERRORS = frozenset({1, 50})


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


def _prune(requests: dict[str, Any], now: float) -> None:
    """Drop completed and cancelled entries past the retention age or count. Prepared ones stay."""
    done = []
    for key, entry in list(requests.items()):
        if not isinstance(entry, dict) or entry.get("state") not in {"written", "cancelled"}:
            continue
        at = entry.get("at")
        if not isinstance(at, (int, float)):
            entry["at"] = at = now  # an entry from before timestamps: its retention starts now
        if now - at > REQUEST_RETENTION_SECONDS:
            del requests[key]
        else:
            done.append((at, key))
    done.sort(reverse=True)
    for _, key in done[REQUEST_RETENTION_COUNT:]:
        del requests[key]


def cancel_requests(ledger_path: Path, memo_id: str) -> dict[str, Any]:
    """Persist explicit deletion intent before unlinking, including prepared publications, and
    drop every hash the journal derived from the memo's bytes.

    The memo's own requests become `cancelled` and `erased`, keeping only the memo id, the state
    and the time; a retry of one gets `memo_missing` whatever its text, since an answer that
    depended on the text would be the same oracle. A correction that named the memo as its target
    loses its identity, which hashed the memo's revision, and is marked `target_forgotten`: its
    retry is compared on the content hash alone, so a lost reply is still replayed, and a write it
    never finished is refused as `target_missing`, as before.

    Returns the changed entries as they were, for `restore_requests` when the unlink fails.
    """
    ledger = load_ledger(ledger_path)
    previous: dict[str, Any] = {}
    now = time.time()
    for key, entry in ledger.get("requests", {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("memo_id") == memo_id and not entry.get("erased"):
            previous[key] = dict(entry)
            for field in ("identity", "content", "supersedes"):
                entry.pop(field, None)
            entry.update(state="cancelled", at=now, erased=True)
        elif entry.get("supersedes") == memo_id and "identity" in entry:
            previous[key] = dict(entry)
            del entry["identity"]
            entry["target_forgotten"] = True
    if previous:
        _save(ledger_path, ledger)
    return previous


def restore_requests(ledger_path: Path, previous: dict[str, Any]) -> None:
    """Undo `cancel_requests` for a memo whose file is still there, so a retry can finish it."""
    if not previous:
        return
    ledger = load_ledger(ledger_path)
    requests = ledger.get("requests", {})
    for key, entry in previous.items():
        current = requests.get(key)
        if isinstance(current, dict) and current.get("memo_id") == entry.get("memo_id"):
            requests[key] = entry
    _save(ledger_path, ledger)


def _conflict(code: str, reason: str) -> dict[str, Any]:
    return {"refused": reason, "conflict": code}


def _target_conflict(folder: Path, target: str | None, expected: str | None) -> dict[str, Any] | None:
    if target is None:
        return _conflict("revision_without_target", "expected_revision requires supersedes") if expected else None
    path = folder / f"{target}.md"
    if not MEMO_ID.fullmatch(target) or path.is_symlink() or not path.is_file():
        return _conflict("target_missing", "the replacement target is not an existing memo")
    if expected is not None and revision(path) != expected:
        return _conflict("revision_mismatch", "the replacement target changed since it was read")
    for other in folder.glob("*.md"):
        if other.is_symlink():
            continue
        try:
            raw = other.read_bytes()
        except OSError:
            continue  # a file nobody can read declares nothing this write could honour
        if b"supersedes" not in raw:
            continue  # parse only the files that can declare a successor
        meta = parse_document(raw.decode("utf-8", errors="replace")).meta
        if any(supersedes_key(ref) == target for ref in supersedes_targets(meta.get("supersedes"))):
            return _conflict("stale_target", "the replacement target already has a successor")
    return None


def _link_unsupported(exc: OSError) -> bool:
    if isinstance(exc, FileExistsError):
        return False
    return exc.errno in _LINK_UNSUPPORTED_ERRNOS or getattr(exc, "winerror", None) in _LINK_UNSUPPORTED_WINERRORS


def _create_exclusive(path: Path, body: bytes) -> None:
    """Exclusive create for a filesystem without hard links: never over an existing file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        path.unlink(missing_ok=True)  # this call created it; a partial memo must not take the id
        raise


def _create(path: Path, body: bytes) -> None:
    """Publish complete bytes exclusively. A human-created destination is never replaced.

    A hard link publishes the staged bytes atomically; where the filesystem has none, an
    exclusive create writes and fsyncs them in place, as the write path did before links.
    """
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".recall-memo-", suffix=".tmp")
    os.close(fd)
    staged = Path(name)
    try:
        atomic_write_bytes(staged, body)
        try:
            os.link(staged, path)
        except OSError as exc:
            if not _link_unsupported(exc):
                raise
            _create_exclusive(path, body)
        _flush_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)


def _next_number(folder: Path, ledger: dict[str, Any]) -> int:
    """One past the largest id in the folder, the authorship ledger or a request reservation."""
    reserved = [e.get("memo_id") for e in ledger.get("requests", {}).values() if isinstance(e, dict)]
    names = [*ledger["authors"], *(p.stem for p in folder.glob("*.md")), *reserved]
    numbers = [int(n.rsplit("-", 1)[1]) for n in names if isinstance(n, str) and MEMO_ID.fullmatch(n)]
    return max(numbers, default=0) + 1


def _publish(
    folder: Path, ledger_path: Path, ledger: dict[str, Any], data: bytes, kind: str, number: int,
    entry: dict[str, Any] | None,
) -> str:
    """Create the memo under the first free id from `number`, journalling the reservation first.

    A name taken between the folder scan and the create moves to the next id (and moves the
    reservation with it), as the write path always did; the taken file is never claimed.
    """
    while True:
        memo_id = f"{kind}-{number:04d}"
        if entry is not None:
            entry["memo_id"] = memo_id
            _save(ledger_path, ledger)  # durable intent before the memo becomes visible
        try:
            _create(folder / f"{memo_id}.md", data)
            return memo_id
        except FileExistsError:
            number += 1


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
    # The identity without the revision precondition, which hashes the target's bytes: what a
    # retry is compared on once that target is forgotten (`cancel_requests`).
    content = hashlib.sha256(json.dumps(
        [digest, kind, author, supersedes], separators=(",", ":"),
    ).encode()).hexdigest()
    ledger = load_ledger(ledger_path)
    authors = ledger["authors"]
    requests = ledger.setdefault("requests", {})
    entry = requests.get(request_id) if request_id is not None else None
    replayed = entry is not None
    changed = True
    if entry is not None:
        if isinstance(entry, dict) and entry.get("erased"):
            # Forgotten: nothing of the memo is left to compare a retry with, and every retry
            # gets this answer, whatever its text.
            return _conflict("memo_missing", "the request's memo was explicitly forgotten; retry cannot recreate it")
        if not isinstance(entry, dict) or (
            entry.get("content") != content if entry.get("target_forgotten") else entry.get("identity") != identity
        ):
            return _conflict("request_mismatch", "request_id was already used for different content or preconditions")
        memo_id = entry.get("memo_id")
        state = entry.get("state")
        # A completed write is authored; a reservation that never completed (prepared, or
        # cancelled before completing) need not be.
        if (not isinstance(memo_id, str) or not MEMO_ID.fullmatch(memo_id)
                or state not in {"prepared", "written", "cancelled"}
                or authors.get(memo_id, author if state != "written" else None) != author):
            raise ValueError("invalid memo request reservation")
        if state == "cancelled":
            return _conflict("memo_missing", "the request's memo was explicitly forgotten; retry cannot recreate it")
        path = folder / f"{memo_id}.md"
        ours = path.exists() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        if state == "written":
            if path.is_symlink() or (path.exists() and not ours):
                return _conflict("memo_changed", "the request's memo changed; retry cannot overwrite it")
            if not path.exists():
                return _conflict("memo_missing", "the request's memo was removed; retry cannot recreate it")
            changed = False  # a pure replay of a completed write rewrites nothing
        elif not ours:
            # Never confirmed: the file under the reserved name, if any, is not provably ours, so
            # it is left alone and the reservation moves to the next free id.
            conflict = _target_conflict(folder, supersedes, expected_revision)
            if conflict:
                return conflict
            folder.mkdir(parents=True, exist_ok=True)
            free = not path.exists() and not path.is_symlink()
            if not free and state == "prepared" and authors.get(memo_id) == author:
                # A ledger from before this module wrote `authors` only after the create recorded
                # the reservation as authored; the file there is not ours, so that row goes too.
                del authors[memo_id]
            number = int(memo_id.rsplit("-", 1)[1]) if free else _next_number(folder, ledger)
            memo_id = _publish(folder, ledger_path, ledger, data, kind, number, entry)
    else:
        conflict = _target_conflict(folder, supersedes, expected_revision)
        if conflict:
            return conflict
        folder.mkdir(parents=True, exist_ok=True)
        if request_id is not None:
            entry = {"identity": identity, "content": content, "state": "prepared"}
            if supersedes is not None:
                entry["supersedes"] = supersedes  # a memo id: which correction a forget must re-key
            requests[request_id] = entry
        memo_id = _publish(folder, ledger_path, ledger, data, kind, _next_number(folder, ledger), entry)
    if changed:
        authors[memo_id] = author
        if entry is not None:
            entry["state"] = "written"
            entry["at"] = time.time()
        _prune(requests, time.time())
        if not requests:
            ledger.pop("requests", None)  # a ledger that never journalled a request keeps its shape
        _save(ledger_path, ledger)
    result: dict[str, Any] = {"memo_id": memo_id}
    if include_receipt or request_id is not None or expected_revision is not None:
        result["receipt"] = {"revision": _indexed_revision(data), "supersedes": supersedes,
                             "write_state": "written", "replayed": replayed}
    return result
