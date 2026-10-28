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
file somebody else put in the folder under a reserved name is never claimed, unless its bytes are
exactly the memo the request would write, which cannot be told apart from that request's own
interrupted publication. The journal keeps every prepared entry, and a completed or cancelled
entry for at least `REQUEST_RETENTION_SECONDS` while it is among the newest
`REQUEST_RETENTION_COUNT`; once a later write prunes it, its request id is a new request.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recall.atomic_write import _flush_directory, atomic_write_bytes
from recall.document import parse_document
from recall.frontmatter import supersedes_key, supersedes_targets

MEMO_ID = re.compile(r"^[a-z]+-\d{4,}$")
_REVISION = re.compile(r"[0-9a-f]{64}")

# Request journal states: reserved, published and recorded, explicitly forgotten.
_PREPARED, _WRITTEN, _CANCELLED = "prepared", "written", "cancelled"
_DONE = frozenset({_WRITTEN, _CANCELLED})
_STATES = _DONE | {_PREPARED}
_ABSENT = object()  # no authors row, as distinct from a row holding JSON null

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
_ERROR_INVALID_FUNCTION = 1
_ERROR_NOT_SUPPORTED = 50
_LINK_UNSUPPORTED_WINERRORS = frozenset({_ERROR_INVALID_FUNCTION, _ERROR_NOT_SUPPORTED})

#: Folders where `os.link` was found unsupported; later writes there skip the staging.
_NO_LINK_FOLDERS: set[str] = set()


def _memo_id(kind: str, number: int) -> str:
    return f"{kind}-{number:04d}"


def _memo_number(memo_id: str) -> int:
    return int(memo_id.rsplit("-", 1)[1])


def revision(path: Path) -> str:
    """Match Markdown content_hash: strip the UTF-8 BOM/NULs and normalize newlines."""
    return _indexed_revision(path.read_bytes())


def _indexed_revision(data: bytes) -> str:
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: The newest ledger layout this module reads: 2 carries the request journal (`requests`). A
#: release older than the journal ignores the version and rewrites the ledger without it, so the
#: version cannot protect against those; it stops a newer layout being read as this one.
LEDGER_VERSION = 2
#: The longest `request_id`, in code points.
REQUEST_ID_MAX_LEN = 128


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "authors": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("authors"), dict):
        raise ValueError("invalid memo authorship ledger")
    version = value.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version > LEDGER_VERSION:
        raise ValueError(f"memo ledger version {version!r} is newer than this release reads")
    if not isinstance(value.get("requests", {}), dict):
        raise ValueError("invalid memo request journal")
    return value


def _save(path: Path, ledger: dict[str, Any]) -> None:
    if "requests" in ledger:
        ledger["version"] = LEDGER_VERSION
    atomic_write_bytes(path, (json.dumps(ledger, sort_keys=True, indent=1) + "\n").encode())


def _prune(requests: dict[str, Any], now: float) -> None:
    """Drop completed and cancelled entries past the retention age or count. Prepared ones stay."""
    done = []
    for key, entry in list(requests.items()):
        if not isinstance(entry, dict) or entry.get("state") not in _DONE:
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
            entry.update(state=_CANCELLED, at=now, erased=True)
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


def _target_conflict(
    folder: Path, target: str | None, expected: str | None,
    has_successor: Callable[[str, str | None], bool] | None = None, own: str | None = None,
) -> dict[str, Any] | None:
    """The refusal a correction of `target` meets, or None. `has_successor(target, own)` asks the
    store whether an indexed document other than memo `own` (the retried request's own memo)
    already supersedes the target, as the store resolves references; the folder scan below also
    sees successors written but not indexed yet."""
    if target is None:
        return _conflict("revision_without_target", "expected_revision requires supersedes") if expected else None
    path = folder / f"{target}.md"
    if not MEMO_ID.fullmatch(target) or path.is_symlink() or not path.is_file():
        return _conflict("target_missing", "the replacement target is not an existing memo")
    if expected is not None:
        try:
            current = revision(path)
        except (OSError, UnicodeDecodeError):
            # Not UTF-8 any more (re-saved in another encoding) or not readable: the indexer
            # cannot have served a revision of these bytes, so the target changed since it was read.
            current = None
        if current != expected:
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
    if has_successor is not None and has_successor(target, own):
        return _conflict("stale_target", "the replacement target already has a successor")
    return None


def sweep_staged(folder: Path) -> None:
    """Remove staging files a crash inside `_create` left behind (they hold a whole memo, or a
    second link to a published one). Call under the write lane, where no `_create` is running."""
    for pattern in (".recall-memo-*.tmp", "..recall-memo-*.tmp.*.tmp"):
        for staged in folder.glob(pattern):
            try:
                staged.unlink()
            except OSError:
                pass  # held open elsewhere (Windows): the next sweep takes it


def _link_unsupported(exc: OSError) -> bool:
    if isinstance(exc, FileExistsError):
        return False
    return exc.errno in _LINK_UNSUPPORTED_ERRNOS or getattr(exc, "winerror", None) in _LINK_UNSUPPORTED_WINERRORS


def _discard(path: Path) -> None:
    """Remove a file this call created; a failure here must not hide the error being raised."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _create_exclusive(path: Path, body: bytes) -> None:
    """Exclusive create for a filesystem without hard links: never over an existing file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        fh = os.fdopen(fd, "wb")
    except BaseException:
        # fdopen owns the descriptor only once it returns, and an open handle makes the unlink
        # fail on Windows (as in recall.atomic_write). io.open may already have closed it, so a
        # failing close must not hide the error being raised.
        try:
            os.close(fd)
        except OSError:
            pass
        _discard(path)
        raise
    try:
        with fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        _discard(path)  # this call created it; a partial memo must not take the id
        raise


def _create(path: Path, body: bytes) -> None:
    """Publish complete bytes exclusively. A human-created destination is never replaced.

    A hard link publishes the staged bytes atomically; where the filesystem has none, an
    exclusive create writes and fsyncs them in place, as the write path did before links. Once a
    folder has refused a link, later writes there go straight to the exclusive create.
    """
    folder = str(path.parent)
    if folder in _NO_LINK_FOLDERS:
        _create_exclusive(path, body)
        _flush_directory(path.parent)
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".recall-memo-", suffix=".tmp")
    staged = Path(name)
    try:
        # Written and fsynced through the descriptor mkstemp returned: one staging file, and the
        # one directory flush below, before the link publishes it.
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            # fdopen owns the descriptor only once it returns, and an open handle makes the
            # unlink below fail on Windows (as in `_create_exclusive`). io.open may already have
            # closed it, so a failing close must not hide the error being raised.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staged, path)
        except OSError as exc:
            if not _link_unsupported(exc):
                raise
            _NO_LINK_FOLDERS.add(folder)
            _create_exclusive(path, body)
        _flush_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)


def _read_regular(path: Path) -> bytes | None:
    """The bytes of a regular file at `path`, or None for no file, a symlink or a directory.

    A regular file that cannot be read now raises: it may be the request's own memo, held by
    another program, and a later retry can still adopt it.
    """
    if path.is_symlink() or not path.is_file():
        return None
    return path.read_bytes()


def _next_number(folder: Path, ledger: dict[str, Any]) -> int:
    """One past the largest id in the folder, the authorship ledger or a request reservation."""
    reserved = [e.get("memo_id") for e in ledger.get("requests", {}).values() if isinstance(e, dict)]
    names = [*ledger["authors"], *(p.stem for p in folder.glob("*.md")), *reserved]
    numbers = [_memo_number(n) for n in names if isinstance(n, str) and MEMO_ID.fullmatch(n)]
    return max(numbers, default=0) + 1


def _publish(
    folder: Path, ledger_path: Path, ledger: dict[str, Any], data: bytes, *, kind: str, number: int,
    entry: dict[str, Any] | None,
) -> str:
    """Create the memo under the first free id from `number`, journalling the reservation first.

    A name taken between the folder scan and the create moves to the next id (and moves the
    reservation with it), as the write path always did; the taken file is never claimed.
    """
    while True:
        memo_id = _memo_id(kind, number)
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
    include_receipt: bool, has_successor: Callable[[str, str | None], bool] | None = None,
) -> dict[str, Any]:
    """Write `body` as a memo. Call while holding the store's write lane.

    Replies `{memo_id}` (legacy calls), `{memo_id, receipt}` when a receipt is asked for or a
    `request_id` or `expected_revision` is given, `{refused}` for a malformed `request_id` or
    `expected_revision`, and `{refused, conflict}` for a precondition: `request_mismatch`,
    `revision_without_target`, `target_missing`, `revision_mismatch`, `stale_target`,
    `memo_missing` or `memo_changed` (docs/MEMO_CORRECTIONS.md).
    """
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()
                                   or len(request_id) > REQUEST_ID_MAX_LEN):
        return {"refused": f"request_id must be a nonempty string of at most {REQUEST_ID_MAX_LEN} characters"}
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
    sweep_staged(folder)
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
        # cancelled before completing) may be unauthored, or authored by this author.
        owner = authors.get(memo_id, _ABSENT) if isinstance(memo_id, str) else _ABSENT
        owner_ok = owner == author if state == _WRITTEN else owner is _ABSENT or owner == author
        if not isinstance(memo_id, str) or not MEMO_ID.fullmatch(memo_id) or state not in _STATES or not owner_ok:
            raise ValueError("invalid memo request reservation")
        if state == _CANCELLED:
            return _conflict("memo_missing", "the request's memo was explicitly forgotten; retry cannot recreate it")
        path = folder / f"{memo_id}.md"
        current = _read_regular(path)
        ours = current is not None and hashlib.sha256(current).hexdigest() == digest
        if state == _WRITTEN:
            if path.is_symlink() or (path.exists() and not ours):
                return _conflict("memo_changed", "the request's memo changed; retry cannot overwrite it")
            if not path.exists():
                return _conflict("memo_missing", "the request's memo was removed; retry cannot recreate it")
            changed = False  # a pure replay of a completed write rewrites nothing
        elif not ours:
            # Never confirmed: the file under the reserved name, if any, is not provably ours, so
            # it is left alone and the reservation moves to the next free id.
            conflict = _target_conflict(
                folder, supersedes, expected_revision, has_successor,
                # Rows under the reserved name are the request's own memo only while no file
                # holds that name; a file there is somebody else's, and so is any claim it makes.
                own=memo_id if current is None and not path.is_symlink() else None,
            )
            if conflict:
                return conflict
            folder.mkdir(parents=True, exist_ok=True)
            if current is not None and len(current) < len(data) and data.startswith(current):
                # The start of this very memo: an exclusive create cut short (the fallback where
                # links are missing). Publishing again would leave a second, partial copy.
                return _conflict("memo_changed", "the request's memo was left incomplete; retry cannot overwrite it")
            free = not path.exists() and not path.is_symlink()
            if not free and state == _PREPARED and authors.get(memo_id) == author:
                # A ledger from before this module wrote `authors` only after the create recorded
                # the reservation as authored; the file there is not ours, so that row goes too.
                del authors[memo_id]
            number = _memo_number(memo_id) if free else _next_number(folder, ledger)
            memo_id = _publish(folder, ledger_path, ledger, data, kind=kind, number=number, entry=entry)
    else:
        conflict = _target_conflict(folder, supersedes, expected_revision, has_successor)
        if conflict:
            return conflict
        folder.mkdir(parents=True, exist_ok=True)
        if request_id is not None:
            entry = {"identity": identity, "content": content, "state": _PREPARED}
            if supersedes is not None:
                entry["supersedes"] = supersedes  # a memo id: which correction a forget must re-key
            requests[request_id] = entry
        memo_id = _publish(folder, ledger_path, ledger, data, kind=kind, number=_next_number(folder, ledger), entry=entry)
    if changed:
        authors[memo_id] = author
        if entry is not None:
            entry["state"] = _WRITTEN
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
