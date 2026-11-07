"""Audit records read straight from a memo directory, matched by the rules `lookup_records` uses.

The auditor brief hook (`recall_hooks.auditor_brief`, `docs/AUDIT_MEMORY_DESIGN.md`, "Auditor brief
hook") runs on the workstation that launches an auditor, where the memo files CCA writes are on
local disk and the corpus is not. An exact key lookup needs no vectors, so this reads the files
through RE-call's own front matter parser and admission check, and answers through
`recall.records.match_candidates`, the same match, best-of, `caused` and ordering code a store
lookup runs. No database, no network, no embedder.

Currency follows the store's rule too: the memos' `supersedes` claims and validity windows are
resolved by `recall.supersession.resolve_supersession_graph`, so a successor hides a record only
inside its own window, and a retraction dated in the future does not hide its target yet.

The differences from a store, stated rather than left to be found:

- A file has no chunk, so a record's `chunk_id` is None and its `chunk_ids` empty, and its `meta`
  is the five front matter fields an entry reports, not a chunk's whole metadata.
- Only memos holding an admitted record take part in the supersession graph. A memo whose record
  was refused still declares its `supersedes` edge to an indexed store; here it declares nothing.
- `match_memo_records` answers an anchor query only, with no result limit (`lookup_records`
  refuses more than `MAX_RECORDS` matches) and no source policy (so a successor is never
  `<withheld>`), and returns the list of C5 record entries, not the envelope around them.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat as stat_module
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from recall.confinement import within
from recall.document import parse_document
from recall.frontmatter import MARKDOWN_SUFFIXES, RECORD_KEY, supersedes_targets, validity_bounds
from recall.records import (
    RecordCandidate,
    admit_record,
    as_utc,
    match_candidates,
    record_parts,
    record_schema,
    validated_anchor_query,
    validated_kinds,
)
from recall.supersession import SupersessionGraph, resolve_supersession_graph

#: The suffixes the indexer reads as markdown documents.
_MEMO_SUFFIXES = MARKDOWN_SUFFIXES


@dataclass(frozen=True)
class MemoRecord:
    """One memo's admitted `recall_record`, with the front matter a C5 entry reports beside it.

    `file` is the memo's path relative to the directory it was loaded from, in posix form: the
    key the indexer stores as `file`, and so the key a `supersedes:` reference resolves against.
    """

    source: str
    record: dict[str, Any]
    valid_from: str | None
    supersedes: tuple[str, ...]
    file: str = ""
    name: str | None = None
    description: str | None = None
    valid_until: str | None = None


class LoadedMemoRecords(list[MemoRecord]):
    """The records `load_memo_records` admitted, and `skipped`: each refused memo and why."""

    skipped: list[tuple[str, str]]

    def __init__(self, records: Sequence[MemoRecord] = (), skipped: Sequence[tuple[str, str]] = ()) -> None:
        super().__init__(records)
        self.skipped = list(skipped)


#: str(path) -> ((mtime_ns, size), a record, a refusal reason, or None for a memo without one).
_CACHE: dict[str, tuple[tuple[int, int], MemoRecord | str | None]] = {}
#: Bumped whenever what a cache entry holds, or how a memo is read into one, changes.
_CACHE_FORMAT = 3
#: The modules that decide what a memo parses into. Their files' stamps are part of the cache tag,
#: so changing how a memo is parsed or admitted re-reads every memo, not only a schema change.
_READER_MODULES = ("recall.document", "recall.frontmatter", "recall.records", "recall.records_files")
#: A temporary file left by a write that was killed is removed by the next full write past this age.
_STALE_TEMP_SECONDS = 60.0
#: The directories the indexer's default walk never enters (`recall.index._DEFAULT_INDEX_EXCLUDED_DIRS`).
#: Restated rather than imported, because importing `recall.index` costs over a second and this module
#: is on a hook's path; `tests/test_records_files.py` pins the two sets equal.
_EXCLUDED_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "site-packages", "__pycache__"}
)
#: The names and suffixes the indexer's link-target rule refuses (`recall.index._safe_link_target`),
#: restated for the same reason and pinned the same way.
_EXCLUDED_NAMES = frozenset({".env", ".env.local", "tokens.json", "credentials.json", "secrets.json", "queries.json"})
_EXCLUDED_SUFFIXES = frozenset({".json", ".toml", ".yaml", ".yml"})

_Entries = dict[str, tuple[tuple[int, int], MemoRecord | str | None]]


def _reader_files() -> list[str]:
    files: list[str] = []
    for name in _READER_MODULES:
        module = sys.modules.get(name) or importlib.import_module(name)
        files.append(str(getattr(module, "__file__", "") or ""))
    return files


def _cache_tag() -> str:
    """What a persisted cache was built by: the entry format, the schema records were admitted
    under, and the size and modification time of every module that reads a memo. A cache with
    another tag is ignored, so any change to the reading code re-reads every memo."""
    parts = [str(_CACHE_FORMAT), json.dumps(record_schema(), sort_keys=True)]
    for file in _reader_files():
        try:
            found = os.stat(file)
            parts.append(f"{file}:{found.st_size}:{found.st_mtime_ns}")
        except OSError:
            parts.append(f"{file}:?")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{_CACHE_FORMAT}:{digest}"


def _to_json(parsed: MemoRecord | str | None) -> object:
    if isinstance(parsed, MemoRecord):
        return {**asdict(parsed), "supersedes": list(parsed.supersedes)}
    if isinstance(parsed, str):
        return {"skip": parsed}
    return None


def _from_json(value: object) -> MemoRecord | str | None:
    if isinstance(value, dict) and isinstance(value.get("skip"), str):
        return str(value["skip"])
    if isinstance(value, dict) and isinstance(value.get("record"), dict):
        return MemoRecord(**{**value, "supersedes": tuple(value.get("supersedes") or ())})
    if value is None:
        return None
    raise ValueError("not a cache entry")


def _dumps(value: object) -> str:
    # ASCII on purpose: `ensure_ascii=False` cannot encode a lone surrogate (which `json.loads`
    # decodes from a memo's `\udc00` escape), and the failure would leave the cache unwritten
    # forever. The escapes read back to the same strings.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _delta_path(path: Path) -> Path:
    return path.with_name(path.name + ".delta")


def _read_cache_file(path: Path, tag: str) -> tuple[_Entries, int]:
    """The persisted entries: the base file, then the delta journal's lines in order (later wins),
    and how many persisted rows were dead weight: a base or journal line from another tag, one
    that does not parse, or one a later row for the same memo replaced.

    A dead row is skipped: an unreadable cache is a cold cache, never an error. Every row the
    writers here emit begins with its tag, so a row from another tag is skipped without decoding
    it; decoding them all cost about a second of a hook's budget (CCA PERF-001, measured
    2026-11-06 on a journal of 9,230 rows, two thirds of them under dead tags).
    """
    prefix = '{"tag":' + _dumps(tag) + ","
    entries: _Entries = {}
    dead = 0
    try:
        text = path.read_text(encoding="utf-8")
        if text.startswith(prefix):
            document = json.loads(text)
            for key, (stamp, entry) in document["entries"].items():
                entries[str(key)] = ((int(stamp[0]), int(stamp[1])), _from_json(entry))
        else:
            dead += 1
    except FileNotFoundError:
        pass
    except Exception:  # BROAD-CATCH: fail-open  (an unreadable cache is a cold cache, never an error)
        entries = {}
        dead += 1
    try:
        lines = _delta_path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return entries, dead
    for line in lines:
        if not line:
            continue
        if not line.startswith(prefix):
            dead += 1
            continue
        try:
            row = json.loads(line)
            stamp, key = row["stamp"], str(row["key"])
            entry = ((int(stamp[0]), int(stamp[1])), _from_json(row["entry"]))
        except Exception:  # BROAD-CATCH: fail-open  (a torn line is skipped)
            dead += 1
            continue
        if key in entries:
            dead += 1  # a later row replaces an earlier one: a memo edited, or two cold launches
        entries[key] = entry
    return entries, dead


def _append_cache_delta(path: Path, tag: str, entries: _Entries) -> None:
    """Append `entries` to the delta journal: a write the size of what changed, which a load
    under a deadline can finish, where rewriting the whole cache could not. Never raises."""
    if not entries:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Leading newline: a line a killed writer left torn ends here, and the reader skips it.
        chunk = "\n" + "".join(
            _dumps({"tag": tag, "key": key, "stamp": list(stamp), "entry": _to_json(parsed)}) + "\n"
            for key, (stamp, parsed) in entries.items()
        )
        with _delta_path(path).open("a", encoding="utf-8") as handle:
            handle.write(chunk)
    except Exception:  # BROAD-CATCH: fail-open  (the cache is an optimisation, never a failure)
        pass


def _write_cache_file(path: Path, tag: str, entries: _Entries) -> None:
    """Replace the persisted cache atomically and drop the delta journal it now holds. Never raises.

    A load without a deadline calls this (the warm command, a library caller), and a load under a
    deadline only to drop dead rows, after appending its own parses and only with as much time
    left as the load took: the write grows with the cache, and a hook killed before the replace
    leaves only a temporary file, which the next write removes.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - _STALE_TEMP_SECONDS
        for stale in path.parent.glob(f".{path.name}.*.tmp"):
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                continue
        document = {
            "tag": tag,
            "entries": {key: [list(stamp), _to_json(parsed)] for key, (stamp, parsed) in entries.items()},
        }
        fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_dumps(document))
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        _delta_path(path).unlink(missing_ok=True)
    except Exception:  # BROAD-CATCH: fail-open  (the cache is an optimisation, never a failure)
        pass


def _is_junction(entry: os.DirEntry[str]) -> bool:
    """Whether `entry` is a Windows directory junction, which `is_symlink()` does not report."""
    probe = getattr(entry, "is_junction", None)  # Python 3.12+
    if probe is not None:
        return bool(probe())
    mount_point = getattr(stat_module, "IO_REPARSE_TAG_MOUNT_POINT", None)
    return mount_point is not None and getattr(entry.stat(follow_symlinks=False), "st_reparse_tag", 0) == mount_point


def _safe_target(resolved_root: Path, target: Path) -> bool:
    """`recall.index._safe_link_target`'s rule for the file a link reads: no excluded directory
    above it, and a name and suffix the default scan would read. Stricter in one place, stated:
    the target must be markdown, where the indexer also reads a link to another document type."""
    relative = target.relative_to(resolved_root)
    if any(part in _EXCLUDED_DIRS for part in relative.parts[:-1]):
        return False
    return (
        not target.name.startswith(".")
        and target.name.casefold() not in _EXCLUDED_NAMES
        and target.suffix.lower() not in _EXCLUDED_SUFFIXES
        and target.suffix.lower() in _MEMO_SUFFIXES
    )


def _memo_files(root: Path) -> list[tuple[Path, os.stat_result | OSError]]:
    """The memo files the indexer's default walk would read under `root`, each with its stat.

    The indexer's rules (`recall.index._default_walk`, `_safe_default_file`, `_confined_with_stats`,
    `_safe_link_target`): no hidden directory or file, none of `_EXCLUDED_DIRS`, no directory
    symlink followed, and a file reached through a link (a file symlink, or any file under a
    Windows junction, which the walk enters like the indexer's) kept only when it resolves inside
    `root` and the file it resolves to passes `_safe_target`. Walked with `os.scandir`, whose
    entries carry their stat on Windows, so each memo costs one directory listing rather than two
    stats; only a linked file is resolved.
    """
    resolved_root = root.resolve()
    found: list[tuple[Path, os.stat_result | OSError]] = []
    pending = [(root, False)]
    while pending:
        directory, linked = pending.pop()
        try:
            with os.scandir(directory) as listing:
                entries = list(listing)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _EXCLUDED_DIRS:
                        pending.append((path, linked or _is_junction(entry)))
                    continue
                if path.suffix.lower() not in _MEMO_SUFFIXES:
                    continue
                if linked or entry.is_symlink():
                    target = within(resolved_root, path)
                    if target is None or not _safe_target(resolved_root, target):
                        continue
                    status = target.stat()
                else:
                    status = entry.stat(follow_symlinks=False)
            except OSError as exc:
                found.append((path, exc))
                continue
            if stat_module.S_ISREG(status.st_mode):
                found.append((path, status))
    return sorted(found, key=lambda pair: pair[0])


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _read(path: Path, file: str) -> MemoRecord | str | None:
    """`path`'s admitted record, the reason it was refused, or None when it declares none.

    An `OSError` is raised, not returned: a sharing violation or a scanner's lock passes, and a
    refusal returned here is cached under the file's unchanged stamp until the file changes.
    """
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\x00", "")
    except UnicodeDecodeError as exc:
        return f"unreadable: {type(exc).__name__}"
    if RECORD_KEY not in text:
        return None
    meta = parse_document(text).meta
    if RECORD_KEY not in meta:
        return None
    admitted = admit_record(meta, str(path))
    record = admitted.get(RECORD_KEY)
    if not isinstance(record, dict):
        return "recall_record refused (see the warning naming this file)"
    try:
        # The indexer refuses a malformed or empty validity window and the whole memo with it
        # (`recall.index`, `recall.generations`), so no store holds such a record; nor does this.
        validity_bounds(admitted)
    except ValueError as exc:
        return f"invalid validity window: {exc}"
    return MemoRecord(
        source=str(path),
        record=record,
        valid_from=_text(admitted.get("valid_from")),
        supersedes=supersedes_targets(admitted.get("supersedes")),
        file=file,
        name=_text(admitted.get("name")),
        description=_text(admitted.get("description")),
        valid_until=_text(admitted.get("valid_until")),
    )


def load_memo_records(
    directory: Path, *, cache_file: Path | None = None, deadline: float | None = None
) -> LoadedMemoRecords:
    """Every admitted `recall_record` under `directory`, in path order.

    The files are the ones the indexer would read there (see `_memo_files`). A memo whose record
    is invalid, or that cannot be read, is skipped and listed in `skipped`, never raised. A
    directory that does not exist or is not a directory raises `NotADirectoryError`: the caller
    decides whether that is silence or an error worth logging.

    Each file is parsed once per `(path, mtime_ns, size)`; a changed file is read again. The cache
    lives in this process and, with `cache_file`, also on disk, because a hook is a new process
    every time. Measured 2026-10-22 on the Windows workstation: the first open of a memo file cost
    about 12 ms (on-access scanning) and a second read about 1 ms, so a cold directory of hundreds
    of memos does not fit a one-second hook.

    `deadline` (a `time.monotonic()` value) bounds the parsing: past it, what was parsed so far is
    appended to the cache's delta journal and `TimeoutError` is raised, so a cold cache warms over
    a few calls instead of never. Under a deadline the cache is appended to, and then, when the
    load read the whole directory with as much time left as it took, over a cache holding dead
    rows, rewritten;
    a load with no deadline rewrites it whole, folding the journal in and dropping removed memos.
    A memo that raised an `OSError` on read is skipped and not cached, so the next load retries it.
    """
    started = time.monotonic()
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"memo directory {root} is not a directory")
    tag = _cache_tag() if cache_file is not None else ""
    persisted, dead = _read_cache_file(cache_file, tag) if cache_file is not None else ({}, 0)
    current: _Entries = {}
    fresh: _Entries = {}
    loaded = LoadedMemoRecords()
    files = _memo_files(root)
    for index, (path, stat) in enumerate(files):
        if isinstance(stat, OSError):
            loaded.skipped.append((str(path), f"unreadable: {type(stat).__name__}"))
            continue
        key, stamp = str(path), (stat.st_mtime_ns, stat.st_size)
        cached = _CACHE.get(key)
        if cached is None or cached[0] != stamp:
            cached = persisted.get(key)
        if cached is not None and cached[0] == stamp:
            parsed = cached[1]
            relative = path.relative_to(root).as_posix()
            if isinstance(parsed, MemoRecord) and parsed.file != relative:
                # The cache is keyed on the absolute path, but `file` is relative to the root the
                # memo was first read under; a load from a parent or child root re-derives it.
                parsed = replace(parsed, file=relative)
        else:
            if deadline is not None and time.monotonic() > deadline:
                if cache_file is not None:
                    _append_cache_delta(cache_file, tag, fresh)
                raise TimeoutError(
                    f"read {index} of {len(files)} memos before the deadline; the cache keeps them"
                )
            try:
                parsed = _read(path, path.relative_to(root).as_posix())
            except OSError as exc:
                # Not cached: the next load tries the file again (CCA STAKES-003).
                loaded.skipped.append((key, f"unreadable: {type(exc).__name__}"))
                continue
            fresh[key] = (stamp, parsed)
        _CACHE[key] = current[key] = (stamp, parsed)
        if isinstance(parsed, MemoRecord):
            loaded.append(parsed)
        elif isinstance(parsed, str):
            loaded.skipped.append((key, parsed))
    if cache_file is not None:
        if deadline is not None:
            # Appended first, so a rewrite killed at the hook's exit loses none of this load's parses.
            _append_cache_delta(cache_file, tag, fresh)
            now = time.monotonic()
            if dead and deadline - now > now - started:
                # A load that read the whole directory with as much time left as it took, over a
                # cache holding dead rows (another tag's, or rows a later one replaced), rewrites
                # it: only `warm` did, and nothing runs it, so every upgrade appended one more copy
                # of the directory for every later launch to read (CCA PERF-001).
                _write_cache_file(cache_file, tag, current)
        elif fresh or set(persisted) != set(current) or _delta_path(cache_file).exists():
            # Written with this directory's entries only, so a removed memo leaves the cache too.
            _write_cache_file(cache_file, tag, current)
    return loaded


def _file(memo: MemoRecord) -> str:
    return memo.file or Path(memo.source).name


def memo_supersession_graph(records: Sequence[MemoRecord]) -> SupersessionGraph:
    """The supersession graph of `records`, resolved as a store resolves its own rows."""
    rows = sorted(
        (
            (_file(memo), target, None, memo.source, memo.valid_from, memo.valid_until)
            for memo in records
            for target in (memo.supersedes or (None,))
        ),
        key=lambda row: (row[0], row[1] or "", row[3]),
    )
    return resolve_supersession_graph(rows)


def _candidate(memo: MemoRecord) -> RecordCandidate:
    record = memo.record
    record_id, kind, anchors = record_parts(record)
    meta = {
        "name": memo.name,
        "description": memo.description,
        "valid_from": memo.valid_from,
        "valid_until": memo.valid_until,
        "supersedes": list(memo.supersedes),
    }
    return RecordCandidate(
        id=record_id,
        record=record,
        kind=kind,
        anchors=anchors,
        meta=meta,
        source=memo.source,
        file=_file(memo),
        chunk_id=None,
        chunk_ids=[],
    )


def match_memo_records(
    records: Sequence[MemoRecord],
    *,
    repo: str,
    anchors: Sequence[dict[str, Any]],
    kinds: Sequence[str],
    now: datetime,
) -> list[dict[str, Any]]:
    """The C5 record entries of `records` matching `anchors` in `repo`, best match first.

    The same query an anchor `lookup_records` call answers, with `superseded_by` read from the
    memos' own supersession graph at `now`. Every record takes part in that graph, whatever
    `kinds` asks for, so a retraction hides its target even when retractions are not wanted.
    """
    query_anchors = validated_anchor_query(repo, anchors)
    wanted_kinds = validated_kinds(kinds)
    instant = as_utc(now)
    graph = memo_supersession_graph(records)
    return match_candidates(
        [_candidate(memo) for memo in records],
        query_anchors=query_anchors,
        repo=repo,
        wanted_ids=None,
        wanted_kinds=wanted_kinds,
        superseded_by=lambda candidate: graph.successor(
            graph.key(candidate.file, candidate.source), now=instant
        ),
        now=instant,
    )


__all__ = [
    "LoadedMemoRecords",
    "MemoRecord",
    "load_memo_records",
    "match_memo_records",
    "memo_supersession_graph",
]
