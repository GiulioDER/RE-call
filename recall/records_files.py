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

Two differences from a store, both stated rather than left to be found:

- A file has no chunk, so a record's `chunk_id` is None and its `chunk_ids` empty.
- Only memos holding an admitted record take part in the supersession graph. A memo whose record
  was refused still declares its `supersedes` edge to an indexed store; here it declares nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.document import parse_document
from recall.frontmatter import RECORD_KEY, supersedes_targets
from recall.records import (
    RECORD_KINDS,
    RecordCandidate,
    RecordsQueryError,
    _string_list,
    _validated_anchors,
    admit_record,
    match_candidates,
    record_schema,
)
from recall.supersession import SupersessionGraph, resolve_supersession_graph

#: The suffixes the indexer reads as markdown documents.
_MEMO_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})


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
_CACHE_FORMAT = 1


def _cache_tag() -> str:
    """What a persisted cache was built by: the entry format and the schema records were admitted
    under. A cache with another tag is ignored, so a schema change re-reads every memo."""
    schema = json.dumps(record_schema(), sort_keys=True).encode("utf-8")
    return f"{_CACHE_FORMAT}:{hashlib.sha256(schema).hexdigest()[:16]}"


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


def _read_cache_file(path: Path) -> dict[str, tuple[tuple[int, int], MemoRecord | str | None]]:
    """The persisted entries, or none when the file is absent, unreadable, or from another tag."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("tag") != _cache_tag():
            return {}
        return {
            str(key): ((int(stamp[0]), int(stamp[1])), _from_json(entry))
            for key, (stamp, entry) in document["entries"].items()
        }
    except Exception:  # BROAD-CATCH: fail-open  (an unreadable cache is a cold cache, never an error)
        return {}


def _write_cache_file(path: Path, entries: dict[str, tuple[tuple[int, int], MemoRecord | str | None]]) -> None:
    """Replace the persisted cache atomically. Never raises: an unwritable cache stays cold."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "tag": _cache_tag(),
            "entries": {key: [list(stamp), _to_json(parsed)] for key, (stamp, parsed) in entries.items()},
        }
        fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
    except Exception:  # BROAD-CATCH: fail-open  (the cache is an optimisation, never a failure)
        pass


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _read(path: Path, file: str) -> MemoRecord | str | None:
    """`path`'s admitted record, the reason it was refused, or None when it declares none."""
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\x00", "")
    except (OSError, UnicodeDecodeError) as exc:
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

    A memo whose record is invalid, or that cannot be read, is skipped and listed in `skipped`,
    never raised. A directory that does not exist or is not a directory raises
    `NotADirectoryError`: the caller decides whether that is silence or an error worth logging.

    Each file is parsed once per `(path, mtime_ns, size)`; a changed file is read again. The cache
    lives in this process and, with `cache_file`, also on disk, because a hook is a new process
    every time: admitting a record is most of the cost (about a millisecond each, measured
    2026-10-22), and a memo directory changes a few files at a time.

    `deadline` (a `time.monotonic()` value) bounds the parsing: past it, what was parsed so far is
    saved to `cache_file` and `TimeoutError` is raised, so a cold cache warms over a few calls
    instead of never.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"memo directory {root} is not a directory")
    persisted = _read_cache_file(cache_file) if cache_file is not None else {}
    current: dict[str, tuple[tuple[int, int], MemoRecord | str | None]] = {}
    changed = False
    loaded = LoadedMemoRecords()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _MEMO_SUFFIXES)
    for index, path in enumerate(paths):
        try:
            stat = path.stat()
        except OSError as exc:
            loaded.skipped.append((str(path), f"unreadable: {type(exc).__name__}"))
            continue
        if not path.is_file():
            continue
        key, stamp = str(path), (stat.st_mtime_ns, stat.st_size)
        cached = _CACHE.get(key)
        if cached is None or cached[0] != stamp:
            cached = persisted.get(key)
        if cached is not None and cached[0] == stamp:
            parsed = cached[1]
        else:
            if deadline is not None and time.monotonic() > deadline:
                if cache_file is not None and changed:
                    _write_cache_file(cache_file, {**persisted, **current})
                raise TimeoutError(
                    f"read {index} of {len(paths)} memos before the deadline; the cache keeps them"
                )
            parsed = _read(path, path.relative_to(root).as_posix())
            changed = True
        _CACHE[key] = current[key] = (stamp, parsed)
        if isinstance(parsed, MemoRecord):
            loaded.append(parsed)
        elif isinstance(parsed, str):
            loaded.skipped.append((key, parsed))
    if cache_file is not None and (changed or set(persisted) != set(current)):
        # Written with this directory's entries only, so a removed memo leaves the cache too.
        _write_cache_file(cache_file, current)
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
    anchors = record.get("anchors")
    meta = {
        "name": memo.name,
        "description": memo.description,
        "valid_from": memo.valid_from,
        "valid_until": memo.valid_until,
        "supersedes": list(memo.supersedes),
    }
    return RecordCandidate(
        id=str(record.get("id")),
        record=record,
        kind=str(record.get("kind")),
        anchors=[a for a in anchors if isinstance(a, dict)] if isinstance(anchors, list) else [],
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
    if not isinstance(repo, str) or not repo:
        raise RecordsQueryError("an anchor query needs `repo`: anchors are paths inside one repository")
    query_anchors = _validated_anchors(anchors)
    wanted_kinds = set(_string_list(kinds, "kinds"))
    if unknown := wanted_kinds - RECORD_KINDS:
        raise RecordsQueryError(f"unknown kinds {sorted(unknown)}; known: {sorted(RECORD_KINDS)}")
    instant = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
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
    )


__all__ = [
    "LoadedMemoRecords",
    "MemoRecord",
    "load_memo_records",
    "match_memo_records",
    "memo_supersession_graph",
]
