"""The dashboard's database pages over a lite store: one SQLite file, opened read-only.

Every function here answers what its namesake in `recall.dashboard.db` answers, in the same shape,
so the pages render a lite store with no branch of their own. The file holds one tenant and one
corpus, so "tenants" is one row and "generations" is the corpus as it is now; a tenant this file
does not hold gets nothing, as an unknown tenant does in Postgres.

The aggregations the Postgres side does in SQL over `jsonb` (retrievals per memo, agent reports per
memo) are done here in Python over the decoded payloads: a local file holds one person's memory,
so its ledger is small, and keeping the rules in one readable place beats a second SQL dialect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from recall.dashboard.db import (
    SEARCH_EVENTS,
    USE_REPORT_EVENT,
    DatabaseUnavailable,
    _search_outcome,
    _summarise,
    stale_row,
)
from recall.lite import LiteStoreError, lite_dsn_path
from recall.stale_reports import STALE_REPORT_EVENT

if TYPE_CHECKING:
    from recall.dashboard.corpus import DetailRelation
    from recall.semantic_graph import SemanticGraphProjection
    from recall.supersession import SupersessionGraph


def store_path(dsn: str) -> Path:
    """The file the server would open, refusing as the server does a DSN it would refuse."""
    try:
        return lite_dsn_path(dsn)
    except LiteStoreError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


#: What `_corpus_stamp` observes of a file and its write-ahead log (CCA CODE-008).
_FileStamp = tuple[tuple[int, ...] | None, ...] | None


class _ReadConnection(sqlite3.Connection):
    file_stamp: _FileStamp = None


@contextmanager
def connect(dsn: str) -> Iterator[sqlite3.Connection]:
    path = store_path(dsn)
    if not path.exists():
        raise DatabaseUnavailable(f"no lite store at {path} yet: `recall index <memory folder>` creates it")
    path = path.resolve()
    before = _corpus_stamp(path)
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, factory=_ReadConnection)
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"{type(exc).__name__}: {exc}") from exc
    connection.file_stamp = before if before == _corpus_stamp(path) else None
    try:
        yield connection
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()


def _when(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _payload(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _sources(payload: dict[str, Any], role: str) -> list[str]:
    """Unsupported containers are not evidence of memo use, even if they are iterable."""
    values = payload.get(role)
    return [str(value) for value in values] if isinstance(values, list) else []


def _tenant(c: sqlite3.Connection) -> str:
    row = c.execute("SELECT value FROM meta WHERE key = 'tenant'").fetchone()
    return str(row[0]) if row else ""


def _events(
    c: sqlite3.Connection, tenant: str, types: tuple[str, ...], *,
    limit: int | None = None, event_id: str | None = None, omit_hits: bool = False,
) -> Iterator[tuple[Any, ...]]:
    """`(event_id, event_type, actor, generation_id, source_uri, payload, created_at)`, newest first."""
    if tenant != _tenant(c):
        return
    # Keep malformed or nonstandard JSON on the existing Python decoding path.
    sql = (
        "SELECT event_id, event_type, actor, generation_id, source_uri, "
        "CASE WHEN ? = 0 THEN payload WHEN typeof(payload) = 'text' AND json_valid(payload) "
        "THEN json_remove(payload, '$.hits') ELSE payload END, created_at FROM audit_events "
        "WHERE event_type IN (SELECT value FROM json_each(?))"
    )
    args: list[Any] = [omit_hits, json.dumps(list(types))]
    if event_id is not None:
        sql += " AND event_id = ?"
        args.append(event_id)
    sql += " ORDER BY created_at DESC, rowid DESC"
    if limit is not None and limit >= 0:
        sql += " LIMIT ?"
        args.append(limit)
    yield from c.execute(sql, args)


def file_tenant(dsn: str) -> str:
    """The tenant the file holds, for the dashboard's default."""
    with connect(dsn) as c:
        return _tenant(c)


def availability(dsn: str) -> dict[str, Any]:
    with connect(dsn) as c:
        version = c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return {
        "engine": "SQLite",
        "server_version": sqlite3.sqlite_version,
        "role": str(store_path(dsn)),
        "read_only": True,
        "migrations": [("lite store", version[0] if version else "?")],
    }


def tenants(dsn: str) -> list[dict[str, Any]]:
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite.inspect import inspect_lite_file

    found = inspect_lite_file(store_path(dsn), include_timestamps=True)
    if found.error is not None:
        raise DatabaseUnavailable(found.error)
    if not found.exists:
        raise DatabaseUnavailable(f"no lite store at {found.path} yet")
    calibration = None
    if found.calibration is not CalibrationStatus.MISSING:
        calibration = {
            "certified": found.calibration is CalibrationStatus.CERTIFIED,
            "threshold": found.threshold,
            "separability": found.separability,
            "published_at": _when(found.calibrated_at),
            "state": found.calibration.value,
        }
    newest = _when(found.newest_indexed_at)
    return [
        {
            "tenant": found.tenant,
            "generation": found.generation_id if found.chunks else None,
            "state": "active" if found.chunks else None,
            "created_at": _when(found.first_indexed_at),
            "activated_at": newest,
            "updated_at": newest,
            "embedder": {"provider": None, "model": found.embedder_model, "dimension": found.dim, "profile_id": None, "verified": None},
            "chunks": found.chunks,
            "sources": found.sources,
            "calibration": calibration,
        }
    ]


def generations(
    dsn: str, tenant: str, limit: int = 15, *, inspected_tenant: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """A lite store keeps no history of builds: its one corpus is the current one."""
    inventory = tenants(dsn) if inspected_tenant is None else [inspected_tenant]
    every = [t for t in inventory if t["tenant"] == tenant and t["generation"]]
    return [
        {"generation": t["generation"], "state": "active", "created_at": t["created_at"], "activated_at": t["activated_at"],
         "retired_at": None, "failure_reason": None, "created_by": "recall-lite"}
        for t in every
    ][:limit]


def lifecycle_events(dsn: str, tenant: str, limit: int = 100) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return []
        rows = c.execute(
            "SELECT event_type, actor, generation_id, source_uri, created_at FROM audit_events "
            "WHERE event_type NOT IN (SELECT value FROM json_each(?)) ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (json.dumps(list(SEARCH_EVENTS)), limit),
        ).fetchall()
    return [{"event": e, "actor": a, "generation": g, "source": s, "created_at": _when(t)} for e, a, g, s, t in rows]


def _search_row(row: tuple[Any, ...]) -> dict[str, Any]:
    event_id, event_type, actor, generation, _source, payload, created = row
    return {"event_id": event_id, "event": event_type, "actor": actor, "generation": generation,
            "created_at": _when(created), **_summarise(_payload(payload), event_type)}


def searches(dsn: str, tenant: str, limit: int = 100, contains: str = "") -> list[dict[str, Any]]:
    if limit == 0:
        return []
    needle = contains.casefold()
    found = []
    with connect(dsn) as c:
        for row in _events(c, tenant, SEARCH_EVENTS, limit=limit if not needle else None):
            item = _search_row(row)
            if needle and needle not in item["query"].casefold():
                continue
            found.append(item)
            if limit > 0 and len(found) >= limit:
                break
    return found[:limit]


def search_count(dsn: str, tenant: str, *, limit: int) -> int:
    """Count search events without reading their payloads."""
    if limit <= 0:
        return 0
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return 0
        count = c.execute(
            "SELECT count(*) FROM (SELECT 1 FROM audit_events "
            "WHERE event_type IN (SELECT value FROM json_each(?)) LIMIT ?)",
            (json.dumps(list(SEARCH_EVENTS)), limit),
        ).fetchone()[0]
        return int(count)


def search(dsn: str, tenant: str, event_id: str) -> dict[str, Any] | None:
    with connect(dsn) as c:
        row = next(_events(c, tenant, SEARCH_EVENTS, limit=1, event_id=event_id), None)
        return _search_row(row) if row is not None else None


def control(
    dsn: str, tenant: str, *, retrievals: bool = True, source: str | None = None, include_summary: bool = True,
) -> dict[str, Any]:
    """`recall.dashboard.db.control`, the same rules: one retrieval per (search, memory), at its best rank."""
    data, _ = _control(dsn, tenant, retrievals=retrievals, source=source, include_summary=include_summary)
    return data


def control_for_memo(
    dsn: str, tenant: str, source: str, *, report_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _control(dsn, tenant, retrievals=True, source=source, include_summary=False, report_limit=report_limit)


def _control(
    dsn: str, tenant: str, *, retrievals: bool, source: str | None, include_summary: bool, report_limit: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    memos: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []

    def memo(source: str) -> dict[str, Any]:
        return memos.setdefault(source, {
            "source": source, "retrieved": 0, "first": 0, "last_retrieved": None, "ok": 0, "superseded": 0,
            "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
            "succeeded": 0, "failed": 0, "last_reported": None,
        })

    summary = {"searches": 0, "answered": 0, "abstained": 0, "refused": 0,
               "reports": 0, "helped": 0, "no_difference": 0, "misled": 0, "succeeded": 0, "failed": 0}
    with connect(dsn) as c:
        c.execute("BEGIN")
        types = SEARCH_EVENTS if include_summary else ("search_decision",)
        search_rows = _events(c, tenant, types, omit_hits=not retrievals) if include_summary or retrievals else ()
        for _id, event_type, _a, _g, _s, raw, created in search_rows:
            payload = _payload(raw)
            if include_summary:
                summary["searches"] += 1
                outcome = _search_outcome(payload, event_type)
                if outcome in ("answered", "abstained", "refused"):
                    summary[outcome] += 1
            if not retrievals or event_type != "search_decision":
                continue
            hits = payload.get("hits")
            best: dict[str, dict[str, Any]] = {}
            for rank, hit in enumerate(hits if isinstance(hits, list) else [], start=1):
                if not isinstance(hit, dict) or hit.get("source") is None:
                    continue
                named = str(hit["source"])
                if source is not None and named != source:
                    continue
                seen = best.setdefault(named, {"rank": rank, "verdicts": set()})
                verdict = hit.get("verdict")
                if isinstance(verdict, str):
                    seen["verdicts"].add(verdict)
            if not best:
                continue
            at = _when(created)
            for found, seen in best.items():
                row = memo(found)
                row["retrieved"] += 1
                row["first"] += seen["rank"] == 1
                row["ok"] += "ok" in seen["verdicts"]
                row["superseded"] += "superseded" in seen["verdicts"]
                row["low_confidence"] += "low_confidence" in seen["verdicts"]
                if at and (row["last_retrieved"] is None or at > row["last_retrieved"]):
                    row["last_retrieved"] = at
        for event_id, _type, actor, _g, _s, raw, created in _events(c, tenant, (USE_REPORT_EVENT,)):
            payload, at = _payload(raw), _when(created)
            used, wrong = _sources(payload, "used"), _sources(payload, "wrong")
            if len(reports) < report_limit and (not source or source in used or source in wrong):
                reports.append(_report_row(event_id, actor, payload, at, used, wrong))
            effect, succeeded = payload.get("effect"), payload.get("task_succeeded")
            if include_summary:
                summary["reports"] += 1
                if effect in ("helped", "no_difference", "misled"):
                    summary[effect] += 1
                if succeeded is True or succeeded is False:
                    summary["succeeded" if succeeded else "failed"] += 1
            for role, names in (("used", used), ("wrong", wrong)):
                for named in names:
                    if source is not None and named != source:
                        continue
                    row = memo(named)
                    row[role] += 1
                    if role == "used" and effect in ("helped", "no_difference"):
                        row[effect] += 1
                    if role == "wrong" and effect == "misled":
                        row["misled"] += 1
                    if role == "used" and (succeeded is True or succeeded is False):
                        row["succeeded" if succeeded else "failed"] += 1
                    if at and (row["last_reported"] is None or at > row["last_reported"]):
                        row["last_reported"] = at
    if include_summary:
        summary["memos_retrieved"] = sum(1 for m in memos.values() if m["retrieved"])
    ordered = sorted(memos.values(), key=lambda m: (-m["retrieved"], -m["used"], m["source"]))
    return {"summary": summary if include_summary else {}, "memos": ordered}, reports


def _report_row(
    event_id: str, actor: str, payload: dict[str, Any], at: datetime | None, used: list[str], wrong: list[str],
) -> dict[str, Any]:
    return {
        "event_id": event_id, "actor": actor, "created_at": at,
        "task": str(payload.get("task", "")), "effect": payload.get("effect"), "used": used, "wrong": wrong,
        "task_succeeded": payload.get("task_succeeded"), "query": payload.get("query"), "note": payload.get("note"),
    }


def use_reports(dsn: str, tenant: str, limit: int = 50, source: str = "") -> list[dict[str, Any]]:
    if limit == 0:
        return []
    out = []
    with connect(dsn) as c:
        for event_id, _type, actor, _g, _s, raw, created in _events(c, tenant, (USE_REPORT_EVENT,), limit=limit if not source else None):
            payload = _payload(raw)
            used, wrong = _sources(payload, "used"), _sources(payload, "wrong")
            if source and source not in used and source not in wrong:
                continue
            out.append(_report_row(event_id, actor, payload, _when(created), used, wrong))
            if limit > 0 and len(out) >= limit:
                break
    return out[:limit]


def top_sources(dsn: str, tenant: str, generation: str, limit: int = 40) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return []
        rows = c.execute(
            "SELECT source, count(*), max(indexed_at) FROM chunks GROUP BY source ORDER BY max(indexed_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"source": s, "chunks": n, "indexed_at": _when(t)} for s, n, t in rows]


def stale_reports(dsn: str, tenants: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    with connect(dsn) as c:
        tenant = _tenant(c)
        rows = _events(c, tenant, (STALE_REPORT_EVENT,), limit=limit) if tenant in tenants else []
        return [stale_row(r[0], tenant, _payload(r[5]), _when(r[6])) for r in rows][:limit]


# ---------------------------------------------------------------------- the corpus graph and View 2


class _CorpusRow(NamedTuple):
    """One chunk of the file as the corpus readers use it, read by name (CCA CODE-004)."""

    id: str
    source: str
    text: str
    metadata: dict[str, Any]
    indexed_at: str
    first_indexed_at: str
    file: str


class _SourceProjection(NamedTuple):
    rows_by_source: dict[str, list[_CorpusRow]]
    sources_by_file: dict[str, set[str]]
    mentions_by_source: dict[str, dict[tuple[str, str], int]]
    relations_by_file: dict[str, list[DetailRelation]]
    replaced_by_source: dict[str, set[str]]
    indexed_keys: dict[str, set[str]]


class _ImportProjection(NamedTuple):
    edges: list[tuple[str, str]]
    by_target: dict[str, list[tuple[str, str]]]
    files: set[str]


def _corpus(c: sqlite3.Connection, tenant: str) -> tuple[list[_CorpusRow], SupersessionGraph, SemanticGraphProjection]:
    """`(chunks, supersession graph, semantic graph)` of the file. Chunks are in order of ordinal,
    then rowid, as Postgres orders them by `chunk_ordinal`.

    The supersession graph is resolved from the rows and records `LiteStore.supersession_graph`
    reads; the semantic graph is the stored one, else one built in memory, since this connection
    is read-only (`recall.lite.graph.graph_of`).
    """
    from recall.lite.store import _supersession_rows
    from recall.related import metadata_ordinal
    from recall.supersession import resolve_supersession_graph
    from recall.supersession_records import sqlite_records

    # One read view for the rows, the records and the graph, so a graph kept under the rows'
    # digest is the graph of those rows (`recall.lite.inspect` opens one for the same reason).
    c.execute("BEGIN")
    key = _file_key(c, tenant)
    files = getattr(c, "file_stamp", None)
    if files != _corpus_stamp(Path(key[0])):
        files = None
    version = str(c.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()[0])
    stamp = (version, files)
    with _CORPUS_LOCK:
        kept = _CORPORA.get(key)
        if files is not None and kept is not None and kept[0] == stamp:
            projection = _GRAPHS.get(key)
            if projection is not None and projection[1] is kept[1][2]:
                return kept[1]
        _CORPORA.pop(key, None)
    digest = hashlib.sha256(version.encode())
    ordered = []
    # Hashes only, for the corpus fingerprint a graph built here needs: reading every vector again
    # for it doubled the read of a large file (CCA PERF-002).
    hashed: list[tuple[str, bytes, bytes]] = []
    for rowid, cid, source, text, raw, indexed, first, embedding in c.execute(
        "SELECT rowid, id, source, text, metadata, indexed_at, first_indexed_at, embedding FROM chunks ORDER BY id"
    ):
        digest.update("\0".join(("", cid, source, text, raw, indexed)).encode())
        digest.update(embedding)
        hashed.append((cid, hashlib.sha256(text.encode("utf-8")).digest(), hashlib.sha256(embedding).digest()))
        metadata = _payload(raw)
        file = metadata["file"] if isinstance(metadata.get("file"), str) else source
        ordinal = metadata_ordinal(metadata)
        ordered.append(((ordinal is None, ordinal or 0, rowid), _CorpusRow(cid, source, text, metadata, indexed, first, file)))
    chunks = [row for _order, row in sorted(ordered, key=lambda pair: pair[0])]
    winner = resolve_supersession_graph(
        _supersession_rows((row.metadata, row.first_indexed_at, row.source) for row in chunks),
        sqlite_records(c, "chunks"),
        tenant,
    )
    result = chunks, winner, _semantic_graph(c, tenant, digest.hexdigest(), chunks, hashed)
    if files is not None and files == _corpus_stamp(Path(key[0])):
        with _CORPUS_LOCK:
            _CORPORA[key] = (stamp, result)
            for old in list(_CORPORA)[:-_MAX_CORPORA]:
                _CORPORA.pop(old, None)
    return result


def _file_key(c: sqlite3.Connection, tenant: str) -> tuple[str, str]:
    """`(the file this connection reads, tenant)`: the key of every per-file cache here."""
    return str(c.execute("PRAGMA database_list").fetchone()[2]), tenant


def _corpus_stamp(path: Path) -> _FileStamp:
    """Observe file replacement and SQLite writes, including commits still in the WAL."""
    stamps: list[tuple[int, ...] | None] = []
    for file in (path, path.with_name(path.name + "-wal")):
        try:
            stat = file.stat()
        except FileNotFoundError:
            stamps.append(None)
        except OSError:
            return None
        else:
            # A read-only connection can create an empty WAL without changing any row.
            if file != path and stat.st_size == 0:
                stamps.append(None)
                continue
            changed = _change_time(file, stat.st_ctime_ns)
            if changed is None:
                return None
            # Run as root, SQLite chowns the WAL each time a connection opens it, which moves only
            # its ctime: kept, every read would look like a write and no view would be reused.
            # A commit still moves the WAL's size or mtime, and `corpus_version` is checked too.
            if file != path and getattr(os, "geteuid", lambda: -1)() == 0:
                changed = 0
            stamps.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, changed))
    return tuple(stamps) if stamps[0] is not None else None


def _change_time(path: Path, posix_ctime: int) -> int | None:
    # Windows stat ctime is creation time; copy2 can preserve every other stat field.
    if sys.platform != "win32":
        return posix_ctime
    import ctypes
    import msvcrt

    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        query.restype = ctypes.c_int
        with path.open("rb") as stream:
            # FILE_BASIC_INFO: four LARGE_INTEGER times followed by DWORD attributes.
            info = (ctypes.c_longlong * 5)()
            if query(msvcrt.get_osfhandle(stream.fileno()), 0, ctypes.byref(info), ctypes.sizeof(info)):
                return int(info[3])
    except OSError:
        pass
    # An unavailable change stamp disables reuse; the reader still builds a fresh view.
    return None


_CORPUS_LOCK = threading.RLock()

#: At most this many decoded read views, reused only while file identity and write stamps match.
_MAX_CORPORA = 8
_CORPORA: dict[
    tuple[str, str],
    tuple[tuple[str, _FileStamp], tuple[list[_CorpusRow], SupersessionGraph, SemanticGraphProjection]],
] = {}
#: `{(file, tenant): (read view, day, View 1 body)}` and `{(file, tenant): (read view, import edges)}`:
#: what is derived from a read view `_corpus` returns, kept while it returns the same one, since the
#: Lite reader caches no response body (CCA PERF-003).
_BODIES: dict[tuple[str, str], tuple[object, str, dict[str, Any]]] = {}
_IMPORTS: dict[tuple[str, str], tuple[object, _ImportProjection]] = {}
_SOURCE_PROJECTIONS: dict[tuple[str, str], tuple[object, _SourceProjection]] = {}


#: `{(file, tenant): (stamp of its rows, graph)}`, one entry per file this process reads.
_GRAPHS: dict[tuple[str, str], tuple[str, SemanticGraphProjection]] = {}


def _semantic_graph(
    c: sqlite3.Connection, tenant: str, stamp: str, rows: list[_CorpusRow], hashed: list[tuple[str, bytes, bytes]]
) -> SemanticGraphProjection:
    """`graph_of`, kept while the file's rows are the ones it was built from (CCA PERF-002): built
    in memory it is the whole build, seconds on a large store, and View 2 asks for it on every page.
    `stamp` digests the corpus version and every row's id, source, text, metadata and `indexed_at`,
    plus its embedding bytes, so the generation follows vector changes too. The version alone is
    not enough (a file deleted and indexed again, or a
    backup copied over it, starts it again and can reach the same number with other rows), nor are
    ids and times (ids are positional, so an edited memo keeps its ids, and under an evaluation
    clock its write time too)."""
    from recall.lite.graph import graph_of
    from recall.lite.store import corpus_fingerprint_of_hashed
    from recall.types import Chunk

    key = _file_key(c, tenant)
    kept = _GRAPHS.get(key)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    graph = graph_of(
        c, tenant, [Chunk(id=r.id, source=r.source, text=r.text, metadata=r.metadata) for r in rows],
        corpus_fingerprint_of_hashed(hashed),
    )
    _GRAPHS[key] = (stamp, graph)
    return graph


def _python_imports(chunks: list[_CorpusRow], files: set[str]) -> list[tuple[str, str]]:
    """`recall.dashboard.corpus._import_edges` over the file's `.py` chunks: the same pattern, run
    by Python's `re` with `^` at each line start, as Postgres runs it newline-sensitive."""
    from recall.dashboard.corpus import _IMPORT_PATTERN, resolve_imports

    pattern = re.compile(_IMPORT_PATTERN, re.M)
    matches = [
        (row.file, found.group(1), found.group(2), found.group(3))
        for row in chunks
        if row.file.endswith(".py")
        for found in pattern.finditer(row.text)
    ]
    return resolve_imports(matches, files)


def corpus_graph(dsn: str, tenant: str, *, today: date | None = None) -> dict[str, Any]:
    """`recall.dashboard.corpus.corpus_graph` for a lite file: the same `assemble_graph`, fed from
    the file's chunks and semantic graph (`recall.lite.graph`) instead of PostgreSQL's tables."""
    from recall.dashboard.corpus import assemble_graph, empty_graph

    now = (today or datetime.now(UTC).date()).isoformat()
    with connect(dsn) as c:
        file_key = _file_key(c, tenant)
        found = _corpus(c, tenant) if tenant == _tenant(c) else None
    if found is None or not found[0]:
        return empty_graph(tenant, f"{tenant} has no corpus in this file")
    with _CORPUS_LOCK:
        body = _BODIES.get(file_key)
    if body is not None and body[0] is found and body[1] == now:
        return body[2]
    rows, winner, graph = found
    by_file: dict[tuple[str, str], list[_CorpusRow]] = {}
    for row in rows:
        by_file.setdefault((row.file, row.source), []).append(row)

    def most(group: list[_CorpusRow], key: str) -> str | None:
        values = [str(r.metadata[key]) for r in group if r.metadata.get(key) is not None]
        return max(values) if values else None  # Postgres's max() over the text values

    files = [
        (file, len(group), min(r.first_indexed_at for r in group), most(group, "type"), most(group, "valid_from"),
         most(group, "valid_until"), group[0].metadata.get("heading_hierarchy"), source, most(group, "indexed_via"))
        for (file, source), group in by_file.items()
    ]
    entities = {entity.id: entity for entity in graph.entities}
    relations = [
        (r.relation, entities[r.subject_id].canonical_name, entities[r.object_id].canonical_name)
        for r in graph.relations
        if r.relation in ("references", "depends_on")
        and entities[r.subject_id].kind == "file" and entities[r.object_id].kind == "file"
    ]
    located = {row.id: (row.file, row.source) for row in rows}
    mentions = sorted({
        (m.entity_id, *located[m.chunk_id])
        for m in graph.mentions
        if m.chunk_id in located and entities[m.entity_id].kind != "file"
    })
    imports = _corpus_imports(file_key, found).edges
    payload = assemble_graph(files, winner, relations, mentions, imports, today=now, tenant=tenant, generation=graph.generation_id)
    _keep(_BODIES, file_key, (found, now, payload))
    return payload


def _corpus_imports(
    key: tuple[str, str], found: tuple[list[_CorpusRow], SupersessionGraph, SemanticGraphProjection]
) -> _ImportProjection:
    """`_python_imports` over every chunk of a read view, once per view: View 2 asked for it on
    every page, a regex over every `.py` chunk of the file (CCA PERF-003)."""
    with _CORPUS_LOCK:
        kept = _IMPORTS.get(key)
    if kept is not None and kept[0] is found:
        return kept[1]
    rows = found[0]
    files = {row.file for row in rows}
    edges = _python_imports(rows, files)
    by_target: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        by_target.setdefault(edge[1], []).append(edge)
    projection = _ImportProjection(edges, by_target, files)
    _keep(_IMPORTS, key, (found, projection))
    return projection


def _keep(cache: dict[tuple[str, str], Any], key: tuple[str, str], value: Any) -> None:
    """Store `value`, keeping at most `_MAX_CORPORA` files, as `_CORPORA` does."""
    with _CORPUS_LOCK:
        cache.pop(key, None)
        cache[key] = value
        for old in list(cache)[:-_MAX_CORPORA]:
            cache.pop(old, None)


def _source_projection(
    key: tuple[str, str], found: tuple[list[_CorpusRow], SupersessionGraph, SemanticGraphProjection],
) -> _SourceProjection:
    """Index source detail lookups once for this read view, with the same eviction limit as the corpus."""
    with _CORPUS_LOCK:
        kept = _SOURCE_PROJECTIONS.get(key)
        if kept is not None and kept[0] is found:
            return kept[1]
    result = _index_sources(*found)
    with _CORPUS_LOCK:
        current = _SOURCE_PROJECTIONS.get(key)
        if current is not None and current[0] is found:
            return current[1]
        if current is kept:
            _keep(_SOURCE_PROJECTIONS, key, (found, result))
    return result


def _index_sources(
    rows: list[_CorpusRow], graph: SupersessionGraph, semantic: SemanticGraphProjection,
) -> _SourceProjection:
    rows_by_source: dict[str, list[_CorpusRow]] = {}
    sources_by_file: dict[str, set[str]] = {}
    indexed_keys: dict[str, set[str]] = {}
    source_by_chunk: dict[str, str] = {}
    for row in rows:
        rows_by_source.setdefault(row.source, []).append(row)
        sources_by_file.setdefault(row.file, set()).add(row.source)
        source_by_chunk[row.id] = row.source
        graph_key = graph.key(row.file, row.source)
        indexed_keys.setdefault(row.file, set()).add(graph_key)
        indexed_keys.setdefault(row.source, set()).add(graph_key)

    entities = {entity.id: entity for entity in semantic.entities}
    mentions_by_source: dict[str, dict[tuple[str, str], int]] = {}
    for mention in semantic.mentions:
        source = source_by_chunk.get(mention.chunk_id)
        entity = entities[mention.entity_id]
        if source is not None and entity.kind != "file":
            counted = mentions_by_source.setdefault(source, {})
            name = (entity.canonical_name, entity.kind)
            counted[name] = counted.get(name, 0) + 1

    relations_by_file: dict[str, list[DetailRelation]] = {}
    for relation in semantic.relations:
        subject, obj = entities[relation.subject_id], entities[relation.object_id]
        detail = (relation.relation, relation.status, subject.canonical_name, subject.kind, obj.canonical_name, obj.kind)
        for file in {entity.canonical_name for entity in (subject, obj) if entity.kind == "file"}:
            relations_by_file.setdefault(file, []).append(detail)

    replaced_by_source: dict[str, set[str]] = {}
    for old, claims in graph.candidates.items():
        for claimant, _when in claims:
            replaced_by_source.setdefault(claimant, set()).add(old)
    return _SourceProjection(
        rows_by_source, sources_by_file, mentions_by_source, relations_by_file, replaced_by_source, indexed_keys,
    )


def source_detail(dsn: str, tenant: str, file: str, *, today: date | None = None) -> dict[str, Any] | None:
    """`recall.dashboard.corpus.source_detail` for a lite file, through the same `assemble_detail`."""
    from recall.dashboard.corpus import MAX_CHUNKS, MAX_ENTITIES, MAX_RELATIONS, _instant, assemble_detail
    from recall.related import metadata_ordinal

    now = (today or datetime.now(UTC).date()).isoformat()
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return None
        file_key = _file_key(c, tenant)
        found = _corpus(c, tenant)
    _, graph, semantic = found
    projection = _source_projection(file_key, found)
    source = file
    if source not in projection.rows_by_source:
        candidates = projection.sources_by_file.get(file)
        if candidates is None or len(candidates) != 1:
            return None
        source = next(iter(candidates))
    mine = projection.rows_by_source[source]
    canonical = str(mine[0].file)
    chunks = [
        (r.id, metadata_ordinal(r.metadata), r.source, r.text, r.metadata, r.indexed_at, r.first_indexed_at, r.file)
        for r in mine[: MAX_CHUNKS + 1]
    ]
    key = graph.key(canonical, source)
    counted = projection.mentions_by_source.get(source, {})
    named_entities = sorted(((n, k, m) for (n, k), m in counted.items()), key=lambda e: (-e[2], e[0]))[:MAX_ENTITIES]
    relations = sorted(
        projection.relations_by_file.get(canonical, []),
        key=lambda r: (r[0], r[2], r[4]),
    )[:MAX_RELATIONS]
    imports: list[tuple[str, str]] = []
    imported_by: list[tuple[str, str]] = []
    if canonical.endswith(".py"):
        # This copy's own statements, as Postgres scans `source_uri = %s`: another root's copy of
        # the same path is the same importer by name (CCA BUG-002).
        import_projection = _corpus_imports(file_key, found)
        imports = [pair for pair in _python_imports(mine, import_projection.files) if pair[0] == canonical]
        imported_by = import_projection.by_target.get(canonical, [])
    successor = graph.successor(key, now=_instant(now))
    replaced = sorted(projection.replaced_by_source.get(key, ()))
    named = {name for name in (successor, *replaced) if name}
    indexed = {graph_key for name in named for graph_key in projection.indexed_keys.get(name, ())}
    return assemble_detail(
        tenant=tenant, generation=semantic.generation_id, key=key, source=source, canonical=canonical, chunks=chunks,
        total=len(mine), first_indexed=min(row.first_indexed_at for row in mine),
        last_indexed=max(row.indexed_at for row in mine),
        graph=graph, successor=successor, replaced=replaced, indexed=indexed, entities=named_entities,
        relations=relations, collided=key != canonical, imports=imports, imported_by=imported_by, today=now,
    )
