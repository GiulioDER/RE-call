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
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from recall.dashboard.db import SEARCH_EVENTS, USE_REPORT_EVENT, DatabaseUnavailable, _summarise, stale_row
from recall.lite import LiteStoreError, lite_dsn_path
from recall.stale_reports import STALE_REPORT_EVENT

if TYPE_CHECKING:
    from recall.semantic_graph import SemanticGraphProjection
    from recall.supersession import SupersessionGraph


def store_path(dsn: str) -> Path:
    """The file the server would open, refusing as the server does a DSN it would refuse."""
    try:
        return lite_dsn_path(dsn)
    except LiteStoreError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


@contextmanager
def connect(dsn: str) -> Iterator[sqlite3.Connection]:
    path = store_path(dsn)
    if not path.exists():
        raise DatabaseUnavailable(f"no lite store at {path} yet: `recall index <memory folder>` creates it")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"{type(exc).__name__}: {exc}") from exc
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
    limit: int | None = None, event_id: str | None = None,
) -> Iterator[tuple[Any, ...]]:
    """`(event_id, event_type, actor, generation_id, source_uri, payload, created_at)`, newest first."""
    if tenant != _tenant(c):
        return
    sql = (
        "SELECT event_id, event_type, actor, generation_id, source_uri, payload, created_at FROM audit_events "
        "WHERE event_type IN (SELECT value FROM json_each(?))"
    )
    args: list[Any] = [json.dumps(list(types))]
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

    found = inspect_lite_file(store_path(dsn))
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
    with connect(dsn) as c:
        newest = c.execute("SELECT max(indexed_at) FROM chunks").fetchone()[0]
    return [
        {
            "tenant": found.tenant,
            "generation": found.generation_id if found.chunks else None,
            "state": "active" if found.chunks else None,
            "activated_at": _when(newest),
            "updated_at": _when(newest),
            "embedder": {"provider": None, "model": found.embedder_model, "dimension": found.dim, "profile_id": None, "verified": None},
            "chunks": found.chunks,
            "sources": found.sources,
            "calibration": calibration,
        }
    ]


def generations(dsn: str, tenant: str, limit: int = 15) -> list[dict[str, Any]]:
    """A lite store keeps no history of builds: its one corpus is the current one."""
    every = [t for t in tenants(dsn) if t["tenant"] == tenant and t["generation"]]
    with connect(dsn) as c:
        first = c.execute("SELECT min(first_indexed_at) FROM chunks").fetchone()[0]
    return [
        {"generation": t["generation"], "state": "active", "created_at": _when(first), "activated_at": t["activated_at"],
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


def search(dsn: str, tenant: str, event_id: str) -> dict[str, Any] | None:
    with connect(dsn) as c:
        row = next(_events(c, tenant, SEARCH_EVENTS, limit=1, event_id=event_id), None)
        return _search_row(row) if row is not None else None


def control(dsn: str, tenant: str, *, retrievals: bool = True, source: str | None = None) -> dict[str, Any]:
    """`recall.dashboard.db.control`, the same rules: one retrieval per (search, memory), at its best rank."""
    memos: dict[str, dict[str, Any]] = {}

    def memo(source: str) -> dict[str, Any]:
        return memos.setdefault(source, {
            "source": source, "retrieved": 0, "first": 0, "last_retrieved": None, "ok": 0, "superseded": 0,
            "low_confidence": 0, "used": 0, "wrong": 0, "helped": 0, "no_difference": 0, "misled": 0,
            "succeeded": 0, "failed": 0, "last_reported": None,
        })

    with connect(dsn) as c:
        search_rows = list(_events(c, tenant, SEARCH_EVENTS))
        report_rows = list(_events(c, tenant, (USE_REPORT_EVENT,)))
    for _id, event_type, _a, _g, _s, raw, created in (search_rows if retrievals else ()):
        if event_type != "search_decision":
            continue
        at = _when(created)
        hits = _payload(raw).get("hits")
        best: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(hits if isinstance(hits, list) else [], start=1):
            if not isinstance(hit, dict) or hit.get("source") is None:
                continue
            if source is not None and str(hit["source"]) != source:
                continue
            seen = best.setdefault(str(hit["source"]), {"rank": rank, "verdicts": set()})
            seen["rank"] = min(seen["rank"], rank)
            verdict = hit.get("verdict")
            if isinstance(verdict, str):
                seen["verdicts"].add(verdict)
        for found, seen in best.items():
            row = memo(found)
            row["retrieved"] += 1
            row["first"] += seen["rank"] == 1
            row["ok"] += "ok" in seen["verdicts"]
            row["superseded"] += "superseded" in seen["verdicts"]
            row["low_confidence"] += "low_confidence" in seen["verdicts"]
            if at and (row["last_retrieved"] is None or at > row["last_retrieved"]):
                row["last_retrieved"] = at
    outcomes = [_summarise(_payload(r[5]), r[1])["outcome"] for r in search_rows]
    summary = {"searches": len(search_rows), "answered": outcomes.count("answered"),
               "abstained": outcomes.count("abstained"), "refused": outcomes.count("refused"),
               "reports": len(report_rows), "helped": 0, "no_difference": 0, "misled": 0, "succeeded": 0, "failed": 0}
    for *_ignored, raw, created in report_rows:
        payload, at = _payload(raw), _when(created)
        effect, succeeded = payload.get("effect"), payload.get("task_succeeded")
        if effect in ("helped", "no_difference", "misled"):
            summary[effect] += 1
        if succeeded is True or succeeded is False:
            summary["succeeded" if succeeded else "failed"] += 1
        for role in ("used", "wrong"):
            for named in _sources(payload, role):
                if source is not None and str(named) != source:
                    continue
                row = memo(str(named))
                row[role] += 1
                if role == "used" and effect in ("helped", "no_difference"):
                    row[effect] += 1
                if role == "wrong" and effect == "misled":
                    row["misled"] += 1
                if role == "used" and (succeeded is True or succeeded is False):
                    row["succeeded" if succeeded else "failed"] += 1
                if at and (row["last_reported"] is None or at > row["last_reported"]):
                    row["last_reported"] = at
    summary["memos_retrieved"] = sum(1 for m in memos.values() if m["retrieved"])
    ordered = sorted(memos.values(), key=lambda m: (-m["retrieved"], -m["used"], m["source"]))
    return {"summary": summary, "memos": ordered}


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
            out.append({
                "event_id": event_id, "actor": actor, "created_at": _when(created),
                "task": str(payload.get("task", "")), "effect": payload.get("effect"), "used": used, "wrong": wrong,
                "task_succeeded": payload.get("task_succeeded"), "query": payload.get("query"), "note": payload.get("note"),
            })
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
    digest = hashlib.sha256(str(c.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()[0]).encode())
    ordered = []
    for rowid, cid, source, text, raw, indexed, first in c.execute(
        "SELECT rowid, id, source, text, metadata, indexed_at, first_indexed_at FROM chunks ORDER BY id"
    ):
        digest.update("\0".join(("", cid, source, text, raw, indexed)).encode())
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
    return chunks, winner, _semantic_graph(c, tenant, digest.hexdigest(), chunks)


#: `{(file, tenant): (stamp of its rows, graph)}`, one entry per file this process reads.
_GRAPHS: dict[tuple[str, str], tuple[str, SemanticGraphProjection]] = {}


def _semantic_graph(c: sqlite3.Connection, tenant: str, stamp: str, rows: list[_CorpusRow]) -> SemanticGraphProjection:
    """`graph_of`, kept while the file's rows are the ones it was built from (CCA PERF-002): built
    in memory it is the whole build, seconds on a large store, and View 2 asks for it on every page.
    `stamp` digests the corpus version and every row's id, source, text, metadata and `indexed_at`,
    everything the graph's entities, mentions and relations are built from, so no write can leave
    a kept graph describing other text. (The vectors are not read: they feed only its generation
    id, which could lag after a file is recreated with the same rows, version and write times but
    other vectors.) The version alone is not enough (a file deleted and indexed again, or a
    backup copied over it, starts it again and can reach the same number with other rows), nor are
    ids and times (ids are positional, so an edited memo keeps its ids, and under an evaluation
    clock its write time too)."""
    from recall.lite.graph import graph_of
    from recall.types import Chunk

    key = (str(c.execute("PRAGMA database_list").fetchone()[2]), tenant)
    kept = _GRAPHS.get(key)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    graph = graph_of(c, tenant, [Chunk(id=r.id, source=r.source, text=r.text, metadata=r.metadata) for r in rows])
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
        found = _corpus(c, tenant) if tenant == _tenant(c) else None
    if found is None or not found[0]:
        return empty_graph(tenant, f"{tenant} has no corpus in this file")
    rows, winner, graph = found
    by_file: dict[tuple[str, str], list[_CorpusRow]] = {}
    for row in rows:
        by_file.setdefault((row.file, row.source), []).append(row)

    def most(group: list[_CorpusRow], key: str) -> str | None:
        values = [str(r.metadata[key]) for r in group if r.metadata.get(key) is not None]
        return max(values) if values else None  # Postgres's max() over the text values

    files = [
        (file, len(group), min(r.first_indexed_at for r in group), most(group, "type"), most(group, "valid_from"),
         most(group, "valid_until"), group[0].metadata.get("heading_hierarchy"), source)
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
    imports = _python_imports(rows, {file for file, _source in by_file})
    return assemble_graph(files, winner, relations, mentions, imports, today=now, tenant=tenant, generation=graph.generation_id)


def source_detail(dsn: str, tenant: str, file: str, *, today: date | None = None) -> dict[str, Any] | None:
    """`recall.dashboard.corpus.source_detail` for a lite file, through the same `assemble_detail`."""
    from recall.dashboard.corpus import MAX_CHUNKS, MAX_ENTITIES, MAX_RELATIONS, _instant, assemble_detail
    from recall.related import metadata_ordinal

    now = (today or datetime.now(UTC).date()).isoformat()
    with connect(dsn) as c:
        if tenant != _tenant(c):
            return None
        rows, graph, semantic = _corpus(c, tenant)
    sources = sorted({row.source for row in rows if file in (row.file, row.source)}, key=lambda s: (s != file, s))
    if not sources or (len(sources) > 1 and sources[0] != file):
        return None
    source = sources[0]
    mine = [row for row in rows if row.source == source]
    canonical = str(mine[0].file)
    chunks = [
        (r.id, metadata_ordinal(r.metadata), r.source, r.text, r.metadata, r.indexed_at, r.first_indexed_at, r.file)
        for r in mine[: MAX_CHUNKS + 1]
    ]
    key = graph.key(canonical, source)
    entities = {entity.id: entity for entity in semantic.entities}
    mine_ids = {row.id for row in mine}
    counted: dict[tuple[str, str], int] = {}
    for mention in semantic.mentions:
        entity = entities[mention.entity_id]
        if mention.chunk_id in mine_ids and entity.kind != "file":
            counted[(entity.canonical_name, entity.kind)] = counted.get((entity.canonical_name, entity.kind), 0) + 1
    named_entities = sorted(((n, k, m) for (n, k), m in counted.items()), key=lambda e: (-e[2], e[0]))[:MAX_ENTITIES]
    relations = sorted(
        (
            (r.relation, r.status, subject.canonical_name, subject.kind, obj.canonical_name, obj.kind)
            for r in semantic.relations
            for subject, obj in [(entities[r.subject_id], entities[r.object_id])]
            if (subject.kind == "file" and subject.canonical_name == canonical)
            or (obj.kind == "file" and obj.canonical_name == canonical)
        ),
        key=lambda r: (r[0], r[2], r[4]),
    )[:MAX_RELATIONS]
    imports: list[tuple[str, str]] = []
    imported_by: list[tuple[str, str]] = []
    if canonical.endswith(".py"):
        files = {row.file for row in rows}
        # This copy's own statements, as Postgres scans `source_uri = %s`: another root's copy of
        # the same path is the same importer by name (CCA BUG-002).
        imports = [pair for pair in _python_imports(mine, files) if pair[0] == canonical]
        imported_by = [pair for pair in _python_imports(rows, files) if pair[1] == canonical]
    successor = graph.successor(key, now=_instant(now))
    replaced = sorted(old for old, claims in graph.candidates.items() if any(f == key for f, _when in claims))
    named = {name for name in (successor, *replaced) if name}
    indexed = {graph.key(row.file, row.source) for row in rows if row.file in named or row.source in named}
    return assemble_detail(
        tenant=tenant, generation=semantic.generation_id, key=key, source=source, canonical=canonical, chunks=chunks,
        total=len(mine), first_indexed=min(row.first_indexed_at for row in mine),
        last_indexed=max(row.indexed_at for row in mine),
        graph=graph, successor=successor, replaced=replaced, indexed=indexed, entities=named_entities,
        relations=relations, collided=key != canonical, imports=imports, imported_by=imported_by, today=now,
    )
