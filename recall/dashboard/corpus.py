"""A corpus read straight from the database: one source file (View 2), and every file as a graph.

These serve tenants that have no memo folder on this machine, such as the code and docs corpora,
and work for the memory tenant too. Everything is read-only: the connection is
`recall.dashboard.db`'s read-only session, every value is a bound parameter, and each query first
sets `recall.tenant_id`, so a role that is subject to row-level security sees exactly the
tenant's rows rather than none.

Supersession is resolved with the same SQL fragment and the same pure resolver the trust layer
uses (`recall.supersession`), so a file drawn as superseded here is one search marks superseded.
The graph has the JSON shape `recall.dashboard.graph.build_graph` returns, so the same page draws
it, plus three fields that say where it came from (`origin`, `tenant`, `generation`) and one more
edge kind:

- `supersedes`: the newer file to the one it replaces, as for memo files;
- `link`: an authored `references` or `depends_on` relation between two files;
- `shares`: two files mentioning the same uncommon entity. A code corpus has no relations, so
  without these it would be stars with no lines. An entity mentioned by more than
  `SHARE_MAX_FILES` files is too common to say anything and is skipped, and each file keeps only
  its `SHARE_PER_FILE` strongest partners, so a common name cannot join everything to everything.

A file is `coalesce(metadata->>'file', source_uri)` throughout: the root-relative path where the
indexer recorded one, else the source itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import PurePosixPath
from typing import Any

from psycopg import sql

from recall.dashboard.db import DashboardDB
from recall.db_constants import TENANT_GUC
from recall.supersession import SUPERSEDES_TARGET_ROWS_SQL, resolve_supersession_candidates

__all__ = ["HUB_LINKS", "SHARE_MAX_FILES", "SHARE_PER_FILE", "assemble_graph", "corpus_graph", "resolve_imports", "source_detail"]

#: An entity named in more files than this is too common to link them (a project name, `self`).
SHARE_MAX_FILES = 8
#: The strongest shared-entity partners each file keeps.
SHARE_PER_FILE = 4
#: A file with at least this many outgoing links is an index page, hidden by default as on the
#: memo graph.
HUB_LINKS = 40
#: Bounds on what one View 2 page reads.
MAX_CHUNKS = 200
MAX_ENTITIES = 60
MAX_RELATIONS = 120

#: `supersession_all`'s query, with the trust layer's own target-expansion fragment.
_SUPERSESSION_ROWS = sql.SQL(
    "SELECT c.metadata->>'file', t.supersedes, min(COALESCE(c.first_indexed_at, c.indexed_at)) "
    "FROM recall_chunks_v1 AS c CROSS JOIN LATERAL ({}) AS t "
    "WHERE c.tenant_id = %s AND c.generation_id = %s AND c.metadata ? 'file' GROUP BY 1, 2 ORDER BY 1, 2"
).format(sql.SQL(SUPERSEDES_TARGET_ROWS_SQL))


@contextmanager
def _tenant(db: DashboardDB, tenant: str) -> Iterator[Any]:
    """A read-only connection whose row-level security tenant is `tenant`."""
    with db.connect() as c:
        c.execute("select set_config(%s, %s, false)", (TENANT_GUC, tenant))
        yield c


def _generation(c: Any, tenant: str) -> str | None:
    row = c.execute("select active_generation_id from recall_tenant_state where tenant_id = %s", (tenant,)).fetchone()
    return str(row[0]) if row and row[0] else None


def _winners(c: Any, tenant: str, generation: str) -> tuple[dict[str, str], frozenset[str]]:
    """`(superseded file -> the file that replaces it, unresolved claims)`, as search resolves them."""
    rows = c.execute(_SUPERSESSION_ROWS, (tenant, generation)).fetchall()
    winner, unresolved, _candidates = resolve_supersession_candidates(rows)
    return winner, unresolved


#: A Python import line: `from MODULE import NAMES` or `import MODULE`. Matched inside Postgres
#: (newline-sensitive, so `^` is a line start), so only the matches leave the database.
_IMPORT_PATTERN = r"^[ \t]*(?:from[ \t]+([.A-Za-z0-9_]+)[ \t]+import[ \t]+\(?([A-Za-z0-9_, \t]+)|import[ \t]+([A-Za-z0-9_.]+))"


def _module_of(file: str) -> str | None:
    """`pkg/sub/mod.py` -> `pkg.sub.mod`; `pkg/__init__.py` -> `pkg`."""
    if not file.endswith(".py"):
        return None
    parts = file[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p) or None


def _import_edges(c: Any, tenant: str, generation: str, files: set[str]) -> list[tuple[str, str]]:
    """`(importer, imported)` pairs for every Python import in the corpus that names one of its files."""
    rows = c.execute(
        "select coalesce(c.metadata->>'file', c.source_uri), m[1], m[2], m[3] from recall_chunks_v1 c, "
        "regexp_matches(c.text, %s, 'gn') as m where c.tenant_id = %s and c.generation_id = %s "
        "and coalesce(c.metadata->>'file', c.source_uri) like %s",
        (_IMPORT_PATTERN, tenant, generation, "%.py"),
    ).fetchall()
    return resolve_imports(rows, files)


def resolve_imports(rows: Any, files: set[str]) -> list[tuple[str, str]]:
    """`(importer, imported)` file pairs from `(importer, from_module, names, plain_module)` matches.

    A module resolves by its full dotted path, else as the dotted tail of exactly one file's module
    (a corpus indexed from a parent directory adds a prefix to every path); a tail two files share
    resolves to neither. `from pkg import name` links to `pkg/name.py` when that is a file, else to
    `pkg`. A relative import resolves against the importing file's package. Pure.
    """
    by_module: dict[str, str] = {}
    by_suffix: dict[str, set[str]] = defaultdict(set)
    for file in files:
        module = _module_of(file)
        if module:
            by_module[module] = file
            parts = module.split(".")
            for i in range(len(parts)):
                by_suffix[".".join(parts[i:])].add(file)

    def resolve(module: str) -> str | None:
        if module in by_module:
            return by_module[module]
        found = by_suffix.get(module)
        return next(iter(found)) if found and len(found) == 1 else None

    pairs: set[tuple[str, str]] = set()
    for importer, from_module, names, plain in rows:
        targets: list[str | None] = []
        if plain:
            targets.append(resolve(plain))
        elif from_module:
            base = from_module
            if base.startswith("."):
                package = (_module_of(importer) or "").split(".")
                if not importer.endswith("__init__.py"):
                    package = package[:-1]
                dots = len(base) - len(base.lstrip("."))
                package = package[: len(package) - (dots - 1)] if dots > 1 else package
                rest = base.lstrip(".")
                base = ".".join([*package, rest] if rest else package)
            for name in (n.strip() for n in (names or "").split(",")):
                if name:
                    targets.append(resolve(f"{base}.{name}") or resolve(base))
            if not names:
                targets.append(resolve(base))
        for target in targets:
            if target and target != importer:
                pairs.add((importer, target))
    return sorted(pairs)


def _day(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).date().isoformat()
    return str(value)[:10] if value else ""


def _title(file: str, hierarchy: Any) -> str:
    if isinstance(hierarchy, list):
        for heading in hierarchy:
            if isinstance(heading, str) and heading.strip():
                return heading.strip()[:160]
    return PurePosixPath(file).stem or file


def _state(file: str, winner: dict[str, str], valid_from: str | None, valid_until: str | None, today: str) -> str:
    if file in winner:
        return "superseded"
    if (valid_until and valid_until[:10] < today) or (valid_from and valid_from[:10] > today):
        return "expired"
    return "current"


def corpus_graph(db: DashboardDB, tenant: str, *, today: date | None = None) -> dict[str, Any]:
    """Every file of `tenant`'s active generation as a node; supersession, relations and shared
    entities as edges. The shape of `build_graph`, see the module docstring."""
    now = (today or datetime.now(UTC).date()).isoformat()
    with _tenant(db, tenant) as c:
        generation = _generation(c, tenant)
        if generation is None:
            return {"nodes": [], "edges": [], "counts": {"memos": 0, "edges": 0}, "notes": [f"{tenant} has no active generation"],
                    "origin": "database", "tenant": tenant, "generation": None}
        files = c.execute(
            "select coalesce(c.metadata->>'file', c.source_uri), count(*), min(coalesce(c.first_indexed_at, c.indexed_at)), "
            "max(c.metadata->>'type'), max(c.metadata->>'valid_from'), max(c.metadata->>'valid_until'), "
            "(array_agg(c.metadata->'heading_hierarchy' order by c.chunk_ordinal))[1] "
            "from recall_chunks_v1 c where c.tenant_id = %s and c.generation_id = %s group by 1",
            (tenant, generation),
        ).fetchall()
        winner, _unresolved = _winners(c, tenant, generation)
        relations = c.execute(
            "select r.relation, s.canonical_name, o.canonical_name from recall_graph_relations_v1 r "
            "join recall_graph_entities_v1 s on s.tenant_id = r.tenant_id and s.generation_id = r.generation_id and s.entity_id = r.subject_id "
            "join recall_graph_entities_v1 o on o.tenant_id = r.tenant_id and o.generation_id = r.generation_id and o.entity_id = r.object_id "
            "where r.tenant_id = %s and r.generation_id = %s and s.entity_kind = 'file' and o.entity_kind = 'file' "
            "and r.relation in ('references', 'depends_on')",
            (tenant, generation),
        ).fetchall()
        mentions = c.execute(
            "select m.entity_id, coalesce(c.metadata->>'file', c.source_uri) from recall_graph_mentions_v1 m "
            "join recall_chunks_v1 c on c.tenant_id = m.tenant_id and c.generation_id = m.generation_id and c.chunk_id = m.chunk_id "
            "join recall_graph_entities_v1 e on e.tenant_id = m.tenant_id and e.generation_id = m.generation_id and e.entity_id = m.entity_id "
            "where m.tenant_id = %s and m.generation_id = %s and e.entity_kind <> 'file' group by 1, 2",
            (tenant, generation),
        ).fetchall()
        imports = _import_edges(c, tenant, generation, {str(row[0]) for row in files})
    return assemble_graph(
        files, winner, relations, mentions, imports, today=now, tenant=tenant, generation=generation,
    )


def assemble_graph(
    files: Any,
    winner: dict[str, str],
    relations: Any,
    mentions: Any,
    imports: list[tuple[str, str]],
    *,
    today: str,
    tenant: str,
    generation: str,
) -> dict[str, Any]:
    """The graph from the queries' rows. Pure, so its rules are testable without a database.

    `files`: `(file, chunks, first indexed, type, valid_from, valid_until, heading hierarchy)`;
    `winner`: superseded file -> the file replacing it; `relations`: `(relation, subject file,
    object file)`; `mentions`: `(entity id, file)`; `imports`: `(importer, imported)`.
    """
    now = today
    nodes: dict[str, dict[str, Any]] = {}
    for file, chunks, first, kind, valid_from, valid_until, hierarchy in files:
        nodes[file] = {
            "id": file, "title": _title(file, hierarchy), "description": f"{chunks} chunk(s)", "type": kind or "",
            "state": _state(file, winner, valid_from, valid_until, now), "born": _day(first) or now,
            "valid_from": valid_from, "valid_until": valid_until,
            "folder": file.split("/", 1)[0] if "/" in file else "", "issues": [], "chunks": chunks,
        }
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source: str, target: str, kind: str, **extra: Any) -> None:
        key = (source, target, kind)
        if source != target and source in nodes and target in nodes and key not in seen:
            seen.add(key)
            edges.append({"source": source, "target": target, "kind": kind, **extra})

    for replaced, newer in sorted(winner.items()):
        add(newer, replaced, "supersedes")
    for relation, subject, obj in sorted(relations):
        add(str(subject), str(obj), "link", relation=relation)
    for importer, imported in imports:
        add(importer, imported, "link", relation="imports")
    by_entity: dict[Any, set[str]] = defaultdict(set)
    for entity, file in mentions:
        by_entity[entity].add(file)
    pair_weight: Counter[tuple[str, str]] = Counter()
    for members in by_entity.values():
        if 2 <= len(members) <= SHARE_MAX_FILES:
            ordered = sorted(members)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    pair_weight[(a, b)] += 1
    partners: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (a, b), weight in pair_weight.items():
        partners[a].append((weight, b))
        partners[b].append((weight, a))
    # Mutual: an edge survives only when each file ranks the other among its strongest, so no file
    # ends up with more than SHARE_PER_FILE of them. Keeping an edge either end ranks would let a
    # file that is the only partner of many small files collect them all, the hub this prevents.
    best = {
        file: {other for _weight, other in sorted(found, key=lambda item: (-item[0], item[1]))[:SHARE_PER_FILE]}
        for file, found in partners.items()
    }
    kept = {(min(a, b), max(a, b)) for a, mine in best.items() for b in mine if a in best.get(b, ())}
    for a, b in sorted(kept):
        add(a, b, "shares", weight=pair_weight[(a, b)])

    out_links: Counter[str] = Counter(e["source"] for e in edges if e["kind"] == "link")
    degree: Counter[str] = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for node in nodes.values():
        node["degree"] = degree[node["id"]]
        node["hub"] = out_links[node["id"]] >= HUB_LINKS
        node["isolated"] = False
    counts = Counter(node["state"] for node in nodes.values())
    kinds = Counter(e["kind"] for e in edges)
    return {
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": edges,
        "counts": {
            "memos": len(nodes), "edges": len(edges), "issues": 0, "isolated": 0, "pending": 0,
            "current": counts["current"], "superseded": counts["superseded"], "expired": counts["expired"],
            "supersedes": kinds["supersedes"], "link": kinds["link"], "shares": kinds["shares"],
            "hubs": sum(1 for n in nodes.values() if n["hub"]),
        },
        "notes": [],
        "origin": "database",
        "tenant": tenant,
        "generation": generation,
    }


def source_detail(db: DashboardDB, tenant: str, file: str, *, today: date | None = None) -> dict[str, Any] | None:
    """One file of `tenant` (matched by its root-relative `file` or its `source_uri`), or None.

    Its identity, state and validity, what replaces it and what it replaces, its chunks, the
    entities they mention and the relations its file entity takes part in.
    """
    now = (today or datetime.now(UTC).date()).isoformat()
    with _tenant(db, tenant) as c:
        generation = _generation(c, tenant)
        if generation is None:
            return None
        chunks = c.execute(
            "select c.chunk_id, c.chunk_ordinal, c.source_uri, c.text, c.metadata, c.indexed_at, "
            "coalesce(c.first_indexed_at, c.indexed_at), coalesce(c.metadata->>'file', c.source_uri) from recall_chunks_v1 c "
            "where c.tenant_id = %s and c.generation_id = %s "
            "and (coalesce(c.metadata->>'file', c.source_uri) = %s or c.source_uri = %s) "
            "order by c.chunk_ordinal limit %s",
            (tenant, generation, file, file, MAX_CHUNKS + 1),
        ).fetchall()
        if not chunks:
            return None
        canonical = str(chunks[0][7])
        total = c.execute(
            "select count(*) from recall_chunks_v1 c where c.tenant_id = %s and c.generation_id = %s "
            "and coalesce(c.metadata->>'file', c.source_uri) = %s",
            (tenant, generation, canonical),
        ).fetchone()[0]
        winner, unresolved = _winners(c, tenant, generation)
        entities = c.execute(
            "select e.canonical_name, e.entity_kind, count(*) from recall_graph_mentions_v1 m "
            "join recall_graph_entities_v1 e on e.tenant_id = m.tenant_id and e.generation_id = m.generation_id and e.entity_id = m.entity_id "
            "join recall_chunks_v1 c on c.tenant_id = m.tenant_id and c.generation_id = m.generation_id and c.chunk_id = m.chunk_id "
            "where m.tenant_id = %s and m.generation_id = %s and coalesce(c.metadata->>'file', c.source_uri) = %s "
            "and e.entity_kind <> 'file' group by 1, 2 order by 3 desc, 1 limit %s",
            (tenant, generation, canonical, MAX_ENTITIES),
        ).fetchall()
        relations = c.execute(
            "select r.relation, r.status, s.canonical_name, s.entity_kind, o.canonical_name, o.entity_kind "
            "from recall_graph_relations_v1 r "
            "join recall_graph_entities_v1 s on s.tenant_id = r.tenant_id and s.generation_id = r.generation_id and s.entity_id = r.subject_id "
            "join recall_graph_entities_v1 o on o.tenant_id = r.tenant_id and o.generation_id = r.generation_id and o.entity_id = r.object_id "
            "where r.tenant_id = %s and r.generation_id = %s and ((s.entity_kind = 'file' and s.canonical_name = %s) "
            "or (o.entity_kind = 'file' and o.canonical_name = %s)) order by 1, 3, 5 limit %s",
            (tenant, generation, canonical, canonical, MAX_RELATIONS),
        ).fetchall()
        imports: list[tuple[str, str]] = []
        if canonical.endswith(".py"):
            # The whole corpus's imports, because "imported by" needs every other file's.
            every = {
                str(row[0]) for row in c.execute(
                    "select distinct coalesce(c.metadata->>'file', c.source_uri) from recall_chunks_v1 c "
                    "where c.tenant_id = %s and c.generation_id = %s",
                    (tenant, generation),
                ).fetchall()
            }
            imports = _import_edges(c, tenant, generation, every)

    first_meta = chunks[0][4] if isinstance(chunks[0][4], dict) else {}
    valid_from, valid_until = first_meta.get("valid_from"), first_meta.get("valid_until")
    declared = first_meta.get("supersedes")
    declared_list = [str(d) for d in (declared if isinstance(declared, list) else ([declared] if declared else []))]
    return {
        "tenant": tenant,
        "generation": generation,
        "file": canonical,
        "source_uri": str(chunks[0][2]),
        "project": first_meta.get("project"),
        "indexed_commit": first_meta.get("indexed_commit"),
        "type": first_meta.get("type"),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "state": _state(canonical, winner, valid_from, valid_until, now),
        "superseded_by": winner.get(canonical),
        "supersedes": sorted(old for old, newer in winner.items() if newer == canonical),
        "declared_supersedes": declared_list,
        "unresolved_claims": sorted(str(t) for t in unresolved if str(t) in set(declared_list)),
        "first_indexed": _day(min(row[6] for row in chunks)),
        "last_indexed": _day(max(row[5] for row in chunks)),
        "chunk_count": total,
        "chunks": [
            {
                "chunk_id": row[0], "ordinal": row[1], "text": row[3] or "",
                "headings": [h for h in ((row[4] or {}).get("heading_hierarchy") or []) if isinstance(h, str)],
            }
            for row in chunks[:MAX_CHUNKS]
        ],
        "entities": [{"name": n, "kind": k, "mentions": m} for n, k, m in entities],
        "relations": [
            {"relation": rel, "status": status, "subject": sn, "subject_kind": sk, "object": on, "object_kind": ok}
            for rel, status, sn, sk, on, ok in relations
        ],
        "imports": sorted(target for importer, target in imports if importer == canonical),
        "imported_by": sorted(importer for importer, target in imports if target == canonical),
    }
