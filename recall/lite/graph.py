"""The semantic graph of a lite store: the PostgreSQL projection (migration 0016) in the same file.

`recall.semantic_graph.build_semantic_graph` builds it from the chunks alone: Markdown and wiki
links, `supersedes`, `depends_on`, `recall_graph` front matter and conservative headings, with no
model. It is keyed by the lite generation id (`recall.lite.store.generation_id_for`). A lite store
has one corpus, so the file holds one graph, the one for the corpus as it is now.

Every corpus write clears it in the same transaction (`LiteStore._bump`), so graph rows never
outlive the chunks they quote, a forgotten memo's included, and the rows present are the current
corpus's. A release from before these tables cannot clear them, which is why the marker also
records the corpus version it was built at: a marker from another version reads as no graph.

It serves navigation (`recall graph`, the dashboard's corpus graph). Search never reads it: graph
expansion is a generation feature of the PostgreSQL install, off by default there too.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from recall.semantic_graph._build import build_semantic_graph
from recall.semantic_graph._model import (
    SEMANTIC_GRAPH_SCHEMA_VERSION,
    GraphReadiness,
    SemanticEntity,
    SemanticGraphDiagnostic,
    SemanticGraphProjection,
    SemanticMention,
    _thaw,
    marker_schema_version,
    semantic_graph_marker,
)
from recall.types import Chunk

__all__ = ["GRAPH_SCHEMA", "build_graph", "clear_graph", "graph_of", "load_graph", "read_marker", "readiness", "write_graph"]

#: The tables of 0016, one generation at a time; JSON columns are TEXT. Created on every open
#: (`LiteStore._create_schema`), so a file from before them gains them with nothing to migrate.
GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_entities (
    generation_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    entity_kind TEXT NOT NULL,
    aliases TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (generation_id, entity_id)
);
CREATE INDEX IF NOT EXISTS graph_entities_lookup ON graph_entities(generation_id, normalized_name);
CREATE TABLE IF NOT EXISTS graph_mentions (
    generation_id TEXT NOT NULL,
    mention_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    mention_text TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (generation_id, mention_id)
);
CREATE INDEX IF NOT EXISTS graph_mentions_chunk ON graph_mentions(generation_id, chunk_id);
CREATE INDEX IF NOT EXISTS graph_mentions_entity ON graph_mentions(generation_id, entity_id);
CREATE TABLE IF NOT EXISTS graph_relations (
    generation_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    uncertainty TEXT NOT NULL,
    pipeline_fingerprint TEXT,
    corpus_fingerprint TEXT,
    metadata TEXT NOT NULL,
    PRIMARY KEY (generation_id, relation_id)
);
CREATE TABLE IF NOT EXISTS graph_relation_evidence (
    generation_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (generation_id, relation_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS graph_relation_evidence_chunk ON graph_relation_evidence(generation_id, chunk_id);
"""
#: The `meta` row holding the readiness marker (`semantic_graph_marker`, plus `generation_id` and
#: `corpus_version`), the place a generation's `validation_summary` holds it on PostgreSQL.
MARKER_KEY = "semantic_graph"
_TABLES = ("graph_relation_evidence", "graph_relations", "graph_mentions", "graph_entities")


def clear_graph(conn: sqlite3.Connection) -> None:
    """Delete the graph and its marker, inside the caller's write transaction."""
    for table in _TABLES:
        conn.execute(f"DELETE FROM {table}")  # noqa: S608  # a constant table name
    conn.execute("DELETE FROM meta WHERE key = ?", (MARKER_KEY,))


def build_graph(chunks: Sequence[Chunk], *, tenant: str, corpus_fingerprint: str) -> SemanticGraphProjection:
    """The graph of `chunks`, the corpus whose fingerprint is `corpus_fingerprint`."""
    from recall.lite.store import generation_id_for

    return build_semantic_graph(
        chunks, tenant_id=tenant, generation_id=generation_id_for(corpus_fingerprint), corpus_fingerprint=corpus_fingerprint,
    )


def write_graph(conn: sqlite3.Connection, graph: SemanticGraphProjection, corpus_version: int) -> None:
    """Replace the file's graph with `graph`, built at `corpus_version`, inside the caller's write
    transaction."""
    clear_graph(conn)
    generation = graph.generation_id
    conn.executemany(
        "INSERT INTO graph_entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (generation, e.id, e.canonical_name, e.normalized_name, e.kind, json.dumps(list(e.aliases)),
             e.extraction_method, e.confidence, json.dumps(_thaw(e.metadata)))
            for e in graph.entities
        ],
    )
    conn.executemany(
        "INSERT INTO graph_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (generation, m.id, m.entity_id, m.chunk_id, m.mention_text, m.extraction_method, m.confidence,
             json.dumps(_thaw(m.metadata)))
            for m in graph.mentions
        ],
    )
    conn.executemany(
        "INSERT INTO graph_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (generation, r.id, r.subject_id, r.object_id, r.relation, r.extraction_method, r.confidence, r.status,
             json.dumps(list(r.uncertainty)), r.pipeline_fingerprint, r.corpus_fingerprint, json.dumps(_thaw(r.metadata)))
            for r in graph.relations
        ],
    )
    conn.executemany(
        "INSERT INTO graph_relation_evidence VALUES (?, ?, ?)",
        [(generation, r.id, chunk_id) for r in graph.relations for chunk_id in r.evidence_chunk_ids],
    )
    marker = {
        **semantic_graph_marker(graph), "generation_id": generation, "corpus_version": corpus_version,
        "corpus_fingerprint": graph.corpus_fingerprint,
    }
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (MARKER_KEY, json.dumps(marker)))


def read_marker(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The marker of a graph built for the corpus as it is now by this builder, else None.

    A marker from another corpus version (a write by a release that does not clear the graph) or
    another builder version describes rows this code would not build, so it is no graph.
    """
    values = dict(conn.execute("SELECT key, value FROM meta WHERE key IN (?, 'corpus_version')", (MARKER_KEY,)).fetchall())
    try:
        marker = json.loads(values.get(MARKER_KEY) or "null")
    except ValueError:
        return None
    if (
        not isinstance(marker, dict)
        or marker.get("ready") is not True
        or not isinstance(marker.get("generation_id"), str)
        or str(marker.get("corpus_version")) != values.get("corpus_version")
        or marker_schema_version(marker) != SEMANTIC_GRAPH_SCHEMA_VERSION
    ):
        return None
    return marker


def readiness(conn: sqlite3.Connection, tenant: str, generation_id: str) -> GraphReadiness:
    """`recall.semantic_graph.read_graph_readiness` for a lite file: the marker, no member rows."""
    marker = read_marker(conn)
    if marker is None or marker["generation_id"] != generation_id:
        return GraphReadiness(
            ready=False, tenant_id=tenant, generation_id=generation_id, graph_id=None, graph_fingerprint=None,
            entity_count=0, mention_count=0, relation_count=0, diagnostic_count=0, reason="GRAPH_NOT_READY",
        )
    return GraphReadiness(
        ready=True, tenant_id=tenant, generation_id=generation_id, graph_id=str(marker["graph_id"]),
        graph_fingerprint=str(marker["graph_fingerprint"]), entity_count=int(marker["entity_count"]),
        mention_count=int(marker["mention_count"]), relation_count=int(marker["relation_count"]),
        diagnostic_count=int(marker["diagnostic_count"]), schema_version=marker_schema_version(marker),
    )


def load_graph(conn: sqlite3.Connection, tenant: str) -> SemanticGraphProjection | None:
    """The stored graph, or None when there is none for the corpus as it is now.

    The loaded graph must reproduce the marker's fingerprint, as on PostgreSQL, so a partial or
    altered set of rows reads as no graph rather than as a smaller one.
    """
    from recall.semantic_graph._store import _load_relation_rows

    marker = read_marker(conn)
    if marker is None:
        return None
    generation = str(marker["generation_id"])
    entities = tuple(
        SemanticEntity(
            id=row[0], tenant_id=tenant, generation_id=generation, canonical_name=row[1], normalized_name=row[2],
            kind=row[3], aliases=tuple(json.loads(row[4])), extraction_method=row[5], confidence=float(row[6]),
            metadata=json.loads(row[7]),
        )
        for row in conn.execute(
            "SELECT entity_id, canonical_name, normalized_name, entity_kind, aliases, extraction_method, confidence, "
            "metadata FROM graph_entities WHERE generation_id = ? ORDER BY entity_id",
            (generation,),
        )
    )
    mentions = tuple(
        SemanticMention(
            id=row[0], tenant_id=tenant, generation_id=generation, entity_id=row[1], chunk_id=row[2],
            mention_text=row[3], extraction_method=row[4], confidence=float(row[5]), metadata=json.loads(row[6]),
        )
        for row in conn.execute(
            "SELECT mention_id, entity_id, chunk_id, mention_text, extraction_method, confidence, metadata "
            "FROM graph_mentions WHERE generation_id = ? ORDER BY mention_id",
            (generation,),
        )
    )
    evidence: dict[str, list[str]] = defaultdict(list)
    for relation_id, chunk_id in conn.execute(
        "SELECT relation_id, chunk_id FROM graph_relation_evidence WHERE generation_id = ? ORDER BY relation_id, chunk_id",
        (generation,),
    ):
        evidence[relation_id].append(chunk_id)
    relation_rows = [
        (*row[:7], json.loads(row[7]), row[8], row[9], json.loads(row[10]), evidence.get(row[0], []))
        for row in conn.execute(
            "SELECT relation_id, subject_id, object_id, relation, extraction_method, confidence, status, uncertainty, "
            "pipeline_fingerprint, corpus_fingerprint, metadata FROM graph_relations WHERE generation_id = ? "
            "ORDER BY relation_id",
            (generation,),
        )
    ]
    relations, problems = _load_relation_rows(
        relation_rows, tenant_id=tenant, generation_id=generation, schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
    )
    if problems:
        return None
    diagnostics = tuple(
        SemanticGraphDiagnostic(
            id=str(item["id"]), tenant_id=tenant, generation_id=generation, kind=item["kind"],
            reference=str(item["reference"]) if item.get("reference") is not None else None,
            message=str(item["message"]), entity_ids=tuple(str(v) for v in item.get("entity_ids", ())),
            relation_ids=tuple(str(v) for v in item.get("relation_ids", ())),
        )
        for item in marker.get("diagnostics", ())
        if isinstance(item, dict)
    )
    graph = SemanticGraphProjection(
        schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
        graph_id=str(marker["graph_id"]),
        tenant_id=tenant,
        generation_id=generation,
        pipeline_fingerprint=None,
        corpus_fingerprint=marker.get("corpus_fingerprint"),
        entities=entities,
        mentions=mentions,
        relations=relations,
        diagnostics=tuple(sorted(diagnostics, key=lambda d: d.id)),
    )
    return graph if graph.fingerprint == marker.get("graph_fingerprint") else None


def graph_of(conn: sqlite3.Connection, tenant: str) -> SemanticGraphProjection:
    """The stored graph, else one built in memory from the file's chunks and not stored.

    For a reader holding a read-only connection (the dashboard), which cannot write the graph but
    must not draw a missing one as an empty corpus.
    """
    loaded = load_graph(conn, tenant)
    if loaded is not None:
        return loaded
    from recall.lite.calibration import corpus_digest

    rows = conn.execute("SELECT id, source, text, metadata, embedding FROM chunks").fetchall()
    chunks = [Chunk(id=row[0], source=row[1], text=row[2], metadata=json.loads(row[3]) or {}) for row in rows]
    return build_graph(chunks, tenant=tenant, corpus_fingerprint=corpus_digest([(r[0], r[2], r[4]) for r in rows]))
