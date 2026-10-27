"""Persisting and loading a semantic graph projection, and reading its readiness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from psycopg.types.json import Jsonb

from recall.semantic_graph._model import (
    SEMANTIC_GRAPH_SCHEMA_VERSION,
    GraphReadiness,
    SemanticEntity,
    SemanticGraphDiagnostic,
    SemanticGraphProjection,
    SemanticMention,
    SemanticRelation,
    _identity,
    _relation_temporal_fields,
    _thaw,
)


def read_graph_readiness(conn: Any, tenant_id: str, generation_id: str) -> GraphReadiness:
    """Read the compact generation marker without loading graph member rows.

    The marker is written in the same transaction as the immutable graph. The serving path
    still validates the loaded projection against this fingerprint before caching it, so this
    fast check removes the repeated payload transfer without making a partial graph acceptable.
    """
    row = conn.execute(
        "SELECT validation_summary FROM recall_generations "
        "WHERE tenant_id = %s AND generation_id = %s",
        (tenant_id, generation_id),
    ).fetchone()
    summary = row[0] if row and isinstance(row[0], Mapping) else None
    marker = summary.get("semantic_graph") if isinstance(summary, Mapping) else None
    required = ("graph_id", "graph_fingerprint")
    counts = ("entity_count", "mention_count", "relation_count", "diagnostic_count")
    if (
        not isinstance(marker, Mapping)
        or marker.get("ready") is not True
        or any(not isinstance(marker.get(field), str) or not marker.get(field) for field in required)
        or any(
            (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for field in counts
            for value in (marker.get(field),)
        )
    ):
        return GraphReadiness(
            ready=False,
            tenant_id=tenant_id,
            generation_id=generation_id,
            graph_id=None,
            graph_fingerprint=None,
            entity_count=0,
            mention_count=0,
            relation_count=0,
            diagnostic_count=0,
            reason="GRAPH_NOT_READY",
        )
    return GraphReadiness(
        ready=True,
        tenant_id=tenant_id,
        generation_id=generation_id,
        graph_id=str(marker["graph_id"]),
        graph_fingerprint=str(marker["graph_fingerprint"]),
        entity_count=int(marker["entity_count"]),
        mention_count=int(marker["mention_count"]),
        relation_count=int(marker["relation_count"]),
        diagnostic_count=int(marker["diagnostic_count"]),
    )


def write_semantic_graph(conn: Any, graph: SemanticGraphProjection) -> None:
    """Replace one generation's graph rows inside the caller's transaction."""
    # The delete below is scoped by the graph's own tenant and generation, so every member
    # must carry that same identity: a foreign member would otherwise be written into a
    # scope the delete never clears, and survive the next rebuild.
    for member_kind, members in (
        ("entity", graph.entities),
        ("mention", graph.mentions),
        ("relation", graph.relations),
    ):
        for member in members:
            if member.tenant_id != graph.tenant_id:
                raise ValueError(f"{member_kind} {member.id} tenant_id does not match graph")
            if member.generation_id != graph.generation_id:
                raise ValueError(f"{member_kind} {member.id} generation_id does not match graph")
    conn.execute(
        "DELETE FROM recall_graph_entities_v1 WHERE tenant_id = %s AND generation_id = %s",
        (graph.tenant_id, graph.generation_id),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO recall_graph_entities_v1 "
            "(tenant_id, generation_id, entity_id, canonical_name, normalized_name, entity_kind, "
            "aliases, extraction_method, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    graph.tenant_id,
                    graph.generation_id,
                    entity.id,
                    entity.canonical_name,
                    entity.normalized_name,
                    entity.kind,
                    Jsonb(list(entity.aliases)),
                    entity.extraction_method,
                    entity.confidence,
                    Jsonb(_thaw(entity.metadata)),
                )
                for entity in graph.entities
            ],
        )

        cur.executemany(
            "INSERT INTO recall_graph_mentions_v1 "
            "(tenant_id, generation_id, mention_id, entity_id, chunk_id, mention_text, "
            "extraction_method, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    graph.tenant_id,
                    graph.generation_id,
                    mention.id,
                    mention.entity_id,
                    mention.chunk_id,
                    mention.mention_text,
                    mention.extraction_method,
                    mention.confidence,
                    Jsonb(_thaw(mention.metadata)),
                )
                for mention in graph.mentions
            ],
        )
        cur.executemany(
            "INSERT INTO recall_graph_relations_v1 "
            "(tenant_id, generation_id, relation_id, subject_id, object_id, relation, "
            "extraction_method, confidence, status, uncertainty, pipeline_fingerprint, "
            "corpus_fingerprint, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    graph.tenant_id,
                    graph.generation_id,
                    relation.id,
                    relation.subject_id,
                    relation.object_id,
                    relation.relation,
                    relation.extraction_method,
                    relation.confidence,
                    relation.status,
                    Jsonb(list(relation.uncertainty)),
                    relation.pipeline_fingerprint,
                    relation.corpus_fingerprint,
                    Jsonb(_thaw(relation.metadata)),
                )
                for relation in graph.relations
            ],
        )
        cur.executemany(
            "INSERT INTO recall_graph_relation_evidence_v1 "
            "(tenant_id, generation_id, relation_id, chunk_id) VALUES (%s, %s, %s, %s)",
            [
                (graph.tenant_id, graph.generation_id, relation.id, chunk_id)
                for relation in graph.relations
                for chunk_id in relation.evidence_chunk_ids
            ],
        )


def delete_semantic_graph(conn: Any, tenant_id: str, generation_id: str) -> int:
    """Delete all graph rows for one tenant and generation inside the caller's transaction.

    The generation's ``semantic_graph`` readiness marker goes too. `read_graph_readiness`
    answers from that marker alone, so leaving it reported a deleted graph as ready with its old
    counts. The rest of ``validation_summary`` is kept.
    """
    result = conn.execute(
        "DELETE FROM recall_graph_entities_v1 WHERE tenant_id = %s AND generation_id = %s",
        (tenant_id, generation_id),
    )
    conn.execute(
        "UPDATE recall_generations SET validation_summary = validation_summary - 'semantic_graph' "
        "WHERE tenant_id = %s AND generation_id = %s AND validation_summary ? 'semantic_graph'",
        (tenant_id, generation_id),
    )
    return int(result.rowcount)


def _load_relation_rows(
    relation_rows: Sequence[Any],
    *,
    tenant_id: str,
    generation_id: str,
) -> tuple[tuple[SemanticRelation, ...], tuple[SemanticGraphDiagnostic, ...]]:
    """Decode persisted relation rows and retain diagnostics for invalid rows."""
    loaded_relations: list[SemanticRelation] = []
    load_diagnostics: list[SemanticGraphDiagnostic] = []
    for row in relation_rows:
        evidence_chunk_ids = tuple(str(item) for item in (row[11] or ()) if item is not None)
        if not evidence_chunk_ids:
            # A relation whose evidence rows are gone cannot satisfy the dataclass invariant.
            relation_id = str(row[0])
            load_diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "missing_evidence",
                            "reference": relation_id,
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="missing_evidence",
                    reference=relation_id,
                    message=f"relation {relation_id} has no surviving evidence rows",
                    relation_ids=(relation_id,),
                )
            )
            continue
        relation_metadata = row[10] or {}
        try:
            effective_at, valid_from, valid_until = _relation_temporal_fields(relation_metadata)
        except ValueError as exc:
            relation_id = str(row[0])
            load_diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "invalid_relation",
                            "reference": relation_id,
                            "value": repr(relation_metadata),
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="invalid_relation",
                    reference=relation_id,
                    message=str(exc),
                )
            )
            continue
        loaded_relations.append(
            SemanticRelation(
                id=str(row[0]),
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=str(row[1]),
                object_id=str(row[2]),
                relation=row[3],
                evidence_chunk_ids=evidence_chunk_ids,
                extraction_method=row[4],
                confidence=float(row[5]),
                status=row[6],
                uncertainty=tuple(row[7] or ()),
                pipeline_fingerprint=str(row[8]) if row[8] else None,
                corpus_fingerprint=str(row[9]) if row[9] else None,
                metadata=relation_metadata,
                effective_at=effective_at,
                valid_from=valid_from,
                valid_until=valid_until,
            )
        )
    return tuple(loaded_relations), tuple(load_diagnostics)


def load_semantic_graph(conn: Any, tenant_id: str, generation_id: str) -> SemanticGraphProjection | None:
    """Load a generation graph, returning ``None`` when no graph has been built."""
    # One transaction, so the four reads observe one snapshot: a concurrent forget() commits
    # between its statements, and four autocommit SELECTs would each see a different state.
    with conn.transaction():
        generation_row = conn.execute(
            "SELECT pipeline_fingerprint, corpus_fingerprint, validation_summary "
            "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
            (tenant_id, generation_id),
        ).fetchone()
        entity_rows = conn.execute(
            "SELECT entity_id, canonical_name, normalized_name, entity_kind, aliases, "
            "extraction_method, confidence, metadata FROM recall_graph_entities_v1 "
            "WHERE tenant_id = %s AND generation_id = %s ORDER BY entity_id",
            (tenant_id, generation_id),
        ).fetchall()
        mention_rows = conn.execute(
            "SELECT mention_id, entity_id, chunk_id, mention_text, extraction_method, "
            "confidence, metadata FROM recall_graph_mentions_v1 "
            "WHERE tenant_id = %s AND generation_id = %s ORDER BY mention_id",
            (tenant_id, generation_id),
        ).fetchall()
        relation_rows = conn.execute(
            "SELECT r.relation_id, r.subject_id, r.object_id, r.relation, r.extraction_method, "
            "r.confidence, r.status, r.uncertainty, r.pipeline_fingerprint, r.corpus_fingerprint, "
            "r.metadata, array_agg(e.chunk_id ORDER BY e.chunk_id) "
            "FROM recall_graph_relations_v1 r LEFT JOIN recall_graph_relation_evidence_v1 e "
            "ON e.tenant_id = r.tenant_id AND e.generation_id = r.generation_id "
            "AND e.relation_id = r.relation_id WHERE r.tenant_id = %s AND r.generation_id = %s "
            "GROUP BY r.relation_id, r.subject_id, r.object_id, r.relation, "
            "r.extraction_method, r.confidence, r.status, r.uncertainty, "
            "r.pipeline_fingerprint, r.corpus_fingerprint, r.metadata "
            "ORDER BY r.relation_id",
            (tenant_id, generation_id),
        ).fetchall()
    marker = generation_row[2].get("semantic_graph") if generation_row and isinstance(generation_row[2], dict) else None
    if not entity_rows and not mention_rows and not relation_rows and not isinstance(marker, dict):
        return None
    entities = tuple(
        SemanticEntity(
            id=str(row[0]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            canonical_name=str(row[1]),
            normalized_name=str(row[2]),
            kind=row[3],
            aliases=tuple(row[4] or ()),
            extraction_method=row[5],
            confidence=float(row[6]),
            metadata=row[7] or {},
        )
        for row in entity_rows
    )
    mentions = tuple(
        SemanticMention(
            id=str(row[0]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            entity_id=str(row[1]),
            chunk_id=str(row[2]),
            mention_text=str(row[3]),
            extraction_method=row[4],
            confidence=float(row[5]),
            metadata=row[6] or {},
        )
        for row in mention_rows
    )
    relations, load_diagnostics = _load_relation_rows(
        relation_rows,
        tenant_id=tenant_id,
        generation_id=generation_id,
    )
    diagnostics = tuple(
        SemanticGraphDiagnostic(
            id=str(item["id"]),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind=item["kind"],
            reference=str(item["reference"]) if item.get("reference") is not None else None,
            message=str(item["message"]),
            entity_ids=tuple(str(value) for value in item.get("entity_ids", ())),
            relation_ids=tuple(str(value) for value in item.get("relation_ids", ())),
        )
        for item in (marker.get("diagnostics", ()) if isinstance(marker, dict) else ())
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("kind"), str)
        and isinstance(item.get("message"), str)
    )
    graph_id = (
        str(marker["graph_id"])
        if isinstance(marker, dict) and marker.get("graph_id")
        else _identity(
            "graph",
            {
                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "generation_id": generation_id,
                "pipeline_fingerprint": generation_row[0] if generation_row else None,
                "corpus_fingerprint": generation_row[1] if generation_row else None,
                "entities": [entity.id for entity in entities],
                "mentions": [mention.id for mention in mentions],
                "relations": [relation.id for relation in relations],
                "diagnostics": (),
            },
        )
    )
    return SemanticGraphProjection(
        schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
        graph_id=graph_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=(
            str(generation_row[0])
            if generation_row and generation_row[0]
            else (relations[0].pipeline_fingerprint if relations else None)
        ),
        corpus_fingerprint=(
            str(generation_row[1])
            if generation_row and generation_row[1]
            else (relations[0].corpus_fingerprint if relations else None)
        ),
        entities=entities,
        mentions=mentions,
        relations=relations,
        diagnostics=tuple(
            sorted((*diagnostics, *load_diagnostics), key=lambda diagnostic: diagnostic.id)
        ),
    )
