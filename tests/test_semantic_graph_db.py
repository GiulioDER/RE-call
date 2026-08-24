import hashlib
import uuid

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from benchmarks.evidence_graph_eval import relation_control
from recall.semantic_graph import (
    build_semantic_graph,
    delete_semantic_graph,
    load_semantic_graph,
    write_semantic_graph,
)
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db


GRAPH_TABLES = (
    "recall_graph_entities_v1",
    "recall_graph_mentions_v1",
    "recall_graph_relations_v1",
    "recall_graph_relation_evidence_v1",
)


@pytest.fixture
def graph_rows():
    tenant = "graph-db-" + uuid.uuid4().hex[:10]
    generation = "gen-" + uuid.uuid4().hex
    source = f"s3://approved/corpora/{tenant}/memo.md"
    source_hash = hashlib.sha256(b"graph evidence").hexdigest()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "INSERT INTO recall_generations "
            "(tenant_id, generation_id, state, pipeline_identity, pipeline_fingerprint, "
            "corpus_fingerprint, manifest, manifest_digest, corpus_version, validation_summary) "
            "VALUES (%s, %s, 'ready', %s, %s, %s, %s, %s, %s, %s)",
            (
                tenant,
                generation,
                Jsonb({}),
                "p" * 64,
                "c" * 64,
                Jsonb({}),
                "m" * 64,
                "v1",
                Jsonb({}),
            ),
        )
        conn.execute(
            "INSERT INTO recall_chunks_v1 "
            "(tenant_id, generation_id, chunk_id, source_uri, object_version_id, source_sha256, "
            "chunk_ordinal, text, metadata, embedding, tsv) "
            "VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, %s, to_tsvector('simple', %s))",
            (
                tenant,
                generation,
                "chunk-1",
                source,
                "v1",
                source_hash,
                "# Graph evidence",
                Jsonb({"project": "RE-call", "service": "API"}),
                Vector([0.0] * 64),
                "graph evidence",
            ),
        )
    yield tenant, generation
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "DELETE FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation),
        )


@requires_db
def test_graph_tables_have_created_at_and_forced_rls():
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for table in GRAPH_TABLES:
            columns = conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = 'created_at'",
                (table,),
            ).fetchone()
            security = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = %s",
                (table,),
            ).fetchone()
            policy = conn.execute(
                "SELECT 1 FROM pg_policies WHERE tablename = %s",
                (table,),
            ).fetchone()
            assert columns is not None
            assert security == (True, True)
            assert policy is not None


@requires_db
def test_graph_persistence_reload_readiness_and_delete(graph_rows):
    tenant, generation = graph_rows
    graph = build_semantic_graph(
        (
            Chunk(
                "chunk-1",
                "memo.md",
                "# Graph evidence",
                {"project": "RE-call", "service": "API", "relations": [
                    {"relation": "supports", "subject": "RE-call", "object": "API"}
                ]},
            ),
        ),
        tenant_id=tenant,
        generation_id=generation,
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        with conn.transaction():
            write_semantic_graph(conn, graph)
        loaded = load_semantic_graph(conn, tenant, generation)
        assert loaded is not None
        assert loaded.fingerprint == graph.fingerprint
        assert loaded.relations[0].evidence_chunk_ids == ("chunk-1",)
        assert delete_semantic_graph(conn, tenant, generation) == len(graph.entities)
        assert load_semantic_graph(conn, tenant, generation) is None


@requires_db
def test_graph_rows_are_generation_scoped_and_relation_controls_are_detached(graph_rows):
    tenant, generation = graph_rows
    graph = build_semantic_graph(
        (Chunk("chunk-1", "memo.md", "", {"project": "A"}),),
        tenant_id=tenant,
        generation_id=generation,
    )
    control = relation_control(graph, "removed_relation_control", seed=11)
    assert control.generation_id == generation
    assert control.relations == ()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        assert conn.execute(
            "SELECT count(*) FROM recall_graph_entities_v1 "
            "WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation),
        ).fetchone()[0] == 0
