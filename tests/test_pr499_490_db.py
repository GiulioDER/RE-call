"""Real PostgreSQL identity and snapshot regressions for BUG-005 and DAT-001.

Each node failed its intended assertion against addaa974d on the remote test host, using a
disposable database. No concurrency timing or sleeps are needed: a second
connection commits precisely after the reader has fetched its first member rows.
"""

from contextlib import nullcontext
import hashlib

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from recall.generation_store import GenerationStore
from recall.semantic_graph import build_semantic_graph, delete_semantic_graph, load_semantic_graph, write_semantic_graph
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db
from tests.test_semantic_graph_db import graph_rows as graph_rows


def _source_collision_chunks():
    return [
        Chunk("seed", "file:///root-a/notes.md", "seed", {"file": "notes.md", "ord": 0}),
        Chunk("aaa-foreign", "file:///root-b/notes.md", "foreign", {"file": "notes.md", "ord": 1}),
        Chunk("zzz-neighbor", "file:///root-a/notes.md", "neighbor", {"file": "notes.md", "ord": 1}),
    ]


@requires_db
@pytest.mark.parametrize("relation", ["source", "ordinal"])
def test_legacy_related_query_limits_within_complete_source(make_store, relation):
    """BUG-005: filtering after SQL LIMIT would still lose the real neighbor."""
    store = make_store(3)
    chunks = _source_collision_chunks()
    store.upsert(chunks, [[1.0, 0.0, 0.0] for _ in chunks])
    _seed, neighbors = store.related_chunks("seed", relation, 1)
    assert [chunk.id for chunk in neighbors] == ["zzz-neighbor"]


@requires_db
@pytest.mark.parametrize("relation", ["source", "ordinal"])
def test_generation_related_query_limits_within_complete_source(graph_rows, relation):
    """BUG-005: the generation table must use its complete source_uri too."""
    tenant, generation = graph_rows
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for chunk in _source_collision_chunks():
            conn.execute(
                "INSERT INTO recall_chunks_v1 (tenant_id,generation_id,chunk_id,source_uri,object_version_id,source_sha256,chunk_ordinal,text,metadata,embedding,tsv) "
                "VALUES (%s,%s,%s,%s,'v1',%s,%s,%s,%s,%s,to_tsvector('simple',%s))",
                (tenant,generation,chunk.id,chunk.source,hashlib.sha256(chunk.text.encode()).hexdigest(),chunk.metadata['ord'],chunk.text,Jsonb(chunk.metadata),Vector([1.0]+[0.0]*63),chunk.text),
            )
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        store._fixed_generation = generation
        _seed, neighbors = store.related_chunks("seed", relation, 1)
    assert [chunk.id for chunk in neighbors] == ["zzz-neighbor"]


@requires_db
@pytest.mark.parametrize("borrowed_transaction", [False, True])
def test_graph_load_has_one_snapshot_during_erasure(graph_rows, borrowed_transaction):
    """DAT-001: READ COMMITTED transactions do not make separate reads atomic."""
    tenant, generation = graph_rows
    graph = build_semantic_graph(
        [Chunk("chunk-1", "memo.md", "", {"project":"Alpha", "service":"Beta", "relations":[{"relation":"supports", "subject":"Alpha", "object":"Beta"}]})],
        tenant_id=tenant, generation_id=generation, pipeline_fingerprint="p"*64, corpus_fingerprint="c"*64,
    )
    with psycopg.connect(TEST_DSN, autocommit=True) as reader, psycopg.connect(TEST_DSN, autocommit=True) as writer:
        for conn in (reader, writer):
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        with writer.transaction():
            write_semantic_graph(writer, graph)
        fired = False

        class Result:
            def __init__(self, cursor):
                self.cursor = cursor

            def _erase_after_fetch(self):
                nonlocal fired
                if not fired:
                    fired = True
                    with writer.transaction():
                        delete_semantic_graph(writer, tenant, generation)

            def fetchone(self):
                row = self.cursor.fetchone()
                self._erase_after_fetch()
                return row

            def fetchall(self):
                rows = self.cursor.fetchall()
                self._erase_after_fetch()
                return rows

        class InterleavedConnection:
            def transaction(self):
                return reader.transaction()

            def execute(self, sql, params):
                cursor = reader.execute(sql, params)
                return Result(cursor) if "recall_graph_entities_v1" in sql else cursor

        with reader.transaction() if borrowed_transaction else nullcontext():
            loaded = load_semantic_graph(InterleavedConnection(), tenant, generation)
        assert fired, "the writer must commit between the relevant reader boundaries"
        assert loaded is not None
        assert (len(loaded.entities), len(loaded.mentions), len(loaded.relations)) == (len(graph.entities), len(graph.mentions), len(graph.relations)), "the loader must not mix pre-erasure entities with post-erasure support"
