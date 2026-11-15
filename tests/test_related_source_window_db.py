"""The bounded `source` relation keeps the chunks the resolver would rank first.

Invariant: ``related_chunks(seed, "source", k)`` on both PostgreSQL stores returns exactly the
first ``k`` chunks of the seed's source under ``recall.related.trusted_related``'s own key,
``(abs((_ordinal(c) or 0) - (_ordinal(seed) or 0)), c.id)``, computed over the WHOLE source.
Failure mode: the store ordered by ordinal ascending before its LIMIT, so for a seed deep inside a
long memo the resolver re-ranked the memo's first chunks and never saw the seed's neighbours.

Red proof: recorded here once CI has run the unfixed query and each mutation.
"""

import hashlib

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from recall.generation_store import GenerationStore
from recall.related import _ordinal
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db
from tests.test_semantic_graph_db import graph_rows as graph_rows

_SOURCE = "file:///notes/long.md"


def _long_memo() -> tuple[Chunk, list[Chunk]]:
    chunks = [Chunk(f"s{i}", _SOURCE, f"part {i}", {"file": "long.md", "ord": i}) for i in range(21)]
    chunks.append(Chunk("other", "file:///notes/other.md", "other", {"file": "other.md", "ord": 9}))
    return chunks[10], chunks


def _odd_ords() -> tuple[Chunk, list[Chunk]]:
    meta = {"file": "odd.md"}
    seed = Chunk("m2", _SOURCE, "seed", {**meta, "ord": 2})
    chunks = [
        seed,
        Chunk("m0", _SOURCE, "m0", {**meta, "ord": 0}),
        Chunk("m1", _SOURCE, "m1", {**meta, "ord": 1}),
        Chunk("m3", _SOURCE, "m3", {**meta, "ord": 3}),
        Chunk("m5", _SOURCE, "m5", {**meta, "ord": 5}),
        Chunk("Z-str", _SOURCE, "string ord", {**meta, "ord": "2"}),
        Chunk("a-bool", _SOURCE, "boolean ord", {**meta, "ord": True}),
        Chunk("Y-float", _SOURCE, "float ord", {**meta, "ord": 2.0}),
        Chunk("X-none", _SOURCE, "no ord", dict(meta)),
        Chunk("n-neg", _SOURCE, "negative ord", {**meta, "ord": -1}),
        Chunk("W-big", _SOURCE, "huge ord", {**meta, "ord": 10**20}),
    ]
    return seed, chunks


def _resolver_top(seed: Chunk, chunks: list[Chunk], k: int) -> list[str]:
    centre = _ordinal(seed) or 0
    same = [c for c in chunks if c.source == seed.source and c.id != seed.id]
    same.sort(key=lambda c: (abs((_ordinal(c) or 0) - centre), c.id))
    return [c.id for c in same[:k]]


def _related_ids(backend, request, seed: Chunk, chunks: list[Chunk], k: int) -> list[str]:
    if backend == "legacy":
        store = request.getfixturevalue("make_store")(3)
        store.upsert(chunks, [[1.0, 0.0, 0.0] for _ in chunks])
        _seed, related = store.related_chunks(seed.id, "source", k)
        return [c.id for c in related]
    tenant, generation = request.getfixturevalue("graph_rows")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for position, chunk in enumerate(chunks):
            conn.execute(
                "INSERT INTO recall_chunks_v1 (tenant_id,generation_id,chunk_id,source_uri,"
                "object_version_id,source_sha256,chunk_ordinal,text,metadata,embedding,tsv) "
                "VALUES (%s,%s,%s,%s,'v1',%s,%s,%s,%s,%s,to_tsvector('simple',%s))",
                (
                    tenant, generation, chunk.id, chunk.source,
                    hashlib.sha256(chunk.text.encode()).hexdigest(), position, chunk.text,
                    Jsonb(chunk.metadata), Vector([1.0] + [0.0] * 63), chunk.text,
                ),
            )
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        store._fixed_generation = generation
        _seed, related = store.related_chunks(seed.id, "source", k)
    return [c.id for c in related]


@requires_db
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("backend", ["legacy", "generation"])
def test_source_window_is_the_resolvers_top_k(backend, k, request):
    """Seed s10 of s0..s20: the window is the seed's neighbours, not the memo's first chunks."""
    seed, chunks = _long_memo()
    expected = _resolver_top(seed, chunks, k)
    assert expected[:2] == ["s11", "s9"][:k]
    assert _related_ids(backend, request, seed, chunks, k) == expected


@requires_db
@pytest.mark.parametrize("k", [3, 10])
@pytest.mark.parametrize("backend", ["legacy", "generation"])
def test_source_window_ranks_odd_ords_as_the_resolver_does(backend, k, request):
    """String, boolean, float and missing ords rank 0; a negative ord ranks as itself.

    k=3 cuts inside the distance-2 tie group, so the tie order decides which chunk is kept; k=10 is
    the whole source, so the order of every chunk is checked.
    """
    seed, chunks = _odd_ords()
    assert _related_ids(backend, request, seed, chunks, k) == _resolver_top(seed, chunks, k)
