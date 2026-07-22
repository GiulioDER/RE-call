"""The supersession map must not go stale across processes.

This is the library's core guarantee, and it was failing OPEN. `supersession()` cached its
result per store instance, invalidated only by that instance's own writes. A long-lived MCP
server holds one store for its lifetime, so once a separate `recall index` run added a
`supersedes:` edge, the server kept serving the superseded memory with verdict `ok` — forever,
or until someone restarted it. No error, no warning; the trust layer simply returned the wrong
answer, which is the exact failure it exists to prevent.

These tests use two store instances against one table: instance B is the writer (the indexer),
instance A is the reader (the running server).
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.embeddings import HashingEmbedder
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

DIM = 64


@pytest.fixture
def shared_table():
    name = "ss_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def _chunk(cid: str, file: str, text: str, supersedes: str | None = None) -> Chunk:
    meta: dict = {"file": file, "ord": 0}
    if supersedes:
        meta["supersedes"] = supersedes
    return Chunk(cid, file, text, meta)


@requires_db
def test_a_reader_sees_an_edge_written_by_another_process(shared_table):
    """The regression. Reader caches the map, writer adds an edge, reader must not stay blind."""
    emb = HashingEmbedder(dim=DIM)
    reader = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    writer = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        reader.ensure_schema()
        writer.upsert([_chunk("v1", "limits_v1.md", "the rate limit is 100 rps")],
                      emb.embed(["the rate limit is 100 rps"]))

        assert reader.supersession_map() == {}  # populates the reader's cache

        writer.upsert(
            [_chunk("v2", "limits_v2.md", "the rate limit is now 250 rps", "limits_v1.md")],
            emb.embed(["the rate limit is now 250 rps"]),
        )

        assert reader.supersession_map() == {"limits_v1.md": "limits_v2.md"}
    finally:
        reader.close()
        writer.close()


@requires_db
def test_a_stale_memory_is_not_served_as_ok_after_another_process_supersedes_it(shared_table):
    """The same bug at the level a user feels it: the verdict on a real search."""
    emb = HashingEmbedder(dim=DIM)
    reader = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    writer = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        reader.ensure_schema()
        writer.upsert([_chunk("v1", "limits_v1.md", "the rate limit is 100 rps")],
                      emb.embed(["the rate limit is 100 rps"]))

        first = trusted_search(reader, emb, "the rate limit is 100 rps", k=5)
        assert [h.verdict for h in first.hits] == ["ok"]  # nothing supersedes it yet

        writer.upsert(
            [_chunk("v2", "limits_v2.md", "revised ceiling for client requests", "limits_v1.md")],
            emb.embed(["revised ceiling for client requests"]),
        )

        after = trusted_search(reader, emb, "the rate limit is 100 rps", k=5)
        verdicts = {h.provenance.file: h.verdict for h in after.hits}
        assert verdicts.get("limits_v1.md") == "superseded", (
            "the reader served a superseded memory as trustworthy because its cached "
            "supersession map predates another process's write"
        )
    finally:
        reader.close()
        writer.close()


@requires_db
def test_a_withdrawn_edge_is_also_noticed(shared_table):
    """Staleness in the other direction: a supersession removed elsewhere must stop applying,
    or the reader keeps demoting a memory that is current again."""
    emb = HashingEmbedder(dim=DIM)
    reader = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    writer = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        reader.ensure_schema()
        writer.upsert(
            [
                _chunk("v1", "limits_v1.md", "the rate limit is 100 rps"),
                _chunk("v2", "limits_v2.md", "revised ceiling", "limits_v1.md"),
            ],
            emb.embed(["the rate limit is 100 rps", "revised ceiling"]),
        )
        assert reader.supersession_map() == {"limits_v1.md": "limits_v2.md"}

        # the successor is deleted elsewhere — the edge no longer exists
        writer.delete_sources(["limits_v2.md"])

        assert reader.supersession_map() == {}
    finally:
        reader.close()
        writer.close()


@requires_db
def test_the_cache_still_avoids_rescanning_when_nothing_changed(shared_table):
    """The fix must not become "scan the whole table on every search".

    Freshness is established by a cheap fingerprint; the expensive DISTINCT scan should run only
    when that fingerprint moves.
    """
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.upsert([_chunk("v1", "a.md", "some memory")], emb.embed(["some memory"]))
        store.supersession_map()

        before = store._supersession_scans
        for _ in range(5):
            store.supersession_map()
        assert store._supersession_scans == before, "rescanned despite an unchanged table"

        store.upsert([_chunk("v2", "b.md", "another memory", "a.md")], emb.embed(["another"]))
        store.supersession_map()
        assert store._supersession_scans == before + 1
    finally:
        store.close()
