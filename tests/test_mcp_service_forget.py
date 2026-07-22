import pytest

from recall.types import Chunk
from recall_mcp.service import forget_memory

from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_forget_removes_all_chunks_for_a_source(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a1", "f.md", "one"), Chunk("a2", "f.md", "two"), Chunk("b1", "g.md", "keep")],
        [[1.0, 0.0, 0.0]] * 3,
    )
    result = forget_memory(store, ["f.md"])
    assert result.chunks_removed == 2
    assert result.sources_removed == ["f.md"]
    assert result.sources_not_found == []
    assert store.count() == 1


@requires_db
def test_forget_reports_a_typo_source_as_not_found_not_as_success(make_store):
    """The failure mode this guards: a typo'd source silently reporting 0 removed as if it
    succeeded. It must come back visibly distinguishable from a real deletion."""
    store = make_store(3)
    store.upsert([Chunk("a1", "f.md", "one")], [[1.0, 0.0, 0.0]])
    result = forget_memory(store, ["f.mdd"])  # typo
    assert result.chunks_removed == 0
    assert result.sources_removed == []
    assert result.sources_not_found == ["f.mdd"]
    assert store.count() == 1  # untouched


@requires_db
def test_forget_partitions_found_and_not_found_in_one_call(make_store):
    store = make_store(3)
    store.upsert(
        [Chunk("a1", "f.md", "one"), Chunk("b1", "g.md", "two")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    result = forget_memory(store, ["f.md", "missing.md"])
    assert result.chunks_removed == 1
    assert result.sources_removed == ["f.md"]
    assert result.sources_not_found == ["missing.md"]
    assert store.count() == 1


@requires_db
def test_forget_rejects_an_empty_source_list(make_store):
    store = make_store(3)
    with pytest.raises(ValueError, match="non-empty"):
        forget_memory(store, [])


@requires_db
def test_forget_is_tenant_scoped(tenant_table):
    """Pins `delete_sources`'s tenant filter through the service layer: a forget issued as
    tenant B must not remove tenant A's chunks, even for the identical source name."""
    from recall.store import PgVectorStore

    a = PgVectorStore(TEST_DSN, dim=3, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=3, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        a.upsert([Chunk("x", "shared.md", "acme note")], [[1.0, 0.0, 0.0]])
        b.upsert([Chunk("y", "shared.md", "globex note")], [[0.0, 1.0, 0.0]])

        result = b.count(), a.count()
        assert result == (1, 1)  # both wrote successfully before the forget

        outcome = forget_memory(b, ["shared.md"])
        assert outcome.chunks_removed == 1
        assert outcome.sources_removed == ["shared.md"]
        assert b.count() == 0
        assert a.count() == 1  # acme's row survives a forget issued as globex
    finally:
        a.close()
        b.close()


@pytest.fixture
def tenant_table():
    import uuid

    import psycopg

    name = "tn_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
