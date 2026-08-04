"""`PgVectorStore` over a process-wide pool, and `for_tenant` as a view rather than a new pool.

`test_shared_pool_tenant_isolation.py` proves the mechanism. This file proves the store actually
uses it: that real reads and writes work through a per-operation transaction, that a view sees its
own tenant's rows and nothing else, and that N tenants cost one pool rather than N.
"""

from __future__ import annotations

import uuid

import pytest

from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall.schema import apply_migrations
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

pytestmark = requires_db


@pytest.fixture
def shared_table():
    name = "shared_" + uuid.uuid4().hex[:8]
    apply_migrations(TEST_DSN, table=name, dim=8)
    yield name
    import psycopg

    from recall.schema import LEDGER_TABLE

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (name,))


@pytest.fixture
def pool():
    p = SharedPool(TEST_DSN, min_size=2, max_size=6)
    yield p
    p.close()


def _vec(x: float) -> list[float]:
    return [x] + [0.0] * 7


class TestReadWriteThroughTheSharedPool:
    def test_upsert_and_read_back(self, pool, shared_table) -> None:
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        store.upsert([Chunk("a", "notes.md", "the rate limit is 100 rps")], [_vec(1.0)])
        hits = store.query_dense(_vec(1.0), k=5)
        assert [h.chunk.id for h in hits] == ["a"]

    def test_count_and_iter_chunks_stream_correctly(self, pool, shared_table) -> None:
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        store.upsert(
            [Chunk(f"c{i}", "notes.md", f"memo {i}") for i in range(5)],
            [_vec(1.0) for _ in range(5)],
        )
        assert sorted(c.id for c in store.iter_chunks(batch_size=2)) == [
            "c0", "c1", "c2", "c3", "c4"
        ]


class TestForTenantIsAViewNotAPool:
    def test_view_shares_the_same_pool_object(self, pool, shared_table) -> None:
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        view = store.for_tenant("globex")
        assert view._shared is store._shared is pool
        assert view.tenant == "globex"
        assert store.tenant == "acme", "the original must not be mutated"

    def test_many_tenants_cost_one_pool(self, pool, shared_table) -> None:
        """The whole reason for section A: 1,000 tenants must not mean 1,000 pools."""
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="t0",
                              shared_pool=pool)
        views = [store.for_tenant(f"t{i}") for i in range(1, 50)]
        for v in views:
            v.upsert([Chunk("x", "n.md", "hello")], [_vec(1.0)])
        assert pool.stats()["max_size"] == 6
        assert pool.stats()["pool_size"] <= 6
        assert all(v._shared is pool for v in views)

    def test_a_view_reads_only_its_own_tenants_rows(self, pool, shared_table) -> None:
        acme = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                             shared_pool=pool)
        globex = acme.for_tenant("globex")
        acme.upsert([Chunk("a1", "acme.md", "acme secret")], [_vec(1.0)])
        globex.upsert([Chunk("g1", "globex.md", "globex secret")], [_vec(1.0)])

        assert [h.chunk.id for h in acme.query_dense(_vec(1.0), k=10)] == ["a1"]
        assert [h.chunk.id for h in globex.query_dense(_vec(1.0), k=10)] == ["g1"]

    def test_closing_a_view_does_not_close_the_pool(self, pool, shared_table) -> None:
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        view = store.for_tenant("globex")
        view.close()
        # The owner is still usable: a view closing must not take the process's connections.
        store.upsert([Chunk("a", "n.md", "still working")], [_vec(1.0)])
        assert store.query_dense(_vec(1.0), k=1)

    def test_view_does_not_inherit_the_supersession_cache(self, pool, shared_table) -> None:
        """A cache keyed by nothing tenant-specific would serve one tenant's edges to another."""
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        store._supersession_cache = ("fingerprint", {"a.md": "b.md"}, frozenset(), {})
        view = store.for_tenant("globex")
        assert view._supersession_cache is None


class TestModeErrors:
    def test_for_tenant_refused_outside_shared_mode(self, shared_table) -> None:
        """In the other modes the tenant is connection session state, so a view would lie."""
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme")
        try:
            with pytest.raises(RuntimeError, match="shared-pool mode"):
                store.for_tenant("globex")
        finally:
            store.close()

    def test_both_pool_modes_at_once_is_refused(self, pool, shared_table) -> None:
        with pytest.raises(ValueError, match="not both"):
            PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                          pool_size=3, shared_pool=pool)

    def test_empty_tenant_view_is_refused(self, pool, shared_table) -> None:
        store = PgVectorStore(TEST_DSN, dim=8, table=shared_table, tenant="acme",
                              shared_pool=pool)
        for bad in ("", "  "):
            with pytest.raises(ValueError):
                store.for_tenant(bad)
