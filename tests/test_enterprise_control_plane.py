from __future__ import annotations

import uuid
from datetime import UTC, datetime

import psycopg
import pytest

from recall.control_plane import ControlPlane, IndexGeneration, TenantRoute
from recall.schema import apply_migrations
from recall.store import PgVectorStore
from recall.types import Chunk
from recall_mcp import stores as stores_module
from recall_mcp.stores import StoreRegistry
from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_control_plane_migration_route_outbox_and_cutover() -> None:
    suffix = uuid.uuid4().hex[:12]
    active = f"g_active_{suffix}"
    shadow = f"g_shadow_{suffix}"
    tenant = f"tenant-{suffix}"
    operation = f"operation-{suffix}"
    control = ControlPlane(TEST_DSN)
    control.apply_migrations()
    control.apply_migrations()  # checksum verified idempotency
    try:
        # Real physical tables holding the SAME corpus. Before the parity gate was wired,
        # this test cut over between two table names that did not exist, which is what made
        # the old `state == 'ready'` check look sufficient.
        active_table, shadow_table = f"chunks_active_{suffix}", f"chunks_shadow_{suffix}"
        for table in (active_table, shadow_table):
            apply_migrations(TEST_DSN, table=table, dim=8)
            with PgVectorStore(TEST_DSN, dim=8, table=table, tenant=tenant) as store:
                store.upsert(
                    [Chunk(id="c1", source="s3://x/one.md", text="body",
                           metadata={"content_hash": "h1"})],
                    [[1.0, 0, 0, 0, 0, 0, 0, 0]],
                )
        control.register_generation(active, active_table, "profile-a", 8)
        control.register_generation(shadow, shadow_table, "profile-b", 8)
        control.set_generation_state(active, "ready", chunk_count=10, source_count=2)
        control.set_route(tenant, active, shadow)
        route = control.route(tenant)
        assert route is not None
        assert route.active.generation_id == active
        assert route.shadow is not None and route.shadow.generation_id == shadow

        sequence = control.append_event(
            tenant, operation, "index", {"sources": ["secret.md"]}, active_count=1
        )
        assert [event.sequence_id for event in control.pending_events(tenant)] == [sequence]
        control.complete_event(tenant, operation, 1)
        assert control.pending_events(tenant) == []
        with psycopg.connect(TEST_DSN) as conn:
            payload = conn.execute(
                "SELECT payload FROM recall_migration_events "
                "WHERE tenant_id = %s AND operation_id = %s", (tenant, operation)
            ).fetchone()
        assert payload is not None and payload[0] is None

        control.set_generation_state(shadow, "ready", chunk_count=10, source_count=2)
        control.cutover(tenant)
        promoted = control.route(tenant)
        assert promoted is not None and promoted.active.generation_id == shadow
        assert promoted.shadow is not None and promoted.shadow.generation_id == active
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("DELETE FROM recall_migration_events WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([active, shadow],),
            )
            for table in (f"chunks_active_{suffix}", f"chunks_shadow_{suffix}"):
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )


def test_set_route_establishes_tenant_before_forced_rls_write() -> None:
    class _Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class _Cursor:
        def __init__(self, rows=()):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _Connection:
        def __init__(self):
            self.sql: list[str] = []
            self.transaction = _Transaction

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql, _params=None):
            self.sql.append(sql)
            if sql.startswith("SELECT generation_id, state"):
                return _Cursor((("active", "ready"),))
            return _Cursor()

    connection = _Connection()
    control = ControlPlane("postgresql:///unused")
    control._connect = lambda: connection  # type: ignore[method-assign]

    control.set_route("acme", "active")

    assert connection.sql[0].startswith("SELECT set_config('recall.tenant_id'")
    assert any(sql.startswith("INSERT INTO recall_tenant_routes") for sql in connection.sql)


def test_enterprise_registry_opens_with_read_only_catalog_validation(monkeypatch) -> None:
    generation = IndexGeneration(
        "active", "chunks_active", "profile-a", 8, "ready", 0, 0,
        datetime.now(UTC), datetime.now(UTC),
    )
    route = TenantRoute("acme", generation, None, datetime.now(UTC))

    class _Control:
        def route(self, _tenant):
            return route

        def watch_routes(self, _callback, _stop):
            return None

    class _Store:
        ensure_called = False

        def __init__(self, *_args, **_kwargs):
            pass

        def ensure_schema(self):
            self.ensure_called = True

        def readiness_facts(self):
            return {"dimension": 8, "rls_enabled": True, "indexes_valid": True}

        def check_rls_effective(self):
            return True

        def close(self):
            return None

    monkeypatch.setattr(stores_module, "PgVectorStore", _Store)
    registry = StoreRegistry(
        dsn="postgresql:///unused", dim=8, allowed_tenants=frozenset({"acme"}),
        pool_size=1, statement_timeout_ms=1000, control_plane=_Control(),
        embedding_profile="profile-a",
    )
    try:
        store = registry.get("acme")
        assert store.ensure_called is False
    finally:
        registry.close()


@requires_db
def test_cutover_refuses_a_shadow_that_does_not_match_the_active_generation() -> None:
    """`state = 'ready'` is an operator assertion, so something must check the tables.

    `mark-ready` stores its `--chunks`/`--sources` argparse ints verbatim and compares them to
    nothing, so an EMPTY generation could be marked ready and cut over, pointing every read
    for the tenant at an empty index. `validate_generation_parity` existed for exactly this
    and had no caller anywhere in the package.
    """
    suffix = uuid.uuid4().hex[:12]
    active, shadow = f"g_active_{suffix}", f"g_shadow_{suffix}"
    active_table, shadow_table = f"chunks_a_{suffix}", f"chunks_s_{suffix}"
    tenant = f"tenant-{suffix}"
    control = ControlPlane(TEST_DSN)
    control.apply_migrations()
    try:
        for table in (active_table, shadow_table):
            apply_migrations(TEST_DSN, table=table, dim=8)
        # The active generation holds a corpus; the shadow is EMPTY but claims otherwise.
        with PgVectorStore(TEST_DSN, dim=8, table=active_table, tenant=tenant) as store:
            store.upsert(
                [Chunk(id="c1", source="s3://x/one.md", text="body",
                       metadata={"content_hash": "h1"})],
                [[1.0, 0, 0, 0, 0, 0, 0, 0]],
            )
        control.register_generation(active, active_table, "profile-a", 8)
        control.register_generation(shadow, shadow_table, "profile-a", 8)
        control.set_generation_state(active, "ready", chunk_count=1, source_count=1)
        # Fabricated counts, exactly as the CLI would accept them.
        control.set_generation_state(shadow, "ready", chunk_count=1_000_000, source_count=120_000)
        control.set_route(tenant, active, shadow)

        with pytest.raises(RuntimeError, match="cutover refused"):
            control.cutover(tenant)
        # The route is untouched: the tenant still serves the generation that has the data.
        route = control.route(tenant)
        assert route is not None and route.active.generation_id == active

        # The escape hatch covers a corpus that legitimately CHANGED, not one that is
        # ABSENT: give the shadow a divergent row, then it promotes. An empty shadow stays
        # refused even under the flag (see the assertion below).
        with PgVectorStore(TEST_DSN, dim=8, table=shadow_table, tenant=tenant) as store:
            store.upsert(
                [Chunk(id="c2", source="s3://x/two.md", text="different",
                       metadata={"content_hash": "h2"})],
                [[0, 1.0, 0, 0, 0, 0, 0, 0]],
            )
        control.cutover(tenant, allow_divergent_corpus=True)
        promoted = control.route(tenant)
        assert promoted is not None and promoted.active.generation_id == shadow
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([active, shadow],),
            )
            for table in (active_table, shadow_table):
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )


@requires_db
def test_cutover_refuses_an_empty_shadow_even_when_the_active_is_also_empty() -> None:
    """Set-difference parity passes vacuously on two empty generations: 0 == 0.

    That is the exact case the gate exists to stop, so it cannot be the case it waves through.
    The declared `mark-ready` counts are never consulted by parity, so without this an empty
    shadow with a fabricated chunk_count of 999999 was promoted whenever the active happened to
    be empty too, and the gate reported success having verified nothing.
    """
    suffix = uuid.uuid4().hex[:12]
    active, shadow = f"g_active_{suffix}", f"g_shadow_{suffix}"
    active_table, shadow_table = f"chunks_ea_{suffix}", f"chunks_es_{suffix}"
    tenant = f"tenant-{suffix}"
    control = ControlPlane(TEST_DSN)
    control.apply_migrations()
    try:
        for table in (active_table, shadow_table):
            apply_migrations(TEST_DSN, table=table, dim=8)  # both remain EMPTY
        control.register_generation(active, active_table, "profile-a", 8)
        control.register_generation(shadow, shadow_table, "profile-a", 8)
        control.set_generation_state(active, "ready", chunk_count=0, source_count=0)
        control.set_generation_state(shadow, "ready", chunk_count=999_999, source_count=999_999)
        control.set_route(tenant, active, shadow)

        with pytest.raises(RuntimeError, match="holds no rows"):
            control.cutover(tenant)
        route = control.route(tenant)
        assert route is not None and route.active.generation_id == active
        # The divergence escape hatch must NOT override emptiness: it is documented for a
        # corpus that changed, and an empty index is never a corpus worth serving.
        with pytest.raises(RuntimeError, match="holds no rows"):
            control.cutover(tenant, allow_divergent_corpus=True)
        still = control.route(tenant)
        assert still is not None and still.active.generation_id == active
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([active, shadow],),
            )
            for table in (active_table, shadow_table):
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )


@requires_db
def test_cutover_refuses_when_the_route_moved_while_parity_was_running(monkeypatch) -> None:
    """Parity now runs OUTSIDE the route transaction, so the swap must re-check the route.

    Moving the comparison out of the locked region is what stops the tenant's route row being
    pinned across two unbounded full-table reads. The cost is that what parity verified may no
    longer be what gets swapped, so the swap re-reads under FOR UPDATE and refuses on a
    mismatch. This drives that branch by re-routing the tenant from inside the parity call.
    """
    suffix = uuid.uuid4().hex[:12]
    first, second, third = (f"g_{n}_{suffix}" for n in ("one", "two", "three"))
    tables = {name: f"chunks_{name}" for name in (first, second, third)}
    tenant = f"tenant-{suffix}"
    control = ControlPlane(TEST_DSN)
    control.apply_migrations()
    try:
        for generation, table in tables.items():
            apply_migrations(TEST_DSN, table=table, dim=8)
            with PgVectorStore(TEST_DSN, dim=8, table=table, tenant=tenant) as store:
                store.upsert(
                    [Chunk(id="c1", source="s3://x/one.md", text="body",
                           metadata={"content_hash": "h1"})],
                    [[1.0, 0, 0, 0, 0, 0, 0, 0]],
                )
            control.register_generation(generation, table, "profile-a", 8)
            control.set_generation_state(generation, "ready", chunk_count=1, source_count=1)
        control.set_route(tenant, first, second)

        original = ControlPlane._require_parity

        def racing(self, conn, tenant_id, active, shadow):
            original(self, conn, tenant_id, active, shadow)
            # A concurrent operator re-points the tenant after parity passed.
            self.set_route(tenant_id, first, third)

        monkeypatch.setattr(ControlPlane, "_require_parity", racing)
        with pytest.raises(RuntimeError, match="route changed while parity"):
            control.cutover(tenant)

        # The swap did not happen: the route is exactly what the racer left it as.
        route = control.route(tenant)
        assert route is not None
        assert route.active.generation_id == first
        assert route.shadow is not None and route.shadow.generation_id == third
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                (list(tables),),
            )
            for table in tables.values():
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )
