from __future__ import annotations

import uuid
from datetime import UTC, datetime

import psycopg

from recall.control_plane import ControlPlane, IndexGeneration, TenantRoute
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
        control.register_generation(active, f"chunks_active_{suffix}", "profile-a", 8)
        control.register_generation(shadow, f"chunks_shadow_{suffix}", "profile-b", 8)
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
