from __future__ import annotations

import uuid

import psycopg

from recall.control_plane import ControlPlane
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
