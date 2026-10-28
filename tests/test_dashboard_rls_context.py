"""DAT-001: scoped dashboard reads must establish the PostgreSQL tenant context.

Baseline red: NOBYPASSRLS reads return empty despite seeded rows. An unscoped
inventory must explain its missing context instead of claiming there are no tenants.
"""
from __future__ import annotations

import json
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from recall.dashboard import db as dbq
from tests.conftest import TEST_DSN, requires_db, role_is_unprivileged


@pytest.fixture
def dashboard_dsn(scoped_rows):
    """A unique SELECT-only role avoids cluster-wide fixture races under xdist."""
    role = "dashboard_probe_" + uuid.uuid4().hex
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        can_provision = conn.execute(
            "select rolsuper or rolcreaterole from pg_roles where rolname=current_user"
        ).fetchone()
        if not can_provision or not can_provision[0]:
            pytest.skip("SELECT-only dashboard role proof requires a test login with CREATEROLE")
        conn.execute(sql.SQL("create role {} login password 'dashboard-probe' nosuperuser nobypassrls").format(sql.Identifier(role)))
        conn.execute(sql.SQL("grant usage on schema public to {}").format(sql.Identifier(role)))
        conn.execute(sql.SQL("grant select on all tables in schema public to {}").format(sql.Identifier(role)))
    parts = conninfo_to_dict(TEST_DSN)
    parts.update(user=role, password="dashboard-probe")
    try:
        yield make_conninfo(**parts)
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("drop owned by {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("drop role {}").format(sql.Identifier(role)))


@pytest.fixture
def scoped_rows(make_store):
    make_store(8)
    tenants = ("dashboard-" + uuid.uuid4().hex, "dashboard-" + uuid.uuid4().hex)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for tenant in tenants:
            conn.execute("select set_config('recall.tenant_id', %s, false)", (tenant,))
            for kind in ("search_decision", "use_report", "stale_report", "generation_created"):
                conn.execute(
                    "insert into recall_audit_events (tenant_id,event_id,event_type,actor,payload) values (%s,%s,%s,'test',%s)",
                    (tenant, tenant + kind, kind, json.dumps({"query": tenant, "task": tenant})),
                )
    yield tenants
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for tenant in tenants:
            conn.execute("select set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("delete from recall_audit_events where tenant_id=%s", (tenant,))


@requires_db
@pytest.mark.parametrize("reader", ["searches", "search", "control", "use_reports", "lifecycle_events", "stale_reports"])
def test_scoped_read_binds_tenant_without_rls_bypass(dashboard_dsn, scoped_rows, reader: str) -> None:
    assert role_is_unprivileged(dashboard_dsn)
    db = dbq.DashboardDB(dashboard_dsn)
    for tenant in scoped_rows:
        if reader == "search":
            found = dbq.search(db, tenant, tenant + "search_decision")
            assert found is not None, "RLS hid the selected tenant's recorded search"
            assert found["query"] == tenant
        elif reader == "control":
            assert dbq.control(db, tenant)["summary"]["searches"] == 1
        elif reader == "stale_reports":
            rows = dbq.stale_reports(db, scoped_rows)
            assert {row["tenant"] for row in rows} == set(scoped_rows)
        else:
            rows = getattr(dbq, reader)(db, tenant)
            assert rows, "RLS hid the selected tenant's recorded rows"
            if reader == "searches":
                assert [row["query"] for row in rows] == [tenant]
            elif reader == "use_reports":
                assert [row["task"] for row in rows] == [tenant]


@requires_db
def test_unscoped_inventory_reports_missing_rls_context(dashboard_dsn, scoped_rows) -> None:
    assert role_is_unprivileged(dashboard_dsn)
    with pytest.raises(dbq.DatabaseUnavailable, match="tenant context"):
        dbq.tenants(dbq.DashboardDB(dashboard_dsn))
