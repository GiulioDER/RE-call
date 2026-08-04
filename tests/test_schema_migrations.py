"""Versioned schema migration and serving-role regression coverage."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from recall.cli import main as cli_main
from recall.schema import (
    CONTROL_PLANE_READ_TABLES,
    CONTROL_PLANE_SEQUENCES,
    CONTROL_PLANE_WRITE_TABLES,
    GENERATION_TABLES,
    GLOBAL_MIGRATION_TARGET,
    LEDGER_TABLE,
    MIGRATION_LOCK_NAME,
    ConcurrentMigrator,
    MigrationChecksumMismatch,
    SchemaTooNew,
    SchemaIncompatible,
    SchemaTooOld,
    apply_migrations,
    load_migrations,
    schema_plan,
    schema_status,
    serving_grants,
)
from recall.store import PgVectorStore
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

DIM = 4


def _name(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:10]


@contextmanager
def _target(prefix: str = "mig_"):
    table = _name(prefix)
    try:
        yield table
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            if conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()[0]:
                conn.execute(
                    f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (table,)
                )


def test_packaged_migrations_have_committed_checksums_and_explicit_modes():
    migrations = load_migrations()
    assert [m.version for m in migrations] == [f"{n:04d}" for n in range(1, 12)]
    assert migrations[0].transactional
    assert migrations[7].transactional
    assert all(m.concurrent_index for m in (*migrations[1:7], *migrations[8:10]))
    assert migrations[10].transactional
    assert len({m.checksum for m in migrations}) == len(migrations)


def test_store_schema_check_delegates_to_read_only_checker(monkeypatch):
    """The serving API cannot accidentally reach the legacy DDL implementation."""
    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "chunks"
    store._dim = DIM
    store._closed = False
    store._pool = None
    store._conn = object()
    seen: list[tuple[object, str, int]] = []

    def fake_check(conn, *, table, dim):
        seen.append((conn, table, dim))

    monkeypatch.setattr("recall.schema.check_schema", fake_check)
    store.check_schema()
    assert seen == [(store._conn, "chunks", DIM)]


@requires_db
def test_serving_check_refuses_pending_migrations_without_creating_the_table():
    with _target() as table, PgVectorStore(TEST_DSN, dim=DIM, table=table) as store:
        with pytest.raises(SchemaTooOld, match="schema migration"):
            store.check_schema()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            assert conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None


@requires_db
def test_fresh_apply_repeated_apply_and_plan_is_read_only():
    with _target() as table:
        before = schema_plan(TEST_DSN, table=table, dim=DIM)
        assert len(before) == 7
        # `plan` must not bootstrap its own ledger.
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            existed_before = conn.execute(
                "SELECT to_regclass(%s)", (LEDGER_TABLE,)
            ).fetchone()[0]
        assert len(schema_plan(TEST_DSN, table=table, dim=DIM)) == 7
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            existed_after = conn.execute(
                "SELECT to_regclass(%s)", (LEDGER_TABLE,)
            ).fetchone()[0]
        assert existed_after == existed_before

        applied = apply_migrations(TEST_DSN, table=table, dim=DIM)
        assert [m.version for m in applied] == [f"{n:04d}" for n in range(1, 8)]
        assert apply_migrations(TEST_DSN, table=table, dim=DIM) == ()
        status = schema_status(TEST_DSN, table=table, dim=DIM)
        assert status.compatible and status.current_version == "0011"

        with PgVectorStore(TEST_DSN, dim=DIM, table=table) as store:
            store.check_schema()


@requires_db
def test_schema_cli_plan_apply_and_status_are_wired(capsys):
    with _target("cli_mig_") as table:
        base = ["--serving-dsn", TEST_DSN, "--table", table]
        cli_main([*base, "schema", "--dim", str(DIM), "plan"])
        planned = capsys.readouterr().out
        assert "would apply 0001" in planned and "would apply 0007" in planned
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            assert conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None
            assert conn.execute(
                f"SELECT count(*) FROM {LEDGER_TABLE} WHERE target_table = %s", (table,)
            ).fetchone()[0] == 0

        cli_main(
            [
                *base,
                "--migration-dsn",
                TEST_DSN,
                "schema",
                "--dim",
                str(DIM),
                "apply",
            ]
        )
        applied = capsys.readouterr().out
        assert "applied 0001" in applied and "applied 0007" in applied
        cli_main([*base, "schema", "--dim", str(DIM), "status"])
        status = capsys.readouterr().out
        assert "current: 0011" in status and "compatible: yes" in status


@requires_db
def test_v08_table_is_adopted_without_rewriting_existing_data():
    with _target("v08_") as table:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE {table} (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({DIM}),
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            )
            conn.execute(
                f"INSERT INTO {table} (id, source, text, embedding) "
                "VALUES ('old', 'memo.md', 'preserve me', '[1,0,0,0]')"
            )

        apply_migrations(TEST_DSN, table=table, dim=DIM)
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            row = conn.execute(
                f"SELECT tenant_id, id, source, text, first_indexed_at FROM {table}"
            ).fetchone()
            key = conn.execute(
                "SELECT array_length(conkey, 1) FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (table,),
            ).fetchone()
        assert row[:4] == ("default", "old", "memo.md", "preserve me")
        assert row[4] is None
        assert key == (2,)


@requires_db
def test_generation_migrations_are_recorded_once_in_the_global_ledger():
    migrations = load_migrations()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        global_versions = conn.execute(
            f"SELECT version FROM {LEDGER_TABLE} WHERE target_table = %s ORDER BY version",
            (GLOBAL_MIGRATION_TARGET,),
        ).fetchall()
        duplicated = conn.execute(
            f"SELECT target_table, version FROM {LEDGER_TABLE} "
            "WHERE version >= '0008' AND target_table != %s",
            (GLOBAL_MIGRATION_TARGET,),
        ).fetchall()

    assert global_versions == [(migration.version,) for migration in migrations[7:]]
    assert duplicated == []


@requires_db
def test_checksum_drift_and_unknown_future_versions_fail_closed():
    with _target() as table:
        apply_migrations(TEST_DSN, table=table, dim=DIM)
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                f"UPDATE {LEDGER_TABLE} SET checksum = %s "
                "WHERE target_table = %s AND version = '0001'",
                ("0" * 64, table),
            )
        with pytest.raises(MigrationChecksumMismatch):
            schema_status(TEST_DSN, table=table, dim=DIM)
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            first = load_migrations()[0]
            conn.execute(
                f"UPDATE {LEDGER_TABLE} SET checksum = %s "
                "WHERE target_table = %s AND version = '0001'",
                (first.checksum, table),
            )
            conn.execute(
                f"INSERT INTO {LEDGER_TABLE} "
                "(target_table, version, filename, checksum, state, applied_at) "
                "VALUES (%s, '9999', '9999_future.sql', %s, 'applied', now())",
                (table, "f" * 64),
            )
        with pytest.raises(SchemaTooNew):
            schema_status(TEST_DSN, table=table, dim=DIM)


@requires_db
def test_advisory_lock_refuses_a_concurrent_migrator():
    with _target() as table, psycopg.connect(TEST_DSN, autocommit=True) as blocker:
        assert blocker.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (MIGRATION_LOCK_NAME,)
        ).fetchone() == (True,)
        with pytest.raises(ConcurrentMigrator):
            apply_migrations(TEST_DSN, table=table, dim=DIM)


@requires_db
def test_interrupted_concurrent_index_phase_is_resumed():
    with _target() as table:
        apply_migrations(TEST_DSN, table=table, dim=DIM)
        migration = load_migrations()[1]
        index = f"{table}_tsv_idx"
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP INDEX {index}")
            conn.execute(
                f"UPDATE {LEDGER_TABLE} SET state = 'failed', error = 'simulated interruption' "
                "WHERE target_table = %s AND version = %s",
                (table, migration.version),
            )
        applied = apply_migrations(TEST_DSN, table=table, dim=DIM)
        assert [m.version for m in applied] == [migration.version]
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            assert conn.execute("SELECT to_regclass(%s)", (index,)).fetchone()[0]


@requires_db
def test_serving_check_detects_policy_and_index_drift_without_repairing_it():
    with _target() as table:
        apply_migrations(TEST_DSN, table=table, dim=DIM)
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                f"ALTER POLICY {table}_tenant_isolation ON {table} "
                "USING (true) WITH CHECK (true)"
            )
        with PgVectorStore(TEST_DSN, dim=DIM, table=table) as store:
            with pytest.raises(SchemaIncompatible, match="policy drift"):
                store.check_schema()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            expression = conn.execute(
                "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = %s",
                (f"{table}_tenant_isolation",),
            ).fetchone()[0]
        assert expression == "true", "serving compatibility check mutated the policy"


@requires_db
def test_serving_role_has_dml_but_cannot_run_ddl():
    with _target() as table:
        apply_migrations(TEST_DSN, table=table, dim=DIM)
        role = _name("recall_serve_")
        parts = urlsplit(TEST_DSN)
        serving_dsn = urlunsplit(
            parts._replace(netloc=f"{role}:test-password@{parts.hostname}:{parts.port or 5432}")
        )
        try:
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute(f'CREATE ROLE "{role}" LOGIN PASSWORD \'test-password\'')
                conn.execute(f'REVOKE CREATE ON SCHEMA public FROM "{role}"')
                conn.execute(f'GRANT USAGE ON SCHEMA public TO "{role}"')
                conn.execute(
                    f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {table} TO "{role}"'
                )
                conn.execute(f'GRANT SELECT ON TABLE {LEDGER_TABLE} TO "{role}"')
            with PgVectorStore(serving_dsn, dim=DIM, table=table) as store:
                store.check_schema()
                assert store.upsert(
                    [Chunk(id="one", source="memo.md", text="hello", metadata={})],
                    [[1.0, 0.0, 0.0, 0.0]],
                ) == 1
                assert store.source_content_hashes() == {"memo.md": ""}
            with pytest.raises(PermissionError, match="migration role"):
                apply_migrations(serving_dsn, table=table, dim=DIM)
            with psycopg.connect(serving_dsn, autocommit=True) as conn:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN forbidden integer")
        finally:
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute(
                    f'REVOKE ALL PRIVILEGES ON TABLE {table}, {LEDGER_TABLE} FROM "{role}"'
                )
                conn.execute(f'REVOKE USAGE ON SCHEMA public FROM "{role}"')
                conn.execute(f'DROP ROLE IF EXISTS "{role}"')


@contextmanager
def _fresh_database(prefix: str = "migdb_"):
    """A throwaway database with no migration ledger, so the global migrations really run.

    The shared test database already carries 0008-0011 in the global ledger, so they are
    skipped for any custom target table. Adopting a *populated* v0.8 install is only
    reachable on a database where they have never been applied.
    """
    name = _name(prefix)
    parts = urlsplit(TEST_DSN)
    admin = urlunsplit(parts._replace(path="/postgres"))
    scratch = urlunsplit(parts._replace(path=f"/{name}"))
    with psycopg.connect(admin, autocommit=True) as conn:
        try:
            conn.execute(f'CREATE DATABASE "{name}"')
        except psycopg.errors.InsufficientPrivilege:
            pytest.skip("RECALL_TEST_DSN role cannot CREATE DATABASE")
    try:
        yield scratch
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            # WITH (FORCE) terminates and drops atomically. pg_terminate_backend returns as
            # soon as SIGTERM is delivered, not once the backend has exited, so a plain DROP
            # could still fail with ObjectInUse from inside this finally, replacing the real
            # test failure and leaking a uniquely-named database on every retry.
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@requires_db
def test_a_populated_v08_install_can_be_adopted_by_the_generation_migrations():
    """Migration 0008 must survive a legacy table that actually contains rows.

    0008 relaxes FORCE RLS, seeds `recall_generations` and `recall_tenant_state` from the
    distinct tenants of the legacy table, then restores FORCE — all in one transaction.
    The seed into `recall_tenant_state` queues its DEFERRABLE INITIALLY DEFERRED foreign
    key trigger events, and PostgreSQL refuses `ALTER TABLE` while any are pending, so the
    restore aborts with `ObjectInUse` and the whole chain rolls back. An empty legacy table
    queues nothing, which is why every fresh database, and CI, passes.
    """
    with _fresh_database() as dsn:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({DIM}),
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            )
            conn.execute(
                "INSERT INTO chunks (id, source, text, embedding) "
                "VALUES ('old', 'memo.md', 'preserve me', '[1,0,0,0]')"
            )

        apply_migrations(dsn, table="chunks", dim=DIM)

        with psycopg.connect(dsn, autocommit=True) as conn:
            tenants = conn.execute("SELECT tenant_id FROM recall_tenant_state").fetchall()
            legacy = conn.execute(
                "SELECT generation_id, state FROM recall_generations"
            ).fetchall()
            forced = conn.execute(
                "SELECT relforcerowsecurity FROM pg_class WHERE relname = 'recall_tenant_state'"
            ).fetchone()
        # The legacy tenant is adopted as evidence, and FORCE RLS is back on afterwards.
        assert tenants == [("default",)]
        assert legacy == [("legacy-v08", "legacy_unverified")]
        assert forced == (True,)


@requires_db
def test_the_generated_serving_grants_are_sufficient_for_the_control_plane():
    """A serving role given exactly `serving_grants(...)` must be able to serve.

    No migration emits a GRANT, so the privilege list lived only as prose and drifted: it
    named ten objects and missed the four enterprise control-plane tables the serving process
    reads on every routed request, plus the one `bigserial` sequence in the schema. Following
    the documentation literally produced `permission denied` at startup readiness. This pins
    the generated list against the operations that actually run.
    """
    from recall.control_plane import ControlPlane

    role = _name("cca_grantee_")
    parts = urlsplit(TEST_DSN)
    as_role = urlunsplit(parts._replace(netloc=f"{role}:probe@{parts.hostname}:{parts.port}"))
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'probe' NOSUPERUSER NOBYPASSRLS")
        conn.execute(f"GRANT CONNECT ON DATABASE {parts.path.lstrip('/')} TO {role}")
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
    try:
        control = ControlPlane(TEST_DSN)
        control.apply_migrations()
        generation = _name("g_")
        physical = _name("cp_chunks_")
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"CREATE TABLE {physical} (id text primary key)")
            # Grant ONLY what the generated list prescribes, nothing more.
            for statement in serving_grants(role, table=physical, enterprise=True):
                conn.execute(statement)
        control.register_generation(generation, physical, "profile-x", DIM)
        control.set_generation_state(generation, "ready")
        control.set_route("probe-tenant", generation)

        # The two operations the panel proved were denied: a routed read and an outbox append.
        as_serving = ControlPlane(as_role)
        assert as_serving.route("probe-tenant") is not None
        # An INSERT here is what needed USAGE on the bigserial sequence.
        assert as_serving.append_event("probe-tenant", _name("op_"), "index", {}) > 0
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM recall_migration_events WHERE tenant_id = %s", ("probe-tenant",)
            )
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", ("probe-tenant",))
            conn.execute(f"DROP TABLE IF EXISTS {physical}")
            conn.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE IF EXISTS {role}")


def test_serving_grants_cover_every_table_the_migrator_manages():
    """A table added to the constants must not be able to fall out of the grant list."""
    statements = " ".join(serving_grants("recall_server", enterprise=True))
    for name in (LEDGER_TABLE, *GENERATION_TABLES, *CONTROL_PLANE_READ_TABLES,
                 *CONTROL_PLANE_WRITE_TABLES, *CONTROL_PLANE_SEQUENCES):
        assert name in statements, f"{name} is created by this project but never granted"
    # The sequence needs USAGE, not table DML: a table-only grant was the near miss.
    assert "GRANT USAGE ON SEQUENCE" in statements
    with pytest.raises(ValueError, match="role"):
        serving_grants("not a role")
