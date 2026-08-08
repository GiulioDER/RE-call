"""Versioned schema migration and serving-role regression coverage."""

from __future__ import annotations

import threading
import time
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
    MIGRATION_LOCK_WAIT_SECONDS,
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
    assert [m.version for m in migrations] == [f"{n:04d}" for n in range(1, 14)]
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
        assert status.compatible and status.current_version == "0013"

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
        assert "current: 0013" in status and "compatible: yes" in status


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
            # Adoption assigns the legacy row to tenant `default` AND turns on forced row level
            # security, so this verification read needs the GUC. Without it the policy matches
            # nothing, `fetchone()` returns None, and the assertion fails with a TypeError that
            # says nothing about tenancy. Invisible while the suite ran as a superuser, for whom
            # the policy is inert.
            conn.execute("SELECT set_config('recall.tenant_id', 'default', false)")
            row = conn.execute(
                f"SELECT tenant_id, id, source, text, first_indexed_at FROM {table}"
            ).fetchone()
            key = conn.execute(
                "SELECT array_length(conkey, 1) FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (table,),
            ).fetchone()
        assert row is not None, "the adopted row is not visible to this connection"
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
    """Take the migration lock for real, then hand it back explicitly rather than by closing.

    A session advisory lock is released when the *server-side* backend exits, and closing the
    connection only schedules that. Leaving the release to `with` therefore handed the rest of
    the suite a lock that was, for a moment, still held. Measured against this container: on an
    idle database another session never once observed the stale lock in 400 rounds, but with 16
    connections churning it observed it in 101 of 400, needing 2 to 4 further attempts before the
    lock cleared. A full run is the loaded case, so a test scheduled next by pytest-randomly could
    lose that race and error with `ConcurrentMigrator` from `apply_migrations`, which is what was
    seen about once in four full runs.

    `pg_advisory_unlock` returning true is the assertion that matters: it is true only when this
    session did hold the lock and has now released it, so it fails if the lock was never taken and
    it cannot pass while the release is still pending. Do not replace it with a check from a second
    connection that the lock is free. That check passes on an idle database whether or not the
    release is explicit, which makes it a guard that cannot fail on the machine you run it on.
    """
    lock_sql = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))"
    unlock_sql = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"
    with _target() as table, psycopg.connect(TEST_DSN, autocommit=True) as blocker:
        assert blocker.execute(lock_sql, (MIGRATION_LOCK_NAME,)).fetchone() == (True,)
        with pytest.raises(ConcurrentMigrator):
            apply_migrations(TEST_DSN, table=table, dim=DIM)
        assert blocker.execute(unlock_sql, (MIGRATION_LOCK_NAME,)).fetchone() == (True,)


@requires_db
def test_migration_lock_waits_out_a_lock_that_is_merely_being_reaped():
    """`ConcurrentMigrator` reports a migrator that is running, not one that has just stopped.

    A session advisory lock outlives the exit of the process holding it, until the server reaps
    the backend. A migrator restarted straight after a kill, a Ctrl-C or a container restart
    therefore raced its own predecessor's lock and was refused for a condition that had already
    passed. The two cases are told apart by how long the lock is held: a real migrator holds for
    seconds to minutes, a lock awaiting reaping clears in milliseconds.

    The lock is genuinely held when `apply_migrations` is called and is only released part way
    through, so returning at all is the whole assertion: refusing on the first failed attempt
    raises here instead. Do not add a check that the call took at least `hold_for`. It cannot
    acquire the lock before the release, so that check passes however the code behaves.

    The companion is `test_advisory_lock_refuses_a_concurrent_migrator`, where the lock is never
    released and the refusal must still arrive. Together they pin the wait as bounded.
    """
    lock_sql = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))"
    unlock_sql = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"
    hold_for = 0.3
    assert hold_for < MIGRATION_LOCK_WAIT_SECONDS, "the lock must clear inside the migrator's wait"
    with _target() as table, psycopg.connect(TEST_DSN, autocommit=True) as blocker:
        assert blocker.execute(lock_sql, (MIGRATION_LOCK_NAME,)).fetchone() == (True,)

        def release_shortly() -> None:
            time.sleep(hold_for)
            blocker.execute(unlock_sql, (MIGRATION_LOCK_NAME,))

        releaser = threading.Thread(target=release_shortly)
        releaser.start()
        try:
            applied = apply_migrations(TEST_DSN, table=table, dim=DIM)
        finally:
            releaser.join(timeout=30)
    assert applied


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
    # This one genuinely needs `CREATEROLE`: its subject is a privilege boundary between two
    # roles, so it has to make the second role. Skipped rather than failed when `RECALL_TEST_DSN`
    # names a role that cannot create one, which is now a supported way to run the suite. It is
    # not weakened: on any DSN that CAN provision (CI's included) it runs exactly as before.
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        may_provision = conn.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    if not (may_provision and may_provision[0]):
        pytest.skip("RECALL_TEST_DSN's role cannot CREATE ROLE, which this test's subject requires")
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
    if not parts.scheme:
        # Same hazard the sibling helper guards: a libpq keyword/value DSN has no URI parts, so
        # rebuilding it yields a string libpq cannot parse.
        pytest.skip("non-URI RECALL_TEST_DSN; cannot derive an admin DSN")
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
    # Like its sibling `test_serving_role_has_dml_but_cannot_run_ddl`, this test's subject is a
    # privilege boundary, so it has to CREATE the second role. Skipped rather than failed when
    # `RECALL_TEST_DSN` names a role that cannot, which is now a supported way to run the suite
    # (an unprivileged DSN is what makes every RLS assertion in it non-vacuous). Unchanged on any
    # DSN that can provision, CI's included.
    with psycopg.connect(TEST_DSN, autocommit=True) as _probe:
        _may = _probe.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    if not (_may and _may[0]):
        pytest.skip("RECALL_TEST_DSN's role cannot CREATE ROLE, which this test's subject requires")

    from recall.control_plane import ControlPlane

    role = _name("cca_grantee_")
    # Bind every name the cleanup touches BEFORE the try. The role is a LOGIN role with a known
    # password, so a NameError in the finally would both mask the real failure and leak it into
    # the cluster permanently, once per failing run.
    generation = _name("g_")
    physical = _name("cp_chunks_")
    parts = urlsplit(TEST_DSN)
    as_role = urlunsplit(parts._replace(netloc=f"{role}:probe@{parts.hostname}:{parts.port}"))
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'probe' NOSUPERUSER NOBYPASSRLS")
            conn.execute(f"GRANT CONNECT ON DATABASE {parts.path.lstrip('/')} TO {role}")
            conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        control = ControlPlane(TEST_DSN)
        control.apply_migrations()
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
            # Each statement independently, so one failure cannot skip DROP ROLE and strand a
            # login role in the cluster.
            failures: list[str] = []
            for statement, params in (
                ("DELETE FROM recall_migration_events WHERE tenant_id = %s", ("probe-tenant",)),
                ("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", ("probe-tenant",)),
                (f"DROP TABLE IF EXISTS {physical}", None),
                (f"REASSIGN OWNED BY {role} TO CURRENT_USER", None),
                (f"DROP OWNED BY {role}", None),
                (f"DROP ROLE IF EXISTS {role}", None),
            ):
                try:
                    conn.execute(statement, params) if params else conn.execute(statement)
                except psycopg.Error as exc:
                    failures.append(f"{statement.split()[0]} {statement.split()[1]}: {exc}")
            leaked = conn.execute(
                "SELECT count(*) FROM pg_roles WHERE rolname = %s", (role,)
            ).fetchone()[0]
        if leaked:
            # Never swallow this silently: it is a LOGIN role with a known password, and
            # `_name()` appends a fresh uuid, so a silent leak accumulates one per failing run.
            raise AssertionError(f"grants test leaked role {role} into the cluster: {failures}")


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
    # PostgreSQL grantee keywords are valid Python identifiers, so `.isidentifier()` alone let
    # `--role public` grant every managed table to every role in the database.
    for keyword in ("public", "PUBLIC", "current_user", "SESSION_USER", "current_role", "user"):
        with pytest.raises(ValueError, match="grantee keyword"):
            serving_grants(keyword)
    # The ROLE is quoted, so PostgreSQL cannot case-fold it onto a different role.
    assert '"Recall_Server"' in " ".join(serving_grants("Recall_Server"))
    # The TABLE must NOT be quoted: the migrator creates it with unquoted DDL, so PostgreSQL
    # folds it to lower case and a quoted GRANT would name an object nothing here can create.
    assert "ON probe_table TO" in " ".join(serving_grants("recall_server", table="probe_table"))
    assert '"probe_table"' not in " ".join(serving_grants("recall_server", table="probe_table"))
    # recall_tenant_routes is read-only for the serving role: only the migration-role CLI
    # writes it, and INSERT/UPDATE would let a tenant repoint its own active generation.
    enterprise = serving_grants("recall_server", enterprise=True)
    assert any("SELECT ON" in s and "recall_tenant_routes" in s for s in enterprise)
    assert not any("INSERT" in s and "recall_tenant_routes" in s for s in enterprise)
