"""Versioned schema migration and serving-role regression coverage."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from recall.cli import main as cli_main
from recall.schema import (
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
    assert [m.version for m in migrations] == [f"{n:04d}" for n in range(1, 11)]
    assert migrations[0].transactional
    assert migrations[7].transactional
    assert all(m.concurrent_index for m in (*migrations[1:7], *migrations[8:]))
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
        assert status.compatible and status.current_version == "0010"

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
        assert "current: 0010" in status and "compatible: yes" in status


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
