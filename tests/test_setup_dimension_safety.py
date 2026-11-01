"""Dimension conflicts must preserve existing objects, rows and serving grants.

Red proof: these tests fail preservation assertions against eddfc9179. The race
uses a real second connection at the old public wizard's rebuild announcement;
the RLS case proves a physical row is hidden from its NOBYPASSRLS owner.
"""

from contextlib import contextmanager
import uuid

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from recall import setup
from recall.schema import apply_migrations
from tests.conftest import TEST_DSN, requires_db
from tests.test_schema_migrations import _fresh_database
from tests.test_setup import _roomy_probe


@contextmanager
def _role():
    name = "setup_probe_" + uuid.uuid4().hex[:12]
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(
            sql.Identifier(name)
        ))
    try:
        yield name
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


def _run_conflict(dsn, table, tmp_path, monkeypatch, *, migration_dsn=None, print_fn=None):
    monkeypatch.setattr(setup, "probe_hardware", lambda: _roomy_probe(10 * 1024**3))
    monkeypatch.setattr(setup, "_module_available", lambda name: False)
    answers = iter(["n", "", "", "", "4"])
    try:
        setup.run_setup_wizard(
            dsn=dsn, migration_dsn=migration_dsn, table=table,
            env_path=tmp_path / ".env", input_fn=lambda prompt="": next(answers),
            print_fn=print_fn or (lambda *args, **kwargs: None),
        )
    except (SystemExit, StopIteration) as exc:
        return str(exc)
    return ""


@requires_db
@pytest.mark.parametrize("table", ["chunks", "setup_custom"])
def test_dimension_conflict_never_drops_after_an_empty_count(
    table, tmp_path, monkeypatch,
):
    """BUG-001: a committed concurrent row must survive an attempted rebuild.

    The unchanged node fails on baseline at the table/row preservation assertion.
    A second connection commits and observes the row before the old DROP resumes.
    A refusal before that announcement must preserve the original relation too.
    """
    with _fresh_database("setup_race_") as dsn:
        apply_migrations(dsn, dim=64)
        apply_migrations(dsn, table=table, dim=64)
        with psycopg.connect(dsn, autocommit=True) as conn:
            original_oid = conn.execute("SELECT %s::regclass::oid", (table,)).fetchone()[0]
        committed = []

        def insert_after_count(*args, **kwargs):
            if "wizard will rebuild it now" not in " ".join(map(str, args)):
                return
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("INSERT INTO {} (id, source, text) VALUES (%s, %s, %s)").format(
                    sql.Identifier(table)
                ), ("concurrent", "writer.md", "committed after the count"))
                committed.append(conn.execute(sql.SQL("SELECT count(*) FROM {}").format(
                    sql.Identifier(table)
                )).fetchone()[0])

        _run_conflict(dsn, table, tmp_path, monkeypatch, print_fn=insert_after_count)
        with psycopg.connect(dsn, autocommit=True) as conn:
            oid = conn.execute("SELECT to_regclass(%s)::oid", (table,)).fetchone()[0]
            assert oid == original_oid, "setup replaced a table after an unprotected empty count"
            if committed:
                assert committed == [1], "the independent writer must have committed a real row"
                assert conn.execute(sql.SQL("SELECT count(*) FROM {} WHERE id = %s").format(
                    sql.Identifier(table)
                ), ("concurrent",)).fetchone()[0] == 1


@requires_db
def test_dimension_conflict_preserves_rows_hidden_by_force_rls(tmp_path, monkeypatch):
    """STAKES-001: baseline drops a populated table its owner sees as empty."""
    with _role() as owner:
        with _fresh_database("setup_rls_") as dsn:
            table = "setup_hidden"
            apply_migrations(dsn, dim=64)
            apply_migrations(dsn, table=table, dim=64)
            owner_dsn = make_conninfo(dsn, options=f"-c role={owner}")
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                    sql.Identifier(table), sql.Identifier(owner)
                ))
                conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(owner)))
                conn.execute(sql.SQL("GRANT SELECT, DELETE ON recall_schema_migrations, recall_sparse_v1 TO {}").format(sql.Identifier(owner)))
                conn.execute(sql.SQL("INSERT INTO {} (tenant_id, id, source, text) VALUES (%s, %s, %s, %s)").format(
                    sql.Identifier(table)
                ), ("other-tenant", "hidden", "hidden.md", "preserve this tenant"))
                assert conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0] == 1
            with psycopg.connect(owner_dsn, autocommit=True) as conn:
                assert conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone() == (False, False)
                assert conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0] == 0
            _run_conflict(owner_dsn, table, tmp_path, monkeypatch)
            with psycopg.connect(dsn, autocommit=True) as conn:
                exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
                assert exists is not None, "setup dropped a populated table hidden by FORCE RLS"
                assert conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0] == 1


@requires_db
def test_dimension_conflict_preserves_global_migration_state(tmp_path, monkeypatch):
    """DAT-001: baseline partially resets globals then fails replaying policy 0034."""
    with _fresh_database("setup_global_") as dsn:
        apply_migrations(dsn, table="chunks", dim=64)
        _run_conflict(dsn, "chunks", tmp_path, monkeypatch)
        assert setup._schema_prepare_state(dsn, 64, "chunks")[0] == "compatible", (
            "a dimension conflict must not leave the existing schema partially reset"
        )


@requires_db
def test_dimension_conflict_preserves_explicit_serving_grants(tmp_path, monkeypatch):
    """DEPLOY-001: baseline recreates the custom relation without its explicit ACL."""
    with _role() as server:
        with _fresh_database("setup_grants_") as dsn:
            table = "setup_granted"
            apply_migrations(dsn, dim=64)
            apply_migrations(dsn, table=table, dim=64)
            serving_dsn = make_conninfo(dsn, options=f"-c role={server}")
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(server)))
                conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(server)))
                conn.execute(sql.SQL("GRANT INSERT ON {} TO {}").format(sql.Identifier(table), sql.Identifier(server)))
                assert conn.execute("SELECT has_table_privilege(%s, %s, 'SELECT, INSERT')", (server, table)).fetchone()[0]
            _run_conflict(serving_dsn, table, tmp_path, monkeypatch, migration_dsn=dsn)
            with psycopg.connect(dsn, autocommit=True) as conn:
                assert conn.execute("SELECT has_table_privilege(%s, %s, 'SELECT, INSERT')", (server, table)).fetchone()[0], (
                    "setup removed previously working explicit serving grants"
                )
