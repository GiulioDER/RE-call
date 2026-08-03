"""Versioned, checksum-verified PostgreSQL schema management.

Serving code calls :func:`check_schema` only.  DDL is reachable solely through the
explicit migrator API and ``recall schema apply``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Literal

import psycopg

if TYPE_CHECKING:
    from psycopg import Connection

from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, TENANT_GUC, _schema_lock_timeout_ms

LEDGER_TABLE = "recall_schema_migrations"
MIGRATION_LOCK_NAME = "recall-schema-migrations-v1"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
_INDEX_MARKER = re.compile(r"^-- recall:concurrent-index ([A-Za-z_][A-Za-z0-9_]*)$", re.M)


class SchemaError(RuntimeError):
    """Base class for a schema state that prevents safe service."""


class SchemaTooOld(SchemaError):
    """The database is missing one or more migrations."""


class SchemaTooNew(SchemaError):
    """The database contains migrations unknown to this package."""


class MigrationChecksumMismatch(SchemaError):
    """A committed or previously-applied migration changed bytes."""


class ConcurrentMigrator(SchemaError):
    """Another process currently owns the schema migration lock."""


class InterruptedConcurrentIndex(SchemaError):
    """A concurrent index phase could not be validated or resumed."""


class SchemaIncompatible(SchemaError, ValueError):
    """The selected table is not a compatible RE-call v0.8 table."""


@dataclass(frozen=True)
class Migration:
    version: str
    filename: str
    checksum: str
    sql: str
    transactional: bool
    concurrent_index: str | None


@dataclass(frozen=True)
class MigrationState:
    version: str
    filename: str
    checksum: str
    state: Literal["pending", "running", "failed", "applied"]
    applied_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SchemaStatus:
    table: str
    current_version: str | None
    required_version: str
    compatible: bool
    migrations: tuple[MigrationState, ...]

    @property
    def pending(self) -> tuple[MigrationState, ...]:
        return tuple(m for m in self.migrations if m.state != "applied")


def _validate_target(table: str, dim: int) -> None:
    if not table.isidentifier():
        raise ValueError("table must be a valid SQL identifier")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive int")


def _migration_root() -> resources.abc.Traversable:
    return resources.files("recall.migrations")


def load_migrations() -> tuple[Migration, ...]:
    """Load migrations and verify their bytes against the committed manifest."""
    root = _migration_root()
    expected_raw = json.loads(root.joinpath("checksums.json").read_text(encoding="utf-8"))
    if not isinstance(expected_raw, dict):
        raise MigrationChecksumMismatch("migration checksum manifest must be a JSON object")
    expected = {str(k): str(v) for k, v in expected_raw.items()}
    sql_root = root.joinpath("sql")
    names = sorted(p.name for p in sql_root.iterdir() if _MIGRATION_NAME.fullmatch(p.name))
    if set(names) != set(expected):
        missing = sorted(set(names) - set(expected))
        extra = sorted(set(expected) - set(names))
        raise MigrationChecksumMismatch(
            f"migration manifest/file mismatch (unlisted={missing}, missing_files={extra})"
        )
    migrations: list[Migration] = []
    versions: set[str] = set()
    for name in names:
        data = sql_root.joinpath(name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected[name]:
            raise MigrationChecksumMismatch(
                f"migration {name} checksum drift: committed {expected[name]}, actual {actual}"
            )
        version = name[:4]
        if version in versions:
            raise MigrationChecksumMismatch(f"duplicate migration version {version}")
        versions.add(version)
        sql = data.decode("utf-8")
        marker = _INDEX_MARKER.search(sql)
        transactional = sql.startswith("-- recall:transactional")
        if transactional == (marker is not None):
            raise MigrationChecksumMismatch(
                f"migration {name} must declare exactly one execution mode"
            )
        migrations.append(
            Migration(
                version=version,
                filename=name,
                checksum=actual,
                sql=sql,
                transactional=transactional,
                concurrent_index=marker.group(1) if marker else None,
            )
        )
    if not migrations:
        raise MigrationChecksumMismatch("no packaged migrations found")
    return tuple(migrations)


def _render(migration: Migration, table: str, dim: int) -> tuple[str, str | None]:
    substitutions = {
        "__RECALL_TABLE__": table,
        "__RECALL_DIM__": str(dim),
        "__RECALL_DEFAULT_TENANT__": DEFAULT_TENANT,
        "__RECALL_TENANT_GUC__": TENANT_GUC,
    }
    sql = migration.sql
    index = migration.concurrent_index
    for token, value in substitutions.items():
        sql = sql.replace(token, value)
        if index is not None:
            index = index.replace(token, value)
    return sql, index


def _connect(dsn: str) -> Connection:
    return psycopg.connect(dsn, autocommit=True, connect_timeout=10)


def _ledger_exists(conn: Connection) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()
    return bool(row and row[0])


def _read_applied(conn: Connection, table: str) -> dict[str, tuple[str, str, str | None, str | None]]:
    if not _ledger_exists(conn):
        return {}
    rows = conn.execute(
        f"SELECT version, checksum, state, applied_at::text, error "
        f"FROM {LEDGER_TABLE} WHERE target_table = %s ORDER BY version",
        (table,),
    ).fetchall()
    return {str(r[0]): (str(r[1]), str(r[2]), r[3], r[4]) for r in rows}


def schema_status(dsn: str, *, table: str = DEFAULT_TABLE, dim: int) -> SchemaStatus:
    """Inspect migration state with SELECTs only; never bootstraps the ledger."""
    _validate_target(table, dim)
    migrations = load_migrations()
    with _connect(dsn) as conn:
        applied = _read_applied(conn, table)
    known = {m.version for m in migrations}
    unknown = sorted(set(applied) - known)
    if unknown:
        raise SchemaTooNew(
            f"table {table!r} has unknown schema migration(s) {unknown}; upgrade RE-call"
        )
    states: list[MigrationState] = []
    for migration in migrations:
        row = applied.get(migration.version)
        if row is None:
            states.append(
                MigrationState(
                    migration.version, migration.filename, migration.checksum, "pending"
                )
            )
            continue
        checksum, state, applied_at, error = row
        if checksum != migration.checksum:
            raise MigrationChecksumMismatch(
                f"applied migration {migration.filename} has checksum {checksum}, "
                f"package has {migration.checksum}"
            )
        if state not in {"running", "failed", "applied"}:
            raise SchemaError(f"migration {migration.version} has invalid ledger state {state!r}")
        states.append(
            MigrationState(
                migration.version,
                migration.filename,
                migration.checksum,
                state,  # type: ignore[arg-type]
                str(applied_at) if applied_at else None,
                str(error) if error else None,
            )
        )
    current = next((m.version for m in reversed(states) if m.state == "applied"), None)
    return SchemaStatus(
        table=table,
        current_version=current,
        required_version=migrations[-1].version,
        compatible=all(m.state == "applied" for m in states),
        migrations=tuple(states),
    )


def schema_plan(dsn: str, *, table: str = DEFAULT_TABLE, dim: int) -> tuple[MigrationState, ...]:
    """Return unapplied phases. This operation is deliberately read-only."""
    return schema_status(dsn, table=table, dim=dim).pending


def _validate_existing_table(conn: Connection, table: str, dim: int) -> None:
    exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    if not exists or exists[0] is None:
        return
    columns_row = conn.execute(
        "SELECT array_agg(attname), max(atttypmod) FILTER (WHERE attname = 'embedding') "
        "FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0 AND NOT attisdropped",
        (table,),
    ).fetchone()
    columns = set(columns_row[0] or ()) if columns_row else set()
    required = {"id", "source", "text", "metadata", "embedding", "indexed_at", "tsv"}
    if missing := sorted(required - columns):
        raise SchemaIncompatible(
            f"table {table!r} is not a recall table compatible with v0.8; "
            f"missing columns {missing}"
        )
    actual_dim = columns_row[1] if columns_row else None
    if actual_dim is None:
        raise SchemaIncompatible(f"table {table!r} has no embedding type modifier")
    if actual_dim > 0 and actual_dim != dim:
        raise SchemaIncompatible(
            f"table {table!r} uses vector({actual_dim}), requested dimension is {dim}"
        )


def _validate_current_schema(conn: Connection, table: str, dim: int) -> None:
    """Validate security and index invariants after the migration ledger is current."""
    _validate_existing_table(conn, table, dim)
    columns = conn.execute(
        "SELECT array_agg(attname) FROM pg_attribute WHERE attrelid = %s::regclass "
        "AND attnum > 0 AND NOT attisdropped",
        (table,),
    ).fetchone()
    present = set(columns[0] or ()) if columns else set()
    if missing := sorted({"tenant_id", "first_indexed_at"} - present):
        raise SchemaIncompatible(f"table {table!r} schema drift: missing columns {missing}")

    policy = f"{table}_tenant_isolation"
    state = conn.execute(
        "SELECT c.relrowsecurity, c.relforcerowsecurity, p.polname IS NOT NULL, "
        "pg_get_expr(p.polqual, p.polrelid), pg_get_expr(p.polwithcheck, p.polrelid), "
        "p.polroles = '{0}'::oid[], p.polcmd "
        "FROM pg_class c LEFT JOIN pg_policy p "
        "ON p.polrelid = c.oid AND p.polname = %s WHERE c.oid = %s::regclass",
        (policy, table),
    ).fetchone()
    want = f"(tenant_id = current_setting('{TENANT_GUC}'::text, true))"
    if not state or state != (True, True, True, want, want, True, "*"):
        raise SchemaIncompatible(
            f"table {table!r} row-level-security policy drift; run a new reviewed migration"
        )

    expected_indexes = {
        f"{table}_tsv_idx",
        f"{table}_emb_idx",
        f"{table}_indexed_at_idx",
        f"{table}_source_idx",
        f"{table}_file_idx",
        f"{table}_tenant_idx",
    }
    rows = conn.execute(
        "SELECT c.relname, i.indisvalid FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid WHERE i.indrelid = %s::regclass",
        (table,),
    ).fetchall()
    valid = {str(name) for name, is_valid in rows if is_valid}
    if missing_indexes := sorted(expected_indexes - valid):
        raise SchemaIncompatible(
            f"table {table!r} schema drift: missing/invalid indexes {missing_indexes}"
        )


def _create_ledger(conn: Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
            target_table TEXT NOT NULL,
            version TEXT NOT NULL,
            filename TEXT NOT NULL,
            checksum CHAR(64) NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('running', 'failed', 'applied')),
            started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            applied_at TIMESTAMPTZ,
            error TEXT,
            applied_by TEXT NOT NULL DEFAULT current_user,
            PRIMARY KEY (target_table, version)
        )
        """
    )


def _require_migration_privileges(conn: Connection) -> None:
    row = conn.execute(
        "SELECT has_schema_privilege(current_user, current_schema(), 'CREATE')"
    ).fetchone()
    if not row or not row[0]:
        raise PermissionError(
            "schema apply requires the migration role (CREATE on the target schema); "
            "use the migration DSN, not the serving DSN"
        )


def _mark_running(conn: Connection, table: str, migration: Migration) -> None:
    conn.execute(
        f"""
        INSERT INTO {LEDGER_TABLE}
            (target_table, version, filename, checksum, state, started_at, applied_at, error)
        VALUES (%s, %s, %s, %s, 'running', clock_timestamp(), NULL, NULL)
        ON CONFLICT (target_table, version) DO UPDATE SET
            filename = EXCLUDED.filename,
            checksum = EXCLUDED.checksum,
            state = 'running',
            started_at = clock_timestamp(),
            applied_at = NULL,
            error = NULL
        """,
        (table, migration.version, migration.filename, migration.checksum),
    )


def _mark_failed(conn: Connection, table: str, migration: Migration, exc: BaseException) -> None:
    conn.execute(
        f"UPDATE {LEDGER_TABLE} SET state = 'failed', error = %s "
        "WHERE target_table = %s AND version = %s",
        (f"{type(exc).__name__}: {exc}"[:2000], table, migration.version),
    )


def _mark_applied(conn: Connection, table: str, migration: Migration) -> None:
    conn.execute(
        f"UPDATE {LEDGER_TABLE} SET state = 'applied', applied_at = clock_timestamp(), error = NULL "
        "WHERE target_table = %s AND version = %s",
        (table, migration.version),
    )


def _run_transactional(
    conn: Connection, table: str, dim: int, migration: Migration
) -> None:
    sql, _ = _render(migration, table, dim)
    with conn.transaction():
        _mark_running(conn, table, migration)
        conn.execute(sql)
        _mark_applied(conn, table, migration)


def _run_concurrent_index(
    conn: Connection, table: str, dim: int, migration: Migration
) -> None:
    sql, index = _render(migration, table, dim)
    if index is None:  # pragma: no cover - load_migrations enforces this shape
        raise AssertionError("concurrent migration has no index marker")
    _mark_running(conn, table, migration)
    try:
        row = conn.execute(
            "SELECT i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
            "WHERE c.oid = to_regclass(%s)",
            (index,),
        ).fetchone()
        if row and not row[0]:
            conn.execute(f"DROP INDEX CONCURRENTLY {index}")
        conn.execute(sql)
        valid = conn.execute(
            "SELECT i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
            "WHERE c.oid = to_regclass(%s)",
            (index,),
        ).fetchone()
        if not valid or not valid[0]:
            raise InterruptedConcurrentIndex(
                f"migration {migration.filename} did not leave valid index {index!r}"
            )
    except BaseException as exc:
        _mark_failed(conn, table, migration, exc)
        raise
    _mark_applied(conn, table, migration)


def apply_migrations(
    migration_dsn: str, *, table: str = DEFAULT_TABLE, dim: int
) -> tuple[MigrationState, ...]:
    """Apply pending migrations under a process-wide PostgreSQL advisory lock."""
    _validate_target(table, dim)
    migrations = load_migrations()
    applied_now: list[MigrationState] = []
    with _connect(migration_dsn) as conn:
        _require_migration_privileges(conn)
        lock = conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (MIGRATION_LOCK_NAME,)
        ).fetchone()
        if not lock or not lock[0]:
            raise ConcurrentMigrator("another RE-call schema migrator is already running")
        lock_row = conn.execute("SHOW lock_timeout").fetchone()
        statement_row = conn.execute("SHOW statement_timeout").fetchone()
        prior_lock_timeout = str(lock_row[0]) if lock_row else "0"
        prior_statement_timeout = str(statement_row[0]) if statement_row else "0"
        try:
            # Migration work, especially HNSW construction, is deliberately unbounded. Waiting
            # for a table lock is not progress and stays bounded independently.
            conn.execute("SET statement_timeout = 0")
            conn.execute(f"SET lock_timeout = {_schema_lock_timeout_ms()}")
            _create_ledger(conn)
            _validate_existing_table(conn, table, dim)
            existing = _read_applied(conn, table)
            known = {m.version for m in migrations}
            if unknown := sorted(set(existing) - known):
                raise SchemaTooNew(
                    f"table {table!r} has unknown schema migration(s) {unknown}; upgrade RE-call"
                )
            for migration in migrations:
                row = existing.get(migration.version)
                if row is not None:
                    checksum, state, _applied_at, _error = row
                    if checksum != migration.checksum:
                        raise MigrationChecksumMismatch(
                            f"applied migration {migration.filename} checksum drift"
                        )
                    if state == "applied":
                        continue
                if migration.transactional:
                    _run_transactional(conn, table, dim, migration)
                else:
                    _run_concurrent_index(conn, table, dim, migration)
                applied_now.append(
                    MigrationState(
                        migration.version,
                        migration.filename,
                        migration.checksum,
                        "applied",
                    )
                )
            _validate_current_schema(conn, table, dim)
        finally:
            # Restore both settings before this connection can be reused by a caller. Use
            # set_config because SHOW returns formatted values such as '5s' and '1min'.
            try:
                conn.execute(
                    "SELECT set_config('statement_timeout', %s, false)",
                    (prior_statement_timeout,),
                )
                conn.execute(
                    "SELECT set_config('lock_timeout', %s, false)", (prior_lock_timeout,)
                )
            finally:
                conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (MIGRATION_LOCK_NAME,)
                )
    return tuple(applied_now)


def check_schema(conn: Connection, *, table: str = DEFAULT_TABLE, dim: int) -> None:
    """Fail unless the schema exactly matches this package. Executes SELECT only."""
    _validate_target(table, dim)
    migrations = load_migrations()
    applied = _read_applied(conn, table)
    known = {m.version for m in migrations}
    if unknown := sorted(set(applied) - known):
        raise SchemaTooNew(
            f"table {table!r} has unknown migration(s) {unknown}; upgrade the application"
        )
    pending: list[str] = []
    for migration in migrations:
        row = applied.get(migration.version)
        if row is None or row[1] != "applied":
            pending.append(migration.version)
            continue
        if row[0] != migration.checksum:
            raise MigrationChecksumMismatch(
                f"migration {migration.filename} checksum does not match the running package"
            )
    if pending:
        raise SchemaTooOld(
            f"table {table!r} needs schema migration(s) {pending}; run `recall schema apply`"
        )
    _validate_current_schema(conn, table, dim)
