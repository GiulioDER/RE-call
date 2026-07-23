"""Schema DDL waits a bounded time for its LOCK, while its WORK stays unbounded.

`ensure_schema()` lifts `statement_timeout` so an HNSW build — minutes on a real corpus — is not
cancelled halfway and left as an INVALID index that `IF NOT EXISTS` then treats as present
forever. But `statement_timeout` also bounded lock WAITING, and lifting it removed the only thing
that broke a wait for a lock that never comes: `CREATE INDEX CONCURRENTLY` waits for every
concurrent transaction on the table, and the tenancy ALTERs take ACCESS EXCLUSIVE. One
`idle in transaction` session elsewhere could park schema setup indefinitely, with every query
arriving afterwards queued behind it and no error to explain why.

These tests pin the split: bounded queueing, unbounded work, and both session settings put back
the way they were found.
"""
from __future__ import annotations

import time
import uuid

import psycopg
import pytest

from recall.store import DEFAULT_SCHEMA_LOCK_TIMEOUT_MS, PgVectorStore, _schema_lock_timeout_ms

from tests.conftest import TEST_DSN, requires_db


@pytest.fixture
def table_name():
    return f"lock_probe_{uuid.uuid4().hex[:8]}"


@requires_db
def test_the_lock_bound_is_in_force_during_the_ddl(table_name, monkeypatch):
    """Not merely restored afterwards — actually applied while the DDL runs.

    A setting that is put back correctly but was never in force protects nothing, and the two are
    indistinguishable from the outside once `ensure_schema` returns.
    """
    seen: list[str] = []
    original = PgVectorStore._ensure_schema_ddl

    def spy(self, conn):
        seen.append(conn.execute("SHOW lock_timeout").fetchone()[0])
        return original(self, conn)

    monkeypatch.setattr(PgVectorStore, "_ensure_schema_ddl", spy)
    with PgVectorStore(TEST_DSN, dim=4, table=table_name, statement_timeout_ms=15000) as store:
        store.ensure_schema()
        _drop(store, table_name)

    assert seen == ["5s"], f"lock_timeout during DDL was {seen}, expected the 5000ms default"


@requires_db
def test_both_session_settings_are_restored_on_the_connection_that_ran_the_ddl(table_name):
    """Asserted on the SAME connection the DDL ran on — the only one that can show the leak.

    A fresh `_connect()` proves nothing here: `_prepare` configures every new connection with
    the right `statement_timeout` and a default `lock_timeout` regardless of what the DDL left
    behind. A test written that way stays green with both restores deleted, which is precisely
    the regression it is supposed to catch. The connection handed back to the pool is the
    subject, so it has to be the thing observed.
    """
    with PgVectorStore(TEST_DSN, dim=4, table=table_name, statement_timeout_ms=15000) as store:
        store.ensure_schema()
        conn = store._conn  # the live single-mode connection ensure_schema just used
        assert conn is not None, "expected single-connection mode for this probe"
        assert conn.execute("SHOW lock_timeout").fetchone()[0] == "0"
        assert conn.execute("SHOW statement_timeout").fetchone()[0] == "15s"
        _drop(store, table_name)


@pytest.mark.timeout(30)
@requires_db
def test_ensure_schema_gives_up_rather_than_queueing_behind_a_held_lock(table_name, monkeypatch):
    """The whole point, driven through `ensure_schema()` itself.

    The timeout mark is load-bearing, not belt-and-braces. The blocking session lives in THIS
    process, so if the lock bound regresses `ensure_schema` waits on a lock only this thread can
    release — a deadlock, not a failure. Bare `pytest -q` (what CI runs) would then read as
    "still running" until the job cap, which is precisely the outcome `pytest-timeout` is
    declared in pyproject.toml to prevent.

    Hand-rolling `SET lock_timeout` on a scratch connection and running your own `ALTER` would
    demonstrate that PostgreSQL implements `lock_timeout` — which was never in doubt. The claim
    under test is that THIS function applies it, so `ensure_schema()` has to be the thing called
    while the lock is held.
    """
    monkeypatch.setenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", "400")
    with PgVectorStore(TEST_DSN, dim=4, table=table_name, statement_timeout_ms=15000) as store:
        store.ensure_schema()  # table + indexes exist, so a re-run is otherwise a no-op
        # Drop one index so the next ensure_schema has real DDL to do and must take a lock.
        with store._connect() as conn:
            conn.execute(f"DROP INDEX IF EXISTS {table_name}_source_idx")

        blocker = psycopg.connect(TEST_DSN)
        try:
            with blocker.transaction():
                blocker.execute(f"LOCK TABLE {table_name} IN ACCESS EXCLUSIVE MODE")
                start = time.monotonic()
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    store.ensure_schema()
                waited = time.monotonic() - start
        finally:
            blocker.close()

        # Bounded, not merely eventual: without the fix this blocks until the blocker commits.
        assert waited < 5.0, f"ensure_schema waited {waited:.1f}s for a lock it should refuse"
        store.ensure_schema()  # idempotent: succeeds once the lock is gone
        _drop(store, table_name)


def test_the_bound_is_configurable_and_malformed_values_fall_back(monkeypatch):
    """It can refuse where the old code waited, so it needs an escape hatch — `0` waits forever."""
    monkeypatch.delenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", raising=False)
    assert _schema_lock_timeout_ms() == DEFAULT_SCHEMA_LOCK_TIMEOUT_MS

    monkeypatch.setenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", "0")
    assert _schema_lock_timeout_ms() == 0  # explicit opt-out: wait forever, as before

    monkeypatch.setenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", "30000")
    assert _schema_lock_timeout_ms() == 30000

    for bad in ("", "soon", "-1", "5.5"):
        monkeypatch.setenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", bad)
        assert _schema_lock_timeout_ms() == DEFAULT_SCHEMA_LOCK_TIMEOUT_MS, bad

    # Above PostgreSQL's int range `SET lock_timeout` raises, so a knob meant to LOOSEN the bound
    # would instead break ensure_schema entirely. Clamped rather than rejected.
    monkeypatch.setenv("RECALL_SCHEMA_LOCK_TIMEOUT_MS", "3000000000")
    assert _schema_lock_timeout_ms() == 2147483647


def _drop(store: PgVectorStore, table: str) -> None:
    with store._connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
