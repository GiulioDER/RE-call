from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from recall.store import PgVectorStore

#: The local dev database from docker-compose.yml — the same one the README quickstart uses.
_LOCAL_DEV_DSN = "postgresql://recall:recall@localhost:5432/recall"

#: The test database. Deliberately NOT read from `RECALL_DSN`.
#:
#: These tests DROP TABLES. `RECALL_DSN` is the variable the README tells users to point at their
#: real database, so resolving the test DSN from it meant that exporting it and running `pytest`
#: destroyed production data — no flag, no prompt, no way back. The suite now reads a dedicated
#: `RECALL_TEST_DSN` and otherwise falls back to the local dev container, so a `RECALL_DSN`
#: pointing at anything real is simply never consulted.
TEST_DSN = os.environ.get("RECALL_TEST_DSN", _LOCAL_DEV_DSN)


def _reject_unsafe_test_dsn() -> None:
    """Refuse to run destructive tests against a database that might not be disposable.

    Two ways to get here: pointing `RECALL_TEST_DSN` at the same database as `RECALL_DSN`, or
    pointing it at a remote host. Both are refused at import time rather than discovered
    afterwards, because the damage is not recoverable from a test report.
    """
    from urllib.parse import urlsplit

    from recall.store import _is_local_host

    configured = os.environ.get("RECALL_TEST_DSN")
    if configured is None:
        return
    if configured == os.environ.get("RECALL_DSN"):
        raise RuntimeError(
            "RECALL_TEST_DSN is the same database as RECALL_DSN. These tests DROP TABLES — "
            "point RECALL_TEST_DSN at a throwaway database."
        )
    host = (urlsplit(configured).hostname or "").lower()
    if not _is_local_host(host) and not os.environ.get("RECALL_ALLOW_REMOTE_TEST_DB"):
        raise RuntimeError(
            f"RECALL_TEST_DSN points at the non-local host {host!r}. These tests DROP TABLES — "
            "set RECALL_ALLOW_REMOTE_TEST_DB=1 only if that database is genuinely disposable."
        )


_reject_unsafe_test_dsn()


def _db_available() -> bool:
    try:
        psycopg.connect(TEST_DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(),
    reason="pgvector DB not reachable (run `docker compose up -d`)",
)


@pytest.fixture
def make_store():
    created: list[PgVectorStore] = []

    def _factory(dim: int) -> PgVectorStore:
        table = "t_" + uuid.uuid4().hex[:8]
        store = PgVectorStore(TEST_DSN, dim=dim, table=table)
        store.ensure_schema()
        created.append(store)
        return store

    yield _factory

    for store in created:
        if store._closed:
            # A test may close its store deliberately (close() is sticky). Still drop the
            # table — skipping teardown entirely would leak a uuid-named table per run.
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {store.table}")
            continue
        store.drop_table()
        store.close()


@pytest.fixture
def cli_table():
    """A uuid-named table for CLI end-to-end tests, dropped afterwards.

    The CLI tests used to run against the default `chunks` table and `DROP TABLE IF EXISTS
    chunks` to isolate themselves — which is what made `pytest` destructive against whatever
    database was configured. A throwaway table per test isolates without dropping anything a
    user owns.
    """
    name = "cli_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
