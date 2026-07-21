from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from recall.store import PgVectorStore

TEST_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


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
        store._conn.execute(f"DROP TABLE IF EXISTS {store.table}")
        store.close()
