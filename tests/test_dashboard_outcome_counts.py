"""NUM-001: canonical abstentions are decisions, but must never count as answers.

Red proof: the unchanged baseline counts three answered decisions instead of two,
on both PostgreSQL and SQLite. The exact nodes then pass with outcome-aware counts.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg

from recall.dashboard import db as dbq
from recall.lite.store import LiteStore
from tests.conftest import TEST_DSN, requires_db

EVENTS = (
    ("search_decision", {"outcome": "answered", "abstained": False}),
    ("search_decision", {"outcome": "abstained", "abstained": True}),
    ("search_refusal", {"outcome": "refused"}),
    ("search_decision", {"query": "legacy answer"}),
)


def _assert_counts(db: dbq.DashboardDB, tenant: str) -> None:
    counts = dbq.control(db, tenant)["summary"]
    assert counts["answered"] == 2, "an abstention counted as an answer"
    assert (counts["searches"], counts["abstained"], counts["refused"]) == (4, 1, 1)


def test_lite_counts_canonical_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    with LiteStore(path, dim=4, tenant="default") as store:
        for kind, payload in EVENTS:
            store.append_audit_event(kind, payload)
    _assert_counts(dbq.DashboardDB("sqlite:///" + path.as_posix()), "default")


@requires_db
def test_postgres_counts_canonical_outcomes(make_store) -> None:
    make_store(8)
    tenant = "outcomes-" + uuid.uuid4().hex
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("select set_config('recall.tenant_id', %s, false)", (tenant,))
        try:
            for kind, payload in EVENTS:
                conn.execute(
                    "insert into recall_audit_events (tenant_id,event_id,event_type,actor,payload) values (%s,%s,%s,'test',%s)",
                    (tenant, uuid.uuid4().hex, kind, json.dumps(payload)),
                )
            _assert_counts(dbq.DashboardDB(TEST_DSN), tenant)
        finally:
            conn.execute("delete from recall_audit_events where tenant_id=%s", (tenant,))
