"""The Control page's numbers, computed by `recall.dashboard.db` against a real audit table.

Invariants and the failure each one catches:
- K1 a memory returned as several chunks of one search counts as ONE retrieval, at the rank of
  its best chunk, so a long memo is not credited once per chunk.
- K2 "first" counts only searches where the memory was ranked first.
- K3 reports credit `used` and `wrong` to the memories they name, split by effect, and count
  task success only where it was reported.
- K4 the summary counts every search beside every report, so a reader can see how much of the
  page is testimony; another tenant's rows are never counted.
- K5 the report list filtered to one memory returns only reports naming it.

Red proof, 2026-10-07, each mutation alone against `recall/dashboard/db.py`, failing in the named
assertion (JUnit XML), then restored and green:
- Q1 (K1) `per_search` grouped by `(event_id, source, rank)`: the three-chunk memo counted 3.
- Q2 (K2) `filter (where rank = 1)` changed to `rank <= 2`: the second-ranked memo counted first.
- Q3 (K3) `wrong` reports credited to `used`: the misleading memo showed as used.
- Q4 (K4) the summary's `tenant_id = %s` on reports dropped: the other tenant's report counted.
- Q5 (K5) the source filter ignored: every report came back.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from recall.dashboard import db as dbq
from tests.conftest import TEST_DSN, requires_db

A, B, C = "file:///m/a.md", "file:///m/b.md", "file:///m/c.md"


def _hit(source: str, verdict: str = "ok") -> dict[str, str]:
    return {"source": source, "verdict": verdict}


@pytest.fixture
def tenant(make_store) -> Iterator[str]:
    make_store(8)  # applies the migrations that create recall_audit_events
    name = "ctl-" + uuid.uuid4().hex[:10]
    other = name + "-other"
    rows = [
        (name, "search_decision", {"query": "q1", "hits": [_hit(A), _hit(B), _hit(A), _hit(A)]}),
        (name, "search_decision", {"query": "q2", "hits": [_hit(B), _hit(A, "superseded"), _hit(C, "low_confidence")]}),
        (name, "search_refusal", {"query": "q3"}),
        (name, "use_report", {"task": "t1", "effect": "helped", "used": [A], "wrong": [], "task_succeeded": True}),
        (name, "use_report", {"task": "t2", "effect": "misled", "used": [B], "wrong": [C], "task_succeeded": False}),
        (name, "use_report", {"task": "t3", "effect": "no_difference", "used": [A, B], "wrong": [], "task_succeeded": None}),
        (other, "use_report", {"task": "elsewhere", "effect": "helped", "used": [A], "wrong": []}),
        (other, "search_decision", {"query": "q", "hits": [_hit(A)]}),
    ]
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for i, (who, kind, payload) in enumerate(rows):
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (who,))
            conn.execute(
                "INSERT INTO recall_audit_events (tenant_id, event_id, event_type, actor, payload) VALUES (%s, %s, %s, 'test', %s)",
                (who, f"evt_{i}_{uuid.uuid4().hex[:6]}", kind, json.dumps(payload)),
            )
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for who in (name, other):
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (who,))
            conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (who,))


@pytest.fixture
def db() -> dbq.DashboardDB:
    return dbq.DashboardDB(TEST_DSN)


@requires_db
def test_retrievals_count_searches_not_chunks_and_first_place(db: dbq.DashboardDB, tenant: str) -> None:
    memos = {m["source"]: m for m in dbq.control(db, tenant)["memos"]}
    assert memos[A]["retrieved"] == 2, "a memory returned as three chunks of one search counted more than once"
    assert memos[A]["first"] == 1 and memos[B]["first"] == 1 and memos[C]["first"] == 0
    assert memos[A]["superseded"] == 1 and memos[C]["low_confidence"] == 1


@requires_db
def test_reports_credit_the_memories_they_name(db: dbq.DashboardDB, tenant: str) -> None:
    memos = {m["source"]: m for m in dbq.control(db, tenant)["memos"]}
    assert (memos[A]["used"], memos[A]["helped"], memos[A]["no_difference"]) == (2, 1, 1)
    assert (memos[C]["used"], memos[C]["wrong"], memos[C]["misled"]) == (0, 1, 1)
    assert (memos[B]["used"], memos[B]["succeeded"], memos[B]["failed"]) == (2, 0, 1)
    assert memos[A]["succeeded"] == 1


@requires_db
def test_the_summary_counts_this_tenant_only(db: dbq.DashboardDB, tenant: str) -> None:
    summary = dbq.control(db, tenant)["summary"]
    assert (summary["searches"], summary["answered"]) == (3, 2)
    assert (summary["reports"], summary["helped"], summary["misled"], summary["no_difference"]) == (3, 1, 1, 1)
    assert (summary["succeeded"], summary["failed"]) == (1, 1)


@requires_db
def test_the_report_list_filters_to_one_memory(db: dbq.DashboardDB, tenant: str) -> None:
    assert len(dbq.use_reports(db, tenant)) == 3
    only_c = dbq.use_reports(db, tenant, source=C)
    assert [r["task"] for r in only_c] == ["t2"]
