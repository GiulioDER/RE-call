"""NUM-002: a refused gate is distinct from an answer and a working abstention.

Red proof: the public retrieval count assertion fails on baseline970dbeb7b.
Canonical payloads pass through the actual summarizer before page rendering.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.dashboard import db as dbq
from recall.dashboard.server import DashboardApp
from tests.test_dashboard_db_pages import FakeCorpus, WHEN

SIGNED = {"Host": "127.0.0.1:8765", "Cookie": "recall_dashboard=tok"}


def test_retrieval_keeps_answer_abstention_and_refusal_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCorpus()
    fake.install(monkeypatch)
    rows = []
    for outcome in ("answered", "abstained", "refused"):
        payload = {"query": outcome + " query", "outcome": outcome, "hits": []}
        if outcome != "refused":
            payload["abstained"] = outcome == "abstained"
        rows.append({"event_id": outcome, "event": "search_refusal" if outcome == "refused" else "search_decision",
                     "actor": "test", "generation": None, "created_at": WHEN, **dbq._summarise(payload)})
    fake.searches["memory"] = rows
    app = DashboardApp(tmp_path, port=8765, token="tok", db=dbq.DashboardDB("postgresql://fake"))
    listing = app.handle("GET", "/retrieval", SIGNED).body.decode()
    assert "<span class='eyebrow'>answered</span><b>1</b>" in listing, "NUM-002: a refusal counted as an answer"
    assert ">refused</span>" in listing
    for outcome in ("answered", "abstained", "refused"):
        detail = app.handle("GET", "/retrieval/event?id=" + outcome, SIGNED).body.decode()
        assert "<dt>outcome</dt><dd>" + outcome in detail
