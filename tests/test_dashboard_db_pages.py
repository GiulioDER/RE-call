"""The database pages: overview, retrieval, a single search, and the corpus half of Activity.

The database is replaced by fakes of the `recall.dashboard.db` query functions, so these tests need
no Postgres; what they pin is the server's side: the session gate, the tenant choice (scoped,
remembered, validated), escaping of agent-written text, and a database that is absent or down.

Invariants and the failure each one catches:
- D1 without a database the memo pages still serve and the database pages say so with 503, not 500.
- D2 a query an agent wrote, and a hit's source, are escaped on the list and on the search page.
- D3 `?tenant=` scopes every query to that tenant and is remembered in a SameSite=Strict cookie.
- D4 a tenant name that is not a slug is ignored, never echoed or stored.
- D5 a search id is looked up within the current tenant, so another tenant's search is a 404.
- D6 a database that stops answering gives the 503 page, not a traceback.
- D7 Activity shows the selected tenant's corpus events beside the memo events.
- D8 the database pages need the session like every other page.

Red proof, 2026-10-06, each mutation alone against `recall/dashboard/server.py`, failing in the
named assertion (JUnit XML), then restored and green:
- M1 (D2) `_e(r['query'][:220])` in `_retrieval_page` rendered raw: `<script>` reached the page.
- M2 (D3) the `Set-Cookie` for the chosen tenant dropped: "the chosen tenant was not remembered".
- M3 (D4) the `TENANT_NAME.fullmatch` check on the query removed: the bad name was queried.
- M4 (D5) `_retrieval_event_page` passed DEFAULT_TENANT instead of the current tenant: 200, not 404.
- M5 (D6) `except dbq.DatabaseUnavailable` in `_overview_page` narrowed to `KeyError`:
  `DatabaseUnavailable` escaped `handle`, which is the defect itself (a 500 for a dead tunnel).
- M6 (D7) the lifecycle merge in `_activity_page` skipped: the corpus event was missing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.dashboard import db as dbq
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
WHEN = datetime(2026, 10, 6, 8, 15, tzinfo=UTC)


def _tenant(name: str, model: str) -> dict[str, Any]:
    return {
        "tenant": name, "generation": f"gen_{name}", "state": "active", "activated_at": WHEN, "updated_at": WHEN,
        "embedder": {"provider": "voyage", "model": model, "dimension": 1024, "profile_id": "p1", "verified": True},
        "chunks": 120, "sources": 12,
        "calibration": {"certified": True, "threshold": 0.5, "separability": 0.97, "published_at": WHEN, "state": "certified"},
    }


def _search(event_id: str, query: str, source: str) -> dict[str, Any]:
    return {
        "event_id": event_id, "event": "search_decision", "actor": "mcp-service", "generation": "gen_memory",
        "created_at": WHEN, "query": query, "outcome": "answered", "abstained": False, "reason": None,
        "trust_state": "trusted", "failure_code": None, "verdict_counts": {"ok": 1},
        "hits": [{"source": source, "verdict": "ok", "confidence": 0.8, "cosine": 0.7, "superseded_by": None, "ord": 0}],
        "k": 5, "stage_ms": None,
    }


class FakeCorpus:
    """Records which tenant each query asked for; `down` makes every call fail like a dead tunnel."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []
        self.down = False
        self.searches = {
            "memory": [_search("e-mem", "<script>alert(1)</script> which port?", "<img src=x onerror=1>.md")],
            "re-call-code-gen": [_search("e-code", "where is the router?", "recall/router.py")],
        }

    def _check(self) -> None:
        if self.down:
            raise dbq.DatabaseUnavailable("connection refused")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def availability(db: DashboardDB) -> dict[str, Any]:
            self._check()
            return {"server_version": "17.10", "role": "reader", "read_only": True, "migrations": [("chunks", "0007")]}

        def tenants(db: DashboardDB) -> list[dict[str, Any]]:
            self._check()
            return [_tenant("memory", "voyage-context-4"), _tenant("re-call-code-gen", "voyage-code-3")]

        def scoped(kind: str, result: Any):
            def query(db: DashboardDB, tenant: str, *args: Any, **kwargs: Any) -> Any:
                self._check()
                self.asked.append((kind, tenant))
                return result(tenant, *args, **kwargs)
            return query

        monkeypatch.setattr(dbq, "availability", availability)
        monkeypatch.setattr(dbq, "tenants", tenants)
        monkeypatch.setattr(dbq, "generations", scoped("generations", lambda t, *a, **k: []))
        monkeypatch.setattr(dbq, "lifecycle_events", scoped("lifecycle", lambda t, *a, **k: [
            {"event": "generation_promoted", "actor": "indexer", "generation": f"gen_{t}", "source": None, "created_at": WHEN}
        ]))
        monkeypatch.setattr(dbq, "searches", scoped("searches", lambda t, *a, **k: self.searches.get(t, [])))
        monkeypatch.setattr(dbq, "search", scoped("search", lambda t, event_id: next(
            (s for s in self.searches.get(t, []) if s["event_id"] == event_id), None)))


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "note.md").write_text("# Note\nThe port is 9090.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeCorpus:
    corpus_db = FakeCorpus()
    corpus_db.install(monkeypatch)
    return corpus_db


@pytest.fixture
def app(corpus: Path, fake: FakeCorpus) -> DashboardApp:
    return DashboardApp(corpus, port=8765, token="tok", db=DashboardDB("postgresql://fake"))


def test_without_a_database_the_memo_pages_serve_and_the_database_pages_say_so(corpus: Path) -> None:
    app = DashboardApp(corpus, port=8765, token="tok")
    assert app.handle("GET", "/graph", SIGNED).status == 200
    for path in ("/overview", "/retrieval", "/retrieval/event?id=x"):
        page = app.handle("GET", path, SIGNED)
        assert page.status == 503, path
        assert "No database is configured" in page.body.decode()


def test_agent_written_query_and_source_are_escaped(app: DashboardApp) -> None:
    listing = app.handle("GET", "/retrieval", SIGNED).body.decode()
    assert "<script>alert(1)</script>" not in listing
    assert "&lt;script&gt;alert(1)&lt;/script&gt; which port?" in listing
    event = app.handle("GET", "/retrieval/event?id=e-mem", SIGNED).body.decode()
    assert "<script>alert(1)</script>" not in event and "<img src=x" not in event
    assert "&lt;img src=x onerror=1&gt;.md" in event


def test_the_chosen_tenant_scopes_the_queries_and_is_remembered(app: DashboardApp, fake: FakeCorpus) -> None:
    chosen = app.handle("GET", "/retrieval?tenant=re-call-code-gen", SIGNED)
    assert ("searches", "re-call-code-gen") in fake.asked
    assert "where is the router?" in chosen.body.decode()
    cookies = [value for name, value in chosen.headers if name == "Set-Cookie" and value.startswith("recall_tenant=")]
    assert len(cookies) == 1, "the chosen tenant was not remembered"
    assert "SameSite=Strict" in cookies[0] and "re-call-code-gen" in cookies[0]
    fake.asked.clear()
    again = app.handle("GET", "/retrieval", {**HOST, "Cookie": "recall_dashboard=tok; recall_tenant=re-call-code-gen"})
    assert fake.asked == [("searches", "re-call-code-gen")]
    assert "where is the router?" in again.body.decode()


def test_a_tenant_that_is_not_a_slug_is_ignored(app: DashboardApp, fake: FakeCorpus) -> None:
    page = app.handle("GET", "/retrieval?tenant=%3Cb%3Eevil%3C%2Fb%3E", SIGNED)
    assert page.status == 200
    assert all(tenant == "memory" for _, tenant in fake.asked)
    assert "evil" not in page.body.decode()
    assert not [value for name, value in page.headers if name == "Set-Cookie" and value.startswith("recall_tenant=")]


def test_a_search_is_looked_up_within_the_current_tenant(app: DashboardApp) -> None:
    assert app.handle("GET", "/retrieval/event?id=e-code&tenant=re-call-code-gen", SIGNED).status == 200
    assert app.handle("GET", "/retrieval/event?id=e-code", SIGNED).status == 404


def test_a_database_that_stops_answering_gives_the_unavailable_page(app: DashboardApp, fake: FakeCorpus) -> None:
    fake.down = True
    for path in ("/overview", "/retrieval", "/retrieval/event?id=e-mem"):
        page = app.handle("GET", path, SIGNED)
        assert page.status == 503, path
        assert "did not answer" in page.body.decode()
    assert app.handle("GET", "/activity", SIGNED).status == 200


def test_activity_shows_the_selected_tenants_corpus_events(app: DashboardApp, fake: FakeCorpus) -> None:
    page = app.handle("GET", "/activity?tenant=re-call-code-gen", SIGNED).body.decode()
    assert ("lifecycle", "re-call-code-gen") in fake.asked
    assert "re-call-code-gen: generation promoted" in page
    assert "gen_re-call-code-gen" in page


def test_the_database_pages_need_the_session(app: DashboardApp, fake: FakeCorpus) -> None:
    for path in ("/overview", "/retrieval", "/retrieval/event?id=e-mem"):
        assert app.handle("GET", path, HOST).status == 403, path
    assert fake.asked == []


def test_switching_corpus_keeps_the_page_you_are_on(app: DashboardApp) -> None:
    """D9: the corpus picker keeps what a page shows (a memo, a filter), except where the thing
    shown belongs to one corpus (a search, a memory's usage), where it drops back to the list.

    Red proof, 2026-10-07: with `keep = {}` for every page in `_tenant_picker` (the rule before
    this test), the reader page's picker linked to `/read?tenant=...` with no memo, which answers
    404; this failed on the first assertion. Restored, green.
    """
    reader = app.handle("GET", "/read?path=note.md", SIGNED).body.decode()
    assert "href='/read?path=note.md&amp;tenant=re-call-code-gen'" in reader, "the picker dropped the memo"
    event = app.handle("GET", "/retrieval/event?id=e-mem", SIGNED).body.decode()
    assert "href='/retrieval?tenant=re-call-code-gen'" in event
    assert "e-mem&amp;tenant" not in event
