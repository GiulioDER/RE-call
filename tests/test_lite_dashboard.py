"""The dashboard's database pages over a lite store, written by the real MCP server.

No fakes on either side: the MCP server indexes, searches (with the decision ledger on) and takes
agent reports into one SQLite file, then the dashboard reads that file, read-only, and renders its
pages. So what is pinned is the whole local loop a user without Postgres has.

Invariants and the failure each one catches:
- W1 the overview names the lite store, its one tenant (chosen by default, not `memory`) and its
  certified calibration.
- W2 every search the server recorded is listed, and one opens to its hits.
- W3 the Control page counts retrievals and agent reports per memo from the lite ledger, and a
  memo's page lists the reports naming it.
- W4 the dashboard opens the file read-only: a page view leaves the file's bytes unchanged.
- W5 `recall dashboard` uses a lite store named by `RECALL_DSN`, and never a Postgres DSN there.

Red proof, 2026-10-08, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green:
- V1 (W1) `DashboardApp.handle` falling back to `DEFAULT_TENANT` instead of the app's
  `default_tenant` when no tenant is chosen: "the lite store's tenant was not shown by default".
  It survived twice first: a mutation of the same fallback in `_current_tenant` is unreachable
  from a request (`handle` sets the tenant first), and the test then asserted only "48 chunks",
  which the overview prints for every tenant's card whichever is selected. W1 now asserts the
  selected card.
- V2 (W2) `lite_db.searches` reading `use_report` rows instead of search events: "a recorded
  search is missing". It SURVIVED a first version of the test, because an agent report carried
  the same query text as the search it looked for; W2 now looks for a query no report repeats.
- V3 (W3) `lite_db.control` counting every hit of a search rather than one retrieval per memo:
  "two chunks of one memo in one search counted as two retrievals".
- V4 (W4) `?mode=ro` dropped from `lite_db.connect`: `DID NOT RAISE OperationalError`.
- V5 (W5) `_database` accepting any DSN from the environment: "a Postgres DSN in RECALL_DSN was
  used by the dashboard".
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from urllib.parse import urlencode
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from recall.cli_commands.dashboard_cmd import _database
from recall.dashboard.db import DashboardDB
from recall.dashboard.lite_db import file_tenant
from recall.dashboard.server import DashboardApp
from tests.test_lite_calibration import _NOUNS, _word, _write_memos
from tests import test_lite_mcp
from tests.test_lite_mcp import _serve

lite_env = test_lite_mcp.lite_env  # the MCP tests' fixture: a lite DSN and an offline embedder
HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
QUESTION = f"who services the {_word(7, 1)} {_NOUNS[7]}"
#: Searched, and named by no agent report, so only the search ledger can put it on a page.
SECOND = f"who services the {_word(3, 1)} {_NOUNS[3]}"


@pytest.fixture
def lite_store(lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """A lite store with 48 memos, a certified calibration, two recorded searches and two reports."""
    monkeypatch.setenv("RECALL_DECISION_LEDGER", "1")
    env = {**lite_env, "RECALL_DECISION_LEDGER": "1"}
    _write_memos(tmp_path / "mem", 48)

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        await call("recall_index", path=str(tmp_path / "mem"))
        await call("recall_search", query=QUESTION)
        await call("recall_search", query=SECOND)
        await call("recall_report_use", task="find the crew", effect="helped", used=["memo-007.md"], query=QUESTION)
        await call("recall_report_use", task="second look", effect="no_difference", used=["memo-007.md"])

    _serve(env, body)
    return env["RECALL_DSN"], tmp_path / "store" / "memory.db"


def _app(root: Path, dsn: str) -> DashboardApp:
    return DashboardApp(root, port=8765, token="tok", db=DashboardDB(dsn), default_tenant=file_tenant(dsn))


def _page(app: DashboardApp, path: str) -> str:
    response = app.handle("GET", path, SIGNED)
    assert response.status == 200, (path, response.status, response.body[:300])
    return response.body.decode()


def test_the_overview_shows_the_lite_store(lite_store: tuple[str, Path], tmp_path: Path) -> None:
    """W1."""
    dsn, _ = lite_store
    page = _page(_app(tmp_path / "mem", dsn), "/overview")
    assert "SQLite" in page
    assert "48 chunks" in page
    assert "class='card tenant on'" in page, "the lite store's tenant was not shown by default"
    assert "certified" in page and "not certified" not in page


def test_recorded_searches_are_listed_and_open(lite_store: tuple[str, Path], tmp_path: Path) -> None:
    """W2."""
    dsn, _ = lite_store
    app = _app(tmp_path / "mem", dsn)
    listing = _page(app, "/retrieval")
    assert SECOND in listing, "a recorded search is missing"
    with sqlite3.connect(lite_store[1]) as raw:
        event_id = raw.execute("SELECT event_id FROM audit_events WHERE event_type = 'search_decision' "
                               "AND json_extract(payload, '$.query') = ?", (QUESTION,)).fetchone()[0]
    event = _page(app, f"/retrieval/event?id={event_id}")
    assert "memo-007.md" in event


def test_control_counts_retrievals_and_reports_per_memo(lite_store: tuple[str, Path], tmp_path: Path) -> None:
    """W3."""
    from recall.dashboard import db as dbq

    dsn, _ = lite_store
    data = dbq.control(DashboardDB(dsn), file_tenant(dsn))
    assert data["summary"]["searches"] == 2 and data["summary"]["reports"] == 2
    target = next(m for m in data["memos"] if m["source"].endswith("memo-007.md"))
    assert target["retrieved"] == 1, f"one search returning memo-007 counted {target['retrieved']} retrievals"
    assert target["used"] == 2 and target["helped"] == 1 and target["no_difference"] == 1
    page = _page(_app(tmp_path / "mem", dsn), "/control/memo?" + urlencode({"source": target["source"]}))
    assert "find the crew" in page and "second look" in page


def test_one_search_counts_once_per_memo_at_its_best_rank(tmp_path: Path) -> None:
    """W3, the counting rule itself: two chunks of one memo in one result are one retrieval."""
    from recall.dashboard import lite_db
    from recall.lite import LiteStore

    with LiteStore(tmp_path / "m.db", dim=4) as store:
        hits = [{"source": "a.md", "verdict": "ok"}, {"source": "a.md", "verdict": "superseded"}, {"source": "b.md", "verdict": "ok"}]
        store.append_audit_event("search_decision", {"query": "q", "hits": hits})
    memos = {m["source"]: m for m in lite_db.control("sqlite:///" + (tmp_path / "m.db").as_posix(), "default")["memos"]}
    assert memos["a.md"]["retrieved"] == 1, "two chunks of one memo in one search counted as two retrievals"
    assert (memos["a.md"]["first"], memos["a.md"]["ok"], memos["a.md"]["superseded"]) == (1, 1, 1)
    assert (memos["b.md"]["retrieved"], memos["b.md"]["first"]) == (1, 0)


def test_page_views_leave_the_file_unchanged(lite_store: tuple[str, Path], tmp_path: Path) -> None:
    """W4."""
    dsn, path = lite_store
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    app = _app(tmp_path / "mem", dsn)
    for page in ("/overview", "/retrieval", "/control", "/activity", "/health", "/"):
        app.handle("GET", page, SIGNED)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before, "a page view changed the store file"
    from recall.dashboard import lite_db

    with lite_db.connect(dsn) as c, pytest.raises(sqlite3.OperationalError):
        c.execute("CREATE TABLE written_by_the_dashboard (x)")


def test_the_dashboard_finds_a_lite_store_but_not_a_postgres_dsn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """W5."""
    monkeypatch.delenv("RECALL_SERVING_DSN", raising=False)
    monkeypatch.setenv("RECALL_DSN", "sqlite:///" + (tmp_path / "m.db").as_posix())
    found = _database("")
    assert found is not None and found.lite
    monkeypatch.setenv("RECALL_DSN", "postgresql://recall:secret@localhost:5432/recall")
    assert _database("") is None, "a Postgres DSN in RECALL_DSN was used by the dashboard"


def test_ledger_rows_are_json(lite_store: tuple[str, Path]) -> None:
    """Sanity: the payloads the dashboard decodes are what the server wrote."""
    with sqlite3.connect(lite_store[1]) as raw:
        kinds = {row[0] for row in raw.execute("SELECT event_type FROM audit_events")}
        payloads = [json.loads(row[0]) for row in raw.execute("SELECT payload FROM audit_events")]
    assert {"search_decision", "use_report"} <= kinds
    assert all(isinstance(p, dict) for p in payloads)
