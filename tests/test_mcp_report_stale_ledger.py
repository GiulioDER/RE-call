"""`recall_report_stale` on a server whose memo files are not on disk: the report reaches the audit ledger.

The served deployment runs the MCP server on a remote host, in a serving checkout that holds no
memos, so the sidecar path refused every report there and a dashboard on the workstation could not
have read the sidecar anyway. A database-backed store now takes the report as one `stale_report`
row in `recall_audit_events`, checked against the text search served.

Invariants and the failure each one catches:
- L1 with no memo files under the root, a report naming served sources with verbatim quotes is one
  `stale_report` row carrying the served names, both quotes, the client and the claim key, and
  nothing is created under the root (no `.recall` litter in a serving checkout).
- L2 a quote that is not verbatim in the served text is refused and nothing is written.
- L3 a name the served generation does not hold is refused and nothing is written.
- L4 a store that cannot return served text refuses an unchecked report rather than storing it.
- L5 with the files present and a database-backed store, the report goes to both places; when the
  served generation lacks the memo, the sidecar still takes it and the result says why the ledger
  did not.
- L6 a quote longer than `MAX_QUOTE_CHARS` is refused before anything is written.
- L7 the registered tool reaches the ledger path, end to end through the server's coroutine.
- L8 `GenerationStore.source_chunk_texts` reads one source of the pinned generation, in chunk order.

Red proof, 2026-10-06, each mutation alone against the named production line, run through
pytest with JUnit XML, then restored and green:
- M1 (L1) `_database_backed` returning False: `test_a_report_with_no_files_here_reaches_the_ledger`
  raised StaleReportRefused "'recall/old.md' is not a file in the corpus", which is what the served
  tool did on a remote serving host before this change; L2, L4, L5 and L7 fail with it.
- M2 (L1) an unconditional `StaleReportQueue(default_queue_path(root)).close()` before the names are
  resolved: the same test, AssertionError "the report left a sidecar in a root that holds no memos".
- M3 (L2) `if not grounded(quote, text)` in `_record_served` made `if not True`:
  `test_a_quote_not_in_the_served_text_is_refused`, DID NOT RAISE StaleReportRefused.
- M4 (L3) both `resolve_sources` calls in `_record_served` replaced by the names as given:
  `test_a_name_the_served_generation_lacks_is_refused`, the regex did not match ("current_quote is
  not verbatim in recall/invented.md"): the unknown name got as far as the quote check.
- M5 (L4) `if checked_locally: continue` made an unconditional `continue`:
  `test_a_store_that_cannot_show_served_text_refuses_an_unchecked_report`, DID NOT RAISE.
- M6 (L5) `if report is None: raise` after `_record_served` made an unconditional `raise`:
  `test_files_here_and_a_database_store_record_both_or_say_why_not`, StaleReportRefused "'new.md' in
  replacing_source is not a memory in this corpus" escaped although the sidecar had taken it.
- M7 (L6) the `MAX_QUOTE_CHARS` check made `if False`: `test_an_overlong_quote_is_refused`, DID NOT RAISE.
- M8 (L8) `generation_id = %s AND` and its parameter dropped from the `source_chunk_texts` query:
  `test_source_chunk_texts_reads_one_source_of_the_pinned_generation`, the params tuple differed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.generation_store import GenerationStore
from recall.rewrite import claim_key
from recall.stale_reports import StaleReportQueue, StaleReportRefused, default_queue_path
from recall_mcp.stale_reports_api import MAX_QUOTE_CHARS, STALE_REPORT_EVENT, report_stale

OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"
SERVED = {
    "recall/old.md": ["The release is 0.12.0 and", f"{OLD_QUOTE}.\n"],
    "recall/new.md": [f"Release 0.12.0 {NEW_QUOTE}.\n"],
    "recall/other.md": ["Something else entirely, long enough to quote."],
}


class LedgerStore:
    """A database-backed store as `report_stale` sees it: served sources, their text, the ledger."""

    tenant = "memory"

    def __init__(self, served: dict[str, list[str]] | None = None, *, texts: bool = True) -> None:
        self.served = dict(SERVED if served is None else served)
        self.rows: list[tuple[str, dict[str, Any], str]] = []
        if texts:
            self.source_chunk_texts = lambda source: list(self.served.get(source, []))

    def source_content_hashes(self) -> dict[str, str]:
        return {source: f"h-{i}" for i, source in enumerate(self.served)}

    def chunks_for_source(self, source: str) -> list[Any]:
        raise NotImplementedError("chunks_for_source reads the legacy chunk table only")

    def append_audit_event(self, event_type: str, payload: dict[str, Any], *, actor: str = "serving", **_: Any) -> str:
        self.rows.append((event_type, payload, actor))
        return f"evt-{len(self.rows)}"


@pytest.fixture
def empty_root(tmp_path: Path) -> Path:
    root = tmp_path / "serving-checkout"
    root.mkdir()
    (root / "README.md").write_text("Not a memo store.\n", encoding="utf-8")
    return root


def _report(store: Any, root: Path, **overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "stale_source": "recall/old.md",
        "replacing_source": "recall/new.md",
        "stale_quote": OLD_QUOTE,
        "current_quote": NEW_QUOTE,
        "task": "which release is current?",
    }
    args.update(overrides)
    return report_stale(store, env={"RECALL_INDEX_ROOT": str(root)}, **args)


def test_a_report_with_no_files_here_reaches_the_ledger(empty_root: Path) -> None:
    store = LedgerStore()
    out = _report(store, empty_root)
    ((event_type, payload, actor),) = store.rows
    assert event_type == STALE_REPORT_EVENT == "stale_report"
    assert actor == "agent-report"
    assert (payload["stale_source"], payload["replacing_source"]) == ("recall/old.md", "recall/new.md")
    assert (payload["stale_quote"], payload["current_quote"]) == (OLD_QUOTE, NEW_QUOTE)
    assert payload["client"] == "mcp:memory" and payload["task"] == "which release is current?"
    assert payload["claim_key"] == claim_key("supersedes", "recall/old.md", "recall/new.md")
    assert out["recorded_in"] == ["audit_ledger"] and out["event_id"] == "evt-1" and out["report_count"] is None
    assert not (empty_root / ".recall").exists(), "the report left a sidecar in a root that holds no memos"


def test_a_file_name_resolves_to_the_one_served_source(empty_root: Path) -> None:
    store = LedgerStore()
    _report(store, empty_root, stale_source="old.md", replacing_source="new.md")
    assert store.rows[0][1]["stale_source"] == "recall/old.md"


def test_a_quote_not_in_the_served_text_is_refused(empty_root: Path) -> None:
    store = LedgerStore()
    with pytest.raises(StaleReportRefused, match="current_quote is not verbatim in recall/new.md"):
        _report(store, empty_root, current_quote="this sentence is not in the newer memo at all")
    assert store.rows == []


def test_a_name_the_served_generation_lacks_is_refused(empty_root: Path) -> None:
    store = LedgerStore()
    with pytest.raises(StaleReportRefused, match="not a memory in this corpus"):
        _report(store, empty_root, replacing_source="recall/invented.md")
    assert store.rows == []


def test_a_store_that_cannot_show_served_text_refuses_an_unchecked_report(empty_root: Path) -> None:
    store = LedgerStore(texts=False)
    with pytest.raises(StaleReportRefused, match="cannot be checked against the served text"):
        _report(store, empty_root)
    assert store.rows == []


def test_files_here_and_a_database_store_record_both_or_say_why_not(tmp_path: Path) -> None:
    (tmp_path / "old.md").write_text(f"The release is 0.12.0 and {OLD_QUOTE}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    both = LedgerStore()
    out = _report(both, tmp_path, stale_source="old.md", replacing_source="new.md")
    assert out["recorded_in"] == ["sidecar", "audit_ledger"] and out["report_count"] == 1
    assert (out["stale_source"], out["replacing_source"]) == ("old.md", "new.md")
    assert len(both.rows) == 1

    unindexed = LedgerStore({"recall/old.md": SERVED["recall/old.md"]})
    out = _report(unindexed, tmp_path, stale_source="old.md", replacing_source="new.md")
    assert out["recorded_in"] == ["sidecar"] and out["report_count"] == 2
    assert "not a memory in this corpus" in out["audit_ledger"]
    assert unindexed.rows == []
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        (report,) = queue.list()
    assert report.report_count == 2


def test_an_overlong_quote_is_refused(empty_root: Path) -> None:
    store = LedgerStore()
    with pytest.raises(StaleReportRefused, match=f"quote at most {MAX_QUOTE_CHARS}"):
        _report(store, empty_root, stale_quote=OLD_QUOTE + " " * MAX_QUOTE_CHARS)
    assert store.rows == []


def test_the_registered_tool_records_to_the_ledger(empty_root: Path, monkeypatch) -> None:
    from recall_mcp.server import build_server

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(empty_root))
    store = LedgerStore()
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"store": store, "embedder": None, "calibration": None}))

    async def run() -> str:
        return await tools["recall_report_stale"].fn(
            ctx=ctx, stale_source="recall/old.md", replacing_source="recall/new.md",
            stale_quote=OLD_QUOTE, current_quote=NEW_QUOTE,
        )

    out = json.loads(asyncio.run(run()))
    assert out["recorded_in"] == ["audit_ledger"]
    assert [row[0] for row in store.rows] == [STALE_REPORT_EVENT]


class _Rows:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[str]]:
        return self._rows


def test_source_chunk_texts_reads_one_source_of_the_pinned_generation(monkeypatch) -> None:
    store = object.__new__(GenerationStore)
    store._tenant = "tenant-a"
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Connection:
        def execute(self, sql: str, params: tuple[Any, ...]) -> _Rows:
            calls.append((sql, params))
            return _Rows([("first chunk",), ("second chunk",)])

    monkeypatch.setattr(store, "_generation_id", lambda: "generation-1")
    monkeypatch.setattr(store, "_with_retry", lambda operation: operation(Connection()))

    assert store.source_chunk_texts("recall/old.md") == ["first chunk", "second chunk"]
    ((sql, params),) = calls
    assert params == ("tenant-a", "generation-1", "recall/old.md")
    assert "ORDER BY chunk_ordinal" in sql and "recall_chunks_v1" in sql
    with pytest.raises(ValueError):
        store.source_chunk_texts("")
