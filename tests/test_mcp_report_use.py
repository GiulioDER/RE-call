"""`recall_report_use`: an agent's account of what memory did for one task, kept honest.

Invariants and the failure each one catches:
- U1 a memory the corpus does not hold is refused and nothing is written, so an invented or
  mistyped name cannot inflate a memory's counts.
- U2 a file name resolves to the one stored source it names; a file name two sources share is
  refused rather than credited to either.
- U3 `helped` must say which memory helped, and `misled` which one misled.
- U4 one memory cannot be both used and wrong in one report.
- U5 the row is a `use_report` in the audit ledger, with the resolved sources and bounded text.

Red proof, 2026-10-07, each mutation alone against `recall_mcp/use_reports_api.py`, failing in the
named assertion (JUnit XML), then restored and green:
- R1 (U1) the "not a memory in this corpus" refusal replaced by accepting the name as given:
  `DID NOT RAISE UseReportRefused`.
- R2 (U2) the ambiguity refusal removed: `DID NOT RAISE UseReportRefused`.
- R3 (U3) the `helped` without `used` refusal removed: `DID NOT RAISE UseReportRefused`.
- R4 (U4) the used-and-wrong refusal removed: `DID NOT RAISE UseReportRefused`.
- R5 (U5) `USE_REPORT_EVENT` written as `search_decision`: the row had the wrong event type.
"""

from __future__ import annotations

from typing import Any

import pytest

from recall_mcp.use_reports_api import MAX_TASK_CHARS, USE_REPORT_EVENT, UseReportRefused, report_use

SOURCES = {
    "file:///memory/port-is-9090.md": "h1",
    "file:///memory/port-was-8080.md": "h2",
    "file:///memory/a/readme.md": "h3",
    "file:///memory/b/readme.md": "h4",
}


class FakeStore:
    tenant = "memory"

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any], str]] = []

    def source_content_hashes(self) -> dict[str, str]:
        return dict(SOURCES)

    def append_audit_event(self, event_type: str, payload: dict[str, Any], *, actor: str = "serving", **_: Any) -> str:
        self.rows.append((event_type, payload, actor))
        return f"evt-{len(self.rows)}"


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def test_a_memory_the_corpus_does_not_hold_is_refused(store: FakeStore) -> None:
    with pytest.raises(UseReportRefused, match="not a memory in this corpus"):
        report_use(store, task="fix the port", effect="helped", used=["port-is-9090.md", "invented.md"])
    assert store.rows == [], "a report naming an unknown memory was written"


def test_a_file_name_resolves_and_a_shared_one_is_refused(store: FakeStore) -> None:
    result = report_use(store, task="fix the port", effect="helped", used=["port-is-9090.md"])
    assert result["used"] == ["file:///memory/port-is-9090.md"]
    with pytest.raises(UseReportRefused, match="matches 2 memories"):
        report_use(store, task="read the docs", effect="no_difference", used=["readme.md"])
    assert len(store.rows) == 1, "an ambiguous name was credited to one of the sources"


def test_helped_and_misled_must_name_the_memory(store: FakeStore) -> None:
    with pytest.raises(UseReportRefused, match="which one helped"):
        report_use(store, task="fix the port", effect="helped", wrong=["port-was-8080.md"])
    with pytest.raises(UseReportRefused, match="which one misled"):
        report_use(store, task="fix the port", effect="misled", used=["port-is-9090.md"])
    assert store.rows == [], "an effect was recorded without the memory it is about"


def test_one_memory_cannot_be_both_used_and_wrong(store: FakeStore) -> None:
    with pytest.raises(UseReportRefused, match="both used and wrong"):
        report_use(store, task="fix the port", effect="no_difference", used=["port-is-9090.md"], wrong=["file:///memory/port-is-9090.md"])
    assert store.rows == [], "a contradictory report was written"


def test_the_row_is_a_bounded_use_report(store: FakeStore) -> None:
    report_use(
        store, task="x" * (MAX_TASK_CHARS + 50), effect="misled", used=["port-is-9090.md"],
        wrong=["port-was-8080.md"], task_succeeded=True, query="  which port?  ", note="",
    )
    ((event_type, payload, actor),) = store.rows
    assert event_type == USE_REPORT_EVENT == "use_report"
    assert actor == "agent-report"
    assert payload["used"] == ["file:///memory/port-is-9090.md"]
    assert payload["wrong"] == ["file:///memory/port-was-8080.md"]
    assert len(payload["task"]) == MAX_TASK_CHARS
    assert payload["query"] == "which port?" and payload["note"] is None and payload["task_succeeded"] is True


@pytest.mark.parametrize("effect", ["", "great", "HELPED"])
def test_an_unknown_effect_is_refused(store: FakeStore, effect: str) -> None:
    with pytest.raises(UseReportRefused, match="effect must be one of"):
        report_use(store, task="fix the port", effect=effect, used=["port-is-9090.md"])
