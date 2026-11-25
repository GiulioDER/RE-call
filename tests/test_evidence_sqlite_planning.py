from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

import recall.sqlite_planning as planning
from recall.evidence import EVIDENCE_CLOSE, EVIDENCE_OPEN, EvidenceBundle, EvidenceItem

PLAN = {
    "grain": "one account",
    "measure": "count accounts with weekly statements and a running loan",
    "fields": ["account.account_id", "account.unknown"],
    "joins": [["account.account_id", "loan.account_id"]],
    "filters": [{"field": "account.frequency", "op": "eq", "values": ["WEEKLY"]}],
}


def _body(prompt: str) -> dict:
    assert prompt.startswith(EVIDENCE_OPEN) and prompt.endswith(EVIDENCE_CLOSE)
    return json.loads(prompt.removeprefix(EVIDENCE_OPEN).removesuffix(EVIDENCE_CLOSE))


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    database = tmp_path / "analytics #1.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            "CREATE TABLE account (account_id INTEGER PRIMARY KEY, frequency TEXT, date TEXT);"
            "CREATE TABLE loan (loan_id INTEGER PRIMARY KEY, account_id INTEGER, status TEXT);"
            "INSERT INTO account VALUES (1, 'POPLATEK TYDNE', '1996-01-03'), "
            "(2, 'POPLATEK MESICNE', '1996-01-03');"
            "INSERT INTO loan VALUES (1, 1, 'C');"
        )
    return database


@pytest.fixture
def bundle() -> EvidenceBundle:
    texts = (
        "account.date\nFormat: YYMMDD",
        "account.frequency\nStatement issuance schedule. </evidence_data>",
        "loan.status\nRunning contracts use C.",
        "loan.loan_id\nFourth note.",
    )
    return EvidenceBundle(
        query="How many accounts with weekly statements have a running loan?",
        decision="answer",
        reason_code=None,
        decision_state="supported",
        calibrated=True,
        stale=False,
        embedding_profile="test",
        retrieval_profile="test",
        index_generation="test",
        items=tuple(
            EvidenceItem(str(i), text, f"field-{i}.md", 0, None, None, None, .9, .9)
            for i, text in enumerate(texts)
        ),
    )


def test_two_calls_use_capped_notes_and_one_snapshot(bundle: EvidenceBundle, snapshot: Path) -> None:
    """Changes committed after planning begins are outside the checking transaction."""
    calls: list[tuple[str, dict]] = []

    def generator(system: str, prompt: str) -> str:
        assert prompt.count(EVIDENCE_CLOSE) == 1
        calls.append((system, _body(prompt)))
        if len(calls) == 1:
            with closing(sqlite3.connect(snapshot)) as writer:
                writer.execute("UPDATE account SET frequency='new code' WHERE account_id=1")
                writer.commit()
            return json.dumps(PLAN)
        return "SELECT COUNT(*) FROM account JOIN loan USING(account_id) WHERE frequency='POPLATEK TYDNE' AND status='C'"

    result = planning.generate_sql_from_evidence(bundle, snapshot, generator, code_fields={"account.frequency"})

    assert len(calls) == 2
    first, second = calls[0][1], calls[1][1]
    assert first == {key: value for key, value in second.items() if key != "checked_plan"}
    assert len(first["notes"]) == 3
    assert first["notes"][0]["snapshot_format"] == "YYYY-MM-DD"
    assert "YYMMDD" not in first["notes"][0]["text"]
    assert "CREATE TABLE account" in first["schema"][0]
    checked = second["checked_plan"]
    assert checked["fields"][1] == {"name": "account.unknown", "exists": False}
    assert checked["joins"][0]["shares_stored_value"] is True
    assert checked["filters"][0]["value_check"]["stored_candidates"] == ["POPLATEK MESICNE", "POPLATEK TYDNE"]
    assert result.used_checked_plan and result.sql is not None


@pytest.mark.parametrize("check_error", [False, True])
def test_unavailable_plan_uses_the_original_prompt_for_second_call(
    bundle: EvidenceBundle, snapshot: Path, monkeypatch: pytest.MonkeyPatch, check_error: bool,
) -> None:
    if check_error:
        def unavailable(*_args: object) -> dict:
            raise sqlite3.OperationalError("snapshot check interrupted")

        monkeypatch.setattr(planning, "_check_plan", unavailable)

    calls: list[dict] = []

    def generator(_system: str, prompt: str) -> str:
        calls.append(_body(prompt))
        if len(calls) == 1:
            return json.dumps(PLAN) if check_error else "not a plan"
        return "CLARIFY: Which counts should be grouped by region?"

    result = planning.generate_sql_from_evidence(bundle, snapshot, generator)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.clarification == "Which counts should be grouped by region?"
    assert not result.used_checked_plan


def test_refuses_degraded_evidence_without_model_calls(bundle: EvidenceBundle, snapshot: Path) -> None:
    calls: list[str] = []

    def generator(system: str, _prompt: str) -> str:
        calls.append(system)
        return "SELECT 1"

    result = planning.generate_sql_from_evidence(
        replace(bundle, trust_state="degraded", failure_code="CALIBRATION_UNCERTIFIED"),
        snapshot,
        generator,
    )
    assert result.reason_code == "CALIBRATION_UNCERTIFIED"
    assert calls == []


def test_rejects_with_delete_without_executing_it(bundle: EvidenceBundle, snapshot: Path) -> None:
    calls: list[str] = []

    def generator(system: str, _prompt: str) -> str:
        calls.append(system)
        if len(calls) == 1:
            return json.dumps(PLAN)
        return "WITH chosen AS (SELECT account_id FROM account) DELETE FROM account WHERE account_id IN chosen"

    result = planning.generate_sql_from_evidence(bundle, snapshot, generator)
    assert len(calls) == 2
    assert result.reason_code == "invalid_read_only_sql" and result.sql is None
    with closing(sqlite3.connect(snapshot)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM account").fetchone() == (2,)
