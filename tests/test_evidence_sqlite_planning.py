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


@pytest.mark.parametrize("sql", ["SELECT * FROM pragma_optimize", "SELECT * FROM PRAGMA_Optimize(0x10002)"])
def test_rejects_pragma_table_functions(bundle: EvidenceBundle, snapshot: Path, sql: str) -> None:
    """STAKES-001 (PR 1512 audit). A pragma table-valued function runs its PRAGMA only when
    stepped, so EXPLAIN under the authorizer passes it, and `pragma_optimize` then writes
    `sqlite_stat1` on the caller's connection. Red proof: against the pre-fix `_read_authorizer`
    (no `pragma_` refusal) the statement is returned as `sql`."""
    replies = iter(["not a plan", sql])
    result = planning.generate_sql_from_evidence(bundle, snapshot, lambda *_: next(replies))
    assert result.reason_code == "invalid_read_only_sql" and result.sql is None


def test_accepts_sql_fenced_as_sqlite(bundle: EvidenceBundle, snapshot: Path) -> None:
    """BUG-001 (PR 1512 audit). Red proof: against the pre-fix `_unfence` the `sql` alternative
    matches first, the body starts with `ite`, and the reply is refused as invalid."""
    replies = iter(["not a plan", "```sqlite\nSELECT COUNT(*) FROM account\n```"])
    result = planning.generate_sql_from_evidence(bundle, snapshot, lambda *_: next(replies))
    assert result.sql == "SELECT COUNT(*) FROM account"


def test_plan_value_that_cannot_be_bound_falls_back(bundle: EvidenceBundle, snapshot: Path) -> None:
    """BUG-002 (PR 1512 audit). A JSON escape decodes to a lone surrogate, which sqlite3 cannot
    bind. Red proof: against the pre-fix `_parse_plan` the plan is accepted, the value check
    raises UnicodeEncodeError and the call fails instead of falling back."""
    plan = {**PLAN, "filters": [{"field": "account.frequency", "op": "eq", "values": ["\ud800"]}]}
    calls: list[dict] = []

    def generator(_system: str, prompt: str) -> str:
        calls.append(_body(prompt))
        return json.dumps(plan) if len(calls) == 1 else "SELECT COUNT(*) FROM account"

    result = planning.generate_sql_from_evidence(bundle, snapshot, generator)
    assert calls[0] == calls[1] and not result.used_checked_plan
    assert result.sql == "SELECT COUNT(*) FROM account"


def test_sql_that_cannot_be_encoded_is_invalid(bundle: EvidenceBundle, snapshot: Path) -> None:
    """BUG-003 (PR 1512 audit). Red proof: against the pre-fix `_validate_sql` EXPLAIN raises
    UnicodeEncodeError, which escapes `except sqlite3.Error`."""
    replies = iter(["not a plan", "SELECT '\ud800'"])
    result = planning.generate_sql_from_evidence(bundle, snapshot, lambda *_: next(replies))
    assert result.reason_code == "invalid_read_only_sql" and result.sql is None


def test_inbound_snapshot_format_is_not_passed_on(bundle: EvidenceBundle, snapshot: Path) -> None:
    """SEC-001 (PR 1512 audit). A `snapshot_format` the item already carries (from a check of
    another snapshot, or decoded from stored data) must not reach the writer as an observation
    of this snapshot. Red proof: against the pre-fix `_check` a note with no format claim is
    kept unchanged, its inbound `snapshot_format` included."""
    stale = replace(bundle.items[1], snapshot_format="YYMMDD")
    calls: list[dict] = []

    def generator(_system: str, prompt: str) -> str:
        calls.append(_body(prompt))
        return "not a plan" if len(calls) == 1 else "SELECT 1"

    planning.generate_sql_from_evidence(replace(bundle, items=(stale,)), snapshot, generator)
    assert [note.get("snapshot_format") for note in calls[0]["notes"]] == [None]


def _first_call(bundle: EvidenceBundle, snapshot: Path, plan: dict | None = None) -> tuple[dict, dict | None]:
    calls: list[dict] = []

    def generator(_system: str, prompt: str) -> str:
        calls.append(_body(prompt))
        if len(calls) > 1:
            return "SELECT 1"
        return "not a plan" if plan is None else json.dumps(plan)

    result = planning.generate_sql_from_evidence(bundle, snapshot, generator)
    assert calls, f"no model call was made: {result.reason_code}"
    return calls[0], calls[1].get("checked_plan") if len(calls) > 1 else None


def test_note_cap_counts_verified_notes_only(bundle: EvidenceBundle, snapshot: Path) -> None:
    """DAT-004 and DOC-003 (PR 1512 deferred pass). The three-note cap applies to notes that
    pass the snapshot check, so prose and missing fields ranked first do not use up the slots
    of an eligible note ranked fourth. Red proof: against the pre-fix
    `generate_sql_from_evidence` (`bundle.items[:3]` before `_verify_rules`) the call abstains
    with no_verified_database_rules and makes no model call."""
    noise = tuple(
        replace(bundle.items[0], chunk_id=f"n{i}", text=text)
        for i, text in enumerate(("Prose with no field header.", "account.missing\nGone.", "More prose."))
    )
    first, _ = _first_call(replace(bundle, items=(*noise, bundle.items[2], bundle.items[1], bundle.items[3], bundle.items[0])), snapshot)
    assert [note["text"].partition("\n")[0] for note in first["notes"]] == ["loan.status", "account.frequency", "loan.loan_id"]


def test_plan_fields_resolve_without_regard_to_ascii_case(bundle: EvidenceBundle, snapshot: Path) -> None:
    """DAT-002 (PR 1512 deferred pass). SQLite resolves `ACCOUNT.Frequency` to account.frequency,
    so the plan check must not report it missing. Red proof: against the pre-fix `_catalog`
    keys and membership tests (exact case) the field is reported `exists: False` and its
    filter gets no value check."""
    plan = {**PLAN, "fields": ["ACCOUNT.Account_ID"], "joins": [["Account.account_id", "LOAN.account_id"]],
            "filters": [{"field": "ACCOUNT.Frequency", "op": "eq", "values": ["POPLATEK TYDNE"]}]}
    _, checked = _first_call(bundle, snapshot, plan)
    assert checked is not None
    assert checked["fields"] == [{"name": "ACCOUNT.Account_ID", "exists": True}]
    assert checked["joins"][0]["fields_exist"] is True and checked["joins"][0]["shares_stored_value"] is True
    assert checked["filters"][0]["value_check"] == {"status": "all proposed values stored"}


def test_plan_check_sees_generated_columns(bundle: EvidenceBundle, snapshot: Path) -> None:
    """DAT-003 (PR 1512 deferred pass). Red proof: against the pre-fix `_catalog`
    (`pragma_table_info`, which omits generated columns) `account.year` is reported missing."""
    with closing(sqlite3.connect(snapshot)) as conn:
        conn.execute("ALTER TABLE account ADD COLUMN year TEXT GENERATED ALWAYS AS (substr(date, 1, 4)) VIRTUAL")
    plan = {**PLAN, "fields": ["account.year"], "filters": [{"field": "account.year", "op": "eq", "values": ["1996"]}]}
    _, checked = _first_call(bundle, snapshot, plan)
    assert checked is not None and checked["fields"] == [{"name": "account.year", "exists": True}]
    assert checked["filters"][0]["value_check"] == {"status": "all proposed values stored"}


def test_value_check_honours_the_column_collation(bundle: EvidenceBundle, snapshot: Path) -> None:
    """FA-1-1 (PR 1512 deferred pass, fix audit of PERF-002). A stored value matches under the
    column's declared collation, as `column = ?` does. Red proof: against the pass 1
    `_value_check` (`v IN (SELECT column ...)`, where the left operand is the CTE column `v`,
    so its BINARY collation governs) 'poplatek tydne' and 'C' are reported missing."""
    with closing(sqlite3.connect(snapshot)) as conn:
        conn.executescript(
            "CREATE TABLE code (name TEXT COLLATE NOCASE, tag TEXT COLLATE RTRIM);"
            "INSERT INTO code VALUES ('POPLATEK TYDNE', 'C  ');"
        )
    plan = {**PLAN, "fields": [], "joins": [], "filters": [
        {"field": "code.name", "op": "eq", "values": ["poplatek tydne"]},
        {"field": "code.tag", "op": "in", "values": ["C", "D"]},
    ]}
    _, checked = _first_call(bundle, snapshot, plan)
    assert checked is not None
    assert checked["filters"][0]["value_check"] == {"status": "all proposed values stored"}
    assert checked["filters"][1]["value_check"]["missing"] == ["D"]
