"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_reconcile_work_cca_981.py::test_cli_reconcile_bounds_payload_batches_and_files_every_event: baseline assertion RED, repaired GREEN."""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import uuid4

import pytest

from recall.cli import main
from recall.fact_current import PostgresCurrentFacts, PostgresTables
from recall.fact_ledger import PostgresFactLedger
from recall.provenance_controller import DecisionCode, FactEvent
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW

pytestmark = requires_db
EVENT_COUNT = 1001
BATCH_LIMIT = 1000


@pytest.mark.timeout(400)
def test_cli_reconcile_bounds_payload_batches_and_files_every_event(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The original backlog fetch must fail the observer's explicit allocation assertion."""
    request.getfixturevalue("make_store")(4)
    monkeypatch.setenv("RECALL_FACT_WRITE_DSN", TEST_DSN)
    monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", "1")
    monkeypatch.setenv("RECALL_IGNORE_BROKEN_DOTENV", "1")
    tenant = f"reconcile-work-{uuid4().hex}"
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant, materialize=False)
    for index in range(EVENT_COUNT):
        ledger.record_decision(
            tenant_id=tenant, generation_id="generation-reconcile-work",
            request_id=f"missing-{index}", writer="reconcile-work-test",
            decision_code=DecisionCode.FRESH_SEARCH_INSUFFICIENT, fact=None, now=NOW,
        )
    facts = PostgresCurrentFacts(TEST_DSN, tenant_id=tenant)
    before = facts.status()
    assert before.ledger_events == EVENT_COUNT and before.filed_events == 0, before

    observed_batch_sizes: list[int] = []
    original_events = PostgresTables.events

    def observe_events(tables: PostgresTables, event_ids: Sequence[str] | None) -> list[FactEvent]:
        if tables.tenant_id == tenant:
            # None requests the entire tenant history, which is also too large in this fixture.
            batch_size = EVENT_COUNT if event_ids is None else len(event_ids)
            observed_batch_sizes.append(batch_size)
            assert batch_size <= BATCH_LIMIT, (
                f"reconciliation requested {batch_size} ledger payloads before batching; "
                f"the payload batch limit is {BATCH_LIMIT}"
            )
        return original_events(tables, event_ids)

    # Verification intentionally replays the complete history, so observe only the CLI operation.
    with monkeypatch.context() as observer:
        observer.setattr(PostgresTables, "events", observe_events)
        main([
            "--serving-dsn", TEST_DSN, "--migration-dsn", TEST_DSN, "--tenant", tenant,
            "provenance", "rebuild", "--reconcile",
        ])
    payload = json.loads(capsys.readouterr().out)
    assert observed_batch_sizes and sum(observed_batch_sizes) >= EVENT_COUNT, observed_batch_sizes
    assert payload["action"] == "reconcile" and payload["reconciled"] == EVENT_COUNT, payload
    assert payload["current"] is True, payload
    after = facts.status()
    assert after.ledger_events == EVENT_COUNT and after.filed_events == EVENT_COUNT, after
    assert after.current
    verification = facts.verify()
    assert verification.matches, verification.to_payload()
