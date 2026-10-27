"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_reconcile_delivery_cca_981.py::test_reconcile_preserves_undelivered_external_work[pending]: baseline assertion RED, repaired GREEN.
test_fact_reconcile_delivery_cca_981.py::test_reconcile_preserves_undelivered_external_work[expired]: baseline assertion RED, repaired GREEN.
test_fact_reconcile_delivery_cca_981.py::test_reconcile_preserves_undelivered_external_work[failed]: baseline assertion RED, repaired GREEN.
test_fact_reconcile_delivery_cca_981.py::test_reconcile_repairs_current_rows_after_external_delivery_was_applied: baseline assertion RED, repaired GREEN."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from recall.cli import main
from recall.fact_current import PostgresCurrentFacts
from recall.fact_ledger import PostgresFactLedger, PostgresMaterializationOutbox
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import (
    DecisionCode,
    FactApplicationRequest,
    FactEvent,
    MaterializationRecovery,
    ProvenanceController,
)
from recall.types import AtomicFact
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW, evidence_card

pytestmark = requires_db


@dataclass(frozen=True)
class FileMaterializer:
    """A separate, persistent fact store, idempotent by ledger event id."""

    directory: Path

    def path(self, event: FactEvent) -> Path:
        return self.directory / f"{event.event_id}.json"

    def materialize(self, event: FactEvent) -> None:
        self.path(event).write_text(
            json.dumps({"event_id": event.event_id, "fact": event.fact.to_payload() if event.fact else None}),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class DeliveryCase:
    tenant: str
    event: FactEvent
    outbox: PostgresMaterializationOutbox
    facts: PostgresCurrentFacts
    sink: FileMaterializer

    def recovery(self) -> MaterializationRecovery:
        return MaterializationRecovery(tenant_id=self.tenant, outbox=self.outbox, materializer=self.sink)


@pytest.fixture
def delivery_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeliveryCase:
    # Every connection, including a future writer-DSN resolver, stays on the suite's database.
    monkeypatch.setenv("RECALL_FACT_WRITE_DSN", TEST_DSN)
    monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", "1")
    monkeypatch.setenv("RECALL_IGNORE_BROKEN_DOTENV", "1")
    tenant = f"reconcile-delivery-{uuid4().hex}"
    generation = "generation-delivery"
    fact = AtomicFact(namespace="memory", subject="service:api", predicate="owner", object="team:platform")
    card = evidence_card(fact, tenant=tenant, generation=generation, chunk="chunk-owner", source="owner.md")
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant, materialize=False)
    decision = ProvenanceController(
        tenant_id=tenant, generation_id=generation, cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="delivery-test",
    ).apply_fact(FactApplicationRequest(fact, (card.card_id,), "seed"))
    assert decision.code == DecisionCode.APPLIED and decision.event is not None
    outbox = PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant, lease_seconds=1)
    outbox.enqueue(decision.event)
    facts = PostgresCurrentFacts(TEST_DSN, tenant_id=tenant)
    assert not facts.status().current
    return DeliveryCase(tenant, decision.event, outbox, facts, FileMaterializer(tmp_path))


def reconcile_cli(case: DeliveryCase, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    main([
        "--serving-dsn", TEST_DSN, "--migration-dsn", TEST_DSN, "--tenant", case.tenant,
        "provenance", "rebuild", "--reconcile",
    ])
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("delivery_state", ("pending", "failed", "expired"))
def test_reconcile_preserves_undelivered_external_work(
    delivery_case: DeliveryCase, delivery_state: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Invariant: repairing current rows neither acknowledges nor consumes external delivery.

    Historical baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. Target symbol:
    recall.cli_commands.provenance_cmd._rebuild_postgres. Each parameter must fail at
    'current-table repair consumed external delivery state': that implementation repairs the
    tables but marks the shared external row applied. Execution and assertion red proof: observed in the module proof receipt.
    After repair, the real external recovery must still write the event to its separate store.
    """
    case = delivery_case
    expected_status = "pending"
    if delivery_state != "pending":
        # A past claim creates an expired lease without sleeping or editing queue rows.
        (leased,) = case.outbox.claim(
            tenant_id=case.tenant, event_id=case.event.event_id, limit=1,
            now=datetime.now(UTC) - timedelta(days=1),
        )
        expected_status = "processing"
        if delivery_state == "failed":
            case.outbox.mark_failed(
                tenant_id=case.tenant, event_id=case.event.event_id,
                error="delivery interrupted", lease_token=leased.lease_token,
            )
            expected_status = "failed"
    assert case.outbox.status(case.event.event_id) == expected_status

    report = reconcile_cli(case, capsys)

    assert case.facts.status().current and report["current"], "current-table repair left rows missing"
    assert case.facts.verify().matches
    assert case.outbox.status(case.event.event_id) == expected_status, (
        "current-table repair consumed external delivery state"
    )
    assert not case.sink.path(case.event).exists(), "the external store received no delivery request"
    assert case.recovery().run_once(limit=1) == 1, "external recovery lost its undelivered event"
    assert case.outbox.status(case.event.event_id) == "applied"
    assert json.loads(case.sink.path(case.event).read_text(encoding="utf-8")) == {
        "event_id": case.event.event_id, "fact": case.event.fact.to_payload() if case.event.fact else None,
    }


def test_reconcile_repairs_current_rows_after_external_delivery_was_applied(
    delivery_case: DeliveryCase, capsys: pytest.CaptureFixture[str],
) -> None:
    """Invariant: an external acknowledgement cannot prevent repair of missing current rows.

    Historical baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. Target symbol:
    recall.cli_commands.provenance_cmd._rebuild_postgres. Expected red assertion:
    'an external acknowledgement prevented current-table repair'. The shared row is already
    applied, so the baseline claims nothing and stops without repairing the tables.
    Execution and assertion red proof: observed in the module proof receipt. External delivery must remain applied afterward.
    """
    case = delivery_case
    assert case.recovery().run_once(limit=1) == 1
    assert case.outbox.status(case.event.event_id) == "applied"
    delivered = case.sink.path(case.event).read_text(encoding="utf-8")

    report = reconcile_cli(case, capsys)

    assert case.facts.status().current and report["current"], (
        "an external acknowledgement prevented current-table repair"
    )
    assert case.facts.verify().matches
    assert case.outbox.status(case.event.event_id) == "applied"
    assert case.recovery().run_once(limit=1) == 0
    assert case.sink.path(case.event).read_text(encoding="utf-8") == delivered
