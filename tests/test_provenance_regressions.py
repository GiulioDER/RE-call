"""Regression pins for defects found by the deterministic provenance audit."""

from datetime import UTC, datetime, timedelta

from recall.provenance_controller import (
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
)
from recall.types import AtomicFact, EvidenceCard


NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def _fact(value: str, start: datetime) -> AtomicFact:
    return AtomicFact(
        namespace="memory",
        subject="service:api",
        predicate="owner",
        object=value,
        valid_from=start,
    )


def _card(fact: AtomicFact) -> EvidenceCard:
    return EvidenceCard(
        card_id="",
        chunk_id=f"chunk-{fact.object}",
        source="owner.md",
        source_digest="digest",
        valid_from=None,
        valid_until=None,
        first_indexed_at=None,
        indexed_at=None,
        tenant_id="tenant-a",
        generation_id="gen-a",
        pipeline_fingerprint=None,
        corpus_fingerprint=None,
        calibration_id="calibration-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=1.0,
        rank=1,
        structured_facts=(fact,),
    )


def test_future_conflict_is_not_hidden_by_current_projection():
    future = NOW + timedelta(days=2)
    first_fact = _fact("team:future", future)
    second_fact = _fact("team:other", future + timedelta(hours=1))
    first_card = _card(first_fact)
    second_card = _card(second_fact)
    cards = EvidenceCardStore()
    cards.put((first_card, second_card))
    controller = ProvenanceController(
        tenant_id="tenant-a",
        generation_id="gen-a",
        cards=cards,
        ledger=InMemoryFactLedger(),
        source_digest_for=lambda _card: "digest",
        now=lambda: NOW,
    )
    assert controller.apply_fact(
        FactApplicationRequest(first_fact, (first_card.card_id,), "future-regression-1")
    ).allowed
    blocked = controller.apply_fact(
        FactApplicationRequest(second_fact, (second_card.card_id,), "future-regression-2")
    )
    assert blocked.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION
    assert not blocked.allowed
