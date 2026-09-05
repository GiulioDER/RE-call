from datetime import UTC, datetime, timedelta

import pytest
from dataclasses import replace

from recall import (
    AtomicFact,
    DecisionCode,
    EvidenceCard,
    EvidenceCardStore,
    FactApplicationRequest,
    InMemoryFactLedger,
    InMemoryMaterializationOutbox,
    MaterializationRecovery,
    ProvenanceController,
    SQLiteFactLedger,
    SQLiteEvidenceCardStore,
    SQLiteMaterializationOutbox,
    fact_identity,
    facts_conflict,
)


NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def fact(value: str, *, valid_from: datetime | None = None, valid_until: datetime | None = None) -> AtomicFact:
    return AtomicFact(
        namespace="memory",
        subject="service:api",
        predicate="owner",
        object=value,
        context={},
        valid_from=valid_from,
        valid_until=valid_until,
    )


def card(
    source_fact: AtomicFact,
    *,
    source="owner.md",
    source_digest="digest",
    supersession_links=(),
    valid_from=None,
    valid_until=None,
) -> EvidenceCard:
    return EvidenceCard(
        card_id="",
        chunk_id=source,
        source=source,
        source_digest=source_digest,
        valid_from=valid_from,
        valid_until=valid_until,
        first_indexed_at=NOW - timedelta(days=1),
        indexed_at=NOW,
        tenant_id="tenant-a",
        generation_id="gen-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="cal-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        supersession_links=tuple(supersession_links),
        structured_facts=(source_fact,),
    )


def controller(cards, ledger=None, *, digest="digest", fresh_search=None):
    store = EvidenceCardStore()
    store.put(cards)
    return ProvenanceController(
        tenant_id="tenant-a",
        generation_id="gen-a",
        cards=store,
        ledger=ledger or InMemoryFactLedger(),
        source_digest_for=lambda _card: digest,
        fresh_search=fresh_search,
        now=lambda: NOW,
        writer="test",
    )


def test_trusted_present_evidence_applies_and_replay_is_idempotent():
    claim = fact("team:platform")
    evidence = card(claim)
    ledger = InMemoryFactLedger()
    ctl = controller([evidence], ledger)

    first = ctl.apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "request-1"))
    replay = ctl.apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "request-1"))

    assert first.allowed and first.code == DecisionCode.APPLIED
    assert replay.allowed and replay.code == DecisionCode.DUPLICATE
    assert len([event for event in ledger.events if event.event_type == "asserted"]) == 1


def test_source_change_is_refused_even_when_the_card_was_valid_when_issued():
    claim = fact("team:platform")
    evidence = card(claim)
    decision = controller([evidence], digest="changed").apply_fact(
        FactApplicationRequest(claim, (evidence.card_id,), "request-2")
    )
    assert not decision.allowed
    assert decision.code == DecisionCode.SOURCE_CHANGED


def test_unsupported_claim_gets_one_fresh_search_and_can_recover():
    original = fact("team:platform")
    fresh_claim = fact("team:security")
    fresh = card(fresh_claim, source="fresh.md")
    calls = 0

    def search(_claim, _request):
        nonlocal calls
        calls += 1
        return (fresh.card_id,)

    ctl = controller([fresh], fresh_search=search)
    decision = ctl.apply_fact(FactApplicationRequest(fresh_claim, ("missing",), "request-3"))
    assert not original.object == fresh_claim.object
    assert calls == 1
    assert decision.code == DecisionCode.APPLIED
    assert decision.retried


def test_contradiction_requires_authored_supersession():
    old = fact("team:platform")
    new = fact("team:security")
    old_card = card(old, source="old.md")
    new_card = card(new, source="new.md")
    ledger = InMemoryFactLedger()
    ctl = controller([old_card], ledger)
    assert ctl.apply_fact(FactApplicationRequest(old, (old_card.card_id,), "request-4")).allowed

    refused = controller([new_card], ledger).apply_fact(
        FactApplicationRequest(new, (new_card.card_id,), "request-5")
    )
    assert refused.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION

    replacing_card = card(new, source="new-authoritative.md", supersession_links=(old_card.card_id,))
    allowed = controller([replacing_card], ledger).apply_fact(
        FactApplicationRequest(new, (replacing_card.card_id,), "request-6")
    )
    assert allowed.allowed and allowed.code == DecisionCode.APPLIED
    assert tuple(event.fact.object for event in ledger.current(tenant_id="tenant-a", now=NOW)) == ("team:security",)


def test_inferred_links_do_not_authorize_replacement():
    old = fact("team:platform")
    new = fact("team:security")
    ledger = InMemoryFactLedger()
    old_card = card(old, source="old.md")
    new_card = card(new, source="new.md", supersession_links=())
    ctl = controller([old_card], ledger)
    ctl.apply_fact(FactApplicationRequest(old, (old_card.card_id,), "request-7"))
    decision = controller([new_card], ledger).apply_fact(
        FactApplicationRequest(new, (new_card.card_id,), "request-8")
    )
    assert decision.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION


def test_expired_card_is_rejected_without_search_when_no_retry_is_configured():
    claim = fact("team:platform")
    evidence = card(claim, valid_until=NOW)
    decision = controller([evidence]).apply_fact(
        FactApplicationRequest(claim, (evidence.card_id,), "request-9")
    )
    assert decision.code == DecisionCode.VALIDITY_EXPIRED


def test_canonical_fact_identity_and_interval_conflict_are_deterministic():
    assert fact_identity(fact("team:platform")) == fact_identity(fact("team:platform"))
    assert facts_conflict(fact("a"), fact("b"))
    assert not facts_conflict(
        fact("a", valid_until=NOW),
        fact("b", valid_from=NOW),
    )


def test_card_identity_rejects_tampering():
    evidence = card(fact("team:platform"))
    with pytest.raises(ValueError, match="card id"):
        EvidenceCard(**{**evidence.__dict__, "card_id": "card_forged"})


def test_authoritative_card_revalidation_rejects_changed_structured_support():
    claim = fact("team:platform")
    evidence = card(claim)
    cards = EvidenceCardStore()
    cards.put((evidence,))
    ctl = ProvenanceController(
        tenant_id="tenant-a",
        generation_id="gen-a",
        cards=cards,
        ledger=InMemoryFactLedger(),
        source_digest_for=lambda _card: "digest",
        card_revalidator=lambda current: replace(current, card_id="", structured_facts=()),
        now=lambda: NOW,
        writer="test",
    )

    decision = ctl.apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "revalidate-1"))

    assert not decision.allowed
    assert decision.code == DecisionCode.SOURCE_CHANGED
    assert decision.event is not None
    assert decision.event.event_type == "rejected"


def test_ledger_read_outage_fails_closed_without_asserting():
    claim = fact("team:platform")
    evidence = card(claim)

    class BrokenLedger(InMemoryFactLedger):
        def current(self, *, tenant_id, now):
            raise OSError("ledger unavailable")

    ledger = BrokenLedger()
    decision = controller([evidence], ledger).apply_fact(
        FactApplicationRequest(claim, (evidence.card_id,), "ledger-read-outage")
    )

    assert not decision.allowed
    assert decision.code == DecisionCode.LEDGER_UNAVAILABLE
    assert not [event for event in ledger.events if event.event_type == "asserted"]


def test_materialization_failure_is_not_success_and_replay_retries_same_event():
    claim = fact("team:platform")
    evidence = card(claim)
    ledger = InMemoryFactLedger()
    cards = EvidenceCardStore()
    cards.put((evidence,))

    class Materializer:
        calls = 0

        def materialize(self, _event):
            self.calls += 1
            if self.calls == 1:
                raise OSError("writer unavailable")

    materializer = Materializer()
    request = FactApplicationRequest(claim, (evidence.card_id,), "materialize-1")

    first = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", materializer=materializer,
        now=lambda: NOW, writer="test",
    ).apply_fact(request)
    second = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", materializer=materializer,
        now=lambda: NOW, writer="test",
    ).apply_fact(request)

    assert not first.allowed
    assert first.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
    assert second.allowed and second.code == DecisionCode.DUPLICATE
    assert first.event is not None and second.event is first.event
    assert materializer.calls == 2


def test_materialization_outbox_recovery_replays_without_a_second_fact_event():
    claim = fact("team:platform")
    evidence = card(claim)
    cards = EvidenceCardStore()
    cards.put((evidence,))
    ledger = InMemoryFactLedger()
    outbox = InMemoryMaterializationOutbox()

    class Materializer:
        calls = 0

        def materialize(self, _event):
            self.calls += 1
            if self.calls == 1:
                raise OSError("downstream unavailable")

    materializer = Materializer()
    request = FactApplicationRequest(claim, (evidence.card_id,), "outbox-1")
    first = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", materializer=materializer,
        materialization_outbox=outbox, now=lambda: NOW, writer="test",
    ).apply_fact(request)

    assert first.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
    assert first.event is not None
    assert outbox.status(first.event.event_id) == "failed"
    assert len([event for event in ledger.events if event.event_type == "asserted"]) == 1

    delivered = MaterializationRecovery(
        tenant_id="tenant-a", outbox=outbox, materializer=materializer,
    ).run_once(now=NOW)
    assert delivered == 1
    assert outbox.status(first.event.event_id) == "applied"
    assert materializer.calls == 2

    replay = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", materializer=materializer,
        materialization_outbox=outbox, now=lambda: NOW, writer="test",
    ).apply_fact(request)
    assert replay.allowed and replay.code == DecisionCode.DUPLICATE
    assert materializer.calls == 2


def test_materialization_reconcile_recovers_ledger_outbox_crash_window():
    claim = fact("team:platform")
    evidence = card(claim)
    cards = EvidenceCardStore()
    cards.put((evidence,))
    ledger = InMemoryFactLedger()
    request = FactApplicationRequest(claim, (evidence.card_id,), "reconcile-1")
    applied = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
    ).apply_fact(request)
    assert applied.allowed and applied.event is not None

    outbox = InMemoryMaterializationOutbox()
    delivered = MaterializationRecovery(
        tenant_id="tenant-a", outbox=outbox,
        materializer=type("Working", (), {"materialize": lambda _self, _event: None})(),
    ).reconcile(ledger.events, now=NOW)
    assert delivered == 1
    assert outbox.status(applied.event.event_id) == "applied"
    assert len([event for event in ledger.events if event.event_type == "asserted"]) == 1


def test_sqlite_materialization_outbox_survives_reopen(tmp_path):
    claim = fact("team:platform")
    evidence = card(claim)
    path = str(tmp_path / "materialization.sqlite")
    outbox = SQLiteMaterializationOutbox(path, tenant_id="tenant-a")
    event = InMemoryFactLedger()
    cards = EvidenceCardStore()
    cards.put((evidence,))
    decision = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=event,
        source_digest_for=lambda _card: "digest", materializer=type("Failing", (), {
            "materialize": lambda _self, _event: (_ for _ in ()).throw(OSError("offline"))
        })(), materialization_outbox=outbox, now=lambda: NOW, writer="test",
    ).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "sqlite-outbox-1"))
    assert decision.event is not None
    event_id = decision.event.event_id
    assert outbox.status(event_id) == "failed"
    outbox.close()

    reopened = SQLiteMaterializationOutbox(path, tenant_id="tenant-a")
    delivered = MaterializationRecovery(
        tenant_id="tenant-a", outbox=reopened,
        materializer=type("Working", (), {"materialize": lambda _self, _event: None})(),
    ).run_once(now=NOW)
    assert delivered == 1
    assert reopened.status(event_id) == "applied"
    reopened.close()


def test_replaying_a_recorded_refusal_cannot_become_an_allowed_duplicate():
    claim = fact("team:platform")
    evidence = card(claim)
    cards = EvidenceCardStore()
    cards.put((evidence,))
    ledger = InMemoryFactLedger()
    controller = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
    )
    request = FactApplicationRequest(fact("team:other"), (evidence.card_id,), "refusal-replay")
    first = controller.apply_fact(request)
    second = controller.apply_fact(request)
    assert not first.allowed and first.code == DecisionCode.UNSUPPORTED_CLAIM
    assert not second.allowed and second.code == DecisionCode.UNSUPPORTED_CLAIM
    assert len(ledger.events) == 1


def test_future_conflict_is_rejected_before_its_validity_window_starts():
    future = NOW + timedelta(days=2)
    first_fact = fact("team:future", valid_from=future)
    second_fact = fact("team:other", valid_from=future + timedelta(hours=1))
    first_card = card(first_fact)
    second_card = card(second_fact)
    cards = EvidenceCardStore()
    cards.put((first_card, second_card))
    controller = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards,
        ledger=InMemoryFactLedger(), source_digest_for=lambda _card: "digest",
        now=lambda: NOW, writer="test",
    )
    assert controller.apply_fact(
        FactApplicationRequest(first_fact, (first_card.card_id,), "future-1")
    ).allowed
    blocked = controller.apply_fact(
        FactApplicationRequest(second_fact, (second_card.card_id,), "future-2")
    )
    assert not blocked.allowed
    assert blocked.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION


def test_expired_outbox_lease_cannot_complete_after_reclaim():
    claim = fact("team:platform")
    evidence = card(claim)
    cards = EvidenceCardStore()
    cards.put((evidence,))
    event = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=cards,
        ledger=InMemoryFactLedger(), source_digest_for=lambda _card: "digest",
        now=lambda: NOW, writer="test",
    ).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "lease-event")).event
    assert event is not None
    outbox = InMemoryMaterializationOutbox(lease_seconds=1)
    outbox.enqueue(event)
    old = outbox.claim(tenant_id="tenant-a", now=NOW)[0]
    new = outbox.claim(tenant_id="tenant-a", now=NOW + timedelta(seconds=2))[0]
    assert old.lease_token != new.lease_token
    with pytest.raises(ValueError, match="lease lost"):
        outbox.mark_applied(
            tenant_id="tenant-a", event_id=event.event_id, lease_token=old.lease_token
        )
def test_sqlite_ledger_survives_reopen_and_remains_append_only(tmp_path):
    claim = fact("team:platform")
    evidence = card(claim)
    path = str(tmp_path / "facts.sqlite")
    with SQLiteFactLedger(path, tenant_id="tenant-a") as ledger:
        cards = EvidenceCardStore()
        cards.put((evidence,))
        ctl = ProvenanceController(
            tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
        )
        applied = ctl.apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "sqlite-1"))
        assert applied.allowed and applied.event is not None
        with pytest.raises(PermissionError, match="controller permit"):
            ledger.apply_assertion(
                tenant_id="tenant-a", generation_id="gen-a", fact=claim,
                cards=(evidence,), request_id="direct-write", writer="test", now=NOW,
            )
        with pytest.raises(Exception, match="append-only"):
            ledger._conn.execute("DELETE FROM recall_fact_ledger_events")
    with SQLiteFactLedger(path, tenant_id="tenant-a") as reopened:
        assert [event.fact.object for event in reopened.current(tenant_id="tenant-a", now=NOW)] == [
            "team:platform"
        ]
        cards = EvidenceCardStore()
        cards.put((evidence,))
        replay = ProvenanceController(
            tenant_id="tenant-a", generation_id="gen-a", cards=cards, ledger=reopened,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
        ).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "sqlite-1"))
        assert replay.allowed and replay.code == DecisionCode.DUPLICATE


def test_sqlite_card_store_survives_reopen_and_remains_immutable(tmp_path):
    evidence = card(fact("team:platform"))
    path = str(tmp_path / "provenance.sqlite")
    with SQLiteEvidenceCardStore(path, tenant_id="tenant-a") as cards:
        cards.put((evidence,))
        assert cards.resolve(evidence.card_id) == evidence
        with pytest.raises(Exception, match="immutable"):
            cards._conn.execute(
                "UPDATE recall_evidence_cards SET source_digest = source_digest WHERE card_id = ?",
                (evidence.card_id,),
            )
    with SQLiteEvidenceCardStore(path, tenant_id="tenant-a") as reopened:
        assert reopened.resolve(evidence.card_id) == evidence
        with SQLiteEvidenceCardStore(path, tenant_id="tenant-b") as other_tenant:
            assert other_tenant.resolve(evidence.card_id) is None
