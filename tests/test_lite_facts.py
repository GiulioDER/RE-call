"""The fact ledger and evidence cards on a lite store (one SQLite file).

Invariants and the failure each one catches:
- F1 reads go to the file: a second handle sees an append with no reopen (the snapshot that
  `SQLiteFactLedger` takes at open would hide it).
- F2 the ledger is append only: its triggers refuse a DELETE.
- F3 a stored card whose content no longer matches its id is refused; an identical put is accepted.
- F4 a write waits for another connection's write transaction instead of failing.
- F5 a file made before the fact tables existed is served: the tables appear on first use.
- F6 a contradiction without supersession is refused, and the check inside the append
  transaction holds on its own.
- F7 a fact applied under two requests is listed once by `current`, as on PostgreSQL.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from recall.types import AtomicFact, EvidenceCard
from tests.fact_fixtures import COFFEE, NOW, TEA, evidence_card

def card_for(fact: AtomicFact, *, chunk: str = "c1", source: str = "memo.md") -> EvidenceCard:
    return evidence_card(fact, tenant="default", generation="g1", chunk=chunk, source=source)


def controller(store: LiteStore, cards: LiteEvidenceCardStore) -> ProvenanceController:
    return ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="lite-test",
    )


@pytest.fixture
def store(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "memory.db", dim=8)


def test_f1_a_second_handle_sees_an_append_without_reopening(store: LiteStore, tmp_path: Path) -> None:
    """Red: `LiteFactLedger.current` reading events cached at construction fails "not visible"."""
    reader = LiteFactLedger(LiteStore(tmp_path / "memory.db", dim=8))
    assert reader.current(tenant_id="default", now=NOW) == ()
    cards = LiteEvidenceCardStore(store)
    tea = card_for(TEA)
    cards.put((tea,))
    decision = controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r1"))
    assert decision.allowed and decision.code == DecisionCode.APPLIED, decision
    seen = [event.fact for event in reader.current(tenant_id="default", now=NOW)]
    assert seen == [TEA], "an append on one handle was not visible on another"


def test_f2_the_ledger_is_append_only(store: LiteStore) -> None:
    """Red: the schema without the no_delete trigger lets the DELETE through."""
    cards = LiteEvidenceCardStore(store)
    tea = card_for(TEA)
    cards.put((tea,))
    controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r1"))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute("DELETE FROM recall_fact_ledger_events")


def test_f3_a_card_id_collision_is_refused(store: LiteStore) -> None:
    """Red: `LiteEvidenceCardStore.put` without its read back (`existing = card`) raises nothing.

    A card id is a hash of the card's content, so a tampered row fails to decode before the
    equality test is reached; the read back is what refuses it, and is what the mutation removes.
    """
    cards = LiteEvidenceCardStore(store)
    tea = card_for(TEA)
    cards.put((tea,))
    cards.put((tea,))  # identical content: accepted
    store._conn.execute("DROP TRIGGER recall_evidence_cards_no_update")
    store._conn.execute(
        "UPDATE recall_evidence_cards SET card = replace(card, 'memo.md', 'other.md') WHERE card_id = ?",
        (tea.card_id,),
    )
    # The tampered row no longer hashes to its id, so the read back refuses it on decoding.
    with pytest.raises(ValueError, match="does not match its canonical content"):
        cards.put((tea,))


def test_f3b_an_unknown_card_id_resolves_to_nothing(store: LiteStore) -> None:
    """A card id the file never stored resolves to None, not to an error."""
    assert LiteEvidenceCardStore(store).resolve("missing") is None


def test_f4_a_write_waits_for_another_writer(store: LiteStore, tmp_path: Path) -> None:
    """Red: the store connection opened with timeout=0 makes the append fail "database is locked"."""
    cards = LiteEvidenceCardStore(store)
    tea = card_for(TEA)
    cards.put((tea,))
    blocker = sqlite3.connect(tmp_path / "memory.db", isolation_level=None, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    release = threading.Timer(1.0, lambda: blocker.execute("COMMIT"))
    release.start()
    started = time.monotonic()
    decision = controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r1"))
    release.join()
    assert decision.allowed, decision
    assert time.monotonic() - started >= 0.9, "the append did not wait for the other writer"


def test_f5_an_old_file_gains_the_fact_tables(tmp_path: Path) -> None:
    """Red: `ensure_fact_tables` as a no-op makes `current` raise "no such table"."""
    path = tmp_path / "old.db"
    LiteStore(path, dim=8).close()
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "recall_fact_ledger_events" not in tables
    assert LiteFactLedger(LiteStore(path, dim=8)).current(tenant_id="default", now=NOW) == ()


def test_f6_contradiction_without_supersession_is_refused(store: LiteStore) -> None:
    """Red: `LiteFactLedger.current` returning () AND the ledger's own conflict check deleted makes
    the second application APPLIED. With only the first mutation it stays refused, which shows the
    check inside the append transaction holds on its own."""
    cards = LiteEvidenceCardStore(store)
    tea, coffee = card_for(TEA), card_for(COFFEE, chunk="c2")
    cards.put((tea, coffee))
    first = controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r1"))
    assert first.allowed, first
    refused = controller(store, cards).apply_fact(FactApplicationRequest(COFFEE, (coffee.card_id,), "r2"))
    assert refused.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, refused


def test_f7_a_fact_applied_under_two_requests_is_listed_once(store: LiteStore) -> None:
    """Red (2026-10-13): `asserts_fact` in recall/provenance_controller.py reduced to
    `event.event_type == "asserted"` (the filter `project_current` used before the fix) fails the
    listing assertion with `("r2", DUPLICATE)` listed BEFORE `("r1", APPLIED)` (equal timestamps
    order by event id). Restored, passes."""
    cards = LiteEvidenceCardStore(store)
    tea = card_for(TEA)
    cards.put((tea,))
    assert controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r1")).code == DecisionCode.APPLIED
    assert controller(store, cards).apply_fact(FactApplicationRequest(TEA, (tea.card_id,), "r2")).code == DecisionCode.DUPLICATE

    current = LiteFactLedger(store).current(tenant_id=store.tenant, now=NOW)
    assert [(event.request_id, event.decision_code) for event in current] == [("r1", DecisionCode.APPLIED)]


def test_f9_events_are_ordered_by_instant_whatever_the_clock_offset(store: LiteStore) -> None:
    """An event stamped with a `+02:00` clock sorts by its instant, as PostgreSQL's timestamptz does.

    Red, 2026-10-13, against the pre-fix `_stamp` (kept the offset): fails on "events were ordered
    by their local clock".
    """
    from datetime import timezone

    ledger = LiteFactLedger(store)
    later = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    earlier = datetime(2026, 1, 1, 1, 0, tzinfo=timezone(timedelta(hours=2)))  # 2025-12-31 23:00 UTC
    for request, moment in (("late", later), ("early", earlier)):
        ledger.record_decision(
            tenant_id=store.tenant, generation_id="g1", request_id=request, writer="lite-test",
            decision_code="UNSUPPORTED_CLAIM", fact=TEA, now=moment,
        )
    order = [event.request_id for event in ledger.events]
    assert order == ["early", "late"], f"events were ordered by their local clock: {order}"
    assert all(event.created_at.utcoffset() == timedelta(0) for event in ledger.events)


def test_f10_a_refusal_without_its_fact_is_refused(store: LiteStore) -> None:
    """A `rejected` event must carry the refused fact, as PostgreSQL's fact_shape CHECK requires.

    Red, 2026-10-13, against the pre-fix SQLite DDL (no such CHECK): fails on "a rejected event with
    no fact was stored".
    """
    import sqlite3

    ledger = LiteFactLedger(store)
    try:
        ledger.record_decision(
            tenant_id=store.tenant, generation_id="g1", request_id="r1", writer="lite-test",
            decision_code="UNSUPPORTED_CLAIM", fact=None, now=NOW,
        )
    except sqlite3.IntegrityError:
        pass
    assert ledger.events == (), "a rejected event with no fact was stored"
