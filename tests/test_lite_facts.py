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

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
TEA = AtomicFact(namespace="memory", subject="user", predicate="drink", object="tea")
COFFEE = AtomicFact(namespace="memory", subject="user", predicate="drink", object="coffee")


def card_for(fact: AtomicFact, *, chunk: str = "c1", source: str = "memo.md") -> EvidenceCard:
    return EvidenceCard(
        card_id="", chunk_id=chunk, source=source, source_digest="digest", valid_from=None,
        valid_until=None, first_indexed_at=NOW - timedelta(days=1), indexed_at=NOW,
        tenant_id="default", generation_id="g1", pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64, calibration_id="cal", calibration_status="certified",
        trust_state="trusted", verdict="ok", confidence=0.99, rank=1, structured_facts=(fact,),
    )


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
    with pytest.raises(ValueError, match="collision|canonical content"):
        cards.put((tea,))
    assert cards.resolve("missing") is None


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
