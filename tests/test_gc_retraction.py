"""`generation gc` retracts, for good, the facts whose memos no remaining generation holds.

A memo a promoted generation leaves out is only hidden at read time (#932), because a `generation
rollback` can still bring it back. Once gc deletes the last generation holding it, nothing can, so
gc appends a permanent `retracted` event (SOURCE_REMOVED, writer `recall_generation_gc`): the bare
ledger stops listing the fact, without any served lookup.

Invariants and the failure each one catches:
- G1 gc that deletes the last generation holding a memo retracts its fact.
- G2 a fact whose memo a remaining generation still holds is not retracted.
- G3 a retraction that failed during one gc is completed by the next, which deletes nothing.
- G4 a fact whose cards come from no collected generation is never retracted by gc, even when its
  memo is in no generation (a legacy or lite card).

Red proof (2026-10-15), each against #932's tip `f64e459a6` (pre-fix) unless a mutation is named,
then restored and green; node ids and failure lines in each test's docstring.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from tests.fact_fixtures import NOW, evidence_card
from tests.test_absent_memo_facts import DOOR, _corpus
from tests.test_absent_memo_facts import manager as manager  # noqa: F401  # the fixture

DOOR_MEMO, OTHER_MEMO = b"alpha door memo", b"beta other memo"


def _ready(manager: Any, memos: dict[str, bytes], version: str) -> str:
    from tests.test_generations import _Embedder, _pipeline

    manifest, reader = _corpus(manager.tenant_id, memos, version)
    generation = manager.create(manifest, _pipeline("model-a"))
    manager.build(generation.generation_id, reader, _Embedder(1), lambda text: [text])
    manager.validate(generation.generation_id)
    return str(generation.generation_id)


def _promote(manager: Any, memos: dict[str, bytes], version: str) -> str:
    generation = _ready(manager, memos, version)
    manager.promote(generation, unsafe_development=True)
    return generation


def _apply_door(manager: Any, generation: str, *, source: str | None = None) -> None:
    """Apply DOOR citing the door memo's chunk in `generation` (or a card naming `source`)."""
    from recall.fact_ledger import PostgresFactLedger
    from recall.generation_store import GenerationStore
    from recall.provenance_cards import PostgresEvidenceCardStore
    from tests.conftest import TEST_DSN

    tenant = manager.tenant_id
    if source is None:
        with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
            chunk = next(h.chunk for h in store.query_sparse("alpha", 5) if h.chunk.text == DOOR_MEMO.decode())
        card = evidence_card(DOOR, tenant=tenant, generation=generation, chunk=chunk.id,
                             source=str(chunk.metadata.get("file") or chunk.source))
    else:
        card = evidence_card(DOOR, tenant=tenant, generation=generation, chunk="chunk-ghost", source=source)
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    decision = ProvenanceController(
        tenant_id=tenant, generation_id=generation, cards=cards,
        ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="gc-test",
    ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), "r1"))
    assert decision.code == DecisionCode.APPLIED, decision


def _ledger(manager: Any) -> Any:
    from recall.fact_ledger import PostgresFactLedger
    from tests.conftest import TEST_DSN

    return PostgresFactLedger(TEST_DSN, tenant_id=manager.tenant_id)


def _bare_current(manager: Any) -> list[object]:
    """The ledger's own answer, with no served lookup: only a ledger event can change it."""
    return [e.fact.object for e in _ledger(manager).current(tenant_id=manager.tenant_id, now=NOW) if e.fact]


def _gc(manager: Any) -> tuple[str, ...]:
    later = datetime.now(UTC) + timedelta(minutes=1)
    return tuple(manager.gc(now=later, retention_days=0, retain_previous=0))


@pytest.fixture
def three(manager: Any) -> Iterator[tuple[Any, str]]:
    """Generation 1 holds the door memo and DOOR is applied from it; 2 and 3 leave it out, so gc
    collects 1 (2 is protected as previous) and with it the last copy of the memo."""
    first = _promote(manager, {"door.md": DOOR_MEMO, "other.md": OTHER_MEMO}, "corpus-v1")
    _apply_door(manager, first)
    _promote(manager, {"other.md": OTHER_MEMO}, "corpus-v2")
    _promote(manager, {"other.md": OTHER_MEMO + b" again"}, "corpus-v3")
    yield manager, first


def test_g1_gc_of_the_last_generation_holding_a_memo_retracts_its_fact(three: tuple[Any, str]) -> None:
    """Node tests/test_gc_retraction.py::test_g1_gc_of_the_last_generation_holding_a_memo_retracts_its_fact.

    Red against #932's tip: gc writes nothing, so it fails on "gc left a fact whose memo no
    generation holds: ['4512']". Red, with migration 0027 removed from the package (on a fresh
    database): the append function refuses SOURCE_REMOVED, the retraction is counted -1, and the
    same assertion fails.
    """
    manager, first = three
    assert _bare_current(manager) == ["4512"], "the fact was retracted before gc"

    assert _gc(manager) == (first,)

    assert _bare_current(manager) == [], f"gc left a fact whose memo no generation holds: {_bare_current(manager)}"
    assert manager.last_gc_facts_retracted == 1
    retractions = [e for e in _ledger(manager).events if e.event_type == "retracted"]
    assert [(e.decision_code, e.writer) for e in retractions] == [
        (DecisionCode.SOURCE_REMOVED, "recall_generation_gc")
    ]


def test_g2_a_memo_a_remaining_generation_holds_keeps_its_fact(manager: Any) -> None:
    """Node tests/test_gc_retraction.py::test_g2_a_memo_a_remaining_generation_holds_keeps_its_fact.

    Red, mutation of `GenerationManager.retract_collected_facts` (recall/generations.py) whose
    `gone` ignores the remaining generations (`return card.source in known`): fails on "gc
    retracted a fact whose memo a remaining generation holds".
    """
    first = _promote(manager, {"door.md": DOOR_MEMO}, "corpus-v1")
    _apply_door(manager, first)
    _promote(manager, {"door.md": DOOR_MEMO, "other.md": OTHER_MEMO}, "corpus-v2")
    _promote(manager, {"door.md": DOOR_MEMO, "other.md": OTHER_MEMO + b" again"}, "corpus-v3")

    assert _gc(manager) == (first,)
    assert _bare_current(manager) == ["4512"], "gc retracted a fact whose memo a remaining generation holds"


def test_g3_the_next_gc_completes_a_failed_retraction(
    three: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_gc_retraction.py::test_g3_the_next_gc_completes_a_failed_retraction.

    Red, mutation of `GenerationManager.gc` (recall/generations.py) passing only this call's
    deleted generations (`retract_collected_facts(collected=delete)`): the second gc deletes
    nothing, so it retracts nothing and fails on "a second gc did not complete the retraction".
    Red against #932's tip: the same failure line.
    """
    from recall.fact_ledger import PostgresFactLedger

    manager, first = three

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise OSError("ledger unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(PostgresFactLedger, "retract_forgotten", refuse)
        assert _gc(manager) == (first,)
    assert _bare_current(manager) == ["4512"]

    assert _gc(manager) == ()
    assert _bare_current(manager) == [], "a second gc did not complete the retraction"
    assert manager.last_gc_facts_retracted == 1


def test_g4_gc_never_retracts_a_fact_from_a_generation_it_did_not_collect(three: tuple[Any, str]) -> None:
    """Node tests/test_gc_retraction.py::test_g4_gc_never_retracts_a_fact_from_a_generation_it_did_not_collect.

    The fact's only card names a memo no generation holds, but it comes from a generation gc never
    deleted (a legacy or lite card): gc is not what removed it, so gc does not retract it. Red,
    mutation of `retract_collected_facts` whose `touched` accepts every card (`return True`):
    fails on "gc retracted a fact it did not remove the evidence of".
    """
    from recall.fact_ledger import PostgresFactLedger
    from recall.provenance_cards import PostgresEvidenceCardStore
    from recall.types import AtomicFact
    from tests.conftest import TEST_DSN

    manager, _first = three
    ghost = AtomicFact(namespace="memory", subject="user", predicate="locker", object="17")
    card = evidence_card(ghost, tenant=manager.tenant_id, generation="legacy-generation",
                         chunk="chunk-ghost", source="ghost.md")
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=manager.tenant_id)
    cards.put((card,))
    assert ProvenanceController(
        tenant_id=manager.tenant_id, generation_id="legacy-generation", cards=cards,
        ledger=PostgresFactLedger(TEST_DSN, tenant_id=manager.tenant_id),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="gc-test",
    ).apply_fact(FactApplicationRequest(ghost, (card.card_id,), "r-ghost")).allowed

    _gc(manager)
    assert "17" in _bare_current(manager), "gc retracted a fact it did not remove the evidence of"
