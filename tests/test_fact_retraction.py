"""Forgetting a memo retracts the facts it was the only evidence for (lite store, no database).

`recall_forget` deleted a source's chunks and left the fact ledger alone, so a fact applied with
`recall_apply_fact` from a memo stayed in `recall_current_facts` after the memo was forgotten: "forget
my door code" was not honoured. `forget_memory` now appends a `retracted` event for each live
assertion of a fact whose every evidence card names a source that is gone, and both projections
(`project_current`, `project_candidates`) skip a retracted assertion.

Invariants and the failure each one catches:
- R1 a fact whose only memo is forgotten is no longer current.
- R2 a fact with evidence in a memo that survives stays current; it goes when its last memo does,
  including when the earlier memo was forgotten by an earlier call.
- R3 a fact learned again from a new memo after forgetting is a new application and is listed.
- R4 a retracted fact no longer blocks a different value of the same predicate.
- R5 forgetting the memo that superseded a fact does not bring the superseded fact back.
- R6 a forget re-run after its retraction failed completes the retraction, though the chunks are
  already gone.
- R7 a ledger table made before `retracted` existed accepts one, keeping its rows and its
  append-only triggers.

Red proof (2026-10-14), each against the chain end `3d7d42b27` (pre-fix) unless a mutation is named,
then restored and green; the node ids and failure lines are in each test's docstring.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import (
    ControllerDecision,
    DecisionCode,
    FactApplicationRequest,
    ProvenanceController,
)
from recall.types import AtomicFact, Chunk
from recall_mcp.lifecycle import forget_memory
from tests.fact_fixtures import COFFEE, NOW, TEA, evidence_card

DIM = 8
DOOR = AtomicFact(namespace="memory", subject="user", predicate="door_code", object="4512")


@pytest.fixture
def store(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "memory.db", dim=DIM)


def remember(store: LiteStore, name: str) -> None:
    """Index one memo: stored under an absolute source, named by its root-relative file."""
    store.upsert(
        [Chunk(f"chunk-{name}", f"/notes/{name}", f"text of {name}", {"file": name})],
        [[1.0] + [0.0] * (DIM - 1)],
    )


def apply(
    store: LiteStore, fact: AtomicFact, name: str, request_id: str, links: tuple[str, ...] = ()
) -> ControllerDecision:
    """Apply `fact` with one card citing memo `name`, as `recall_apply_fact` does."""
    card = evidence_card(
        fact, tenant=store.tenant, generation="g1", chunk=f"chunk-{name}", source=name, links=links
    )
    cards = LiteEvidenceCardStore(store)
    cards.put((card,))
    return ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="retraction-test",
    ).apply_fact(FactApplicationRequest(fact, (card.card_id,), request_id))


def current(store: LiteStore) -> list[object]:
    return [event.fact.object for event in LiteFactLedger(store).current(tenant_id=store.tenant, now=NOW) if event.fact]


def test_r1_forgetting_a_facts_only_memo_retracts_it(store: LiteStore) -> None:
    """Node tests/test_fact_retraction.py::test_r1_forgetting_a_facts_only_memo_retracts_it.

    Red against the pre-fix chain end: `forget_memory` never touches the ledger, so it fails on
    "the fact outlived its forgotten memo: ['4512']".
    """
    remember(store, "door.md")
    assert apply(store, DOOR, "door.md", "r1").code == DecisionCode.APPLIED
    assert current(store) == ["4512"]

    result = forget_memory(store, ["door.md"])

    assert result.sources_removed == ["door.md"]
    assert current(store) == [], f"the fact outlived its forgotten memo: {current(store)}"
    assert result.facts_retracted == 1
    retractions = [e for e in LiteFactLedger(store).events if e.event_type == "retracted"]
    assert [(e.decision_code, e.writer) for e in retractions] == [(DecisionCode.SOURCE_FORGOTTEN, "recall_forget")]


def test_r2_a_fact_stays_while_any_of_its_memos_survives(store: LiteStore) -> None:
    """Node tests/test_fact_retraction.py::test_r2_a_fact_stays_while_any_of_its_memos_survives.

    Red, mutation of `plan_retractions` (recall/fact_ledger.py): the surviving-evidence test
    removed (`all(gone(card) ...)` dropped, so a touched fact is retracted), fails on "a fact with
    a surviving memo was retracted". Red, mutation of `_forget_retractions`
    (recall_mcp/lifecycle.py): a card counted as gone only when this call names its source
    (`gone=lambda card: touched(card) and card.source in named`), fails on "the fact outlived
    its last memo". Red against the pre-fix chain end: fails on "the fact outlived its last memo".
    """
    remember(store, "a.md")
    remember(store, "b.md")
    assert apply(store, DOOR, "a.md", "r1").code == DecisionCode.APPLIED
    assert apply(store, DOOR, "b.md", "r2").code == DecisionCode.DUPLICATE

    first = forget_memory(store, ["a.md"])
    assert current(store) == ["4512"], "a fact with a surviving memo was retracted"

    forget_memory(store, ["b.md"])
    assert current(store) == [], f"the fact outlived its last memo: {current(store)}"
    assert first.facts_retracted == 0


def test_r3_a_fact_learned_again_after_forgetting_is_a_new_application(store: LiteStore) -> None:
    """Node tests/test_fact_retraction.py::test_r3_a_fact_learned_again_after_forgetting_is_a_new_application.

    Red against the pre-fix chain end: the forgotten assertion is still a candidate, so the new
    one is coded DUPLICATE and fails on "re-learning was decided against the forgotten fact".
    Red, mutation of `retraction_key` and `retracted_keys` (recall/provenance_controller.py) to
    keep only the fact id (generation and cards blanked), so a retraction hides every assertion of
    the fact: fails on "the re-learned fact is hidden by the old retraction".
    """
    remember(store, "door.md")
    apply(store, DOOR, "door.md", "r1")
    forget_memory(store, ["door.md"])

    remember(store, "door-again.md")
    again = apply(store, DOOR, "door-again.md", "r2")
    assert again.code == DecisionCode.APPLIED, f"re-learning was decided against the forgotten fact: {again.code}"
    assert current(store) == ["4512"], "the re-learned fact is hidden by the old retraction"


def test_r4_a_retracted_fact_no_longer_blocks_a_new_value(store: LiteStore) -> None:
    """Node tests/test_fact_retraction.py::test_r4_a_retracted_fact_no_longer_blocks_a_new_value.

    Red against the pre-fix chain end: fails on "a forgotten value still contradicts the new one:
    CONTRADICTION_WITHOUT_SUPERSESSION". Red, mutation of `project_candidates`
    (recall/fact_ledger.py) without its retraction filter: the same failure.
    """
    remember(store, "tea.md")
    apply(store, TEA, "tea.md", "r1")
    forget_memory(store, ["tea.md"])

    remember(store, "coffee.md")
    coffee = apply(store, COFFEE, "coffee.md", "r2")
    assert coffee.code == DecisionCode.APPLIED, f"a forgotten value still contradicts the new one: {coffee.code}"
    assert current(store) == ["coffee"]


def test_r5_forgetting_a_successor_does_not_revive_what_it_superseded(store: LiteStore) -> None:
    """Node tests/test_fact_retraction.py::test_r5_forgetting_a_successor_does_not_revive_what_it_superseded.

    Forgetting may only remove facts, never make one current: the supersession was a statement
    that the old value stopped holding, and nothing that remains says it holds again.
    Red against the pre-fix chain end: fails on "the forgotten successor is still listed:
    ['coffee']".
    """
    remember(store, "tea.md")
    remember(store, "coffee.md")
    apply(store, TEA, "tea.md", "r1")
    assert apply(store, COFFEE, "coffee.md", "r2", links=("tea.md",)).code == DecisionCode.APPLIED

    forget_memory(store, ["coffee.md"])
    assert current(store) == [], f"the forgotten successor is still listed: {current(store)}"


def test_r6_a_rerun_completes_a_failed_retraction(store: LiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_fact_retraction.py::test_r6_a_rerun_completes_a_failed_retraction.

    Red, mutation of `_forget_retractions` (recall_mcp/lifecycle.py): `touched` matching only the
    sources this call deletes (the requested names dropped), so a re-run, which deletes nothing,
    touches nothing: fails on "the re-run did not retract the fact".
    """
    remember(store, "door.md")
    apply(store, DOOR, "door.md", "r1")

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as patch:
        patch.setattr(LiteFactLedger, "retract_forgotten", refuse)
        failed = forget_memory(store, ["door.md"])
    assert failed.sources_removed == ["door.md"] and failed.facts_retracted == -1
    assert "WARNING" in failed.message and "re-run this forget" in failed.message
    assert current(store) == ["4512"]

    rerun = forget_memory(store, ["door.md"])
    assert rerun.sources_not_found == ["door.md"]
    assert current(store) == [], "the re-run did not retract the fact"
    assert rerun.facts_retracted == 1


#: `recall_fact_ledger_events` as the lite stack created it before `retracted` existed.
_OLD_LEDGER_DDL = """
CREATE TABLE recall_fact_ledger_events (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('asserted', 'superseded', 'rejected', 'abstained')),
    fact_id TEXT,
    fact TEXT,
    evidence_cards TEXT NOT NULL,
    supersedes_fact_ids TEXT NOT NULL,
    request_id TEXT NOT NULL,
    writer TEXT NOT NULL,
    decision_code TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    controller_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (tenant_id, request_id),
    CHECK ((event_type IN ('asserted', 'rejected') AND fact IS NOT NULL) OR event_type IN ('superseded', 'abstained'))
);
CREATE INDEX recall_fact_ledger_current_idx ON recall_fact_ledger_events (tenant_id, fact_id, created_at);
CREATE TRIGGER recall_fact_ledger_no_update BEFORE UPDATE ON recall_fact_ledger_events
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
CREATE TRIGGER recall_fact_ledger_no_delete BEFORE DELETE ON recall_fact_ledger_events
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
"""


def test_r7_a_ledger_made_before_retraction_accepts_one(tmp_path: Path) -> None:
    """Node tests/test_fact_retraction.py::test_r7_a_ledger_made_before_retraction_accepts_one.

    Red, mutation of `ensure_fact_tables` (recall/lite/facts.py) without its upgrade call: the
    old CHECK refuses the `retracted` row, the forget reports -1, and it fails on "a ledger made
    before retraction refused it". Red against the pre-fix chain end: the same failure line.
    """
    path = tmp_path / "old.db"
    LiteStore(path, dim=DIM).close()
    with sqlite3.connect(path) as conn:
        conn.executescript(_OLD_LEDGER_DDL)
    store = LiteStore(path, dim=DIM)
    remember(store, "tea.md")
    remember(store, "door.md")
    apply(store, TEA, "tea.md", "r1")
    apply(store, DOOR, "door.md", "r2")

    forget_memory(store, ["door.md"])

    assert current(store) == ["tea"], f"a ledger made before retraction refused it: {current(store)}"
    assert {"r1", "r2"} <= {e.request_id for e in LiteFactLedger(store).events}
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute("DELETE FROM recall_fact_ledger_events")
