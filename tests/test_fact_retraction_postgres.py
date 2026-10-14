"""Forgetting a memo retracts its facts on PostgreSQL, exactly as on a lite store.

The lite file is covered by `tests/test_fact_retraction.py`; this module covers what only a database
can show: migration 0026 admits a `retracted` event through the protected append function, and the
same forget sequence writes the same retraction events (ids, request ids, codes, cards) on both
backends.

Invariants and the failure each one catches:
- P1 a fact whose only memo is forgotten is no longer current on PostgreSQL.
- P2 PostgreSQL and lite record the same events for apply, duplicate, forget one memo, forget the
  other, and re-learn, and project the same current facts after each step.

Red proof (2026-10-14), each then restored and green; node ids and failure lines in each docstring.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from recall.fact_ledger import PostgresFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from recall.store import PgVectorStore
from recall.types import AtomicFact, Chunk
from recall_mcp.lifecycle import forget_memory
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW, evidence_card

pytestmark = requires_db
DIM = 4
GEN = "generation-retraction"
DOOR = AtomicFact(namespace="memory", subject="user", predicate="door_code", object="4512")


def _remember(store: Any, name: str) -> None:
    store.upsert(
        [Chunk(f"chunk-{name}", f"/notes/{name}", f"text of {name}", {"file": name})],
        [[1.0] + [0.0] * (DIM - 1)],
    )


def _apply(store: Any, cards: Any, ledger: Any, name: str, request_id: str) -> str:
    card = evidence_card(DOOR, tenant=store.tenant, generation=GEN, chunk=f"chunk-{name}", source=name)
    cards.put((card,))
    decision = ProvenanceController(
        tenant_id=store.tenant, generation_id=GEN, cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="retraction-test",
    ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), request_id))
    return str(decision.code)


def _view(event: Any) -> tuple[Any, ...]:
    """Everything but `created_at`: a forget stamps the wall clock, which two runs cannot share."""
    return (
        event.event_id, event.event_type, event.fact_id, event.generation_id,
        [card.card_id for card in event.evidence_cards], list(event.supersedes_fact_ids),
        event.request_id, event.writer, event.decision_code,
    )


def _pg(make_store: Callable[[int], PgVectorStore], tenant: str) -> PgVectorStore:
    table = make_store(DIM).table  # a migrated throwaway table, dropped after the test
    return PgVectorStore(TEST_DSN, dim=DIM, table=table, tenant=tenant)


def test_p1_forgetting_a_facts_only_memo_retracts_it_on_postgres(make_store: Callable[[int], PgVectorStore]) -> None:
    """Node tests/test_fact_retraction_postgres.py::test_p1_forgetting_a_facts_only_memo_retracts_it_on_postgres.

    Red against the pre-fix chain end `3d7d42b27`: fails on "the fact outlived its forgotten
    memo". Red, with migration 0026 removed from the package (its SQL file and checksum entry):
    the append function refuses the `retracted` envelope, the forget reports -1 and the same
    assertion fails.
    """
    tenant = f"retract-{uuid4().hex}"
    store = _pg(make_store, tenant)
    try:
        _remember(store, "door.md")
        ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
        assert _apply(store, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant), ledger, "door.md", "r1") == "APPLIED"

        result = forget_memory(store, ["door.md"])

        listed = [e.fact_id for e in ledger.current(tenant_id=tenant, now=NOW)]
        assert listed == [], f"the fact outlived its forgotten memo: {listed} ({result.message})"
        assert result.facts_retracted == 1
        assert [e.decision_code for e in ledger.events if e.event_type == "retracted"] == [DecisionCode.SOURCE_FORGOTTEN]
    finally:
        store.close()


def _scenario(store: Any, cards: Any, ledger: Callable[[], Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    def step(label: str, value: Any) -> None:
        out.append((label, value))
        out.append((f"{label}: current", [e.event_id for e in ledger().current(tenant_id=store.tenant, now=NOW)]))

    _remember(store, "a.md")
    _remember(store, "b.md")
    step("apply a", _apply(store, cards, ledger(), "a.md", "r1"))
    step("duplicate b", _apply(store, cards, ledger(), "b.md", "r2"))
    step("forget a", forget_memory(store, ["a.md"]).facts_retracted)
    step("forget b", forget_memory(store, ["b.md"]).facts_retracted)
    _remember(store, "c.md")
    step("re-learn c", _apply(store, cards, ledger(), "c.md", "r3"))
    out.append(("events", sorted(_view(e) for e in ledger().events)))
    return out


def test_p2_postgres_and_lite_retract_the_same(tmp_path: Path, make_store: Callable[[int], PgVectorStore]) -> None:
    """Node tests/test_fact_retraction_postgres.py::test_p2_postgres_and_lite_retract_the_same.

    Red, mutation of `LiteFactLedger.retract_forgotten` (recall/lite/facts.py) minting its event id
    from `retraction_ids(...)` plus a suffix: fails at step "events". Red against the pre-fix
    chain end: `ForgetResult` has no `facts_retracted`, so the scenario raises AttributeError at
    "forget a" on both backends; that is not an assertion failure, so the mutation is the proof.
    """
    tenant = f"retract-parity-{uuid4().hex}"
    pg_store = _pg(make_store, tenant)
    lite_store = LiteStore(tmp_path / "parity.db", dim=DIM, tenant=tenant)
    try:
        expected = _scenario(
            pg_store, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant),
            lambda: PostgresFactLedger(TEST_DSN, tenant_id=tenant),
        )
        actual = _scenario(lite_store, LiteEvidenceCardStore(lite_store), lambda: LiteFactLedger(lite_store))
    finally:
        pg_store.close()
        lite_store.close()
    assert [label for label, _ in expected] == [label for label, _ in actual]
    for (label, want), (_, got) in zip(expected, actual, strict=True):
        assert got == want, f"step {label}: postgres {want!r} != lite {got!r}"
    steps = dict(expected)
    # The scenario must exercise what it claims: a kept fact, a retraction, a new application.
    assert steps["apply a"] == "APPLIED" and steps["duplicate b"] == "DUPLICATE"
    assert steps["forget a"] == 0 and steps["forget a: current"] != []
    assert steps["forget b"] == 1 and steps["forget b: current"] == []
    assert steps["re-learn c"] == "APPLIED" and len(steps["re-learn c: current"]) == 1
