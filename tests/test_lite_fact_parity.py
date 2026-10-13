"""Parity: the fact ledger, evidence cards and receipts answer identically on PostgreSQL and lite.

One scripted sequence runs against each backend with the same tenant, generation and cards, and
the two transcripts must be equal step by step, with nothing normalised: decision codes, event
ids, card ids, fact payloads, the current projection at three instants, the full event stream,
and the receipt replay and conflict answers. Invariant: a client of `recall_apply_fact` and
`recall_current_facts` cannot tell the backends apart.

Red proof (2026-10-12), each mutation alone, then restored and green:
- `LiteFactLedger.apply_assertion` minting `assertion_event_id(...) + "x"`: fails at step
  "apply tea" (the event id differs).
- the lite `superseded` event written with `fact=fact` (the shape `SQLiteFactLedger` inherits from
  the in-memory ledger): fails at step "events".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from recall.errors import IdempotencyConflict
from recall.fact_ledger import PostgresFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import FactApplicationRequest, ProvenanceController
from recall.store import PgVectorStore
from recall.types import AtomicFact, EvidenceCard
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import COFFEE, NOW, TEA, evidence_card

pytestmark = requires_db
GEN = "generation-parity"
TRIP = AtomicFact(
    namespace="memory", subject="user", predicate="trip", object="lisbon",
    valid_from=NOW + timedelta(days=2), valid_until=NOW + timedelta(days=9),
)


def card(tenant: str, fact: AtomicFact, chunk: str, source: str, links: tuple[str, ...] = ()) -> EvidenceCard:
    return evidence_card(fact, tenant=tenant, generation=GEN, chunk=chunk, source=source, links=links)


@dataclass
class Backend:
    tenant: str
    cards: Any
    ledger: Callable[[], Any]
    receipts: Any  # has get_operation_receipt and record_operation_receipt


def _event_view(event: Any) -> tuple[Any, ...]:
    return (
        event.event_id, event.event_type, event.fact_id,
        event.fact.to_payload() if event.fact else None,
        [c.card_id for c in event.evidence_cards], list(event.supersedes_fact_ids),
        event.request_id, event.writer, event.decision_code, event.created_at.isoformat(),
    )


def scenario(b: Backend) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    tea = card(b.tenant, TEA, "chunk-tea", "tea.md")
    coffee_bare = card(b.tenant, COFFEE, "chunk-coffee-1", "coffee-draft.md")
    coffee = card(b.tenant, COFFEE, "chunk-coffee-2", "coffee.md", links=("tea.md",))
    trip = card(b.tenant, TRIP, "chunk-trip", "trip.md")
    b.cards.put((tea, coffee_bare, coffee, trip))
    out.append(("card ids", [c.card_id for c in (tea, coffee_bare, coffee, trip)]))
    out.append(("resolve", b.cards.resolve(coffee.card_id)))

    def controller() -> ProvenanceController:
        return ProvenanceController(
            tenant_id=b.tenant, generation_id=GEN, cards=b.cards, ledger=b.ledger(),
            source_digest_for=lambda _c: "digest", now=lambda: NOW, writer="parity",
        )

    def apply(fact: AtomicFact, ids: tuple[str, ...], rid: str) -> tuple[Any, ...]:
        d = controller().apply_fact(FactApplicationRequest(fact, ids, rid))
        return (
            d.allowed, str(d.code), d.fact_id, d.retried, d.detail,
            d.event.event_id if d.event else None, [c.card_id for c in d.cards],
        )

    out.append(("apply tea", apply(TEA, (tea.card_id,), "r1")))
    out.append(("replay r1", apply(TEA, (tea.card_id,), "r1")))
    out.append(("duplicate tea", apply(TEA, (tea.card_id,), "r2")))
    out.append(("contradiction", apply(COFFEE, (coffee_bare.card_id,), "r3")))
    out.append(("replay refused r3", apply(COFFEE, (coffee_bare.card_id,), "r3")))
    out.append(("supersede", apply(COFFEE, (coffee.card_id,), "r4")))
    out.append(("future trip", apply(TRIP, (trip.card_id,), "r5")))
    out.append(("unknown card", apply(TEA, ("card-missing",), "r6")))
    ledger = b.ledger()
    for label, at in (
        ("now", NOW), ("during trip", NOW + timedelta(days=3)), ("after trip", NOW + timedelta(days=10)),
    ):
        out.append((f"current {label}", [_event_view(e) for e in ledger.current(tenant_id=b.tenant, now=at)]))
    out.append(("events", [_event_view(e) for e in ledger.events]))
    out.append(("request r3", _event_view(ledger.request_event(tenant_id=b.tenant, request_id="r3"))))
    out.append(("request none", ledger.request_event(tenant_id=b.tenant, request_id="nope")))

    rec = b.receipts
    rec.record_operation_receipt("k1", '{"ok": true}', operation="recall_apply_fact", request_fingerprint="f1")
    rec.record_operation_receipt("k1", '{"ok": false}', operation="recall_apply_fact", request_fingerprint="f1")
    out.append(("receipt replay", rec.get_operation_receipt("k1", operation="recall_apply_fact", request_fingerprint="f1")))
    for label, kwargs in (
        ("other arguments", {"operation": "recall_apply_fact", "request_fingerprint": "f2"}),
        ("other operation", {"operation": "recall_forget", "request_fingerprint": "f1"}),
    ):
        try:
            rec.get_operation_receipt("k1", **kwargs)
            out.append((label, "replayed"))
        except IdempotencyConflict:
            out.append((label, "IdempotencyConflict"))
    out.append(("receipt missing", rec.get_operation_receipt("k2", operation="recall_apply_fact", request_fingerprint="f1")))
    return out


def test_postgres_and_lite_answer_the_same(tmp_path: Path, make_store: Callable[[int], PgVectorStore]) -> None:
    tenant = f"parity-{uuid4().hex}"
    table_store = make_store(4)  # a migrated throwaway table, dropped after the test
    pg_store = PgVectorStore(TEST_DSN, dim=4, table=table_store.table, tenant=tenant)
    postgres = Backend(
        tenant, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant),
        lambda: PostgresFactLedger(TEST_DSN, tenant_id=tenant), pg_store,
    )
    lite_store = LiteStore(tmp_path / "parity.db", dim=4, tenant=tenant)
    lite = Backend(tenant, LiteEvidenceCardStore(lite_store), lambda: LiteFactLedger(lite_store), lite_store)

    expected, actual = scenario(postgres), scenario(lite)
    assert [label for label, _ in expected] == [label for label, _ in actual]
    for (label, want), (_, got) in zip(expected, actual, strict=True):
        assert got == want, f"step {label}: postgres {want!r} != lite {got!r}"
    # The scenario must exercise what it claims: an applied fact, a refusal, a supersession.
    steps = dict(expected)
    assert steps["apply tea"][1] == "APPLIED" and steps["contradiction"][1] == "CONTRADICTION_WITHOUT_SUPERSESSION"
    assert steps["supersede"][0] is True and [v[3]["object"] for v in steps["current now"]] == ["coffee"]
