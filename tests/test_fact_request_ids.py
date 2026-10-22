"""A request id names one application, and the ids the ledger mints are not a client's to send.

Two defects from the 2026-10-21 supersession audit, re-measured still open on master `52be498cc`
on 2026-10-22:

- P1 #7: a request id sent again with another fact, or with the same fact on other cards, was
  answered allowed DUPLICATE with the old event; nothing new was recorded while the decision named
  the new fact. It is now refused `REQUEST_ID_REUSED` and not recorded, and the same request sent
  again stays an idempotent DUPLICATE.
- The ledger's own ids (`<id>:supersession`, `retraction:<event id>`, `stated:<hash>`) shared the
  client's namespace. A client request under one took the server's slot: PostgreSQL, SQLite and
  lite answered DUPLICATE and recorded nothing, while the in-memory ledger, which never
  registered the supersession id, appended a second row under it. It is now refused
  `REQUEST_ID_RESERVED` before the ledger is read, and every ledger holds the id it minted.

Every test runs on the in-memory, SQLite and lite ledgers, and on PostgreSQL where a test database
is configured; `test_q6` compares the backends' transcripts step by step.

Red proof (2026-10-22): each test's docstring names its baseline or mutation and the assertion that
failed. "Master" means `52be498cc` with this file copied in. The new decision codes are compared
as strings, so a pre-fix tree fails in the assertion rather than on a missing name.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from recall.fact_ledger import PostgresFactLedger, SQLiteFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import (
    _PERMIT_SENTINEL,
    ControllerDecision,
    DecisionCode,
    EvidenceCardStore,
    FactApplicationPermit,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
    fact_identity,
)
from recall.types import AtomicFact, EvidenceCard
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import evidence_card

GEN = "generation-request-ids"
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)


def fact(value: str, predicate: str = "city") -> AtomicFact:
    return AtomicFact(namespace="memory", subject="person:alice", predicate=predicate, object=value)


PARIS, ROME, LISBON = fact("Paris"), fact("Rome"), fact("Lisbon")
PHONE, EYES = fact("+33 1 23", "phone"), fact("blue", "eyes")


class Env:
    """A controller over one ledger, one tenant, a fixed clock."""

    def __init__(self, ledger: Any, tenant: str) -> None:
        self.ledger = ledger
        self.tenant = tenant
        self.cards = EvidenceCardStore()

    def card(self, claim: AtomicFact, memo: str, links: tuple[str, ...] = ()) -> EvidenceCard:
        card = evidence_card(
            claim, tenant=self.tenant, generation=GEN, chunk=f"chunk-{memo}", source=memo, links=links
        )
        self.cards.put([card])
        return card

    def apply(
        self, claim: AtomicFact, memo: str, rid: str, links: tuple[str, ...] = ()
    ) -> ControllerDecision:
        card = self.card(claim, memo, links)
        return ProvenanceController(
            tenant_id=self.tenant, generation_id=GEN, cards=self.cards, ledger=self.ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="request-ids",
        ).apply_fact(FactApplicationRequest(claim, (card.card_id,), rid))

    def current(self) -> list[str]:
        return sorted(str(e.fact.object) for e in self.ledger.current(tenant_id=self.tenant, now=NOW))

    def event_ids(self) -> list[str]:
        return [event.event_id for event in self.ledger.events]


def _env(backend: str, tmp_path: Path, request: pytest.FixtureRequest) -> Env:
    tenant = f"tenant-request-ids-{uuid4().hex[:12]}"
    if backend == "memory":
        return Env(InMemoryFactLedger(), tenant)
    if backend == "sqlite":
        return Env(SQLiteFactLedger(str(tmp_path / f"{tenant}.db"), tenant_id=tenant), tenant)
    if backend == "lite":
        return Env(LiteFactLedger(LiteStore(tmp_path / f"{tenant}-lite.db", dim=4, tenant=tenant)), tenant)
    request.getfixturevalue("make_store")(4)  # a migrated test database, or a skip
    return Env(PostgresFactLedger(TEST_DSN, tenant_id=tenant), tenant)


BACKENDS = ("memory", "sqlite", "lite", pytest.param("postgres", marks=requires_db))


@pytest.fixture(params=BACKENDS)
def env(request: pytest.FixtureRequest, tmp_path: Path) -> Env:
    return _env(request.param, tmp_path, request)


@pytest.mark.parametrize("shape", ["another predicate", "a superseding value", "a contradicting value"])
def test_q1_a_used_request_id_with_another_fact_is_refused_and_records_nothing(env: Env, shape: str) -> None:
    """Node tests/test_fact_request_ids.py::test_q1_a_used_request_id_with_another_fact_is_refused_and_records_nothing.

    Red against master, every backend: the first two shapes fail on "a used request id was
    allowed for another fact" (allowed DUPLICATE, carrying the Paris event); the third, refused
    on master too, fails on the code (CONTRADICTION_WITHOUT_SUPERSESSION). Red, mutation removing
    the controller's fact-id check in `ProvenanceController.apply_fact`: the third shape fails on
    the code the same way, the ledger refusing the other two by itself.
    """
    assert env.apply(PARIS, "m1.md", "req").code == DecisionCode.APPLIED
    before = env.event_ids()
    if shape == "another predicate":
        decision = env.apply(PHONE, "m9.md", "req")
    elif shape == "a superseding value":
        decision = env.apply(ROME, "m2.md", "req", links=("m1.md",))
    else:
        decision = env.apply(ROME, "m2.md", "req")
    assert not decision.allowed, f"a used request id was allowed for another fact: {decision.code}"
    assert decision.code == "REQUEST_ID_REUSED", decision
    assert decision.event is not None and decision.event.fact_id == fact_identity(PARIS)
    assert env.event_ids() == before, "a refusal under a used request id was recorded"
    assert env.current() == ["Paris"]


def test_q2_the_same_fact_on_other_cards_under_a_used_id_is_refused(env: Env) -> None:
    """Node tests/test_fact_request_ids.py::test_q2_the_same_fact_on_other_cards_under_a_used_id_is_refused.

    The controller cannot see this one (the fact id matches), so every ledger refuses it on the
    resolved cards (`replays_request`). Red against master, every backend: fails on "the same fact
    on another memo's card was allowed under a used id" (allowed DUPLICATE, nothing recorded).
    Red, mutation dropping the card-id term from `replays_request`: the same line.
    """
    assert env.apply(PARIS, "m1.md", "req").code == DecisionCode.APPLIED
    before = env.event_ids()
    decision = env.apply(PARIS, "m2.md", "req")
    assert not decision.allowed, (
        f"the same fact on another memo's card was allowed under a used id: {decision.code}"
    )
    assert decision.code == "REQUEST_ID_REUSED", decision
    assert env.event_ids() == before, "a refusal under a used request id was recorded"
    # Under an id of its own the second memo is recorded, as a DUPLICATE support.
    assert env.apply(PARIS, "m2.md", "req-2").code == DecisionCode.DUPLICATE


def test_q3_the_same_request_sent_again_stays_an_idempotent_duplicate(env: Env) -> None:
    """Node tests/test_fact_request_ids.py::test_q3_the_same_request_sent_again_stays_an_idempotent_duplicate.

    The control for q1 and q2: green on master. Red, mutation of `replays_request`
    (recall/provenance_controller.py) replaying only an APPLIED event (`asserts_fact(prior)` in
    place of `prior.event_type == "asserted"`): fails on "a replay of a DUPLICATE request was
    refused" on every backend.
    """
    first = env.apply(PARIS, "m1.md", "r1")
    second = env.apply(PARIS, "m2.md", "r2")
    assert (first.code, second.code) == (DecisionCode.APPLIED, DecisionCode.DUPLICATE)
    before = env.event_ids()
    replay_first = env.apply(PARIS, "m1.md", "r1")
    replay_second = env.apply(PARIS, "m2.md", "r2")
    assert replay_first.allowed and replay_first.code == DecisionCode.DUPLICATE, replay_first
    assert replay_second.allowed and replay_second.code == DecisionCode.DUPLICATE, (
        f"a replay of a DUPLICATE request was refused: {replay_second.code}"
    )
    assert replay_first.event is not None and first.event is not None
    assert replay_first.event.event_id == first.event.event_id
    assert env.event_ids() == before, "a replay recorded a second event"


@pytest.mark.parametrize("form", ["supersession", "retraction", "stated"])
def test_q4_a_client_request_under_a_reserved_id_is_refused_and_records_nothing(env: Env, form: str) -> None:
    """Node tests/test_fact_request_ids.py::test_q4_a_client_request_under_a_reserved_id_is_refused_and_records_nothing.

    Red against master, every backend and form: fails on "a client request under the reserved id
    ... was allowed" (`supersession`: APPLIED on the in-memory ledger, DUPLICATE elsewhere;
    `retraction` and `stated`: APPLIED, taking the id a later retraction or statement needs).
    Red, mutation disabling the `reserved_request_id` check in `ProvenanceController.apply_fact`:
    `retraction` and `stated` fail on the same line, `supersession` on the code
    (`REQUEST_ID_REUSED`, the ledger refusing what the controller let through).
    """
    paris = env.apply(PARIS, "m1.md", "r1")
    assert env.apply(ROME, "m2.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    assert paris.event is not None
    rid = {
        "supersession": "r2:supersession",
        "retraction": "retraction:" + paris.event.event_id,
        "stated": "stated:" + "0" * 32,
    }[form]
    before = env.event_ids()
    decision = env.apply(EYES, "m5.md", rid)
    assert not decision.allowed, f"a client request under the reserved id {rid!r} was allowed: {decision.code}"
    assert decision.code == "REQUEST_ID_RESERVED", decision
    assert env.event_ids() == before, "a refusal under a reserved id was recorded"
    assert env.current() == ["Rome"]


def test_q5_every_ledger_holds_the_supersession_id_it_minted(env: Env) -> None:
    """Node tests/test_fact_request_ids.py::test_q5_every_ledger_holds_the_supersession_id_it_minted.

    Below the controller: the ledger API itself. Red against master: on the in-memory ledger
    fails on "the ledger does not hold the id it minted" (`request_event` answered None); on
    SQLite, lite and PostgreSQL on "a raw assertion under the supersession id was answered with
    the superseded event" (DUPLICATE, no error). Red, mutation of `replays_request` dropping its
    `prior.event_type == "asserted"` term: the in-memory and SQLite ledgers, whose `superseded`
    event carries the successor's fact and cards, fail on the same line.
    """
    env.apply(PARIS, "m1.md", "r1")
    rome = env.apply(ROME, "m2.md", "r2", links=("m1.md",))
    assert rome.code == DecisionCode.APPLIED
    minted = env.ledger.request_event(tenant_id=env.tenant, request_id="r2:supersession")
    assert minted is not None and minted.event_type == "superseded", "the ledger does not hold the id it minted"
    card = rome.cards[0]
    permit = FactApplicationPermit(
        tenant_id=env.tenant, generation_id=GEN, request_id="r2:supersession",
        fact_id=fact_identity(ROME), card_ids=(card.card_id,), _token=_PERMIT_SENTINEL,
    )
    before = env.event_ids()
    try:
        result = env.ledger.apply_assertion(
            tenant_id=env.tenant, generation_id=GEN, fact=ROME, cards=(card,),
            request_id="r2:supersession", writer="raw", permit=permit, now=NOW,
        )
    except ValueError as exc:
        assert str(exc) == "REQUEST_ID_REUSED"
    else:
        pytest.fail(f"a raw assertion under the supersession id was answered with the {result.event.event_type} event")
    assert env.event_ids() == before


def _transcript(env: Env) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    def step(label: str, decision: ControllerDecision) -> None:
        out.append((label, (decision.allowed, str(decision.code))))

    step("paris", env.apply(PARIS, "m1.md", "r1"))
    step("replay", env.apply(PARIS, "m1.md", "r1"))
    step("paris from a second memo", env.apply(PARIS, "m2.md", "r2"))
    step("another fact, used id", env.apply(PHONE, "m9.md", "r1"))
    step("other cards, used id", env.apply(PARIS, "m3.md", "r1"))
    step("rome superseding the first memo", env.apply(ROME, "m4.md", "r3", links=("m1.md",)))
    step("supersession id", env.apply(EYES, "m5.md", "r3:supersession"))
    step("retraction id", env.apply(EYES, "m5.md", "retraction:evt_0"))
    step("stated id", env.apply(EYES, "m5.md", "stated:" + "0" * 32))
    step("rome from a second memo", env.apply(ROME, "m6.md", "r4"))
    # Linking the memo that states Rome only through the DUPLICATE (the third fix of the audit).
    step("lisbon superseding the second memo", env.apply(LISBON, "m7.md", "r5", links=("m6.md",)))
    out.append(("current", env.current()))
    out.append(("rows per request id", sorted(Counter(e.request_id for e in env.ledger.events).items())))
    out.append(("rows per type", sorted(Counter(e.event_type for e in env.ledger.events).items())))
    minted = env.ledger.request_event(tenant_id=env.tenant, request_id="r3:supersession")
    out.append(("minted id", None if minted is None else minted.event_type))
    return out


@pytest.mark.parametrize("backend", ("sqlite", "lite", pytest.param("postgres", marks=requires_db)))
def test_q6_every_ledger_answers_reused_and_reserved_ids_alike(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Node tests/test_fact_request_ids.py::test_q6_every_ledger_answers_reused_and_reserved_ids_alike.

    The in-memory ledger is the reference. Red against master, each backend: fails at step
    "supersession id" (in-memory APPLIED, the others DUPLICATE, and both allowed). Red, mutation
    removing the in-memory ledger's registration of the supersession id: fails at step "minted id".
    """
    expected = _transcript(_env("memory", tmp_path, request))
    actual = _transcript(_env(backend, tmp_path, request))
    assert [label for label, _ in expected] == [label for label, _ in actual]
    for (label, want), (_, got) in zip(expected, actual, strict=True):
        assert got == want, f"step {label}: memory {want!r} != {backend} {got!r}"
    steps = dict(expected)
    # The script must exercise what it names.
    assert steps["replay"] == (True, "DUPLICATE")
    assert steps["another fact, used id"] == (False, "REQUEST_ID_REUSED")
    assert steps["supersession id"] == (False, "REQUEST_ID_RESERVED")
    assert steps["lisbon superseding the second memo"] == (True, "APPLIED")
    assert steps["current"] == ["Lisbon"]
