"""The fact ledger's supersession: assertion events, the successor's window, cardinality, NFC.

Owner decision, 2026-10-21: a successor hides its predecessor only inside the successor's own
validity window, so an expired bounded successor gives the old value back. Defects from the
2026-10-21 supersession audit: P1 #2 (a `superseded` event hid the predecessor for all valid
time), P1 #3 (supersession keyed on the fact id, so Paris, Rome, Paris again was never current),
P2 #11 (`facts_conflict` compared raw strings while the key is NFC), P2 #12 (supersedes derived
from what is current now, conflicts from all time), and the set-valued predicate case.

Every test here drives the real controller and real ledgers through names the pre-fix tree has,
so each red proof is master ``67ffcce59`` with this file copied in, unless the docstring names a
mutation instead.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from recall.fact_ledger import SQLiteFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import (
    CONTROLLER_SCHEMA_VERSION,
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    FactEvent,
    InMemoryFactLedger,
    ProvenanceController,
    fact_identity,
    fact_supersession,
    facts_conflict,
    project_current,
    split_contested,
)
from recall.types import AtomicFact
from tests.fact_fixtures import evidence_card

TENANT = "tenant-windows"
GEN = "generation-windows"
NOW = datetime(2026, 10, 21, 12, tzinfo=UTC)


def city(value: str, *, start: datetime | None = None, end: datetime | None = None) -> AtomicFact:
    return AtomicFact(
        namespace="memory", subject="person:alice", predicate="city", object=value,
        valid_from=start, valid_until=end,
    )


class Env:
    """A controller over one ledger, with a clock the test moves."""

    def __init__(self, ledger: Any) -> None:
        self.ledger = ledger
        self.cards = EvidenceCardStore()
        self.clock = NOW

    def apply(self, fact: AtomicFact, memo: str, rid: str, links: tuple[str, ...] = ()) -> Any:
        card = evidence_card(fact, tenant=TENANT, generation=GEN, chunk=f"chunk-{memo}", source=memo, links=links)
        self.cards.put([card])
        controller = ProvenanceController(
            tenant_id=TENANT, generation_id=GEN, cards=self.cards, ledger=self.ledger,
            source_digest_for=lambda _card: "digest", now=lambda: self.clock, writer="windows",
        )
        return controller.apply_fact(FactApplicationRequest(fact, (card.card_id,), rid))

    def current(self, at: datetime | None = None) -> list[Any]:
        return sorted(str(e.fact.object) for e in self.ledger.current(tenant_id=TENANT, now=at or self.clock))


LEDGERS: dict[str, Callable[[Path], Any]] = {
    "memory": lambda _tmp: InMemoryFactLedger(),
    "sqlite": lambda tmp: SQLiteFactLedger(str(tmp / "ledger.db"), tenant_id=TENANT),
    "lite": lambda tmp: LiteFactLedger(LiteStore(tmp / "lite.db", dim=4, tenant=TENANT)),
}


@pytest.fixture(params=sorted(LEDGERS))
def env(request: pytest.FixtureRequest, tmp_path: Path) -> Env:
    return Env(LEDGERS[request.param](tmp_path))


def test_f1_a_future_dated_successor_leaves_the_present_value_current(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f1_a_future_dated_successor_leaves_the_present_value_current.

    Invariant: Rome from +90 days superseding Paris hides Paris only from then, and never in
    the past. Red on ``67ffcce59``: fails on "a successor not yet valid emptied the present"
    (current was ``[]`` now and a year back, audit P1 #2).
    """
    assert env.apply(city("Paris"), "m1.md", "r1").code == DecisionCode.APPLIED
    move = city("Rome", start=NOW + timedelta(days=90))
    assert env.apply(move, "move.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    assert env.current() == ["Paris"], f"a successor not yet valid emptied the present: {env.current()}"
    assert env.current(NOW + timedelta(days=91)) == ["Rome"]
    assert env.current(NOW - timedelta(days=365)) == ["Paris"]


def test_f2_an_expired_bounded_successor_gives_the_old_value_back(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f2_an_expired_bounded_successor_gives_the_old_value_back.

    Invariant (the cost the owner accepted): a stay in Rome from 60 to 30 days ago hides Paris
    only then. Red on ``67ffcce59``: fails on "an expired successor still hides the old value".
    """
    env.apply(city("Paris"), "m1.md", "r1")
    stay = city("Rome", start=NOW - timedelta(days=60), end=NOW - timedelta(days=30))
    assert env.apply(stay, "stay.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    assert env.current() == ["Paris"], f"an expired successor still hides the old value: {env.current()}"
    assert env.current(NOW - timedelta(days=45)) == ["Rome"]


def test_f3_a_retroactive_correction_leaves_the_open_ended_present_alone(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f3_a_retroactive_correction_leaves_the_open_ended_present_alone.

    Invariant: a correction for 2020 to 2022 of an open-ended value hides it in 2021 only. Red on
    ``67ffcce59``: fails on "a correction of the past emptied the present".
    """
    since_2020 = datetime(2020, 1, 1, tzinfo=UTC)
    env.apply(city("Paris", start=since_2020), "m1.md", "r1")
    fix = city("Rome", start=since_2020, end=datetime(2022, 1, 1, tzinfo=UTC))
    assert env.apply(fix, "fix.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    assert env.current() == ["Paris"], f"a correction of the past emptied the present: {env.current()}"
    assert env.current(datetime(2021, 6, 1, tzinfo=UTC)) == ["Rome"]


def test_f4_a_chain_whose_middle_member_is_temporary(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f4_a_chain_whose_middle_member_is_temporary.

    Invariant: Paris; then Rome (January to April) superseding it; then Milan (from March)
    superseding Rome. Milan names only Rome, yet hides Paris through it wherever Milan holds, so
    Milan is no contradiction of Paris and is applied; Paris is current before January only.
    Red on ``67ffcce59``: fails on "a successor of a temporary value was refused" (Milan was
    refused as contradicting Paris, which it never names).
    """
    year = 2026
    env.apply(city("Paris"), "m1.md", "r1")
    rome = city("Rome", start=datetime(year, 1, 1, tzinfo=UTC), end=datetime(year, 4, 1, tzinfo=UTC))
    env.apply(rome, "rome.md", "r2", links=("m1.md",))
    milan = city("Milan", start=datetime(year, 3, 1, tzinfo=UTC))
    decision = env.apply(milan, "milan.md", "r3", links=("rome.md",))
    assert decision.code == DecisionCode.APPLIED, f"a successor of a temporary value was refused: {decision.code}"
    assert env.current(datetime(year, 2, 1, tzinfo=UTC)) == ["Rome"]
    assert env.current(datetime(year, 3, 15, tzinfo=UTC)) == ["Milan"]
    assert env.current(datetime(year, 6, 1, tzinfo=UTC)) == ["Milan"]
    assert env.current(datetime(year - 1, 12, 1, tzinfo=UTC)) == ["Paris"]


def test_f5_paris_then_rome_then_paris_again_leaves_paris_current(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f5_paris_then_rome_then_paris_again_leaves_paris_current.

    Invariant: supersession hides the assertion EVENTS recorded before it, never the fact id for
    good. All three applications share one fixed clock, which is the case `ordered_after` keeps
    in order. Red on ``67ffcce59``: fails on "a value asserted again after its successor is never
    current" (the third application was APPLIED and current was ``[]``, audit P1 #3).
    """
    env.apply(city("Paris"), "m1.md", "r1")
    assert env.apply(city("Rome"), "m2.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    again = env.apply(city("Paris"), "m3.md", "r3", links=("m2.md",))
    assert again.code == DecisionCode.APPLIED
    assert env.current() == ["Paris"], f"a value asserted again after its successor is never current: {env.current()}"
    assert fact_identity(again.event.fact) == fact_identity(city("Paris"))


def test_f6_an_authored_link_supersedes_an_expired_overlapping_value(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f6_an_authored_link_supersedes_an_expired_overlapping_value.

    Invariant: the controller derives what a claim supersedes from the same set the ledger
    checks conflicts against (every value in force at some instant of the claim's window), so a
    correction linking an expired value is applied. Red on ``67ffcce59``: fails on "an authored
    link could not supersede an expired value" (CONTRADICTION_WITHOUT_SUPERSESSION, audit P2 #12).
    """
    past = NOW - timedelta(days=400)
    env.clock = past - timedelta(days=10)
    env.apply(city("Paris", end=past), "m1.md", "r1")
    env.clock = NOW
    correction = env.apply(city("Rome"), "m2.md", "r2", links=("m1.md",))
    assert correction.code == DecisionCode.APPLIED, f"an authored link could not supersede an expired value: {correction.code}"
    assert env.current() == ["Rome"]
    assert env.current(past - timedelta(days=5)) == ["Rome"], "the correction holds over the old window too"


def test_f7_a_future_value_can_be_corrected_before_it_starts(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f7_a_future_value_can_be_corrected_before_it_starts.

    Invariant: as f6, for a value that has not started. Red on ``67ffcce59``: fails on "a future
    value could not be corrected" (CONTRADICTION_WITHOUT_SUPERSESSION).
    """
    start = NOW + timedelta(days=30)
    env.apply(city("Rome", start=start), "plan.md", "r1")
    decision = env.apply(city("Milan", start=start), "plan2.md", "r2", links=("plan.md",))
    assert decision.code == DecisionCode.APPLIED, f"a future value could not be corrected: {decision.code}"
    assert env.current(start + timedelta(days=1)) == ["Milan"]


def test_f8_forgetting_a_successor_still_does_not_revive_what_it_hid(tmp_path: Path) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f8_forgetting_a_successor_still_does_not_revive_what_it_hid.

    A CONTROL for `test_r5_forgetting_a_successor_does_not_revive_what_it_superseded`, now that
    hiding depends on the successor's window: the window, not whether the successor is still
    live, decides. Green on ``67ffcce59``. Red proof by mutation of
    `recall.provenance_controller.project_current`: build the supersession index from the live
    (unretracted) assertions only; fails on "forgetting the successor revived the old value:
    ['Paris']" (2026-10-21). A mutation skipping retracted successors inside `fact_supersession`
    did NOT go red: the successor's own `superseded` event still hides as an unknown-window
    successor, which is the backstop working.
    """
    store = LiteStore(tmp_path / "lite.db", dim=4, tenant=TENANT)
    env = Env(LiteFactLedger(store))
    env.apply(city("Paris"), "m1.md", "r1")
    env.apply(city("Rome"), "m2.md", "r2", links=("m1.md",))
    retracted = env.ledger.retract_forgotten(
        tenant_id=TENANT, touched=lambda c: c.source == "m2.md", gone=lambda c: c.source == "m2.md", now=NOW,
    )
    assert len(retracted) == 1
    assert env.current() == [], f"forgetting the successor revived the old value: {env.current()}"


def test_f9_two_applications_at_one_clock_reading_are_recorded_in_order(tmp_path: Path) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f9_two_applications_at_one_clock_reading_are_recorded_in_order.

    Invariant: the events of one conflict key are strictly ordered by `created_at` however the
    clock reads (`recall.provenance_controller.ordered_after`), so ledger order is write order.
    Red on ``67ffcce59``: fails on "two applications of one key share a timestamp".
    """
    env = Env(SQLiteFactLedger(str(tmp_path / "ledger.db"), tenant_id=TENANT))
    env.apply(city("Paris"), "m1.md", "r1")
    env.apply(city("Rome"), "m2.md", "r2", links=("m1.md",))
    env.apply(city("Paris"), "m3.md", "r3", links=("m2.md",))
    stamps = [e.created_at for e in env.ledger.events if e.event_type == "asserted"]
    assert len(set(stamps)) == len(stamps), f"two applications of one key share a timestamp: {stamps}"
    assert stamps == sorted(stamps)


# ------------------------------------------------------------------ cardinality and NFC


def test_c1_a_second_email_is_not_a_contradiction(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_c1_a_second_email_is_not_a_contradiction.

    Invariant: `email` is set-valued (`DEFAULT_PREDICATE_CARDINALITY`), so a second value is a
    second value, listed beside the first and contesting nothing. Red on ``67ffcce59``: fails on
    "a second email was refused as a contradiction".
    """
    first = AtomicFact(namespace="memory", subject="person:alice", predicate="email", object="a@x.example")
    second = AtomicFact(namespace="memory", subject="person:alice", predicate="email", object="b@y.example")
    env.apply(first, "e1.md", "e1")
    decision = env.apply(second, "e2.md", "e2")
    assert decision.code == DecisionCode.APPLIED, f"a second email was refused as a contradiction: {decision.code}"
    current = env.ledger.current(tenant_id=TENANT, now=NOW)
    standing, contested = split_contested(current)
    assert sorted(str(e.fact.object) for e in standing) == ["a@x.example", "b@y.example"]
    assert contested == ()


def test_c2_a_fact_may_declare_its_own_cardinality() -> None:
    """Node tests/test_fact_supersession_windows.py::test_c2_a_fact_may_declare_its_own_cardinality.

    Invariant: an event predicate (`visited`) never conflicts, a fact's declared `cardinality`
    overrides its predicate's, and a fact declaring none keeps the identity it always had. Red on
    ``67ffcce59`` at the first assertion (two visits conflicted), and, for the declared half (a
    field the pre-fix tree lacks), by mutation of
    `recall.provenance_controller.predicate_cardinality` ignoring `fact.cardinality`: fails on
    "a declared set-valued predicate still conflicts" (2026-10-21).
    """
    visit = dict(namespace="memory", subject="person:alice", predicate="visited")
    single_city = dict(namespace="memory", subject="person:alice", predicate="city")
    assert not facts_conflict(AtomicFact(**visit, object="Rome"), AtomicFact(**visit, object="Oslo"))
    assert facts_conflict(AtomicFact(**single_city, object="Rome"), AtomicFact(**single_city, object="Oslo"))
    hobby = dict(namespace="memory", subject="person:alice", predicate="hobby", cardinality="set")
    assert not facts_conflict(AtomicFact(**hobby, object="chess"), AtomicFact(**hobby, object="go")), (
        "a declared set-valued predicate still conflicts"
    )
    plain = AtomicFact(namespace="memory", subject="person:alice", predicate="city", object="Rome")
    assert "cardinality" not in plain.to_payload(), "a fact declaring none changed its identity"
    assert AtomicFact.from_payload(AtomicFact(**hobby, object="go").to_payload()).cardinality == "set"
    with pytest.raises(ValueError, match="cardinality"):
        AtomicFact(namespace="memory", subject="s", predicate="p", object=1, cardinality="many")


def test_c3_an_nfd_subject_conflicts_with_its_nfc_twin(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_c3_an_nfd_subject_conflicts_with_its_nfc_twin.

    Invariant: facts are NFC-normalised at construction, so `José` spelled NFD shares the
    conflict key of its NFC twin AND conflicts with it, and an unlinked second value is refused.
    Red on ``67ffcce59``: fails on "an NFD subject did not conflict with its NFC twin" (audit
    P2 #11: both were current, zero contested).
    """
    nfc = unicodedata.normalize("NFC", "person:josé")
    nfd = unicodedata.normalize("NFD", nfc)
    paris = AtomicFact(namespace="memory", subject=nfc, predicate="city", object="Paris")
    rome = AtomicFact(namespace="memory", subject=nfd, predicate="city", object="Rome")
    assert facts_conflict(paris, rome), "an NFD subject did not conflict with its NFC twin"
    assert rome.subject == nfc
    env.apply(paris, "m1.md", "u1")
    assert env.apply(rome, "m2.md", "u2").code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION


def test_c4_declaring_a_cardinality_does_not_slip_a_second_value_past_a_single_one(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_c4_declaring_a_cardinality_does_not_slip_a_second_value_past_a_single_one.

    Invariant: two values conflict unless both are of ONE multi-valued cardinality, so Rome
    declaring `cardinality: set` still contradicts single-valued Paris and, unlinked, is refused
    as Rome is; `set` against `event` conflicts too. Red on ``0f714299d``: fails on "a declared
    cardinality slipped a second value past a single one" (audit STAKES-002: APPLIED, both
    standing, none contested).
    """
    env.apply(city("Paris"), "m1.md", "k1")
    rome = AtomicFact(namespace="memory", subject="person:alice", predicate="city", object="Rome", cardinality="set")
    assert facts_conflict(city("Paris"), rome) and facts_conflict(rome, city("Paris")), (
        "a declared cardinality slipped a second value past a single one"
    )
    decision = env.apply(rome, "m2.md", "k2")
    assert decision.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, decision.code
    hobby = dict(namespace="memory", subject="person:alice", predicate="hobby")
    assert facts_conflict(
        AtomicFact(**hobby, object="chess", cardinality="set"),
        AtomicFact(**hobby, object="go", cardinality="event"),
    )


# ------------------------------------------------------------------ the 2026-10-21 audit of this change


def _assertion(index: int, value: str, supersedes: tuple[str, ...], at: datetime) -> FactEvent:
    """An applied assertion of `city(value)`, built directly, for the index's own scale tests."""
    fact = city(value)
    return FactEvent(
        event_id=f"e{index}", event_type="asserted", tenant_id=TENANT, generation_id=GEN, fact=fact,
        fact_id=fact_identity(fact), evidence_cards=(), supersedes_fact_ids=supersedes,
        request_id=f"q{index}", writer="windows", decision_code=DecisionCode.APPLIED,
        policy_version="test", controller_version=CONTROLLER_SCHEMA_VERSION, created_at=at,
    )


def test_f10_a_long_supersession_chain_resolves_without_recursion() -> None:
    """Node tests/test_fact_supersession_windows.py::test_f10_a_long_supersession_chain_resolves_without_recursion.

    Invariant: 3,000 values, each superseding the one before, resolve, and every one but the
    last is hidden. Every ledger read and write builds this index over the tenant's whole
    history, so a chain long enough to exhaust the interpreter's recursion took the tenant down,
    for unrelated keys too (audit SEC-002: refused from link 995). Red on ``0f714299d``: fails on
    "the index recursed once per link of the chain".
    """
    events: list[FactEvent] = []
    previous: tuple[str, ...] = ()
    for i in range(3000):
        event = _assertion(i, f"city-{i}", previous, NOW + timedelta(microseconds=i))
        events.append(event)
        previous = (str(event.fact_id),)
    try:
        index = fact_supersession(events)
    except RecursionError:
        pytest.fail("the index recursed once per link of the chain")
    later = NOW + timedelta(days=1)
    assert [e.event_id for e in events if not index.hidden_at(e, later)] == ["e2999"]


def test_f11_a_value_flipped_back_and_forth_keeps_one_window_per_assertion() -> None:
    """Node tests/test_fact_supersession_windows.py::test_f11_a_value_flipped_back_and_forth_keeps_one_window_per_assertion.

    Invariant: Paris and Rome alternating 24 times, each superseding the other, leave every
    assertion but the last hidden inside ONE merged window. A value asserted again keeps its
    fact id, so each later successor hides every earlier assertion of it; listed rather than
    merged, the windows grew about 1.6 times per flip (24 million at 33 flips, audit SEC-002).
    The pre-fix tree has no `hidden` map, so the red proof is a mutation of
    `recall.provenance_controller.fact_supersession`: store `tuple(spans)` in place of
    `_merge(spans)`; fails on "the windows hiding one assertion were listed, not merged".
    """
    events = []
    for i in range(24):
        value, other = ("Paris", "Rome") if i % 2 == 0 else ("Rome", "Paris")
        supersedes = (fact_identity(city(other)),) if i else ()
        events.append(_assertion(i, value, supersedes, NOW + timedelta(microseconds=i)))
    index = fact_supersession(events)
    widest = max(len(index.hidden.get(e.event_id, ())) for e in events)
    assert widest == 1, f"the windows hiding one assertion were listed, not merged: {widest}"
    assert [e.event_id for e in events if not index.hidden_at(e, NOW)] == ["e23"]


def test_f12_a_value_asserted_again_over_a_bounded_successor_supersedes_it(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f12_a_value_asserted_again_over_a_bounded_successor_supersedes_it.

    Invariant: Paris; Rome from ten days ago to sixty days ahead, superseding it; Paris again,
    linking Rome. The third application adds a supersession the first lacks, so it is APPLIED,
    not DUPLICATE, and Paris is current through Rome's window; a fourth that adds nothing is a
    DUPLICATE. Red on ``0f714299d``: fails on "a re-assertion naming its successor was recorded
    DUPLICATE" (audit STAKES-001: the link was dropped and Rome stayed current).
    """
    env.apply(city("Paris"), "m1.md", "r1")
    summer = city("Rome", start=NOW - timedelta(days=10), end=NOW + timedelta(days=60))
    assert env.apply(summer, "rome.md", "r2", links=("m1.md",)).code == DecisionCode.APPLIED
    again = env.apply(city("Paris"), "m3.md", "r3", links=("rome.md",))
    assert again.code == DecisionCode.APPLIED, f"a re-assertion naming its successor was recorded {again.code}"
    assert env.current() == ["Paris"]
    assert env.current(NOW + timedelta(days=30)) == ["Paris"]
    assert env.current(NOW + timedelta(days=90)) == ["Paris"]
    assert env.apply(city("Paris"), "m4.md", "r4", links=("rome.md",)).code == DecisionCode.DUPLICATE


def test_f13_a_fact_in_force_under_two_assertions_is_listed_once() -> None:
    """Node tests/test_fact_supersession_windows.py::test_f13_a_fact_in_force_under_two_assertions_is_listed_once.

    Invariant: now that a re-assertion adding a supersession is recorded anew (f12), one fact
    can be in force under two assertions, and the projection lists it once, under the first.
    Red on ``0f714299d``: fails on "a fact in force under two assertions was listed twice".
    """
    first = _assertion(0, "Paris", (), NOW)
    second = _assertion(1, "Paris", (fact_identity(city("Rome")),), NOW + timedelta(microseconds=1))
    listed = [e.event_id for e in project_current([first, second], NOW)]
    assert listed == ["e0"], f"a fact in force under two assertions was listed twice: {listed}"


def test_f14_a_memo_stating_a_fact_through_a_duplicate_can_supersede_it(env: Env) -> None:
    """Node tests/test_fact_supersession_windows.py::test_f14_a_memo_stating_a_fact_through_a_duplicate_can_supersede_it.

    Invariant: a fact applied from m1 (APPLIED) and then from m2 (DUPLICATE) has both memos as
    sources, so a new value whose memo links m2 supersedes it, and the fact is listed once
    beforehand. Re-measured open on master ``52be498cc`` (2026-10-22): links were matched against
    the APPLIED event's cards only. Red against ``52be498cc`` with this test copied in, every
    ledger: fails on "a link to the memo behind the DUPLICATE did not supersede"
    (CONTRADICTION_WITHOUT_SUPERSESSION). Red, mutation of `_superseding_fact_ids`
    (recall/provenance_controller.py) reading the APPLIED event's cards only: the same line.
    """
    assert env.apply(city("Paris"), "m1.md", "r1").code == DecisionCode.APPLIED
    assert env.apply(city("Paris"), "m2.md", "r2").code == DecisionCode.DUPLICATE
    assert env.current() == ["Paris"], "a fact applied from two memos was not listed once"
    decision = env.apply(city("Rome"), "m3.md", "r3", links=("m2.md",))
    assert decision.code == DecisionCode.APPLIED, (
        f"a link to the memo behind the DUPLICATE did not supersede: {decision.code}"
    )
    assert env.current() == ["Rome"]
