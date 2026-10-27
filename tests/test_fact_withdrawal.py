"""Withdrawing a memo gives back the facts it superseded; forgetting one does not (fact ledger).

Owner decision, 2026-10-22: `recall_forget` keeps retired what a forgotten memo retired, in both
layers (pinned for the ledger by `tests/test_fact_retraction.py::test_r5_...`), and the new
`recall_withdraw` is the explicit operation for "this memo was a mistake". It appends a `withdrawn`
event per assertion the memo alone supported (`recall.fact_ledger.plan_withdrawals`): the
assertion is no longer listed, and `fact_supersession` stops counting it as a successor while still
passing on what hides it, so a chain is spliced rather than opened.

Invariants, on a lite store and on PostgreSQL alike:
- L1 withdrawing the head of a fact's line makes the value it superseded current again, and the
  materialized current-facts tables say so (the read is answered from them, not a replay).
- L2 withdrawing the middle of a chain splices it: the first value comes back only where nothing
  later hides it, and the line never has two current values.
- L3 withdrawing a memo forgotten earlier gives back what the forget kept retired.
- L4 a value another memo still states survives withdrawing one of its memos, supersession and all.
- L5 the two layers agree: after a forget `recall_current_facts` lists neither value; after a
  withdraw it lists the old one, through the document graph's card liveness and the ledger alike.
- L6 PostgreSQL and lite record the same withdrawal events.
- L7 a lite ledger table made before `withdrawn` existed accepts one, keeping its rows.
- L8 a client request id in the form a withdrawal's takes is refused, so no request can take a
  withdrawal's slot.
- L9 a withdrawn assertion whose value was erased (PR 950's shape: no fact, so its `superseded`
  event stands alone) is withdrawn through that event too, and the chain is still spliced.

Red proof: in each docstring; master (`e5d87caba`) has no withdrawal at all, so every red proof
here is a mutation, run on a Linux test host and restored, named with its target and failure line.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import (
    DecisionCode,
    FactApplicationRequest,
    ProvenanceController,
    project_current,
)
from recall.types import AtomicFact, Chunk
from recall_mcp.lifecycle import forget_memory, withdraw_memory
from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason
from tests.fact_fixtures import NOW, evidence_card

DIM = 8
GEN = "g1"
FEBRUARY = datetime(2026, 2, 1, tzinfo=UTC)
TEA = AtomicFact(namespace="memory", subject="user", predicate="drink", object="tea")
COFFEE = AtomicFact(namespace="memory", subject="user", predicate="drink", object="coffee")
#: Coffee for the first quarter, then juice from March: a successor with a window, so a splice shows.
COFFEE_Q1 = AtomicFact(
    namespace="memory", subject="user", predicate="drink", object="coffee",
    valid_from=datetime(2026, 1, 1, tzinfo=UTC), valid_until=datetime(2026, 4, 1, tzinfo=UTC),
)
JUICE = AtomicFact(
    namespace="memory", subject="user", predicate="drink", object="juice",
    valid_from=datetime(2026, 3, 1, tzinfo=UTC),
)
PARIS = AtomicFact(namespace="memory", subject="user", predicate="home_city", object="Paris")
ROME = AtomicFact(namespace="memory", subject="user", predicate="home_city", object="Rome")


class Ledgered:
    """A store with its fact ledger and card store, the same calls on lite and PostgreSQL."""

    def __init__(self, backend: str, tmp_path: Path, make_store: Callable[[int], Any] | None) -> None:
        self.backend = backend
        if backend == "lite":
            self.store: Any = LiteStore(tmp_path / "memory.db", dim=DIM)
            self.ledger: Any = LiteFactLedger(self.store)
            self.cards: Any = LiteEvidenceCardStore(self.store)
        else:
            from recall.fact_ledger import PostgresFactLedger
            from recall.provenance_cards import PostgresEvidenceCardStore
            from recall.store import PgVectorStore

            assert make_store is not None
            tenant = f"withdraw-{uuid4().hex}"
            self.store = PgVectorStore(TEST_DSN, dim=DIM, table=make_store(DIM).table, tenant=tenant)
            self.ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
            self.cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)

    def remember(self, name: str, supersedes: str | None = None) -> None:
        metadata: dict[str, Any] = {"file": name}
        if supersedes:
            metadata["supersedes"] = supersedes
        self.store.upsert(
            [Chunk(f"chunk-{name}", f"/notes/{name}", f"text of {name}", metadata)],
            [[1.0] + [0.0] * (DIM - 1)],
        )

    def apply(self, fact: AtomicFact, name: str, request_id: str, links: tuple[str, ...] = ()) -> str:
        card = evidence_card(fact, tenant=self.store.tenant, generation=GEN, chunk=f"chunk-{name}",
                             source=name, links=links)
        self.cards.put((card,))
        return str(ProvenanceController(
            tenant_id=self.store.tenant, generation_id=GEN, cards=self.cards, ledger=self.ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="withdrawal-test",
        ).apply_fact(FactApplicationRequest(fact, (card.card_id,), request_id)).code)

    def current(self, now: datetime = NOW) -> list[object]:
        return [e.fact.object for e in self.ledger.current(tenant_id=self.store.tenant, now=now) if e.fact]

    def replayed(self, now: datetime = NOW) -> list[object]:
        return [e.fact.object for e in project_current(self.ledger.events, now) if e.fact]


@pytest.fixture(params=["lite", "postgres"])
def ledgered(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Ledgered]:
    make_store = None
    if request.param == "postgres":
        if not _db_available():
            pytest.skip(db_unreachable_reason())
        make_store = request.getfixturevalue("make_store")
    yield Ledgered(request.param, tmp_path, make_store)


def test_l1_withdrawing_the_head_gives_back_what_it_superseded(ledgered: Ledgered) -> None:
    """Node tests/test_fact_withdrawal.py::test_l1_withdrawing_the_head_gives_back_what_it_superseded.

    Red, mutation of `fact_supersession` (recall/provenance_controller.py) counting a withdrawn
    successor's window (`if holder is None:` made `if True:`): fails on
    "the withdrawn successor still hides tea", both backends. The read is answered by the
    materialized tables (`last_read`); red, mutation of `project_key` (recall/fact_current.py)
    reading retractions without the withdrawals: the tables keep coffee listed beside tea and it
    fails on the same line, while the replay alone would have answered tea. Since PR 950 the
    withdraw also erases coffee's value: red (2026-10-24), mutation of `derive_tables`
    (recall/fact_current.py) leaving a redacted fact keyless (`named[event.fact_id] = key`
    removed), so its withdrawal is filed under no key: fails on the same line, both backends.
    """
    ledgered.remember("tea.md")
    ledgered.remember("coffee.md")
    assert ledgered.apply(TEA, "tea.md", "r1") == DecisionCode.APPLIED
    assert ledgered.apply(COFFEE, "coffee.md", "r2", links=("tea.md",)) == DecisionCode.APPLIED
    assert ledgered.current() == ["coffee"]

    result = withdraw_memory(ledgered.store, ["coffee.md"])
    assert result.sources_withdrawn == ["coffee.md"] and result.facts_withdrawn == 1, result.message
    assert ledgered.current() == ["tea"], f"the withdrawn successor still hides tea: {ledgered.current()}"
    assert ledgered.ledger.last_read == "table", "the tables did not answer"
    assert ledgered.replayed() == ["tea"]


def test_l2_withdrawing_the_middle_splices_the_chain(ledgered: Ledgered) -> None:
    """Node tests/test_fact_withdrawal.py::test_l2_withdrawing_the_middle_splices_the_chain.

    Tea, then coffee for the first quarter superseding it, then juice from March superseding
    coffee. Withdrawing coffee gives tea back for February (nothing else hides it then) and leaves
    juice hiding it from March, so today juice alone is current.
    Red, mutation of `fact_supersession` counting a withdrawn successor's window: fails on "the
    splice did not give tea back for February". Red, mutation of `fact_supersession` not passing
    on what hides a withdrawn successor (`and holder is None` added to its `asserted` branch and
    `and successor.event_type != "asserted"` to the branch after it): fails on "two current
    values after withdrawing the middle".
    """
    for name in ("tea.md", "coffee.md", "juice.md"):
        ledgered.remember(name)
    assert ledgered.apply(TEA, "tea.md", "r1") == DecisionCode.APPLIED
    assert ledgered.apply(COFFEE_Q1, "coffee.md", "r2", links=("tea.md",)) == DecisionCode.APPLIED
    assert ledgered.apply(JUICE, "juice.md", "r3", links=("coffee.md",)) == DecisionCode.APPLIED
    assert ledgered.current(FEBRUARY) == ["coffee"] and ledgered.current() == ["juice"]

    withdraw_memory(ledgered.store, ["coffee.md"])
    assert ledgered.current(FEBRUARY) == ["tea"], (
        f"the splice did not give tea back for February: {ledgered.current(FEBRUARY)}"
    )
    assert ledgered.current() == ["juice"], f"two current values after withdrawing the middle: {ledgered.current()}"


def test_l3_withdrawing_a_forgotten_memo_gives_back_what_the_forget_kept(ledgered: Ledgered) -> None:
    """Node tests/test_fact_withdrawal.py::test_l3_withdrawing_a_forgotten_memo_gives_back_what_the_forget_kept.

    The forget keeps tea retired (owner decision); the memo then turns out to have been a mistake.
    Its chunks are gone, so the withdrawal finds it by name, as a forget re-run does.
    Red, mutation of `plan_withdrawals` (recall/fact_ledger.py) skipping retracted applications
    (`and retraction_key(event) not in retracted` added to its collection): fails on
    "withdrawing a forgotten memo did not give tea back".
    Since PR 950 the forget erases coffee's value first: red (2026-10-24), mutation of both
    tables' `named_key` (recall/fact_current.py) ignoring the filed key of a redacted fact, so the
    later withdrawal is filed under no key: fails on the same line, both backends.
    """
    ledgered.remember("tea.md")
    ledgered.remember("coffee.md")
    ledgered.apply(TEA, "tea.md", "r1")
    ledgered.apply(COFFEE, "coffee.md", "r2", links=("tea.md",))
    forget_memory(ledgered.store, ["coffee.md"])
    assert ledgered.current() == []

    result = withdraw_memory(ledgered.store, ["coffee.md"])
    assert ledgered.current() == ["tea"], f"withdrawing a forgotten memo did not give tea back: {result.message}"
    assert result.facts_withdrawn == 1


def test_l4_a_value_another_memo_states_survives_withdrawing_one_memo(ledgered: Ledgered) -> None:
    """Node tests/test_fact_withdrawal.py::test_l4_a_value_another_memo_states_survives_withdrawing_one_memo.

    Coffee is stated by two memos; withdrawing one leaves coffee current and tea retired.
    Red, mutation of `plan_withdrawals` without its support check (the `if any(...): continue`
    removed): the first application is withdrawn and the DUPLICATE is never listed, so it fails
    on "a value another memo states was withdrawn".
    """
    for name in ("tea.md", "coffee.md", "coffee-again.md"):
        ledgered.remember(name)
    ledgered.apply(TEA, "tea.md", "r1")
    ledgered.apply(COFFEE, "coffee.md", "r2", links=("tea.md",))
    assert ledgered.apply(COFFEE, "coffee-again.md", "r3", links=("tea.md",)) == DecisionCode.DUPLICATE

    result = withdraw_memory(ledgered.store, ["coffee.md"])
    assert ledgered.current() == ["coffee"], f"a value another memo states was withdrawn: {ledgered.current()}"
    assert result.facts_withdrawn == 0


@pytest.mark.parametrize("operation", ["forget", "withdraw"])
def test_l5_the_two_layers_agree(ledgered: Ledgered, operation: str) -> None:
    """Node tests/test_fact_withdrawal.py::test_l5_the_two_layers_agree.

    Rome supersedes Paris in the memo (`supersedes:`) and in the fact (its card links Paris's
    memo). After a forget, `recall_current_facts` lists neither city and the document graph keeps
    paris.md retired; after a withdraw it lists Paris and the graph has nothing hiding paris.md.
    (An applied fact's card does not consult the document graph; a stated one does, through
    `ServedState.superseded`, so for stated facts the two answers are one.)
    Red, mutation of `_forget_memory_unlocked` (recall_mcp/lifecycle.py) preparing a retraction
    for a withdrawal (`withdraw` dropped from its `_forget_retractions` call): fails on "the layers
    disagree after withdraw", the ledger keeping Paris hidden. Red, mutation of `record_kind`
    (recall/supersession_records.py) answering FORGOTTEN_RECORD: the document graph keeps
    paris.md retired and it fails on "the document graph disagrees after withdraw".
    """
    from recall_mcp.provenance import current_facts_memory

    ledgered.remember("paris.md")
    ledgered.remember("rome.md", supersedes="paris.md")
    ledgered.apply(PARIS, "paris.md", "r1")
    ledgered.apply(ROME, "rome.md", "r2", links=("paris.md",))

    remove = withdraw_memory if operation == "withdraw" else forget_memory
    remove(ledgered.store, ["rome.md"])
    listed = sorted(str(item["fact"]["object"]) for item in current_facts_memory(ledgered.store)["facts"])  # type: ignore[index, union-attr]
    expected = ["Paris"] if operation == "withdraw" else []
    assert listed == expected, f"the layers disagree after {operation}: {listed}"
    successor = ledgered.store.supersession_graph().successor("paris.md")
    assert successor == (None if operation == "withdraw" else "rome.md"), (
        f"the document graph disagrees after {operation}: {successor}"
    )


def test_l6_postgres_and_lite_record_the_same_withdrawal(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """Node tests/test_fact_withdrawal.py::test_l6_postgres_and_lite_record_the_same_withdrawal.

    Red, with migrations 0032 and 0033 removed from the package (their SQL files and checksum
    entries): the append function refuses the `withdrawn` envelope, PostgreSQL's withdrawal
    reports -1 and it fails on "PostgreSQL and lite recorded different withdrawals".

    A withdraw erases the values it took back (`recall.fact_redaction`), which re-keys the
    withdrawn fact under a fresh random id on each side, so the ids compared are their count.
    """
    if not _db_available():
        pytest.skip(db_unreachable_reason())
    runs: list[list[tuple[Any, ...]]] = []
    for backend in ("lite", "postgres"):
        make_store = request.getfixturevalue("make_store") if backend == "postgres" else None
        (tmp_path / backend).mkdir(exist_ok=True)
        side = Ledgered(backend, tmp_path / backend, make_store)
        side.remember("tea.md")
        side.remember("coffee.md")
        side.apply(TEA, "tea.md", "r1")
        side.apply(COFFEE, "coffee.md", "r2", links=("tea.md",))
        withdraw_memory(side.store, ["coffee.md"])
        # Event and request ids hash the tenant, which differs between the two stores here.
        runs.append([
            (e.event_type, e.writer, e.decision_code, len(e.supersedes_fact_ids),
             [c.card_id for c in e.evidence_cards], e.request_id.startswith("withdrawal:evt_"))
            for e in side.ledger.events
            if e.event_type == "withdrawn"
        ])
        runs[-1].append(tuple(side.current()))
    assert runs[0] == runs[1] and len(runs[0]) == 2, f"PostgreSQL and lite recorded different withdrawals: {runs}"


#: `recall_fact_ledger_events` as a lite file made after `retracted` (migration 0026) and before
#: `withdrawn` holds it.
_RETRACTION_ERA_DDL = """
CREATE TABLE recall_fact_ledger_events (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('asserted', 'superseded', 'rejected', 'abstained', 'retracted')),
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
    CHECK ((event_type IN ('asserted', 'rejected') AND fact IS NOT NULL) OR event_type IN ('superseded', 'abstained', 'retracted'))
);
CREATE INDEX recall_fact_ledger_current_idx ON recall_fact_ledger_events (tenant_id, fact_id, created_at);
CREATE TRIGGER recall_fact_ledger_no_update BEFORE UPDATE ON recall_fact_ledger_events
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
CREATE TRIGGER recall_fact_ledger_no_delete BEFORE DELETE ON recall_fact_ledger_events
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
"""


def test_l7_a_ledger_made_before_withdrawal_accepts_one(tmp_path: Path) -> None:
    """Node tests/test_fact_withdrawal.py::test_l7_a_ledger_made_before_withdrawal_accepts_one.

    Red, mutation of `_admits_retraction` (recall/lite/facts.py) still asking for `'retracted'`:
    the old table is not rebuilt, its CHECK refuses the `withdrawn` row, the withdrawal reports
    -1, and it fails on "a ledger made before withdrawal refused it".
    """
    path = tmp_path / "old.db"
    LiteStore(path, dim=DIM).close()
    with sqlite3.connect(path) as conn:
        conn.executescript(_RETRACTION_ERA_DDL)
    side = Ledgered("lite", tmp_path, None)
    side.store.close()
    side.store = LiteStore(path, dim=DIM)
    side.ledger = LiteFactLedger(side.store)
    side.cards = LiteEvidenceCardStore(side.store)
    side.remember("tea.md")
    side.remember("coffee.md")
    side.apply(TEA, "tea.md", "r1")
    side.apply(COFFEE, "coffee.md", "r2", links=("tea.md",))

    result = withdraw_memory(side.store, ["coffee.md"])
    assert result.facts_withdrawn == 1 and side.current() == ["tea"], (
        f"a ledger made before withdrawal refused it: {result.message}"
    )
    with sqlite3.connect(path) as conn:
        triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {"recall_fact_ledger_no_update", "recall_fact_ledger_no_delete"} <= triggers


def test_l8_a_request_id_shaped_like_a_withdrawal_is_reserved(tmp_path: Path) -> None:
    """Node tests/test_fact_withdrawal.py::test_l8_a_request_id_shaped_like_a_withdrawal_is_reserved.

    `withdrawal:<event id>` is the request id the ledger mints for a withdrawal, unique per
    tenant, so a client sending it would pre-empt that withdrawal. Red, mutation of
    `reserved_request_id` (recall/provenance_controller.py) without `WITHDRAWAL_REQUEST_PREFIX`:
    fails on "a withdrawal's request id was accepted".
    """
    side = Ledgered("lite", tmp_path, None)
    side.remember("tea.md")
    code = side.apply(TEA, "tea.md", "withdrawal:evt_" + "0" * 32)
    assert code == DecisionCode.REQUEST_ID_RESERVED, f"a withdrawal's request id was accepted: {code}"


def test_l9_a_withdrawn_assertion_whose_value_was_erased_is_withdrawn_through_its_superseded_event() -> None:
    """Node tests/test_fact_withdrawal.py::test_l9_a_withdrawn_assertion_whose_value_was_erased_is_withdrawn_through_its_superseded_event.

    PR 950 erases a forgotten fact's value in place: the assertion keeps its event and its fact id
    but no fact, so it no longer pairs with its `superseded` event, which then hides on its own
    with an unknown window. Withdrawing that assertion must void the event too, and pass on what
    hides the assertion: tea, coffee (erased) superseding it, juice from March superseding coffee,
    coffee withdrawn. Hand-built, since the redaction is not on this branch.
    Red, mutation of `fact_supersession`'s `withdrawn_for` (recall/provenance_controller.py)
    without its `_sup` branch: fails on "the erased successor still hides tea in February". Red,
    mutation dropping the `elif holder is not None` splice: fails on "two current values".
    """
    from datetime import timedelta

    from recall.provenance_controller import FactEvent, fact_identity

    def event(event_id: str, kind: str, fact: AtomicFact | None, *, fact_id: str | None,
              supersedes: tuple[str, ...] = (), request: str, day: int) -> FactEvent:
        return FactEvent(
            event_id=event_id, event_type=kind, tenant_id="t", generation_id=GEN, fact=fact,
            fact_id=fact_id, evidence_cards=(), supersedes_fact_ids=supersedes, request_id=request,
            writer="w", decision_code=DecisionCode.WITHDRAWN if kind == "withdrawn" else DecisionCode.APPLIED,
            policy_version="p", controller_version=1,
            created_at=datetime(2025, 12, 1, tzinfo=UTC) + timedelta(days=day),
        )

    tea = fact_identity(TEA)
    events = [
        event("evt_tea", "asserted", TEA, fact_id=tea, request="r1", day=0),
        event("evt_coffee", "asserted", None, fact_id="fact_erased", supersedes=(tea,), request="r2", day=1),
        event("evt_coffee_sup", "superseded", None, fact_id=None, supersedes=(tea,),
              request="r2:supersession", day=1),
        event("evt_juice", "asserted", JUICE, fact_id=fact_identity(JUICE), supersedes=("fact_erased",),
              request="r3", day=2),
    ]
    assert [e.fact.object for e in project_current(events, FEBRUARY) if e.fact] == []
    events.append(event("evt_w", "withdrawn", None, fact_id=None, supersedes=("fact_erased",),
                        request="withdrawal:evt_coffee", day=3))

    february = [e.fact.object for e in project_current(events, FEBRUARY) if e.fact]
    assert february == ["tea"], f"the erased successor still hides tea in February: {february}"
    today = [e.fact.object for e in project_current(events, NOW) if e.fact]
    assert today == ["juice"], f"two current values: {today}"
