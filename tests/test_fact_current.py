"""The materialized current facts answer every read and conflict check as the ledger's replay does.

Invariant: on any ledger, `current()` answered from the tables (`recall.fact_current`) equals
`project_current` over the same events, at every instant and for every served set, and a write
whose conflict check reads one conflict key's history from the tables decides exactly as one that
reads the tenant's whole history. Where the tables cannot answer (an event they do not file, an
id the ledger no longer holds, an event naming another key's fact) the read replays, and says so
in `last_read`.

Runs on a lite file; `tests/test_fact_current_postgres.py` runs the same harness on PostgreSQL.

Red proof, 2026-10-22: nineteen mutations across this file and its PostgreSQL sibling, each alone
on the remote test host, every one red in the assertion its test's docstring names, then restored
and green.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from recall.fact_current import CURRENT_TABLE, SQLiteCurrentFacts
from recall.fact_ledger import GC_RETRACTION_WRITER
from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import (
    _PERMIT_SENTINEL,
    DecisionCode,
    FactApplicationPermit,
    FactApplicationRequest,
    FactEvent,
    ProvenanceController,
    fact_identity,
    live_sources,
    project_current,
)
from recall.types import AtomicFact, EvidenceCard
from tests.fact_fixtures import NOW, evidence_card
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

TENANT = "current-facts"
GEN = "generation-current"
DAY = timedelta(days=1)


def city(value: str, start: datetime | None = None, end: datetime | None = None, *, predicate: str = "city") -> AtomicFact:
    return AtomicFact(
        namespace="memory", subject="user", predicate=predicate, object=value, valid_from=start, valid_until=end,
    )


def card(fact: AtomicFact, source: str, tenant: str = TENANT) -> EvidenceCard:
    return evidence_card(fact, tenant=tenant, generation=GEN, chunk=f"chunk-{source}-{fact_identity(fact)}", source=source)


def apply(
    ledger: Any, fact: AtomicFact, request_id: str, *, supersedes: Sequence[AtomicFact] = (),
    source: str | None = None, now: datetime = NOW, stated: bool = False, tenant: str = TENANT,
) -> FactEvent:
    """`apply_assertion` as the controller calls it, with a valid permit."""
    evidence = card(fact, source or f"{request_id}.md", tenant)
    permit = FactApplicationPermit(
        tenant_id=tenant, generation_id=GEN, request_id=request_id, fact_id=fact_identity(fact),
        card_ids=(evidence.card_id,), _token=_PERMIT_SENTINEL,
    )
    return ledger.apply_assertion(
        tenant_id=tenant, generation_id=GEN, fact=fact, cards=(evidence,), request_id=request_id,
        writer="current-facts-test", permit=permit,
        supersedes_fact_ids=tuple(fact_identity(old) for old in supersedes), now=now, stated=stated,
    ).event


@contextmanager
def lite(path: Path, *, materialize: bool = True) -> Iterator[tuple[LiteStore, LiteFactLedger]]:
    store = LiteStore(path, dim=4, tenant=TENANT)
    try:
        yield store, LiteFactLedger(store, materialize=materialize)
    finally:
        store.close()


def objects(events: Sequence[FactEvent]) -> list[object]:
    return [event.fact.object if event.fact else None for event in events]


def assert_table_is_replay(
    ledger: Any, instants: Sequence[datetime], served: Callable[[frozenset[str]], Collection[str]] | None = None,
) -> None:
    """At every instant, the table's answer is the replay's, and the table gave it."""
    events = ledger.events
    for at in instants:
        table = ledger.current(tenant_id=ledger.tenant_id, now=at, served=served)
        assert ledger.last_read == "table", f"{at}: the read replayed"
        assert table == project_current(events, at, served), f"{at}: table != replay"


def intervals(store: LiteStore, event_id: str) -> list[tuple[str, str]]:
    return [
        (str(low), str(high)) for low, high in store._conn.execute(
            f"SELECT valid_from, valid_until FROM {CURRENT_TABLE} WHERE event_id = ? ORDER BY valid_from",
            (event_id,),
        ).fetchall()
    ]


def stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


EARLIEST = stamp(datetime.min.replace(tzinfo=UTC))
LATEST = stamp(datetime.max.replace(tzinfo=UTC))
PARIS = city("paris")


def test_a_future_dated_successor_leaves_today_current(tmp_path: Path) -> None:
    """Rome from next month supersedes Paris: today Paris, from next month Rome.

    Red: `project_key` storing the fact's own window instead of `index.in_force(event)`
    (`recall/fact_current.py`) leaves Paris's row open-ended, so the read 40 days out lists Paris
    and Rome, and the value assertion fails.
    """
    rome = city("rome", NOW + 30 * DAY)
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1")
        apply(ledger, rome, "r2", supersedes=(PARIS,))
        assert objects(ledger.current(tenant_id=TENANT, now=NOW)) == ["paris"]
        assert objects(ledger.current(tenant_id=TENANT, now=NOW + 40 * DAY)) == ["rome"]
        assert intervals(store, first.event_id) == [(EARLIEST, stamp(NOW + 30 * DAY))]
        assert_table_is_replay(ledger, (NOW, NOW + 29 * DAY, NOW + 30 * DAY, NOW + 400 * DAY))


def test_an_expired_bounded_successor_revives_its_predecessor(tmp_path: Path) -> None:
    """Rome for one month supersedes Paris: Paris before it and again after it, in two rows.

    Red: the `project_key` own-window mutation above lists Paris and Rome 40 days out, and the
    value assertion fails.
    """
    rome = city("rome", NOW + 30 * DAY, NOW + 60 * DAY)
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1")
        apply(ledger, rome, "r2", supersedes=(PARIS,))
        assert objects(ledger.current(tenant_id=TENANT, now=NOW + 40 * DAY)) == ["rome"]
        assert objects(ledger.current(tenant_id=TENANT, now=NOW + 70 * DAY)) == ["paris"]
        assert intervals(store, first.event_id) == [
            (EARLIEST, stamp(NOW + 30 * DAY)), (stamp(NOW + 60 * DAY), LATEST),
        ]
        assert_table_is_replay(ledger, (NOW, NOW + 40 * DAY, NOW + 60 * DAY, NOW + 70 * DAY))


def test_a_retroactive_correction_splits_the_open_fact(tmp_path: Path) -> None:
    """Oslo for 2020 to 2022 supersedes the open-ended Paris: Paris is split into two rows.

    Red: the `project_key` own-window mutation above stores one Paris row, so the row-count
    assertion fails (one row, not two).
    """
    oslo = city("oslo", datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC))
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1")
        apply(ledger, oslo, "r2", supersedes=(PARIS,))
        assert intervals(store, first.event_id) == [
            (EARLIEST, stamp(datetime(2020, 1, 1, tzinfo=UTC))),
            (stamp(datetime(2022, 1, 1, tzinfo=UTC)), LATEST),
        ]
        assert objects(ledger.current(tenant_id=TENANT, now=datetime(2021, 6, 1, tzinfo=UTC))) == ["oslo"]
        assert objects(ledger.current(tenant_id=TENANT, now=NOW)) == ["paris"]
        assert_table_is_replay(ledger, (datetime(2019, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC), NOW))


def test_a_value_flipped_back_is_current_again(tmp_path: Path) -> None:
    """Paris, Rome superseding it, Paris again superseding Rome: the third assertion is current.

    Red: `materialize_events` skipping the recompute for an event that names facts (mutation:
    `if key is not None and event.event_type not in _INERT:` to `... _INERT and not
    event.supersedes_fact_ids:`) leaves the first Paris row in place, and the listed-event
    assertion fails.
    """
    rome = city("rome")
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1")
        apply(ledger, rome, "r2", supersedes=(PARIS,))
        third = apply(ledger, PARIS, "r3", supersedes=(rome,))
        assert third.decision_code == DecisionCode.APPLIED
        listed = ledger.current(tenant_id=TENANT, now=NOW + DAY)
        assert [event.event_id for event in listed] == [third.event_id]
        assert intervals(store, first.event_id) == []
        assert_table_is_replay(ledger, (NOW - DAY, NOW, NOW + DAY))


def test_a_set_valued_predicate_keeps_both_values(tmp_path: Path) -> None:
    """Two emails are no contradiction: both are applied and both are current.

    Red: `project_key` keeping one assertion's rows per key (mutation: `rows.extend(` to
    `rows = list(`) lists the second email alone, and the value assertion fails.
    """
    with lite(tmp_path / "f.db") as (_store, ledger):
        apply(ledger, city("a@example.org", predicate="email"), "r1")
        apply(ledger, city("b@example.org", predicate="email"), "r2")
        assert objects(ledger.current(tenant_id=TENANT, now=NOW)) == ["a@example.org", "b@example.org"]
        assert_table_is_replay(ledger, (NOW,))


def test_two_assertions_of_one_fact_list_the_first(tmp_path: Path) -> None:
    """A fact in force under two assertions is listed once, under the first, as `project_current`.

    Paris; Rome stated (contested); Paris again superseding Rome, which is a new assertion
    (`repeats_assertion`): both Paris assertions are in force, the first is listed.

    Red: `current_view` keeping the LAST assertion per fact (`first.setdefault` replaced by
    `first[fact_id] = event_id`) lists the third, and the event-id assertion fails.
    """
    rome = city("rome")
    with lite(tmp_path / "f.db") as (_store, ledger):
        first = apply(ledger, PARIS, "r1")
        apply(ledger, rome, "r2", stated=True)
        third = apply(ledger, PARIS, "r3", supersedes=(rome,), stated=True)
        assert third.decision_code == DecisionCode.APPLIED
        listed = ledger.current(tenant_id=TENANT, now=NOW)
        assert [event.event_id for event in listed] == [first.event_id]
        assert_table_is_replay(ledger, (NOW,))


def test_a_retraction_removes_every_interval_of_its_assertion(tmp_path: Path) -> None:
    """Forgetting Paris's memo retracts it: both of its split rows go, and nothing is current.

    Red: `_INERT` in `recall/fact_current.py` holding "retracted" leaves Paris's rows in place,
    and the empty-rows assertion fails.
    """
    oslo = city("oslo", datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC))
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1", source="paris.md")
        apply(ledger, oslo, "r2", supersedes=(PARIS,), source="oslo.md")
        written = ledger.retract_forgotten(
            tenant_id=TENANT, touched=lambda c: c.source == "paris.md", gone=lambda c: c.source == "paris.md",
        )
        assert [event.event_type for event in written] == ["retracted"]
        assert intervals(store, first.event_id) == []
        assert ledger.current(tenant_id=TENANT, now=NOW) == ()
        assert_table_is_replay(ledger, (datetime(2021, 1, 1, tzinfo=UTC), NOW))


def test_a_gc_retraction_is_materialized_like_a_forget(tmp_path: Path) -> None:
    """`generation gc`'s retraction (SOURCE_REMOVED) removes the fact the same way.

    Red: the `_INERT` mutation above fails the empty-rows assertion here too.
    """
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1", source="paris.md")
        ledger.retract_forgotten(
            tenant_id=TENANT, touched=lambda _c: True, gone=lambda _c: True,
            writer=GC_RETRACTION_WRITER, decision_code=DecisionCode.SOURCE_REMOVED,
        )
        assert intervals(store, first.event_id) == []
        assert_table_is_replay(ledger, (NOW,))


def test_a_duplicate_keeps_a_fact_whose_first_memo_is_not_served(tmp_path: Path) -> None:
    """Paris applied from a.md and again (DUPLICATE) from b.md: with a.md not served, Paris stays.

    The served check reads every live assertion of the fact, the DUPLICATE included.

    Red: `current_view` computing `absent_fact_ids` over the listed events only (mutation:
    `absent_fact_ids(support, ...)` to `absent_fact_ids([e for e in support if e.event_id in
    listed], ...)`) drops Paris, and the value assertion fails.
    """
    with lite(tmp_path / "f.db") as (_store, ledger):
        apply(ledger, PARIS, "r1", source="a.md")
        duplicate = apply(ledger, PARIS, "r2", source="b.md")
        assert duplicate.decision_code == DecisionCode.DUPLICATE

        def served(names: frozenset[str]) -> Collection[str]:
            return names - {"a.md"}

        assert objects(ledger.current(tenant_id=TENANT, now=NOW, served=served)) == ["paris"]
        assert_table_is_replay(ledger, (NOW,), served)
        current, support = ledger.current_view(tenant_id=TENANT, now=NOW, served=served)
        assert live_sources(support, served, NOW)[str(current[0].fact_id)] == ("b.md",)


def test_an_event_the_tables_do_not_file_sends_reads_to_replay(tmp_path: Path) -> None:
    """A ledger event written without materializing (an older build) makes reads replay.

    Red: `usable` in `recall/fact_current.py` ignoring the counts (mutation: `return mode !=
    "replay"`) makes the read use the stale tables, which list Paris alone, and the value
    assertion fails.
    """
    rome = city("rome")
    with lite(tmp_path / "f.db") as (store, ledger):
        apply(ledger, PARIS, "r1")
        apply(LiteFactLedger(store, materialize=False), rome, "r2", supersedes=(PARIS,))
        assert objects(ledger.current(tenant_id=TENANT, now=NOW)) == ["rome"]
        assert ledger.last_read == "replay"
        assert SQLiteCurrentFacts(store._conn, tenant_id=TENANT, write=store._write).reconcile() == 2
        assert_table_is_replay(ledger, (NOW,))


def test_a_write_while_the_tables_lag_checks_the_whole_history(tmp_path: Path) -> None:
    """The conflict check never reads a lagging table: Rome contradicts an unfiled Paris.

    Red: `key_history` in `recall/fact_current.py` skipping its `usable` check reads the key from
    the tables, which do not hold Paris, so Rome is APPLIED and `pytest.raises` fails.
    """
    with lite(tmp_path / "f.db") as (store, ledger):
        apply(LiteFactLedger(store, materialize=False), PARIS, "r1")
        with pytest.raises(ValueError, match=DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION):
            apply(ledger, city("rome"), "r2")
        assert ledger.last_read == "replay"


def test_an_id_the_ledger_no_longer_holds_sends_reads_to_replay(tmp_path: Path) -> None:
    """Counts that match while a filed id names nothing (a rewrite with no rebuild) replay.

    Red: `current_view` without its consistency checks (mutation: `if rows is None:` to
    `if False:` and the support check's condition to `False`) answers from the table, which can
    list neither Paris, filed under the stale id, nor the unfiled email: the value assertion fails.
    """
    with lite(tmp_path / "f.db") as (store, ledger):
        first = apply(ledger, PARIS, "r1")
        # An unfiled event and a filed id the ledger lacks: the counts agree, the sets do not.
        apply(LiteFactLedger(store, materialize=False), city("e@example.org", predicate="email"), "r2")
        with store._write():
            store._conn.execute(
                "UPDATE recall_fact_current SET event_id = 'evt_gone' WHERE event_id = ?", (first.event_id,)
            )
            store._conn.execute(
                "UPDATE recall_fact_current_events SET event_id = 'evt_gone' WHERE event_id = ?", (first.event_id,)
            )
            store._conn.execute(
                "INSERT INTO recall_fact_current_events (tenant_id, event_id, conflict_key, fact_id, event_type) "
                "VALUES (?, 'evt_padding', NULL, NULL, 'abstained')", (TENANT,),
            )
        assert sorted(map(str, objects(ledger.current(tenant_id=TENANT, now=NOW)))) == ["e@example.org", "paris"]
        assert ledger.last_read == "replay"


def test_an_event_naming_another_keys_fact_puts_the_tenant_in_replay(tmp_path: Path) -> None:
    """A supersession across conflict keys (only the raw ledger API can write one) replays.

    Red: `event_keys` ignoring the keys of named facts (mutation: its loop over
    `event.supersedes_fact_ids` made empty) files the event under its own key alone, the tables
    answer and list the language the other key's successor hides, and the value assertion fails.
    """
    language = AtomicFact(namespace="memory", subject="user", predicate="language", object="it")
    with lite(tmp_path / "f.db") as (store, ledger):
        apply(ledger, language, "r1")
        apply(ledger, PARIS, "r2", supersedes=(language,))
        assert objects(ledger.current(tenant_id=TENANT, now=NOW)) == ["paris"]
        assert ledger.last_read == "replay"
        status = SQLiteCurrentFacts(store._conn, tenant_id=TENANT, write=store._write).rebuild()
        assert status.mode == "replay" and "conflict keys" in str(status.reason)


def test_rebuild_reproduces_what_materializing_each_event_wrote(tmp_path: Path) -> None:
    """The tables are derived state: a rebuild from the ledger writes the same rows.

    Red: `rebuild_tables` grouping assertions only (mutation: `if key is not None:` to
    `if key is not None and event.event_type == "asserted":` in its grouping loop) loses the
    retraction, so Oslo's interval comes back and the row equality fails.
    """
    oslo = city("oslo", datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC))
    with lite(tmp_path / "f.db") as (store, ledger):
        apply(ledger, PARIS, "r1", source="paris.md")
        apply(ledger, oslo, "r2", supersedes=(PARIS,), source="oslo.md")
        apply(ledger, city("rome", NOW + 30 * DAY, NOW + 60 * DAY), "r3", supersedes=(PARIS,))
        ledger.record_decision(
            tenant_id=TENANT, generation_id=GEN, request_id="r4", writer="t",
            decision_code=DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, fact=city("lima"),
        )
        ledger.retract_forgotten(
            tenant_id=TENANT, touched=lambda c: c.source == "oslo.md", gone=lambda c: c.source == "oslo.md",
        )

        def rows() -> list[tuple[Any, ...]]:
            return sorted(store._conn.execute(
                "SELECT * FROM recall_fact_current UNION ALL "
                "SELECT tenant_id, event_id, conflict_key, fact_id, event_type, live FROM recall_fact_current_events"
            ).fetchall(), key=repr)

        incremental = rows()
        facts = SQLiteCurrentFacts(store._conn, tenant_id=TENANT, write=store._write)
        status = facts.rebuild()
        assert rows() == incremental
        assert status.current and status.ledger_events == len(ledger.events) == 7
        facts.materialize(ledger.events[0])  # idempotent by event id
        assert rows() == incremental


def test_a_refusal_is_filed_so_the_tables_stay_current(tmp_path: Path) -> None:
    """A refused claim is a ledger event too: the tables file it and keep answering.

    Red: `LiteFactLedger.record_decision` not calling `_materialized` leaves the refusal unfiled,
    the counts differ, and `last_read` reads "replay".
    """
    with lite(tmp_path / "f.db") as (store, ledger):
        cards = LiteEvidenceCardStore(store)
        paris_card, rome_card = card(PARIS, "paris.md"), card(city("rome"), "rome.md")
        cards.put((paris_card, rome_card))
        controller = ProvenanceController(
            tenant_id=TENANT, generation_id=GEN, cards=cards, ledger=ledger,
            source_digest_for=lambda _c: "digest", now=lambda: NOW, writer="t",
        )
        assert controller.apply_fact(FactApplicationRequest(PARIS, (paris_card.card_id,), "r1")).allowed
        refused = controller.apply_fact(FactApplicationRequest(city("rome"), (rome_card.card_id,), "r2"))
        assert refused.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION
        assert ledger.conflict_history(tenant_id=TENANT, fact=PARIS) and ledger.last_read == "table"
        assert_table_is_replay(ledger, (NOW,))


def test_rebuild_on_a_lite_dsn(tmp_path: Path) -> None:
    """`recall provenance rebuild` serves a `sqlite:///` DSN: check, reconcile, rebuild.

    Red: `_cmd_provenance` without the `rebuild` dispatch falls through to `apply`, which needs
    `--claim`, so the JSON parse of the output fails.
    """
    path = tmp_path / "f.db"
    with lite(path) as (store, ledger):
        apply(ledger, PARIS, "r1")
        apply(LiteFactLedger(store, materialize=False), city("rome"), "r2", supersedes=(PARIS,))
    base = ["--dsn", "sqlite:///" + path.as_posix(), "--tenant", TENANT, "provenance", "rebuild"]
    checked = json.loads(run_cli([*base, "--check"]))
    assert (checked["ledger_events"], checked["filed_events"], checked["reads"]) == (3, 1, "replay")
    reconciled = json.loads(run_cli([*base, "--reconcile"]))
    assert reconciled["reconciled"] == 2 and reconciled["current"] is True
    rebuilt = json.loads(run_cli(base))
    assert rebuilt["action"] == "rebuild" and rebuilt["current"] is True and rebuilt["intervals"] == 1


# ------------------------------------------------------------------ the property

OFFSETS = (-400, -30, 0, 30, 60, 400)
INSTANTS = tuple(NOW + days * DAY for days in (-500, -400, -31, -30, -1, 0, 1, 29, 30, 31, 59, 60, 61, 399, 400, 401))
SOURCES = ("a.md", "b.md", "c.md")
VALUES = {"city": ("paris", "rome", "oslo"), "email": ("a@x.org", "b@x.org")}


@dataclass(frozen=True)
class Op:
    kind: str  # "assert", "refuse" or "retract"
    predicate: str = "city"
    value: int = 0
    window: tuple[int | None, int | None] = (None, None)
    supersede: int = 0  # 0: none; 1: the latest applied value of the key; 2: every earlier value
    stated: bool = False
    source: int = 0
    clock: int = 0  # days after NOW the write is stamped at (0 often, so stamps tie)


@st.composite
def windows(draw: st.DrawFn) -> tuple[int | None, int | None]:
    start = draw(st.sampled_from((None, *OFFSETS)))
    later = [offset for offset in OFFSETS if start is None or offset > start]
    end = draw(st.sampled_from((None, *later)))
    return start, end


OPS = st.one_of(
    st.builds(
        Op, kind=st.just("assert"), predicate=st.sampled_from(("city", "email")), value=st.integers(0, 2),
        window=windows(), supersede=st.integers(0, 2), stated=st.booleans(), source=st.integers(0, 2),
        clock=st.sampled_from((0, 0, 1, -1)),
    ),
    st.builds(Op, kind=st.just("refuse"), value=st.integers(0, 2)),
    st.builds(Op, kind=st.just("retract"), source=st.integers(0, 2)),
)


def op_fact(op: Op) -> AtomicFact:
    values = VALUES[op.predicate]
    start, end = op.window
    return city(
        values[op.value % len(values)],
        None if start is None else NOW + start * DAY,
        None if end is None else NOW + end * DAY,
        predicate=op.predicate,
    )


def run_ops(ledger: Any, ops: Sequence[Op], *, tenant: str = TENANT) -> list[object]:
    """Apply `ops` and return what each answered, so two ledgers can be compared op by op."""
    applied: list[AtomicFact] = []
    outcomes: list[object] = []
    for index, op in enumerate(ops):
        rid = f"op-{index}"
        try:
            if op.kind == "assert":
                fact = op_fact(op)
                same_key = [old for old in applied if old.predicate == fact.predicate and old != fact]
                names = () if op.supersede == 0 or not same_key else (
                    same_key[-1:] if op.supersede == 1 else same_key
                )
                event = apply(
                    ledger, fact, rid, supersedes=names, source=SOURCES[op.source],
                    now=NOW + op.clock * DAY, stated=op.stated, tenant=tenant,
                )
                if event.decision_code == DecisionCode.APPLIED:
                    applied.append(fact)
                outcomes.append((event.event_id, event.decision_code, event.created_at, event.supersedes_fact_ids))
            elif op.kind == "refuse":
                event = ledger.record_decision(
                    tenant_id=tenant, generation_id=GEN, request_id=rid, writer="t",
                    decision_code=DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION,
                    fact=city(VALUES["city"][op.value]), now=NOW,
                )
                outcomes.append((event.event_id, event.event_type))
            else:
                gone = SOURCES[op.source]
                written = ledger.retract_forgotten(
                    tenant_id=tenant, touched=lambda c, gone=gone: c.source == gone,
                    gone=lambda c, gone=gone: c.source == gone, now=NOW,
                )
                outcomes.append(tuple(event.event_id for event in written))
        except ValueError as exc:
            outcomes.append(("refused", str(exc)))
    return outcomes


def served_variants() -> tuple[Callable[[frozenset[str]], Collection[str]] | None, ...]:
    def without(name: str) -> Callable[[frozenset[str]], Collection[str]]:
        return lambda names: names - {name}

    return (None, lambda names: names, without("a.md"), without("b.md"), lambda _names: ())


def event_view(event: FactEvent) -> tuple[object, ...]:
    return (
        event.event_id, event.event_type, event.fact_id, event.fact.to_payload() if event.fact else None,
        [c.card_id for c in event.evidence_cards], list(event.supersedes_fact_ids), event.request_id,
        event.decision_code, event.created_at.isoformat(),
    )


def check_equivalence(table: Any, replay: Any, ops: Sequence[Op], *, tenant: str = TENANT) -> None:
    """Run `ops` on a materializing ledger and on one that replays; compare everything."""
    assert run_ops(table, ops, tenant=tenant) == run_ops(replay, ops, tenant=tenant)
    events = table.events
    assert [event_view(e) for e in events] == [event_view(e) for e in replay.events]
    for served in served_variants():
        for at in INSTANTS:
            current, support = table.current_view(tenant_id=tenant, now=at, served=served)
            assert table.last_read == "table"
            assert current == project_current(events, at, served), (at, served)
            sources = live_sources(support, served, at)
            everything = live_sources(events, served, at)
            assert {e.fact_id: sources[str(e.fact_id)] for e in current} == {
                e.fact_id: everything[str(e.fact_id)] for e in current
            }


# The window-rule cases, each guaranteed rather than left to the search: a future-dated
# successor, an expired bounded successor, a retroactive correction, a value flipped back, a
# set-valued predicate, and a retraction of a split fact.
CASES = (
    [Op("assert"), Op("assert", value=1, window=(30, None), supersede=1)],
    [Op("assert"), Op("assert", value=1, window=(30, 60), supersede=1)],
    [Op("assert"), Op("assert", value=2, window=(-400, -30), supersede=1)],
    [Op("assert"), Op("assert", value=1, supersede=1), Op("assert", value=0, supersede=1)],
    [Op("assert", predicate="email"), Op("assert", predicate="email", value=1)],
    [Op("assert"), Op("assert", value=2, window=(-400, -30), supersede=1, source=1), Op("retract")],
)


@settings(
    max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(st.lists(OPS, max_size=12))
@example(CASES[0])
@example(CASES[1])
@example(CASES[2])
@example(CASES[3])
@example(CASES[4])
@example(CASES[5])
def test_the_tables_answer_as_the_replay_on_any_ledger(ops: list[Op]) -> None:
    """Random ledgers: every write decides the same with and without the tables, and every read
    from them equals the replay at sixteen instants and five served sets.

    Red: the `project_key` own-window mutation fails "current == project_current" on the first
    explicit example (a future-dated successor), at the instant the successor starts, and on three
    other explicit examples.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with lite(Path(tmp) / "table.db") as (_a, table), lite(Path(tmp) / "replay.db", materialize=False) as (_b, replay):
            check_equivalence(table, replay, ops)
