"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_history_purge_cca_981.py::test_purge_during_history_read_preserves_duplicate_decision: baseline assertion RED, repaired GREEN.
test_fact_history_purge_cca_981.py::test_purge_during_history_read_preserves_backward_clock_order: baseline assertion RED, repaired GREEN."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

import psycopg
import pytest

from recall.fact_current import PostgresCurrentFacts, PostgresTables
from recall.fact_ledger import PostgresFactLedger
from recall.provenance_controller import (
    ControllerDecision,
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    FactEvent,
    ProvenanceController,
)
from recall.types import AtomicFact
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW, evidence_card

pytestmark = requires_db
GENERATION = "generation-history-purge"


def _city(value: str) -> AtomicFact:
    return AtomicFact(namespace="memory", subject="person:alice", predicate="city", object=value)


def _apply(
    ledger: PostgresFactLedger,
    claim: AtomicFact,
    memo: str,
    request_id: str,
    *,
    now: datetime = NOW,
    links: tuple[str, ...] = (),
) -> ControllerDecision:
    card = evidence_card(
        claim, tenant=ledger.tenant_id, generation=GENERATION,
        chunk=f"chunk-{memo}", source=memo, links=links,
    )
    cards = EvidenceCardStore()
    cards.put([card])
    controller = ProvenanceController(
        tenant_id=ledger.tenant_id, generation_id=GENERATION, cards=cards, ledger=ledger,
        source_digest_for=lambda _card: "digest", now=lambda: now, writer="history-purge",
    )
    return controller.apply_fact(FactApplicationRequest(claim, (card.card_id,), request_id))


@contextmanager
def _purge_between_history_queries(
    monkeypatch: pytest.MonkeyPatch, tenant: str,
) -> Iterator[list[str]]:
    """Delegate every real query; coordinate only the append's READ COMMITTED boundary."""
    purger = PostgresCurrentFacts(TEST_DSN, tenant_id=tenant, statement_timeout_ms=8000)
    connected, finished = Event(), Event()
    purger_pids: list[int] = []
    errors: list[BaseException] = []
    outcomes: list[str] = []
    workers: list[Thread] = []
    original_connect = purger._connect
    original_key_events = PostgresTables.key_events

    def connect_purger() -> psycopg.Connection:
        conn = original_connect()
        purger_pids.append(conn.info.backend_pid)
        connected.set()
        return conn

    def purge() -> None:
        try:
            purger.purge()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    with psycopg.connect(TEST_DSN, autocommit=True, connect_timeout=5) as observer:
        observer.execute("SET statement_timeout = 1000")

        def key_events(tables: PostgresTables, key: str) -> list[FactEvent] | None:
            isolation = tables._conn.execute("SHOW transaction_isolation").fetchone()
            if tables.tenant_id == tenant and not workers and isolation == ("read committed",):
                worker = Thread(target=purge, name="fact-history-purge", daemon=True)
                workers.append(worker)
                worker.start()
                assert connected.wait(5), "purger did not establish its independent connection"
                deadline = monotonic() + 5
                while True:
                    if finished.is_set():
                        assert not errors, f"public purge failed: {errors!r}"
                        outcomes.append("purge-committed")
                        break
                    waiting = observer.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_locks "
                        "WHERE pid = %s AND locktype = 'advisory' AND NOT granted) "
                        "AND %s = ANY(pg_blocking_pids(%s))",
                        (purger_pids[0], tables._conn.info.backend_pid, purger_pids[0]),
                    ).fetchone()
                    if waiting == (True,):
                        outcomes.append("purge-waits-for-writer")
                        break
                    assert monotonic() < deadline, "purger neither committed nor waited on writer"
                    finished.wait(0.01)
            return original_key_events(tables, key)

        with monkeypatch.context() as patch:
            patch.setattr(purger, "_connect", connect_purger)
            patch.setattr(PostgresTables, "key_events", key_events)
            try:
                yield outcomes
            finally:
                for worker in workers:
                    worker.join(timeout=12)
                    assert not worker.is_alive(), "public purge did not finish after writer returned"
                assert not errors, f"public purge failed: {errors!r}"
                assert len(outcomes) == 1, "write-side materialized history boundary was not exercised"


def test_purge_during_history_read_preserves_duplicate_decision(
    make_store: Callable[[int], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second public application remains DUPLICATE while a purge races its history read.

    Node: tests/test_fact_history_purge_cca_981.py::test_purge_during_history_read_preserves_duplicate_decision.
    Expected baseline RED: the decision-code assertion sees APPLIED after purge empties the
    tables between usable and key_events. Execution observed in the module proof receipt; no result is claimed here.
    """
    make_store(4)
    tenant = f"history-purge-{uuid4().hex}"
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
    claim = _city("Paris")
    first = _apply(ledger, claim, "first.md", "first")
    assert first.allowed and first.code == DecisionCode.APPLIED and first.event is not None
    assert PostgresCurrentFacts(TEST_DSN, tenant_id=tenant).status().current

    with _purge_between_history_queries(monkeypatch, tenant):
        repeated = _apply(ledger, claim, "second.md", "second")

    assert repeated.code == DecisionCode.DUPLICATE, (
        "purging derived rows turned a repeat of the same fact into a new APPLIED assertion: "
        f"{repeated.code}"
    )
    assert repeated.allowed and repeated.event is not None
    assertions = [event for event in ledger.events if event.event_type == "asserted"]
    assert len(assertions) == 2
    assert [event.decision_code for event in assertions] == [DecisionCode.APPLIED, DecisionCode.DUPLICATE]
    current = ledger.current(tenant_id=tenant, now=NOW)
    assert [event.event_id for event in current] == [first.event.event_id]


def test_purge_during_history_read_preserves_backward_clock_order(
    make_store: Callable[[int], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correction remains ordered after its predecessor despite a deliberately earlier clock.

    Node: tests/test_fact_history_purge_cca_981.py::test_purge_during_history_read_preserves_backward_clock_order.
    Public controller clock target: NOW minus one day on the second request only. The ledger
    must advance that timestamp past the predecessor using its real history. Expected baseline
    RED: the strict created_at assertion fails after purge supplies an empty key history.
    Execution observed in the module proof receipt; the clock injection is deliberate, not an elapsed-time measurement.
    """
    make_store(4)
    tenant = f"history-purge-{uuid4().hex}"
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
    first = _apply(ledger, _city("Paris"), "first.md", "first")
    assert first.allowed and first.code == DecisionCode.APPLIED and first.event is not None
    assert PostgresCurrentFacts(TEST_DSN, tenant_id=tenant).status().current

    with _purge_between_history_queries(monkeypatch, tenant):
        correction = _apply(
            ledger, _city("Rome"), "second.md", "second",
            now=NOW - timedelta(days=1), links=("first.md",),
        )

    assert correction.allowed and correction.code == DecisionCode.APPLIED, correction
    assert correction.event is not None
    assert correction.event.created_at > first.event.created_at, (
        "purging derived rows lost the predecessor used to order the correction: "
        f"{correction.event.created_at} <= {first.event.created_at}"
    )
    assert correction.event.supersedes_fact_ids == (first.event.fact_id,)
    current = ledger.current(tenant_id=tenant, now=NOW)
    assert [event.event_id for event in current] == [correction.event.event_id]
