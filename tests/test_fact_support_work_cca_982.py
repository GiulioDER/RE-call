"""Deterministic testbench for avoiding support aggregation without a conflicting fact.

Only the public controller application is the action under test. The support-helper spy
delegates to the real implementation and is installed after the DUPLICATE history exists.
No duration or throughput measurement is involved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

import recall.provenance_controller as provenance
from recall.fact_ledger import SQLiteFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import DecisionCode, FactEvent, InMemoryFactLedger
from recall.types import AtomicFact, EvidenceCard
from tests.test_fact_supersession_windows import Env, TENANT, city


@pytest.mark.parametrize(
    "backend,duplicate_count",
    [
        pytest.param(backend, count, id=f"{backend}-{count}")
        for backend in ("memory", "sqlite", "lite")
        for count in (16, 32)
    ],
)
def test_unrelated_predicate_does_not_aggregate_support_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    duplicate_count: int,
) -> None:
    """Node IDs:
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[memory-16]
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[memory-32]
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[sqlite-16]
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[sqlite-32]
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[lite-16]
    tests/test_fact_support_work_cca_982.py::test_unrelated_predicate_does_not_aggregate_support_history[lite-32]

    Invariant: an unrelated predicate with no conflicts does not aggregate the cards of
    the existing fact's genuinely recorded DUPLICATE applications. Its application must
    still be APPLIED, leave both facts current, and preserve every supporting assertion.

    Baseline: 1c34dcccb1ad10876146bf8b848525d6fa0650e2 with this file copied in.
    Target: ProvenanceController.apply_fact eagerly calls assertion_cards(in_force.history)
    even when its conflicts tuple is empty. Expected RED: the final assertion reports a
    support-history aggregation over 17 or 33 real assertion/card snapshots instead of none.
    Observed 2026-10-26 on the isolated testbench: all six cases failed the intended final
    aggregation assertion on the baseline (17 or 33 real event/card snapshots), then passed
    after repair. The scoped run passed 387 tests with one unrelated Windows-only skip;
    no case in this file was skipped and no fixture error was counted as RED.

    served_sources is None through Env, so legitimate absence/liveness checks return before
    assertion_cards. The spy is active for the unrelated application only, not setup or
    the final current-facts control. It observes and delegates, never substitutes a result.
    """
    with ExitStack() as resources:
        ledger: Any
        if backend == "memory":
            ledger = InMemoryFactLedger()
        elif backend == "sqlite":
            ledger = resources.enter_context(
                SQLiteFactLedger(str(tmp_path / "ledger.db"), tenant_id=TENANT)
            )
        else:
            store = resources.enter_context(LiteStore(tmp_path / "lite.db", dim=4, tenant=TENANT))
            ledger = LiteFactLedger(store)
        env = Env(ledger)
        claim = city("Paris")
        first = env.apply(claim, "original.md", "original")
        assert first.allowed and first.code == DecisionCode.APPLIED
        support_ids: set[str] = set()
        for number in range(duplicate_count):
            duplicate = env.apply(claim, f"support-{number}.md", f"support-{number}")
            assert duplicate.allowed and duplicate.code == DecisionCode.DUPLICATE
            assert duplicate.event is not None
            support_ids.update(card.card_id for card in duplicate.event.evidence_cards)

        before = tuple(ledger.events)
        assert len(before) == duplicate_count + 1
        assert Counter(event.decision_code for event in before) == {
            DecisionCode.APPLIED: 1,
            DecisionCode.DUPLICATE: duplicate_count,
        }
        assert len(support_ids) == duplicate_count, "setup did not record distinct supporting cards"
        assert all(event.event_type == "asserted" for event in before)
        assert env.current() == ["Paris"]

        original_aggregate = provenance.assertion_cards
        aggregated_history: list[tuple[int, int]] = []

        def observe_aggregation(events: Sequence[FactEvent]) -> dict[str, tuple[EvidenceCard, ...]]:
            aggregated_history.append((len(events), sum(len(event.evidence_cards) for event in events)))
            return original_aggregate(events)

        unrelated = AtomicFact(
            namespace=claim.namespace,
            subject=claim.subject,
            predicate="eyes",
            object="blue",
        )
        with monkeypatch.context() as observation:
            observation.setattr(provenance, "assertion_cards", observe_aggregation)
            decision = env.apply(unrelated, "unrelated.md", "unrelated")

        assert decision.allowed and decision.code == DecisionCode.APPLIED, decision
        assert decision.event is not None and decision.event.request_id == "unrelated"
        assert env.current() == ["Paris", "blue"]
        after = tuple(ledger.events)
        assert len(after) == len(before) + 1
        assert {event.event_id for event in before} <= {event.event_id for event in after}
        assert aggregated_history == [], (
            "an unrelated predicate aggregated support history without any conflict: "
            f"(events, cards)={aggregated_history}, DUPLICATE supports={duplicate_count}"
        )
