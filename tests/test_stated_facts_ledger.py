"""The stated path's rules at the ledger and the controller, with no index involved.

A memo that states a fact makes it a candidate the moment the store records it
(`recall.stated_facts`). Two memos may state different values for one key with no supersession
between them; the ledger records both and the projection lists them as contested, so the append
must not refuse. A model's `recall_apply_fact` keeps refusing that same disagreement.

Invariants and the failure each one catches:
- L1 a stated assertion that disagrees with a current fact is APPLIED on a lite ledger; an
  ordinary one is still refused.
- L2 the lite ledger records the policy version a stated assertion carries.
- L3 the in-memory ledger follows L1.

Red proof: recorded per test below.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import (
    _PERMIT_SENTINEL,
    DecisionCode,
    FactApplicationPermit,
    InMemoryFactLedger,
    fact_identity,
)
from recall.types import AtomicFact, EvidenceCard
from tests.fact_fixtures import COFFEE, NOW, TEA, evidence_card

STATED_POLICY = "stated-facts-v1"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LiteStore]:
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=8) as opened:
        yield opened


def _permit(tenant: str, fact: AtomicFact, card: EvidenceCard, request_id: str) -> FactApplicationPermit:
    return FactApplicationPermit(
        tenant_id=tenant, generation_id="g1", request_id=request_id,
        fact_id=fact_identity(fact), card_ids=(card.card_id,), _token=_PERMIT_SENTINEL,
    )


def _append(ledger: object, tenant: str, fact: AtomicFact, source: str, request_id: str, **extra: object) -> object:
    card = evidence_card(fact, tenant=tenant, generation="g1", chunk=f"chunk-{source}", source=source)
    return ledger.apply_assertion(  # type: ignore[attr-defined]
        tenant_id=tenant, generation_id="g1", fact=fact, cards=(card,), request_id=request_id,
        writer="recall_stated", permit=_permit(tenant, fact, card, request_id), now=NOW, **extra,
    )


def test_l1_a_stated_disagreement_appends_on_a_lite_ledger(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l1_a_stated_disagreement_appends_on_a_lite_ledger.

    Red, mutation of `assertion_is_duplicate` (recall/fact_ledger.py): delete its `if stated:
    return False` branch. Fails on "a stated disagreement was refused: CONTRADICTION_WITHOUT_SUPERSESSION".
    """
    ledger = LiteFactLedger(store)
    _append(ledger, store.tenant, TEA, "a.md", "s1", stated=True)
    try:
        result = _append(ledger, store.tenant, COFFEE, "b.md", "s2", stated=True)
    except ValueError as exc:
        pytest.fail(f"a stated disagreement was refused: {exc}")
    assert result.event.decision_code == DecisionCode.APPLIED  # type: ignore[attr-defined]
    listed = sorted(e.fact.object for e in ledger.current(tenant_id=store.tenant, now=NOW) if e.fact)
    assert listed == ["coffee", "tea"], f"both stated values should be candidates: {listed}"
    water = AtomicFact(namespace="memory", subject="user", predicate="drink", object="water")
    with pytest.raises(ValueError, match="CONTRADICTION_WITHOUT_SUPERSESSION"):
        _append(ledger, store.tenant, water, "c.md", "a3")


def test_l2_the_lite_ledger_records_a_stated_policy_version(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l2_the_lite_ledger_records_a_stated_policy_version.

    Red, mutation of `LiteFactLedger.apply_assertion` (recall/lite/facts.py): pass
    `policy_version=CONTROLLER_POLICY_VERSION` to `_insert` in place of `policy_version`. Fails on
    "the stated event lost its policy version: provenance-controller-v1".
    """
    ledger = LiteFactLedger(store)
    _append(ledger, store.tenant, TEA, "a.md", "s1", stated=True, policy_version=STATED_POLICY)
    (event,) = [e for e in ledger.events if e.request_id == "s1"]
    assert event.policy_version == STATED_POLICY, f"the stated event lost its policy version: {event.policy_version}"


def test_l3_the_in_memory_ledger_follows_the_same_rule() -> None:
    """Node tests/test_stated_facts_ledger.py::test_l3_the_in_memory_ledger_follows_the_same_rule.

    Red, mutation of `InMemoryFactLedger.apply_assertion` (recall/provenance_controller.py): change
    `if not stated:` above its conflict checks to `if True:`. Fails on "a stated disagreement was
    refused in memory: CONTRADICTION_WITHOUT_SUPERSESSION".
    """
    ledger = InMemoryFactLedger()
    _append(ledger, "t", TEA, "a.md", "s1", stated=True)
    try:
        result = _append(ledger, "t", COFFEE, "b.md", "s2", stated=True)
    except ValueError as exc:
        pytest.fail(f"a stated disagreement was refused in memory: {exc}")
    assert result.event.decision_code == DecisionCode.APPLIED  # type: ignore[attr-defined]
