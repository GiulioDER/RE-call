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
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import (
    _PERMIT_SENTINEL,
    DecisionCode,
    FactApplicationPermit,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
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


def _stated_card(fact: AtomicFact, *, tenant: str, source: str, digest: str = "h1") -> EvidenceCard:
    """A card as `recall.stated_facts.stated_card` builds one: no search, no calibration."""
    return EvidenceCard(
        card_id="", chunk_id=f"chunk-{source}", source=source, source_digest=digest,
        valid_from=None, valid_until=None, first_indexed_at=None, indexed_at=None,
        tenant_id=tenant, generation_id="g1", pipeline_fingerprint=None, corpus_fingerprint=None,
        calibration_id=None, calibration_status="not_applicable", trust_state="stated",
        verdict="unverified", confidence=1.0, rank=1, structured_facts=(fact,),
    )


def _controller(store: LiteStore) -> ProvenanceController:
    return ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=LiteEvidenceCardStore(store),
        ledger=LiteFactLedger(store), now=lambda: NOW, writer="recall_stated",
    )


def test_l4_a_stated_card_applies_its_fact_once(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l4_a_stated_card_applies_its_fact_once.

    Red, mutation of `ProvenanceController.apply_stated` (recall/provenance_controller.py): return
    `self._decision(request, DecisionCode.UNSUPPORTED_CLAIM, cards=(card,))` unconditionally.
    Fails on "a stated card did not apply its fact: UNSUPPORTED_CLAIM".
    """
    controller = _controller(store)
    first = controller.apply_stated(TEA, _stated_card(TEA, tenant=store.tenant, source="a.md"), "s1")
    assert first.code == DecisionCode.APPLIED, f"a stated card did not apply its fact: {first.code}"
    second = controller.apply_stated(TEA, _stated_card(TEA, tenant=store.tenant, source="b.md"), "s2")
    assert second.code == DecisionCode.DUPLICATE
    (event,) = [e for e in LiteFactLedger(store).events if e.request_id == "s1"]
    assert (event.writer, event.policy_version) == ("recall_stated", "stated-facts-v1")


def test_l5_a_card_that_is_not_stated_or_lacks_the_fact_writes_nothing(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l5_a_card_that_is_not_stated_or_lacks_the_fact_writes_nothing.

    Red, mutation of `ProvenanceController.apply_stated`: delete the
    `card.trust_state != STATED_TRUST_STATE or` clause. Fails on "a trusted search card took the
    stated path: APPLIED".
    """
    controller = _controller(store)
    searched = evidence_card(TEA, tenant=store.tenant, generation="g1", chunk="c", source="a.md")
    decision = controller.apply_stated(TEA, searched, "s1")
    assert decision.code == DecisionCode.UNSUPPORTED_CLAIM, f"a trusted search card took the stated path: {decision.code}"
    other = controller.apply_stated(COFFEE, _stated_card(TEA, tenant=store.tenant, source="a.md"), "s2")
    assert other.code == DecisionCode.UNSUPPORTED_CLAIM
    assert LiteFactLedger(store).events == (), "a refused stated application wrote an event"


def test_l6_a_stated_disagreement_is_applied_not_refused(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l6_a_stated_disagreement_is_applied_not_refused.

    Red, mutation of `ProvenanceController.apply_stated`: remove `"stated": True` from the
    assertion's keyword arguments. Fails on "the controller refused a stated disagreement:
    CONTRADICTION_WITHOUT_SUPERSESSION".
    """
    controller = _controller(store)
    controller.apply_stated(TEA, _stated_card(TEA, tenant=store.tenant, source="a.md"), "s1")
    decision = controller.apply_stated(COFFEE, _stated_card(COFFEE, tenant=store.tenant, source="b.md"), "s2")
    assert decision.code == DecisionCode.APPLIED, f"the controller refused a stated disagreement: {decision.code}"


def test_l7_a_stated_card_cannot_authorize_apply_fact(store: LiteStore) -> None:
    """Node tests/test_stated_facts_ledger.py::test_l7_a_stated_card_cannot_authorize_apply_fact.

    Spec test 10. Red, mutation of `EvidenceCard.trusted_for_application` (recall/types.py): accept
    `self.trust_state in {"trusted", "stated"}` and drop the verdict and calibration conditions.
    Fails on "a model applied a fact on a stated card: APPLIED".
    """
    card = _stated_card(TEA, tenant=store.tenant, source="a.md")
    cards = LiteEvidenceCardStore(store)
    cards.put((card,))
    decision = ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        now=lambda: NOW, writer="model",
    ).apply_fact(FactApplicationRequest(TEA, (card.card_id,), "m1"))
    assert decision.code == DecisionCode.TRUST_UNAVAILABLE, f"a model applied a fact on a stated card: {decision.code}"
