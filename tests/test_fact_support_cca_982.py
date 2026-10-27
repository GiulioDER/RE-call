"""PR982 BUG-001 / FIX-001: retired support cannot retarget a later assertion episode.

Nodes (each on memory, SQLite, lite and configured PostgreSQL):
tests/test_fact_support_cca_982.py::test_retired_support_cannot_retire_reasserted_fact
  [original-support] and [old-duplicate-support].
tests/test_fact_support_cca_982.py::test_current_episode_support_can_retire_reasserted_fact
  [current-assertion] and [current-duplicate-support].

Baseline: 1c34dcccb1ad10876146bf8b848525d6fa0650e2.
Production boundary: ProvenanceController.apply_fact, through real public ledger APIs.
Invariant: Paris from A; Rome superseding A; Paris from C superseding Rome establishes a new
Paris assertion. Milan linking only retired A or its old DUPLICATE cannot retire C. Milan may
link C or a genuine DUPLICATE support of C. One fixed clock intentionally exercises ledger order.

Predicted negative baseline RED: the explicit refusal assertion sees APPLIED rather than
CONTRADICTION_WITHOUT_SUPERSESSION. Positive controls are expected to pass the baseline. Their
observed RED mutation inverted the APPLIED owner's episode_in_force status in assertion_cards.
Observed 2026-10-26 on the isolated testbench: all eight negative cases failed the intended
baseline refusal assertion and then passed after repair. All eight positive controls failed their
APPLIED assertion under that mutation, then passed after hash-verified restoration. The scoped run
passed 387 tests with one unrelated Windows-only skip; none of these sixteen cases was skipped.
"""

from __future__ import annotations

import pytest

from recall.provenance_controller import ControllerDecision, DecisionCode
from tests.test_fact_request_ids import NOW, Env, env as env
from tests.test_fact_supersession_windows import city


def _current_event_ids(env: Env) -> list[str]:
    return [
        event.event_id
        for event in env.ledger.current(tenant_id=env.tenant, now=NOW)
    ]


def _reasserted_paris(env: Env, *, old_duplicate: bool = False) -> ControllerDecision:
    first = env.apply(city("Paris"), "a.md", "r-a")
    assert first.allowed and first.code == DecisionCode.APPLIED, "initial Paris was not applied"
    assert first.event is not None
    if old_duplicate:
        duplicate = env.apply(city("Paris"), "a2.md", "r-a2")
        assert duplicate.allowed and duplicate.code == DecisionCode.DUPLICATE, (
            "A2 must be genuine DUPLICATE support in the original Paris episode"
        )
        assert duplicate.event is not None and duplicate.event.event_id != first.event.event_id
        assert duplicate.cards[0].card_id != first.cards[0].card_id
        assert _current_event_ids(env) == [first.event.event_id]

    middle = env.apply(city("Rome"), "b.md", "r-b", links=("a.md",))
    assert middle.allowed and middle.code == DecisionCode.APPLIED, "Rome did not retire original Paris"
    assert middle.event is not None
    assert env.current() == ["Rome"]
    assert _current_event_ids(env) == [middle.event.event_id]

    current = env.apply(city("Paris"), "c.md", "r-c", links=("b.md",))
    assert current.allowed and current.code == DecisionCode.APPLIED, (
        "C must establish a new Paris assertion, not replay the retired one"
    )
    assert current.event is not None
    assert current.event.fact_id == first.event.fact_id
    assert current.event.event_id != first.event.event_id
    assert first.event.created_at < middle.event.created_at < current.event.created_at
    assert env.current() == ["Paris"]
    assert _current_event_ids(env) == [current.event.event_id], "C must be the current Paris assertion"
    return current


@pytest.mark.parametrize(
    "retired_source", ["a.md", "a2.md"], ids=["original-support", "old-duplicate-support"]
)
def test_retired_support_cannot_retire_reasserted_fact(env: Env, retired_source: str) -> None:
    """An authored link to an earlier episode must not authorize superseding the current one."""
    current = _reasserted_paris(env, old_duplicate=retired_source == "a2.md")
    assert current.event is not None
    decision = env.apply(city("Milan"), "d.md", "r-d", links=(retired_source,))
    assert decision.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, (
        "fully retired support authorized retirement of a later assertion: "
        f"{retired_source} produced {decision.code}"
    )
    assert not decision.allowed
    assert not any(
        event.request_id == "r-d" and event.event_type == "asserted"
        for event in env.ledger.events
    ), "the refused Milan request appended a successful assertion"
    assert env.current() == ["Paris"]
    assert _current_event_ids(env) == [current.event.event_id], "refusal displaced C's Paris assertion"


@pytest.mark.parametrize(
    "current_source", ["c.md", "c2.md"], ids=["current-assertion", "current-duplicate-support"]
)
def test_current_episode_support_can_retire_reasserted_fact(env: Env, current_source: str) -> None:
    """Current direct support and genuine current DUPLICATE support must both remain usable."""
    current = _reasserted_paris(env)
    assert current.event is not None
    if current_source == "c2.md":
        duplicate = env.apply(city("Paris"), "c2.md", "r-c2")
        assert duplicate.allowed and duplicate.code == DecisionCode.DUPLICATE, (
            "C2 must be genuine DUPLICATE support in the current Paris episode"
        )
        assert duplicate.event is not None and duplicate.event.event_id != current.event.event_id
        assert duplicate.cards[0].card_id != current.cards[0].card_id
        assert _current_event_ids(env) == [current.event.event_id]

    decision = env.apply(city("Milan"), "d.md", "r-d", links=(current_source,))
    assert decision.code == DecisionCode.APPLIED, (
        f"genuine current support {current_source} could not retire Paris: {decision.code}"
    )
    assert decision.allowed and decision.event is not None
    assert decision.event.supersedes_fact_ids == (current.event.fact_id,)
    assert env.current() == ["Milan"]
    assert _current_event_ids(env) == [decision.event.event_id]
