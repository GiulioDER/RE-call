"""The fact ledger's projection and identity rules, shared by the PostgreSQL and lite ledgers.

Invariants and the failure each catches:
- event ids are a pure function of (tenant, fact, request), so both backends mint the same id for
  the same application; a backend that hashed anything else would hand a client a different id.
- the current projection drops superseded, expired and not yet valid facts; the candidate
  projection (what a new claim may conflict with) keeps expired and future ones.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from recall.fact_ledger import (
    assertion_event_id,
    decision_event_id,
    decision_event_type,
    project_candidates,
    project_current,
)
from recall.provenance_controller import DecisionCode, FactEvent, fact_identity
from recall.types import AtomicFact

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
TEA = AtomicFact(namespace="memory", subject="user", predicate="drink", object="tea")
COFFEE = AtomicFact(namespace="memory", subject="user", predicate="drink", object="coffee")


def _event(
    fact: AtomicFact | None,
    event_type: str = "asserted",
    supersedes: tuple[str, ...] = (),
    request: str = "r1",
) -> FactEvent:
    return FactEvent(
        event_id=f"e-{request}-{event_type}",
        event_type=event_type,
        tenant_id="t",
        generation_id="g",
        fact=fact,
        fact_id=fact_identity(fact) if fact else None,
        evidence_cards=(),
        supersedes_fact_ids=supersedes,
        request_id=request,
        writer="w",
        decision_code="applied",
        policy_version="p",
        controller_version=1,
        created_at=NOW,
    )


def test_assertion_event_id_is_the_postgres_derivation() -> None:
    """Red (2026-10-12): `assertion_event_id` hashing without `request_id` fails the equality."""
    expected = "evt_" + hashlib.sha256(
        f"t\0{fact_identity(TEA)}\0r1\0asserted".encode("utf-8")
    ).hexdigest()[:32]
    assert assertion_event_id("t", TEA, "r1") == expected


def test_decision_event_rules() -> None:
    """Red (2026-10-12): `decision_event_type` always returning "rejected" fails the abstained case."""
    assert decision_event_type(DecisionCode.FRESH_SEARCH_INSUFFICIENT) == "abstained"
    assert decision_event_type(DecisionCode.CARD_NOT_FOUND) == "rejected"
    expected = "evt_" + hashlib.sha256("t\0r1\0rejected".encode("utf-8")).hexdigest()[:32]
    assert decision_event_id("t", "r1", "rejected") == expected


def test_projection_drops_superseded_expired_and_future_facts() -> None:
    """Red (2026-10-12): `project_current` without the valid_until test returns the expired fact."""
    expired = AtomicFact(
        namespace="memory", subject="user", predicate="lives", object="rome",
        valid_until=NOW - timedelta(days=1),
    )
    future = AtomicFact(
        namespace="memory", subject="user", predicate="plans", object="trip",
        valid_from=NOW + timedelta(days=1),
    )
    events = [
        _event(TEA, request="r1"),
        _event(COFFEE, request="r2"),
        _event(None, "superseded", supersedes=(fact_identity(TEA),), request="r2:supersession"),
        _event(expired, request="r3"),
        _event(future, request="r4"),
    ]
    assert [e.fact for e in project_current(events, NOW)] == [COFFEE]
    assert [e.fact for e in project_candidates(events)] == [COFFEE, expired, future]
