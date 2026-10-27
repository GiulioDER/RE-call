"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_maximum_instant_cca_981.py::test_unbounded_fact_is_current_at_maximum_instant[lite]: baseline assertion RED, repaired GREEN.
test_fact_maximum_instant_cca_981.py::test_unbounded_fact_is_current_at_maximum_instant[postgres]: baseline assertion RED, repaired GREEN.
test_fact_maximum_instant_cca_981.py::test_explicit_maximum_end_remains_exclusive[lite]: baseline control GREEN, production mutation RED, restored GREEN.
test_fact_maximum_instant_cca_981.py::test_explicit_maximum_end_remains_exclusive[postgres]: baseline control GREEN, production mutation RED, restored GREEN.
Control mutation: provenance_controller._holds: instant < end changed to instant <= end.
Both controls failed their intended assertion and passed after restoration."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from recall.fact_ledger import PostgresFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import project_current
from recall.types import AtomicFact
from tests.conftest import TEST_DSN, requires_db
from tests.test_fact_request_ids import Env

MAXIMUM = datetime.max.replace(tzinfo=UTC)
BEFORE_MAXIMUM = MAXIMUM - timedelta(microseconds=1)


@pytest.fixture(params=("lite", pytest.param("postgres", marks=requires_db)))
def maximum_env(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Env]:
    """A materializing ledger behind the existing public controller fixture."""
    tenant = f"maximum-instant-{uuid4().hex}"
    if request.param == "lite":
        with LiteStore(tmp_path / "maximum.db", dim=4, tenant=tenant) as store:
            yield Env(LiteFactLedger(store), tenant)
    else:
        request.getfixturevalue("make_store")(4)
        yield Env(PostgresFactLedger(TEST_DSN, tenant_id=tenant), tenant)


def test_unbounded_fact_is_current_at_maximum_instant(maximum_env: Env) -> None:
    """Baseline RED must be the absent unbounded fact at MAXIMUM, on both backends."""
    env = maximum_env
    claim = AtomicFact(namespace="memory", subject="person:alice", predicate="city", object="Paris")
    decision = env.apply(claim, "unbounded.md", "maximum-unbounded")
    assert decision.allowed and decision.code == "APPLIED", decision
    assert decision.event is not None
    expected_ids = (decision.event.fact_id,)

    before = env.ledger.current(tenant_id=env.tenant, now=BEFORE_MAXIMUM)
    assert tuple(event.fact_id for event in before) == expected_ids
    actual = env.ledger.current(tenant_id=env.tenant, now=MAXIMUM)
    assert tuple(event.fact_id for event in actual) == expected_ids, (
        "an unbounded fact disappeared at the maximum supported instant"
    )
    replay = project_current(env.ledger.events, MAXIMUM)
    assert tuple(event.fact_id for event in actual) == tuple(event.fact_id for event in replay)


def test_explicit_maximum_end_remains_exclusive(maximum_env: Env) -> None:
    """Control RED needs the documented inclusive-upper-bound mutation, then restore GREEN."""
    env = maximum_env
    claim = AtomicFact(
        namespace="memory", subject="person:alice", predicate="city", object="Paris",
        valid_until=MAXIMUM,
    )
    decision = env.apply(claim, "bounded.md", "maximum-bounded")
    assert decision.allowed and decision.code == "APPLIED", decision
    assert decision.event is not None
    before = env.ledger.current(tenant_id=env.tenant, now=BEFORE_MAXIMUM)
    assert tuple(event.fact_id for event in before) == (decision.event.fact_id,)

    actual = env.ledger.current(tenant_id=env.tenant, now=MAXIMUM)
    assert tuple(event.fact_id for event in actual) == (), (
        "an explicit maximum upper bound became inclusive"
    )
    replay = project_current(env.ledger.events, MAXIMUM)
    assert tuple(event.fact_id for event in actual) == tuple(event.fact_id for event in replay)
