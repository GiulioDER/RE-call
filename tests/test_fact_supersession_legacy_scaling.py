"""PERF-001: legacy ties must not make later ordered replay dense.

Both nodes were red-proved by restoring the candidate-1 production module,
SHA-256 74309e0cc5f7d43afba0b4807edbcf89391b616c77c08bedd6c3b710c8c868b8.
Its tenant-wide equal-time fallback failed both 24 MiB allocation assertions.
Production bytes were restored and hash-verified.
"""
from __future__ import annotations

import tracemalloc
from datetime import UTC, datetime, timedelta

from recall.provenance_controller import (
    CONTROLLER_SCHEMA_VERSION,
    DecisionCode,
    FactEvent,
    fact_identity,
    project_current,
)
from recall.types import AtomicFact

NOW = datetime(2026, 10, 21, 12, tzinfo=UTC)
ALLOCATION_BUDGET = 24 * 1024 * 1024
SUFFIX_LENGTH = 3000


def _fact(value: str, subject: str = "person:alice") -> AtomicFact:
    return AtomicFact(namespace="memory", subject=subject, predicate="city", object=value)


def _assertion(
    event_id: str, fact: AtomicFact, supersedes: tuple[str, ...], written: datetime
) -> FactEvent:
    return FactEvent(
        event_id=event_id,
        event_type="asserted",
        tenant_id="legacy-scaling",
        generation_id="generation",
        fact=fact,
        fact_id=fact_identity(fact),
        evidence_cards=(),
        supersedes_fact_ids=supersedes,
        request_id="request-" + event_id,
        writer="fixture",
        decision_code=DecisionCode.APPLIED,
        policy_version="fixture",
        controller_version=CONTROLLER_SCHEMA_VERSION,
        created_at=written,
    )


def _suffix(prior: AtomicFact | None = None) -> list[FactEvent]:
    facts = (_fact("Paris"), _fact("Rome"))
    identities = tuple(fact_identity(fact) for fact in facts)
    result = []
    for index in range(SUFFIX_LENGTH):
        targets = (fact_identity(prior),) if index == 0 and prior is not None else (
            (identities[(index - 1) % 2],) if index else ()
        )
        result.append(_assertion(
            f"suffix-{index}", facts[index % 2], targets, NOW + timedelta(microseconds=index + 1),
        ))
    return result


def _measure_projection(events: list[FactEvent]) -> tuple[tuple[FactEvent, ...], int]:
    # Input construction is outside the trace; only replay's temporary allocation is bounded.
    tracemalloc.start()
    try:
        current = project_current(events, NOW)
        _retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return current, peak


def test_unrelated_legacy_tie_keeps_ordered_replay_compact() -> None:
    """Node tests/test_fact_supersession_legacy_scaling.py::test_unrelated_legacy_tie_keeps_ordered_replay_compact.

    An unrelated legacy tie must not expand 3,000 ordered alternating assertions.
    Red-proved by restoring the candidate-1 module identified above;
    its tenant-wide fallback exceeded the 24 MiB allocation assertion.
    The tied successor and the ordered key's last assertion must remain current.
    """
    first, second = _fact("Oslo", "legacy:other"), _fact("Milan", "legacy:other")
    events = [
        _assertion("legacy-first", first, (), NOW),
        _assertion("legacy-second", second, (fact_identity(first),), NOW),
        *_suffix(),
    ]

    current, peak = _measure_projection(events)

    assert peak < ALLOCATION_BUDGET, (
        f"unrelated legacy tie expanded ordered replay: {peak} bytes"
    )
    assert [event.event_id for event in current] == ["legacy-second", "suffix-2999"]


def test_same_key_legacy_tie_keeps_ordered_suffix_compact() -> None:
    """Node tests/test_fact_supersession_legacy_scaling.py::test_same_key_legacy_tie_keeps_ordered_suffix_compact.

    A same-key legacy tie must not make a 3,000-assertion ordered suffix dense.
    Red-proved by restoring the candidate-1 module identified above;
    its fallback exceeded the 24 MiB allocation assertion.
    Only the suffix's last assertion may remain current.
    """
    first, second = _fact("Paris"), _fact("Rome")
    events = [
        _assertion("legacy-first", first, (), NOW),
        _assertion("legacy-second", second, (fact_identity(first),), NOW),
        *_suffix(prior=second),
    ]

    current, peak = _measure_projection(events)

    assert peak < ALLOCATION_BUDGET, (
        f"same-key legacy tie expanded ordered suffix: {peak} bytes"
    )
    assert [event.event_id for event in current] == ["suffix-2999"]
