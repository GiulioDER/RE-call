"""Packaging regression: withdrawal preserves compact fact-history replay."""
from __future__ import annotations

from dataclasses import replace

from recall.provenance_controller import DecisionCode
from tests.test_fact_supersession_legacy_scaling import (
    ALLOCATION_BUDGET,
    _measure_projection,
    _suffix,
)


def test_withdrawing_the_latest_alternating_value_keeps_replay_compact() -> None:
    """Node tests/test_fact_supersession_withdrawal_scaling.py::test_withdrawing_the_latest_alternating_value_keeps_replay_compact.

    The 3,000-assertion history must stay inside the existing 24 MiB replay budget after
    withdrawing its head, and the immediately preceding assertion must become current.
    Red-proof plan: restore origin/master's provenance_controller.py; its dense graph must
    fail the allocation assertion. Independently mutate own_windows to retain a withdrawn
    claim's own span; the final current-event assertion must fail. Both exact red states
    and the restored green node must be recorded before treating this as proven.
    """
    events = _suffix()
    latest = events[-1]
    events.append(replace(
        latest,
        event_id="withdraw-latest",
        event_type="withdrawn",
        fact=None,
        fact_id=None,
        supersedes_fact_ids=(str(latest.fact_id),),
        request_id="withdrawal:" + latest.event_id,
        decision_code=DecisionCode.WITHDRAWN,
    ))

    current, peak = _measure_projection(events)

    assert peak < ALLOCATION_BUDGET, (
        f"withdrawing the head expanded alternating replay: {peak} bytes"
    )
    assert [event.event_id for event in current] == ["suffix-2998"], (
        "the withdrawn head still hides the immediately preceding assertion"
    )
