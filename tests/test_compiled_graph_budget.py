"""The USD cap: a call is sent only when its worst case fits what is left.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.budget``):

- ``test_call_refused_when_worst_case_exceeds_remaining``: the reservation ignoring
  ``max_tokens`` fails on the refusal.
- ``test_spend_never_exceeds_cap_over_a_run``: input reserved at ``bytes // 3`` fails on the call
  count (token-dense replies overrun that reservation and exhaust the budget after one call).
- ``test_settle_uses_reported_usage``: settling the reservation instead of the usage fails on
  ``spent_usd``.
- ``test_missing_usage_charges_the_reservation``: missing usage charged as zero fails.
- ``test_overrun_exhausts_the_budget``: a reply over its reservation not marking the budget
  exhausted fails on the next ``reserve``.
- ``test_from_env_requires_cap_and_prices``: a default price fails on ``pytest.raises``.
"""

from __future__ import annotations

import math

import pytest

from recall.compiled_graph.budget import Budget
from recall.compiled_graph.provider import CompiledGraphConfigError, CompileReply

def _budget(max_usd: float) -> Budget:
    return Budget(max_usd=max_usd, cost_in_per_1k=0.002, cost_out_per_1k=0.008)


def _reply(tokens_in: int | None, tokens_out: int | None) -> CompileReply:
    return CompileReply(content={"records": []}, input_tokens=tokens_in, output_tokens=tokens_out)


def test_call_refused_when_worst_case_exceeds_remaining() -> None:
    budget = _budget(0.02)
    # Worst case: 1,000 input tokens at 0.002 = 0.002, 2,400 output at 0.008 = 0.0192.
    assert budget.reserve(1_000) is False
    assert budget.reserve(100) is True


def test_spend_never_exceeds_cap_over_a_run() -> None:
    budget = _budget(0.5)
    sent = 0
    while budget.reserve(30_000):
        # A token-dense source: one token per byte, every output token used.
        budget.settle(_reply(30_000, 2_400))
        sent += 1
    # Each call reserves exactly what it then costs (0.0792), so six fit under 0.5 and none overran.
    assert sent == 6
    assert not budget.exhausted
    assert budget.spent_usd <= 0.5


def test_settle_uses_reported_usage() -> None:
    budget = _budget(1.0)
    assert budget.reserve(10_000)
    budget.settle(_reply(2_000, 100))
    assert math.isclose(budget.spent_usd, 2 * 0.002 + 0.1 * 0.008)


def test_missing_usage_charges_the_reservation() -> None:
    budget = _budget(1.0)
    assert budget.reserve(10_000)
    budget.settle(None)
    assert math.isclose(budget.spent_usd, 10 * 0.002 + 2.4 * 0.008)
    assert budget.reserve(10_000)
    budget.settle(_reply(None, 50))
    assert math.isclose(budget.spent_usd, 2 * (10 * 0.002 + 2.4 * 0.008))


def test_overrun_exhausts_the_budget() -> None:
    budget = _budget(1.0)
    assert budget.reserve(100)
    # A reasoning model billing past max_tokens: 0.16, under the cap but over the reservation.
    budget.settle(_reply(100, 20_000))
    assert budget.exhausted
    assert budget.reserve(1) is False


def test_reserve_without_settle_is_refused() -> None:
    budget = _budget(1.0)
    assert budget.reserve(100)
    with pytest.raises(RuntimeError):
        budget.reserve(100)


@pytest.mark.parametrize(
    "missing",
    [
        "RECALL_COMPILED_GRAPH_MAX_USD",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT",
    ],
)
def test_from_env_requires_cap_and_prices(missing: str) -> None:
    env = {
        "RECALL_COMPILED_GRAPH_MAX_USD": "2",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "0.002",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0.008",
    }
    budget = Budget.from_env(env)
    assert budget.max_usd == 2.0
    del env[missing]
    with pytest.raises(CompiledGraphConfigError, match=missing):
        Budget.from_env(env)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "two"])
def test_from_env_rejects_unusable_cap(value: str) -> None:
    env = {
        "RECALL_COMPILED_GRAPH_MAX_USD": value,
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "0.002",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0.008",
    }
    with pytest.raises(CompiledGraphConfigError):
        Budget.from_env(env)


def test_output_cap_is_not_a_budget_setting() -> None:
    """The output reserved is the cap the request sends; no caller can set a smaller one.

    Red proof, 2026-11-26: before the fix ``Budget`` took ``max_tokens`` as a field that
    ``compile_source`` never sent, so a ``Budget(max_tokens=100)`` under-reserved every call; this
    fails there on ``pytest.raises`` (DID NOT RAISE).
    """
    with pytest.raises(TypeError):
        Budget(max_usd=1.0, cost_in_per_1k=0.002, cost_out_per_1k=0.008, max_tokens=100)  # type: ignore[call-arg]  # the field is gone; the call is the one a caller could make before
