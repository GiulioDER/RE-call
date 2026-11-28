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

Red proof for the stated reserve, recorded 2026-11-28. Baseline: ``budget.py`` at ``8fe4d168``,
before ``extra_input_tokens`` and ``output_tokens`` existed, so ``from_env`` ignored the two
``RESERVE`` variables.

- ``test_stated_reserve_lets_a_reasoning_model_build``: fails at the baseline on ``assert 1 == 3``
  (the first reply overruns and the second attempt raises ``BudgetExhausted``), and with
  ``worst_case`` not adding ``extra_input_tokens`` likewise.
- ``test_stated_reserve_is_the_worst_case``: fails at the baseline and with that mutation on
  ``math.isclose``.
- ``test_unusable_stated_reserve_is_refused``: every case fails at the baseline with DID NOT
  RAISE; the ``2399`` case fails so with the ``output_tokens < MAX_OUTPUT_TOKENS`` check removed.
- ``test_without_stated_reserve_a_reasoning_model_stops_the_build`` is the control that the stub
  overruns the default reservation; it passes at the baseline by design.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pytest

from recall.compiled_graph.budget import Budget
from recall.compiled_graph.provider import (
    BudgetExhausted,
    CompiledGraphConfigError,
    CompileReply,
    ProviderConfig,
    compile_source,
)

PRICES = {
    "RECALL_COMPILED_GRAPH_MAX_USD": "1",
    "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "0.002",
    "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0.008",
}
#: Figures that cover the reasoning model below: 2,344 extra prompt tokens, 3,076 output tokens.
STATED_RESERVE = {
    "RECALL_COMPILED_GRAPH_RESERVE_EXTRA_INPUT_TOKENS": "2500",
    "RECALL_COMPILED_GRAPH_RESERVE_OUTPUT_TOKENS": "3200",
}

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
    """The output reserved is at least the cap the request sends; no caller can set a smaller one.

    Red proof, 2026-11-26: before the fix ``Budget`` took ``max_tokens`` as a field that
    ``compile_source`` never sent, so a ``Budget(max_tokens=100)`` under-reserved every call; this
    fails there on ``pytest.raises`` (DID NOT RAISE).
    """
    with pytest.raises(TypeError):
        Budget(max_usd=1.0, cost_in_per_1k=0.002, cost_out_per_1k=0.008, max_tokens=100)  # type: ignore[call-arg]  # the field is gone; the call is the one a caller could make before


def test_negative_price_message_allows_zero() -> None:
    """Red proof, 2026-11-26: the one message before the fix ("above zero") fails on ``match``."""
    env = {
        "RECALL_COMPILED_GRAPH_MAX_USD": "1",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "-1",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0",
    }
    with pytest.raises(CompiledGraphConfigError, match="zero or above"):
        Budget.from_env(env)


def _reasoning_model(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    """A stub of ``openai/gpt-6-luna-pro`` through OpenRouter, from the largest figures measured
    on 2026-11-27: 2,344 prompt tokens beyond the request's bytes, 3,076 completion tokens."""
    reply = {
        "choices": [{"message": {"content": json.dumps({"records": []})}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(body) + 2_344, "completion_tokens": 3_076},
    }
    return 200, {}, json.dumps(reply).encode()


def _replies_before_stop(budget: Budget, calls: int) -> int:
    config = ProviderConfig(api_key="sk-test-not-a-real-key", model="openai/gpt-6-luna-pro")
    payload = {"source": "notes/a.md", "anchors": [{"id": "a000_x", "excerpt": "text"}]}
    for done in range(calls):
        try:
            compile_source(config, _reasoning_model, payload, budget, sleep=lambda _: None)
        except BudgetExhausted:
            return done
    return calls


def test_without_stated_reserve_a_reasoning_model_stops_the_build() -> None:
    """Control: the stub overruns the default reservation, as the real model did."""
    budget = Budget.from_env(PRICES)
    assert _replies_before_stop(budget, 3) == 1
    assert budget.exhausted


def test_stated_reserve_lets_a_reasoning_model_build() -> None:
    budget = Budget.from_env({**PRICES, **STATED_RESERVE})
    assert _replies_before_stop(budget, 3) == 3
    assert not budget.exhausted


def test_stated_reserve_is_the_worst_case() -> None:
    budget = Budget.from_env({**PRICES, **STATED_RESERVE})
    assert math.isclose(budget.worst_case(1_000), 3.5 * 0.002 + 3.2 * 0.008)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RECALL_COMPILED_GRAPH_RESERVE_OUTPUT_TOKENS", "2399"),
        ("RECALL_COMPILED_GRAPH_RESERVE_OUTPUT_TOKENS", "3k"),
        ("RECALL_COMPILED_GRAPH_RESERVE_EXTRA_INPUT_TOKENS", "-1"),
        ("RECALL_COMPILED_GRAPH_RESERVE_EXTRA_INPUT_TOKENS", "1.5"),
    ],
)
def test_unusable_stated_reserve_is_refused(name: str, value: str) -> None:
    with pytest.raises(CompiledGraphConfigError, match=name):
        Budget.from_env({**PRICES, name: value})
