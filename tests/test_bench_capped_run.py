"""`benchmarks.capped_run`: the spend ceiling stops a paid run, and the harness reads it as terminal.

Red proof (2026-11-06), each against a mutation of `benchmarks/capped_run.py`:

- `test_the_call_after_the_ceiling_is_refused`: the guard in `SpendCap.wrap` changed from `>=` to
  `>`, with the ceiling exactly two calls' priced cost. Failed with `DID NOT RAISE SpendCapReached`.
- `test_the_refusal_is_terminal_to_the_harness`: `SpendCapReached.status_code` changed to 429 AND
  the "requires more credits" phrase removed from its message. Failed on `assert is_terminal`.
  Either signal alone keeps it terminal (changing only the status stayed green), which is why the
  error carries both.

Both pass on the restored module.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.capped_run import SpendCap, SpendCapReached
from benchmarks.llm import is_terminal


def _fake_create(calls: list[int]):
    def create(_this, *args, **kwargs):
        calls.append(1)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=500_000, completion_tokens=0))

    return create


def test_the_call_after_the_ceiling_is_refused() -> None:
    # 500,000 in at USD 1/M is exactly USD 0.5 a call; the ceiling is exactly two calls.
    cap = SpendCap(1.0, usd_per_m_in=1.0, usd_per_m_out=2.0)
    calls: list[int] = []
    capped = cap.wrap(_fake_create(calls))
    capped(None)
    capped(None)
    with pytest.raises(SpendCapReached):
        capped(None)
    assert len(calls) == 2
    assert cap.spent_usd == 1.0


def test_the_refusal_is_terminal_to_the_harness() -> None:
    cap = SpendCap(0.1, usd_per_m_in=1.0, usd_per_m_out=1.0)
    capped = cap.wrap(_fake_create([]))
    capped(None)
    with pytest.raises(SpendCapReached) as caught:
        capped(None)
    assert is_terminal(caught.value)
