"""Run `benchmarks.run` under a hard USD spend ceiling, for an unattended job.

::

    RECALL_BENCH_MAX_USD=2.5 RECALL_BENCH_USD_PER_M_IN=0.15 RECALL_BENCH_USD_PER_M_OUT=0.60 \\
        python -m benchmarks.capped_run --arm recall --conversations 10 --model <id>

`benchmarks.run` meters tokens but has no ceiling, and a CI job has nobody to press Ctrl-C. This
wraps the OpenAI SDK's chat completion so that, once the tokens already returned price at or above
the ceiling, the next call raises before it is sent. The error carries HTTP status 402 and the
phrase "requires more credits", which `benchmarks.llm.is_terminal` reads as an exhausted account,
so the harness stops the run and publishes what it already paid for instead of retrying or
dropping questions one at a time.

The price is the caller's statement of the model's per-million-token rates, not something read
from the provider, so the ceiling is only as good as those two numbers. Overshoot is bounded by one
call, because the check runs before each request and the cost is added after it returns.
"""
from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from typing import Any


class SpendCapReached(Exception):
    """The configured ceiling is spent; shaped like an empty OpenRouter balance on purpose."""

    status_code = 402


class SpendCap:
    def __init__(self, max_usd: float, usd_per_m_in: float, usd_per_m_out: float) -> None:
        if max_usd <= 0 or usd_per_m_in < 0 or usd_per_m_out < 0:
            raise ValueError("spend cap and prices must be positive")
        self.max_usd = max_usd
        self._in = usd_per_m_in / 1_000_000
        self._out = usd_per_m_out / 1_000_000
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def wrap(self, create: Callable[..., Any]) -> Callable[..., Any]:
        def capped(this: Any, *args: Any, **kwargs: Any) -> Any:
            with self._lock:
                if self.spent_usd >= self.max_usd:
                    raise SpendCapReached(
                        f"spend cap of USD {self.max_usd:.2f} reached "
                        f"(USD {self.spent_usd:.4f} spent): requires more credits"
                    )
            resp = create(this, *args, **kwargs)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                cost = (int(getattr(usage, "prompt_tokens", 0) or 0) * self._in
                        + int(getattr(usage, "completion_tokens", 0) or 0) * self._out)
                with self._lock:
                    self.spent_usd += cost
            return resp

        return capped


def _env_float(name: str) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raise SystemExit(f"{name} is not set; a capped run needs the ceiling and both prices")
    return float(raw)


def main(argv: list[str] | None = None) -> int:
    cap = SpendCap(
        _env_float("RECALL_BENCH_MAX_USD"),
        _env_float("RECALL_BENCH_USD_PER_M_IN"),
        _env_float("RECALL_BENCH_USD_PER_M_OUT"),
    )
    import openai.resources.chat.completions as completions

    completions.Completions.create = cap.wrap(completions.Completions.create)
    from benchmarks.run import main as run_main

    try:
        return run_main(argv)
    finally:
        print(f"capped_run: USD {cap.spent_usd:.4f} priced against a USD {cap.max_usd:.2f} cap",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
