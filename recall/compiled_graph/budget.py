"""The per-build USD cap, priced from figures the operator states.

A call is sent only when its worst case fits what is left. Input is reserved at one token per UTF-8
byte of the request, which a byte-level tokenizer cannot exceed; the AML service measured pasted
data at about one character per token, so a characters-per-token guess is not a bound. Output is
reserved at the request's ``max_tokens``. After the call the reservation is replaced by the
reported usage, or kept when the provider reports none.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from recall.compiled_graph.provider import (
    ENV_PREFIX,
    MAX_OUTPUT_TOKENS,
    CompiledGraphConfigError,
    CompileReply,
)


def _positive_float(env: Mapping[str, str], name: str, *, allow_zero: bool) -> float:
    variable = f"{ENV_PREFIX}_{name}"
    text = env.get(variable, "").strip()
    if not text:
        raise CompiledGraphConfigError(f"{variable} is required for a compile")
    try:
        value = float(text)
    except ValueError as exc:
        raise CompiledGraphConfigError(f"{variable} must be a number") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise CompiledGraphConfigError(f"{variable} must be a finite number above zero")
    return value


@dataclass
class Budget:
    max_usd: float
    cost_in_per_1k: float
    cost_out_per_1k: float
    max_tokens: int = MAX_OUTPUT_TOKENS
    spent_usd: float = 0.0
    exhausted: bool = False
    _reserved_usd: float | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Budget:
        return cls(
            max_usd=_positive_float(env, "MAX_USD", allow_zero=False),
            cost_in_per_1k=_positive_float(env, "COST_PER_1K_INPUT", allow_zero=True),
            cost_out_per_1k=_positive_float(env, "COST_PER_1K_OUTPUT", allow_zero=True),
        )

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1000 * self.cost_in_per_1k + tokens_out / 1000 * self.cost_out_per_1k

    def worst_case(self, request_bytes: int) -> float:
        """The USD one call of ``request_bytes`` is reserved at."""
        return self._cost(request_bytes, self.max_tokens)

    def reserve(self, request_bytes: int) -> bool:
        """Hold the worst case of one call; False when it does not fit what is left."""
        if self._reserved_usd is not None:
            raise RuntimeError("settle the previous call before reserving another")
        worst = self.worst_case(request_bytes)
        if self.exhausted or self.spent_usd + worst > self.max_usd:
            return False
        self._reserved_usd = worst
        return True

    def settle(self, reply: CompileReply | None) -> None:
        """Charge the reported usage, or the whole reservation when usage is unknown."""
        reserved = self._reserved_usd if self._reserved_usd is not None else 0.0
        self._reserved_usd = None
        if reply is None or reply.input_tokens is None or reply.output_tokens is None:
            charged = reserved
        else:
            charged = self._cost(reply.input_tokens, reply.output_tokens)
        self.spent_usd += charged
        if charged > reserved or self.spent_usd >= self.max_usd:
            self.exhausted = True
