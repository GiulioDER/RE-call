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
    required_env,
)


def _positive_float(env: Mapping[str, str], name: str, *, allow_zero: bool) -> float:
    variable = f"{ENV_PREFIX}_{name}"
    text = required_env(env, name)
    try:
        value = float(text)
    except ValueError as exc:
        raise CompiledGraphConfigError(f"{variable} must be a number") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "zero or above" if allow_zero else "above zero"
        raise CompiledGraphConfigError(f"{variable} must be a finite number {bound}")
    return value


@dataclass
class Budget:
    """One build's spend: ``reserve`` before each attempt, then ``settle`` it exactly once.

    A second ``reserve`` before the ``settle`` raises. ``exhausted`` latches for the rest of the
    build once the spend reaches ``max_usd`` or a reply's reported usage exceeds its reservation,
    and every later ``reserve`` then returns False.
    """

    max_usd: float
    cost_in_per_1k: float
    cost_out_per_1k: float
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

    def reserve(self, input_token_bound: int) -> bool:
        """Hold the worst case of one call; False when it does not fit what is left."""
        if self._reserved_usd is not None:
            raise RuntimeError("settle the previous call before reserving another")
        worst = self._cost(input_token_bound, MAX_OUTPUT_TOKENS)
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
