"""The per-build USD cap, priced from figures the operator states.

A call is sent only when its worst case fits what is left. Input is reserved at one token per UTF-8
byte of the request, which a byte-level tokenizer cannot exceed; the AML service measured pasted
data at about one character per token, so a characters-per-token guess is not a bound. Output is
reserved at the request's ``max_tokens``. After the call the reservation is replaced by the
reported usage, or kept when the provider reports none.

Some models bill tokens the request does not carry, so the operator states them per model, as with
the prices: extra input tokens per call, and output tokens reserved at or above ``max_tokens``.
Measured 2026-11-27 for ``openai/gpt-6-luna-pro`` through OpenRouter, over three sources: up to
2,344 prompt tokens beyond the request's bytes, and up to 3,076 completion tokens against
``max_tokens`` 2,400 (its reasoning tokens are billed past the cap). A reply over its reservation
still stops the build, since then even the stated figures were not a bound.
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


def _tokens(env: Mapping[str, str], name: str, default: int) -> int:
    variable = f"{ENV_PREFIX}_{name}"
    text = env.get(variable, "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise CompiledGraphConfigError(f"{variable} must be a whole number of tokens") from exc


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
    extra_input_tokens: int = 0
    output_tokens: int = MAX_OUTPUT_TOKENS
    spent_usd: float = 0.0
    exhausted: bool = False
    _reserved_usd: float | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Budget:
        return cls(
            max_usd=_positive_float(env, "MAX_USD", allow_zero=False),
            cost_in_per_1k=_positive_float(env, "COST_PER_1K_INPUT", allow_zero=True),
            cost_out_per_1k=_positive_float(env, "COST_PER_1K_OUTPUT", allow_zero=True),
            extra_input_tokens=_tokens(env, "RESERVE_EXTRA_INPUT_TOKENS", 0),
            output_tokens=_tokens(env, "RESERVE_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS),
        )

    def __post_init__(self) -> None:
        if self.extra_input_tokens < 0:
            raise CompiledGraphConfigError(f"{ENV_PREFIX}_RESERVE_EXTRA_INPUT_TOKENS must be zero or above")
        # Below the request's max_tokens the reservation would not cover a reply the cap allows.
        if self.output_tokens < MAX_OUTPUT_TOKENS:
            raise CompiledGraphConfigError(
                f"{ENV_PREFIX}_RESERVE_OUTPUT_TOKENS must be at least {MAX_OUTPUT_TOKENS}, the request's max_tokens"
            )

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1000 * self.cost_in_per_1k + tokens_out / 1000 * self.cost_out_per_1k

    def worst_case(self, input_token_bound: int) -> float:
        """The USD one call is reserved at, given a bound on its request's input tokens."""
        return self._cost(input_token_bound + self.extra_input_tokens, self.output_tokens)

    def reserve(self, input_token_bound: int) -> bool:
        """Hold the worst case of one call; False when it does not fit what is left."""
        if self._reserved_usd is not None:
            raise RuntimeError("settle the previous call before reserving another")
        worst = self.worst_case(input_token_bound)
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
