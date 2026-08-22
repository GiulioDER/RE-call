"""Canonical JSON compatible records for agent comparison runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

RECALL_ON = "recall_on"
RECALL_OFF = "recall_off"
VARIANTS = (RECALL_ON, RECALL_OFF)


def _check_optional_nonnegative(name: str, value: float | int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative number or None")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite or None")


def _tuple_of_strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    return result


def _tuple_of_mappings(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of mappings")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} items must be mappings")
        result.append(dict(item))
    return tuple(result)


@dataclass(frozen=True)
class SessionRecord:
    """One completed or failed task execution in one benchmark arm.

    ``None`` means that a value was not observed. It must not be replaced with zero,
    because an unmeasured value must remain distinguishable from a measured zero.
    """

    task_id: str
    variant: str
    success: bool
    user_input: str = ""
    response: str = ""
    reference: str | None = None
    retrieved_contexts: tuple[str, ...] = ()
    reference_contexts: tuple[str, ...] = ()
    conversation: tuple[dict[str, Any], ...] = ()
    reference_tool_calls: tuple[dict[str, Any], ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    recall_call_count: int = 0
    recall_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_turns: int | None = None
    wall_time_ms: float | None = None
    system_cost_usd: float | None = None
    evaluator_cost_usd: float | None = None
    abstained: bool = False
    trust_verdicts: tuple[str, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if self.recall_call_count < 0:
            raise ValueError("recall_call_count must be nonnegative")
        for name in (
            "recall_latency_ms",
            "input_tokens",
            "output_tokens",
            "model_turns",
            "wall_time_ms",
            "system_cost_usd",
            "evaluator_cost_usd",
        ):
            _check_optional_nonnegative(name, getattr(self, name))
        if self.variant == RECALL_OFF and self.recall_call_count:
            raise ValueError("recall_off records cannot contain RE-call calls")

    @property
    def total_tokens(self) -> int | None:
        """Return measured model tokens, or None when either side is unavailable."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def is_complete(self) -> bool:
        """Whether the runner completed without an execution error."""

        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serializable representation."""

        return {
            "record_version": 1,
            "task_id": self.task_id,
            "variant": self.variant,
            "success": self.success,
            "user_input": self.user_input,
            "response": self.response,
            "reference": self.reference,
            "retrieved_contexts": list(self.retrieved_contexts),
            "reference_contexts": list(self.reference_contexts),
            "conversation": [dict(item) for item in self.conversation],
            "reference_tool_calls": [dict(item) for item in self.reference_tool_calls],
            "tool_calls": [dict(item) for item in self.tool_calls],
            "recall_call_count": self.recall_call_count,
            "recall_latency_ms": self.recall_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_turns": self.model_turns,
            "wall_time_ms": self.wall_time_ms,
            "system_cost_usd": self.system_cost_usd,
            "evaluator_cost_usd": self.evaluator_cost_usd,
            "abstained": self.abstained,
            "trust_verdicts": list(self.trust_verdicts),
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SessionRecord":
        """Construct a record from a runner result or a JSON object."""

        return cls(
            task_id=str(value["task_id"]),
            variant=str(value["variant"]),
            success=bool(value["success"]),
            user_input=str(value.get("user_input", "")),
            response=str(value.get("response", "")),
            reference=value.get("reference"),
            retrieved_contexts=_tuple_of_strings(
                value.get("retrieved_contexts"), "retrieved_contexts"
            ),
            reference_contexts=_tuple_of_strings(
                value.get("reference_contexts"), "reference_contexts"
            ),
            conversation=_tuple_of_mappings(value.get("conversation"), "conversation"),
            reference_tool_calls=_tuple_of_mappings(
                value.get("reference_tool_calls"), "reference_tool_calls"
            ),
            tool_calls=_tuple_of_mappings(value.get("tool_calls"), "tool_calls"),
            recall_call_count=int(value.get("recall_call_count", 0)),
            recall_latency_ms=value.get("recall_latency_ms"),
            input_tokens=value.get("input_tokens"),
            output_tokens=value.get("output_tokens"),
            model_turns=value.get("model_turns"),
            wall_time_ms=value.get("wall_time_ms"),
            system_cost_usd=value.get("system_cost_usd"),
            evaluator_cost_usd=value.get("evaluator_cost_usd"),
            abstained=bool(value.get("abstained", False)),
            trust_verdicts=_tuple_of_strings(value.get("trust_verdicts"), "trust_verdicts"),
            error=value.get("error"),
            metadata=value.get("metadata", {}),
        )
