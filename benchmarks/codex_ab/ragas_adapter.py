"""Optional conversion of session records into Ragas evaluation samples."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .schema import SessionRecord


class RagasUnavailable(RuntimeError):
    """Raised when the optional Ragas dependency is not installed."""


def _imports() -> tuple[Any, ...]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
        from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
    except ImportError as error:
        raise RagasUnavailable(
            "Install it directly with `pip install 'ragas>=0.4,<1'`. It is deliberately "
            "not a recall-rag extra: see benchmarks/codex_ab/README.md."
        ) from error
    return (
        EvaluationDataset,
        evaluate,
        MultiTurnSample,
        SingleTurnSample,
        AIMessage,
        HumanMessage,
        ToolCall,
        ToolMessage,
    )


def to_single_turn_sample(record: SessionRecord) -> Any:
    """Convert one record to a Ragas ``SingleTurnSample``."""

    _, _, _, single_turn, _, _, _, _ = _imports()
    return single_turn(
        user_input=record.user_input or record.task_id,
        response=record.response,
        reference=record.reference,
        retrieved_contexts=list(record.retrieved_contexts),
        reference_contexts=list(record.reference_contexts),
    )


def _message_from_mapping(item: dict[str, Any], classes: tuple[Any, Any, Any, Any]) -> Any:
    ai_message, human_message, tool_call, tool_message = classes
    role = item.get("role") or item.get("type")
    content = str(item.get("content", ""))
    if role in {"user", "human"}:
        return human_message(content=content)
    if role in {"assistant", "ai"}:
        calls = [
            tool_call(name=str(call["name"]), args=dict(call.get("args", {})))
            for call in item.get("tool_calls", [])
        ]
        return ai_message(content=content, tool_calls=calls or None)
    if role == "tool":
        return tool_message(content=content)
    raise ValueError(f"unsupported conversation role: {role!r}")


def to_multi_turn_sample(record: SessionRecord) -> Any:
    """Convert one record to a Ragas ``MultiTurnSample``."""

    _, _, multi_turn, _, ai_message, human_message, tool_call, tool_message = _imports()
    messages = [
        _message_from_mapping(item, (ai_message, human_message, tool_call, tool_message))
        for item in record.conversation
    ]
    reference_calls = [
        tool_call(name=str(call["name"]), args=dict(call.get("args", {})))
        for call in record.reference_tool_calls
    ]
    return multi_turn(
        user_input=messages,
        reference_tool_calls=reference_calls or None,
    )


def build_dataset(records: Iterable[SessionRecord], *, multi_turn: bool = False) -> Any:
    """Build a homogeneous Ragas evaluation dataset."""

    evaluation_dataset, _, _, _, _, _, _, _ = _imports()
    materialized = list(records)
    if not materialized:
        raise ValueError("at least one session record is required")
    samples = [
        to_multi_turn_sample(record) if multi_turn else to_single_turn_sample(record)
        for record in materialized
    ]
    return evaluation_dataset(samples=samples)


def evaluate_records(
    records: Iterable[SessionRecord],
    metrics: Sequence[Any],
    *,
    multi_turn: bool = False,
    **kwargs: Any,
) -> Any:
    """Run Ragas metrics, keeping the dependency and evaluator cost explicit."""

    _, evaluate, _, _, _, _, _, _ = _imports()
    dataset = build_dataset(records, multi_turn=multi_turn)
    return evaluate(dataset=dataset, metrics=list(metrics), **kwargs)
