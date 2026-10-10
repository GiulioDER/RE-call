"""ST-1a: answer feedback written into memory, failed records dropped at Add (default off).

AgentMemoryBench (``solomoon313/AgentMemoryBench`` at ``9dfda8de``), the closest public protocol to
AML's Streaming branch, writes every answered question back into memory with its outcome: the
question, the model's answer and a 0/1 judge reward. Published streaming results (StreamBench,
Evo-Memory) find that showing a model its failed attempts lowers accuracy while its correct ones
help. Under ``RECALL_AML_FEEDBACK_GUARD=1`` an Add message that is such a record and reports a
failure is not stored; a record that reports a success, and every other message, is stored exactly
as before.

Two shapes are recognised, and only these:

1. One message whose whole text is a JSON object, optionally after one short label line ending in a
   colon (``Streaming online QA feedback:``), carrying a question, the model's answer and an outcome.
   Flat (``{"question", "predicted_answer", "score"}``) or AgentMemoryBench's result shape
   (``{"reward", "result": {"question", "predicted_answer", ...}}``).
2. AgentMemoryBench's judge message, text that starts ``response: {"label": "CORRECT"|"WRONG"}``. An
   Add that holds one beside at least one other message is a feedback unit as a whole, and a WRONG
   verdict drops every message of it.

A failed record that also carries a reference answer (``gold_answer`` and the like) is kept: it holds
the correct answer, which is evidence. Anything else, including a JSON object that lacks one of the
three parts, or whose outcome cannot be read or contradicts itself, is not a feedback record and is
stored unchanged. Which format AML uses, if any, is not public and its organiser does not answer
questions about it, so the guard is built so that a wrong guess costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Literal

FEEDBACK_GUARD_ENV = "RECALL_AML_FEEDBACK_GUARD"

Verdict = Literal["passed", "failed"]

_QUESTION_KEYS = frozenset({"question", "query"})
_ANSWER_KEYS = frozenset(
    {"predicted_answer", "prediction", "predicted", "answer", "model_answer", "response", "output"}
)
_OUTCOME_KEYS = (
    "score", "reward", "llm_score", "correct", "is_correct", "label", "verdict", "judgement", "judgment",
)
_REFERENCE_KEYS = frozenset(
    {"gold_answer", "gold", "gold_answers", "reference", "reference_answer", "ground_truth",
     "expected_answer", "correct_answer", "matched_answer"}
)
_PASSED_WORDS = frozenset({"correct", "right", "true", "yes", "pass", "passed", "success", "successful"})
_FAILED_WORDS = frozenset({"wrong", "incorrect", "false", "no", "fail", "failed", "failure"})
#: One short label line before the JSON object, such as ``Streaming online QA feedback:``.
_LABEL_LINE = re.compile(r"[^\n{}]{1,80}:[ \t]*\n")
#: AgentMemoryBench's ``llm_judge`` tool message: ``response: {"label": "WRONG"}\nreasoning: ...``.
_JUDGE = re.compile(r"\s*response:\s*(\{[^{}\n]*\})")


@dataclass(frozen=True)
class FeedbackFilter:
    kept: list[Any]
    dropped: int
    passed: int


def _key(name: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _text(content: object) -> str | None:
    """The message text, or None when it is not plain text (an image part makes it not a record)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts = []
    for part in content:
        if hasattr(part, "model_dump"):
            part = part.model_dump(mode="python")
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None
        texts.append(part["text"])
    return "\n".join(texts)


def _json_object(text: str) -> dict[str, Any] | None:
    body = text.strip()
    if not body.startswith("{"):
        label = _LABEL_LINE.match(body)
        if label is None:
            return None
        body = body[label.end():].strip()
    if not (body.startswith("{") and body.endswith("}")):
        return None
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _outcome(value: object) -> Verdict | None:
    if isinstance(value, bool):
        return "passed" if value else "failed"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        return "passed" if value > 0 else "failed"
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _PASSED_WORDS:
            return "passed"
        if word in _FAILED_WORDS:
            return "failed"
        try:
            return _outcome(float(word))
        except ValueError:
            return None
    return None


def _filled(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def record_verdict(record: dict[str, Any]) -> tuple[Verdict, bool] | None:
    """(verdict, carries a reference answer) for a feedback record, None for anything else."""
    fields = {_key(k): v for k, v in record.items()}
    inner = fields.get("result")
    if isinstance(inner, dict):
        # AgentMemoryBench's shape: the question and answer inside ``result``, the reward outside.
        nested = {_key(k): v for k, v in inner.items()}
        metrics = nested.get("metrics")
        merged = {**({_key(k): v for k, v in metrics.items()} if isinstance(metrics, dict) else {}), **nested}
        merged.update({k: fields[k] for k in _OUTCOME_KEYS if k in fields})
        merged.update({k: fields[k] for k in _REFERENCE_KEYS if k in fields and k not in merged})
        fields = merged
    if not any(_filled(fields.get(k)) for k in _QUESTION_KEYS):
        return None
    if not any(k in fields and fields[k] is not None for k in _ANSWER_KEYS):
        return None
    present = [k for k in _OUTCOME_KEYS if k in fields]
    if not present:
        return None
    verdicts = {_outcome(fields[k]) for k in present}
    if None in verdicts or len(verdicts) != 1:
        return None
    verdict = verdicts.pop()
    assert verdict is not None
    return verdict, any(_filled(fields.get(k)) for k in _REFERENCE_KEYS)


def message_verdict(message: Any) -> tuple[Verdict, bool] | None:
    """The verdict of one message that is a whole JSON feedback record, else None."""
    text = _text(getattr(message, "content", None))
    if text is None:
        return None
    record = _json_object(text)
    return None if record is None else record_verdict(record)


def judge_verdict(message: Any) -> Verdict | None:
    """The verdict of an AgentMemoryBench ``llm_judge`` message, else None."""
    text = _text(getattr(message, "content", None))
    if text is None:
        return None
    match = _JUDGE.match(text)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(value, dict) or set(map(_key, value)) != {"label"}:
        return None
    label = str(next(iter(value.values()))).strip().upper()
    return {"CORRECT": "passed", "WRONG": "failed"}.get(label)  # type: ignore[return-value]


def filter_feedback(messages: list[Any]) -> FeedbackFilter:
    """Drop the failed feedback records of one Add; keep every other message, in order."""
    judged = [v for v in (judge_verdict(m) for m in messages) if v is not None]
    if len(messages) > 1 and judged and len(set(judged)) == 1:
        if judged[0] == "failed":
            return FeedbackFilter(kept=[], dropped=len(messages), passed=0)
        return FeedbackFilter(kept=list(messages), dropped=0, passed=1)
    kept: list[Any] = []
    dropped = passed = 0
    for message in messages:
        verdict = message_verdict(message)
        if verdict is not None and verdict == ("failed", False):
            dropped += 1
            continue
        if verdict is not None and verdict[0] == "passed":
            passed += 1
        kept.append(message)
    return FeedbackFilter(kept=kept, dropped=dropped, passed=passed)
