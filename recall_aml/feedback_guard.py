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
the correct answer, which is evidence.

**A record is dropped only if its question is one this tenant was searched with earlier** (``asked``,
fed by ``AskedQuestions``). Feedback can only be about a question the platform already asked, and
outside Streaming a tenant is written before it is searched, so on ordinary data the guard cannot act
at all. The memory of asked questions lives in the process (C9 serves from one worker): a restart
empties it, which only switches the guard off for what follows, never on. Anything else, including a JSON object that lacks one of the
three parts, or whose outcome cannot be read or contradicts itself, is not a feedback record and is
stored unchanged. Which format AML uses, if any, is not public and its organiser does not answer
questions about it, so the guard is built so that a wrong guess costs nothing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
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
#: The question line of an AgentMemoryBench answer turn: ``Question: ...`` then ``Answer:``.
_QUESTION_LINE = re.compile(r"Question:\s*(.+?)\s*(?:\nAnswer:|$)", re.DOTALL)
#: Below this many characters a question must match a searched query exactly; above it, one may
#: contain the other (a query can carry the options or an instruction around the question).
MIN_CONTAINED_CHARS = 20


@dataclass(frozen=True)
class FeedbackFilter:
    kept: list[Any]
    dropped: int
    passed: int


@dataclass(frozen=True)
class FeedbackRecord:
    verdict: Verdict
    has_reference: bool
    question: str


def normalize_question(text: str) -> str:
    return " ".join(text.casefold().split())


class AskedQuestions:
    """The questions each tenant was searched with in this process, newest kept, bounded."""

    def __init__(self, *, max_tenants: int = 20_000, per_tenant: int = 2_048) -> None:
        self._max_tenants, self._per_tenant = max_tenants, per_tenant
        self._by_tenant: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()

    def remember(self, tenant: str, query: str) -> None:
        question = normalize_question(query)
        if not question:
            return
        seen = self._by_tenant.pop(tenant, None) or OrderedDict()
        seen.pop(question, None)
        seen[question] = None
        while len(seen) > self._per_tenant:
            seen.popitem(last=False)
        self._by_tenant[tenant] = seen
        while len(self._by_tenant) > self._max_tenants:
            self._by_tenant.popitem(last=False)

    def asked(self, tenant: str, question: str) -> bool:
        wanted = normalize_question(question)
        seen = self._by_tenant.get(tenant)
        if not wanted or not seen:
            return False
        if wanted in seen:
            return True
        if len(wanted) < MIN_CONTAINED_CHARS:
            return False
        return any(wanted in query or (len(query) >= MIN_CONTAINED_CHARS and query in wanted) for query in seen)


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


def record_verdict(record: dict[str, Any]) -> FeedbackRecord | None:
    """The verdict, reference flag and question of a feedback record, None for anything else."""
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
    questions = [fields[k] for k in ("question", "query") if _filled(fields.get(k))]
    if not questions:
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
    return FeedbackRecord(verdict, any(_filled(fields.get(k)) for k in _REFERENCE_KEYS), str(questions[0]))


def message_verdict(message: Any) -> FeedbackRecord | None:
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


def unit_question(messages: list[Any]) -> str | None:
    """The question of an AgentMemoryBench judge unit, from its ``Question: ...`` line."""
    for message in messages:
        match = _QUESTION_LINE.search(_text(getattr(message, "content", None)) or "")
        if match:
            return match.group(1)
    return None


def filter_feedback(messages: list[Any], asked: Callable[[str], bool] | None = None) -> FeedbackFilter:
    """Drop the failed feedback records of one Add; keep every other message, in order.

    ``asked`` says whether a question was searched earlier in this tenant, and a failed record about
    any other question is kept. None is the detector alone, for tests and scans.
    """

    def was_asked(question: str | None) -> bool:
        return asked is None or (question is not None and asked(question))

    judged = [v for v in (judge_verdict(m) for m in messages) if v is not None]
    if len(messages) > 1 and judged and len(set(judged)) == 1:
        if judged[0] == "failed" and was_asked(unit_question(messages)):
            return FeedbackFilter(kept=[], dropped=len(messages), passed=0)
        return FeedbackFilter(kept=list(messages), dropped=0, passed=int(judged[0] == "passed"))
    kept: list[Any] = []
    dropped = passed = 0
    for message in messages:
        record = message_verdict(message)
        if (
            record is not None
            and record.verdict == "failed"
            and not record.has_reference
            and was_asked(record.question)
        ):
            dropped += 1
            continue
        if record is not None and record.verdict == "passed":
            passed += 1
        kept.append(message)
    return FeedbackFilter(kept=kept, dropped=dropped, passed=passed)
