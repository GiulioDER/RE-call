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
2. AgentMemoryBench's judge unit (``src/server/tasks/locomo/task.py``): an optional system prompt,
   the user message that ends with ``Question: ...`` and ``Answer:`` (memories may come before it),
   the model's answer, then the judge message, text that starts
   ``response: {"label": "CORRECT"|"WRONG"}``. The unit is the span from the nearest question turn
   before the judge to the judge, plus a system message directly before that turn; a WRONG verdict
   drops that span and nothing else, so dialogue sent in the same Add is stored. When the judge
   messages of one Add disagree, none of them acts.

A failed record that also carries a reference answer (``gold_answer`` and the like) is kept: it holds
the correct answer, which is evidence, and so is a judge unit that holds such a record. ``answer``
names the model's answer when it is the only answer field; beside a key that can only be the model's
(``response``, ``predicted_answer``, ...), as in mem0's evaluation records, it is the reference.

**A record is dropped only if its question is one this tenant was searched with earlier** (``asked``,
fed by ``AskedQuestions``). Feedback can only be about a question the platform already asked, and
outside Streaming a tenant is written before it is searched, so on ordinary data the guard cannot act
at all. The memory of asked questions lives in the process (C9 serves from one worker): a restart
empties it, and so does deleting the user, which only switches the guard off for what follows, never
on. Anything else, including a JSON object that lacks one of the
three parts, or whose outcome cannot be read or contradicts itself, is not a feedback record and is
stored unchanged. Which format AML uses, if any, is not public and its organiser does not answer
questions about it, so the guard is built so that a wrong guess costs nothing.

References:

- AgentMemoryBench, "Benchmarking Continual Agent Memory for Online Learning, Transfer, and
  Forgetting", https://github.com/solomoon313/AgentMemoryBench, MIT licence (Copyright (c) 2026
  Zihang Ma). Read at commit ``9dfda8dede15bf16518524e4614ed89344795413``; only its record shapes
  are recognised here, none of its code is copied.
- C.-K. Wu, Z. R. Tam, C.-Y. Lin, Y.-N. Chen and H.-y. Lee, "StreamBench: Towards Benchmarking
  Continuous Improvement of Language Agents", arXiv:2406.08747, 2024.
- T. Wei, N. Sachdeva, B. Coleman, Z. He, Y. Bei, X. Ning, M. Ai, Y. Li, J. He, E. H. Chi, C. Wang,
  S. Chen, F. Pereira, W.-C. Kang and D. Z. Cheng, "Evo-Memory: Benchmarking LLM Agent Test-time
  Learning with Self-Evolving Memory", arXiv:2511.20857, 2025.
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
#: Keys that can only hold the model's answer; ``answer`` is read beside them (``record_verdict``).
_MODEL_ANSWER_KEYS = frozenset(
    {"predicted_answer", "prediction", "predicted", "model_answer", "response", "output"}
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
#: The start of a question line, as AgentMemoryBench writes it after any memories.
_QUESTION_START = re.compile(r"^[ \t]*Question:", re.MULTILINE)
#: What follows the last ``Question:`` of an answer turn: the question, then ``Answer:``.
_FRAMED_QUESTION = re.compile(r"\s*(\S.*?)[ \t]*\nAnswer:", re.DOTALL)
#: Below this many characters a question must match a searched query exactly; above it, one may
#: contain the other (a query can carry the options or an instruction around the question).
MIN_CONTAINED_CHARS = 20
#: A remembered query is cut to this many characters, the bound a plain-text Search query already
#: has (``models.MAX_QUERY_CHARS``); a query sent as text parts has no bound of its own.
MAX_REMEMBERED_CHARS = 20_000
#: All remembered queries together, in characters. Past it the tenants searched longest ago are
#: forgotten first, which only switches the guard off for them.
MAX_REMEMBERED_TOTAL_CHARS = 64_000_000


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

    def __init__(
        self,
        *,
        max_tenants: int = 20_000,
        per_tenant: int = 2_048,
        max_total_chars: int = MAX_REMEMBERED_TOTAL_CHARS,
    ) -> None:
        self._max_tenants, self._per_tenant = max_tenants, per_tenant
        self._max_total_chars = max_total_chars
        self._by_tenant: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
        self._chars = 0

    def remember(self, tenant: str, query: str) -> None:
        question = normalize_question(query)[:MAX_REMEMBERED_CHARS]
        if not question:
            return
        seen = self._by_tenant.pop(tenant, None) or OrderedDict()
        if question in seen:
            seen.pop(question)
            self._chars -= len(question)
        seen[question] = None
        self._chars += len(question)
        while len(seen) > self._per_tenant:
            self._chars -= len(seen.popitem(last=False)[0])
        self._by_tenant[tenant] = seen
        while len(self._by_tenant) > 1 and (
            len(self._by_tenant) > self._max_tenants or self._chars > self._max_total_chars
        ):
            _, oldest = self._by_tenant.popitem(last=False)
            self._chars -= sum(map(len, oldest))

    def forget(self, tenant: str) -> None:
        """Drop what ``tenant`` was asked, so a user created again under its id starts empty."""
        seen = self._by_tenant.pop(tenant, None)
        if seen:
            self._chars -= sum(map(len, seen))

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
    except (ValueError, RecursionError):
        # RecursionError: nesting deeper than the interpreter's stack is not a record either.
        return None
    return value if isinstance(value, dict) else None


def _outcome(value: object) -> Verdict | None:
    if isinstance(value, bool):
        return "passed" if value else "failed"
    if isinstance(value, (int, float)):
        # An int is always finite and math.isfinite overflows on one past float range.
        if isinstance(value, float) and not math.isfinite(value):
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
    model_answered = any(fields.get(k) is not None for k in _MODEL_ANSWER_KEYS)
    if not model_answered and fields.get("answer") is None:
        return None
    present = [k for k in _OUTCOME_KEYS if k in fields]
    if not present:
        return None
    verdicts = {_outcome(fields[k]) for k in present}
    if None in verdicts or len(verdicts) != 1:
        return None
    verdict = verdicts.pop()
    assert verdict is not None
    # Beside a key that can only be the model's answer, ``answer`` is the reference (mem0's
    # evaluation records: ``{"question", "answer": gold, "response": model, "llm_score"}``).
    reference = any(_filled(fields.get(k)) for k in _REFERENCE_KEYS) or (
        model_answered and _filled(fields.get("answer"))
    )
    return FeedbackRecord(verdict, reference, str(questions[0]))


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


def turn_question(message: Any) -> str | None:
    """The question of a message that is an AgentMemoryBench answer turn, else None.

    A turn ends with its question: after its last ``Question:`` line comes the question and then
    ``Answer:`` (and the answer, when it is written in the same message), or the question alone on
    that last line. Earlier ``Question:`` lines are memories, and a question line followed by
    anything else is not a turn.
    """
    text = _text(getattr(message, "content", None)) or ""
    starts = list(_QUESTION_START.finditer(text))
    if not starts:
        return None
    tail = text[starts[-1].end():]
    framed = _FRAMED_QUESTION.match(tail)
    if framed:
        return framed.group(1)
    line = tail.strip()
    return line if line and "\n" not in line else None


def _carries_reference(message: Any) -> bool:
    record = message_verdict(message)
    return record is not None and record.has_reference


def _judge_units(messages: list[Any], judged: list[Verdict | None]) -> list[tuple[range, str]]:
    """Each judge message's unit: the span from its question turn to it, and that question.

    A unit never reaches back past an earlier judge message, and a judge with no question turn
    before it has no unit.
    """
    units: list[tuple[range, str]] = []
    floor = 0
    for end, verdict in enumerate(judged):
        if verdict is None:
            continue
        for start in range(end - 1, floor - 1, -1):
            question = turn_question(messages[start])
            if question is not None:
                if start > floor and getattr(messages[start - 1], "role", None) == "system":
                    start -= 1
                units.append((range(start, end + 1), question))
                break
        floor = end + 1
    return units


def filter_feedback(messages: list[Any], asked: Callable[[str], bool] | None = None) -> FeedbackFilter:
    """Drop the failed feedback records of one Add; keep every other message, in order.

    ``asked`` says whether a question was searched earlier in this tenant, and a failed record about
    any other question is kept. None is the detector alone, for tests and scans.
    """

    def was_asked(question: str | None) -> bool:
        return asked is None or (question is not None and asked(question))

    judged = [judge_verdict(m) for m in messages]
    verdicts = {v for v in judged if v is not None}
    units = _judge_units(messages, judged) if len(verdicts) == 1 else []
    in_unit: set[int] = set()
    drop: set[int] = set()
    dropped = passed = 0
    for span, question in units:
        in_unit.update(span)
        if verdicts == {"passed"}:
            passed += 1
            continue
        if not any(_carries_reference(messages[i]) for i in span) and was_asked(question):
            drop.update(span)
    kept: list[Any] = []
    for index, message in enumerate(messages):
        if index in drop:
            dropped += 1
            continue
        if index in in_unit:
            kept.append(message)
            continue
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
