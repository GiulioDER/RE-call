"""Load the BEAM dataset and flatten it into questions, matching upstream's ids exactly.

Two things here are load-bearing and easy to get subtly wrong:

- **Question ids.** Pairing our RE-call run against Mem0's published per-question artifact joins on
  `question_id`, so this module reproduces upstream's construction (`{size}_{conv}_q{qi}_{type}`
  where `qi` is the position in the flattened list) rather than inventing its own. The flattening
  order comes from `BEAM_QUESTION_TYPES` in the vendored `prompts.py`, which is why that dict is
  imported instead of being re-listed: one source of truth for the order means the join cannot
  silently shift by one.
- **Reading the parquet without pandas.** `datasets.load_dataset` pulls pandas in, and pandas is
  DLL-blocked on this machine. pyarrow reads the identical file. `--data` therefore points at a
  parquet path (or at a pre-converted JSON), never at a HuggingFace loader.
"""
from __future__ import annotations

import json
from ast import literal_eval as _python_literal  # safe: builds literals only, never executes code
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.beam.prompts import BEAM_QUESTION_TYPES


@dataclass(frozen=True)
class Question:
    """One BEAM probing question, keyed the way the published artifact keys it."""

    question_id: str
    chat_size: str
    conversation_idx: int
    question_type: str
    question: str
    rubric: list[str]
    difficulty: str = "unknown"
    #: Present only on `abstention` questions: upstream's reason the question is unanswerable.
    why_unanswerable: str = ""

    @property
    def is_abstention(self) -> bool:
        """True when withholding an answer is the CORRECT behaviour for this question.

        The complement matters as much: a refusal on any other type is a false abstain, which is
        the failure mode a trust layer is most likely to introduce and the one our methodology
        requires reporting next to the abstention rate.
        """
        return self.question_type == "abstention"


@dataclass
class Conversation:
    """One BEAM conversation: its dialogue turns and its 20 probing questions."""

    chat_size: str
    index: int
    conversation_id: str
    #: Flat, chronological. Each entry is ``{"role", "content", "date"}``; `date` may be "".
    turns: list[dict[str, str]] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


def load_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a BEAM split's parquet into plain dicts, with no pandas in the import graph."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _parse_probing(raw: Any) -> dict[str, Any]:
    """HuggingFace stores `probing_questions` as a Python-repr string; fall back to JSON.

    JSON is tried first and the Python-literal parser second. The literal parser accepts only
    literals (dicts, lists, strings, numbers) — it cannot call anything — so parsing this
    third-party field executes no code from the dataset.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = _python_literal(raw)
        except (ValueError, SyntaxError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _flatten_turns(chat: Any) -> list[dict[str, str]]:
    """Flatten BEAM's nested chat into an ordered list of ``{"role", "content", "date"}``.

    BEAM stores chat three ways depending on bucket (2-D list, list of ``{"turns": ...}`` batches,
    or 10M's plan-keyed session dicts). All three are handled so a 10M run later needs no new
    parsing. A turn's `time_anchor` is carried through: BEAM's temporal and event-ordering
    questions ask *when*, and the answerer prompt prefixes each memory with its date — a retriever
    whose chunks have no date is answering those questions blind.
    """

    def _one(turn: Any) -> dict[str, str] | None:
        if not isinstance(turn, dict):
            return None
        content = turn.get("content", "") or ""
        if not content.strip():
            return None
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user" if str(role).lower() in ("human", "user") else "assistant"
        return {"role": role, "content": content, "date": turn.get("time_anchor", "") or ""}

    def _from_batch(batch: Any) -> list[dict[str, str]]:
        """Turns of one batch, every one of them stamped with the BATCH's date.

        BEAM puts `time_anchor` on the first turn of a batch, not on all of them: in the 1M bucket
        that is 10 dated turns out of 1,710. Upstream reads one anchor per batch and applies it to
        everything ingested from that batch (`get_time_anchor_epoch`), so every Mem0 memory carries
        a date. Reading the field per-turn instead would leave 99.4 % of RE-call's memories
        undated, and the vendored answerer prompt — which prefixes each memory with its date and is
        told to reason about dates and ordering — would be answering BEAM's temporal and
        event-ordering categories blind on our side only. That is a harness handicap masquerading
        as a retrieval result, so the anchor is propagated here.
        """
        items = batch if isinstance(batch, list) else batch.get("turns", [])
        out: list[dict[str, str]] = []
        for item in items:
            for candidate in item if isinstance(item, list) else [item]:
                parsed = _one(candidate)
                if parsed is not None:
                    out.append(parsed)
        anchor = next((t["date"] for t in out if t["date"]), "")
        if anchor:
            for turn in out:
                turn["date"] = anchor
        return out

    if not chat or not isinstance(chat, list):
        return []

    turns: list[dict[str, str]] = []
    for element in chat:
        if isinstance(element, list) or (isinstance(element, dict) and "turns" in element):
            turns.extend(_from_batch(element))
        elif isinstance(element, dict):
            # 10M plan format: a session dict whose values are lists of batches.
            plan_keys = sorted(
                element,
                key=lambda k: int(k.split("-")[-1]) if k.split("-")[-1].isdigit() else 0,
            )
            handled = False
            for key in plan_keys:
                batches = element[key]
                if isinstance(batches, list):
                    for batch in batches:
                        turns.extend(_from_batch(batch))
                    handled = True
            if not handled:
                parsed = _one(element)
                if parsed is not None:
                    turns.append(parsed)
    return turns


def _rubric_of(raw: Any) -> list[str]:
    """Rubric nuggets as a flat list of strings, whatever shape the row stores them in."""
    if isinstance(raw, dict):
        nuggets = raw.get("nuggets", [])
        return [n.get("description", str(n)) if isinstance(n, dict) else str(n) for n in nuggets]
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return [str(raw)] if raw else []


def _questions_of(probing: dict[str, Any], chat_size: str, conv_idx: int) -> list[Question]:
    """Flatten one conversation's probing questions, reproducing upstream's `qi` numbering."""
    flat: list[tuple[str, Any]] = []
    for q_type in BEAM_QUESTION_TYPES:
        entry = probing.get(q_type, [])
        if isinstance(entry, list):
            flat.extend((q_type, q) for q in entry if isinstance(q, (dict, str)))
        elif isinstance(entry, dict):
            flat.append((q_type, entry))

    questions: list[Question] = []
    for qi, (q_type, raw) in enumerate(flat):
        if isinstance(raw, str):
            text, rubric, difficulty, why = raw, [], "unknown", ""
        else:
            text = raw.get("question_text", raw.get("question", ""))
            rubric = _rubric_of(raw.get("rubric", []))
            difficulty = raw.get("difficulty", "unknown")
            why = raw.get("why_unanswerable", "")
        questions.append(
            Question(
                question_id=f"{chat_size}_{conv_idx}_q{qi}_{q_type}",
                chat_size=chat_size,
                conversation_idx=conv_idx,
                question_type=q_type,
                question=text,
                rubric=rubric,
                difficulty=difficulty,
                why_unanswerable=why,
            )
        )
    return questions


def load_conversations(
    path: Path, chat_size: str, indices: list[int] | None = None
) -> list[Conversation]:
    """Load a BEAM split into `Conversation` objects.

    `path` is either the HuggingFace parquet for the split or a JSON file previously converted
    from it. `indices` selects a subset by position — the same positions upstream's
    `--conversations` uses, so a subset run's question ids still join against the published
    artifact.
    """
    if path.suffix == ".parquet":
        rows = load_parquet(path)
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    wanted = set(indices) if indices is not None else None

    conversations: list[Conversation] = []
    for idx, row in enumerate(rows):
        if wanted is not None and idx not in wanted:
            continue
        conversations.append(
            Conversation(
                chat_size=chat_size,
                index=idx,
                conversation_id=row.get("conversation_id") or f"{chat_size}_{idx}",
                turns=_flatten_turns(row.get("chat", [])),
                questions=_questions_of(
                    _parse_probing(row.get("probing_questions", "{}")), chat_size, idx
                ),
            )
        )
    return conversations
