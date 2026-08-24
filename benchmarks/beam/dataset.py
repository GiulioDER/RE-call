"""Load the BEAM dataset and flatten it into questions, matching upstream's ids exactly.

Two things here are load-bearing and easy to get subtly wrong:

- **Question ids.** Pairing our RE-call run against Mem0's published per-question artifact joins on
  `question_id`, so this module reproduces upstream's construction (`{size}_{conv}_q{qi}_{type}`
  where `qi` is the position in the flattened list) rather than inventing its own. The flattening
  order comes from `BEAM_QUESTION_TYPES` in the vendored `prompts.py`, which is why that dict is
  imported instead of being re-listed: one source of truth for the order means the join cannot
  silently shift by one.
- **Reading the parquet without pandas.** `datasets.load_dataset` pulls in pandas and a network
  fetch; pyarrow reads the identical file directly, with no loader script and no download.
  `--data` therefore points at a parquet path (or at a pre-converted JSON), never at a
  HuggingFace loader.

  🔁 Corrected 2026-08-23: the reason recorded here used to be "pandas is DLL-blocked on this
  machine". It is not, and `pandas 3.0.3` imports fine. The design is unchanged and still right,
  for the plainer reason above; only the justification was wrong.
"""
from __future__ import annotations

import json
#: Builds literals only — it executes NO code from the dataset. That is not a general
#: safety clearance, though: CPython's own docs warn `literal_eval` is not safe against
#: untrusted data, because a deeply nested literal can exhaust the parser stack. Hence the
#: size cap at the call site and the widened except below.
from ast import literal_eval as _python_literal
from collections.abc import Iterator
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
    """Read a BEAM split's parquet into plain dicts, with no pandas in the import graph.

    Materialises the WHOLE split. For the 1M bucket that is ~40 M tokens of dialogue expanding to
    >3 GB of Python objects and many minutes before the first conversation is available — measured.
    Prefer `iter_rows`, which converts one row at a time; this stays for callers that genuinely
    want the list (tests, ad-hoc inspection of a small split).
    """
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def iter_rows(path: Path, indices: set[int] | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, row)`` for a split, converting ONE row to Python at a time.

    The Arrow table itself is cheap to hold (columnar, compressed); what is expensive is turning
    160 M characters of nested dialogue into Python objects. Doing that per row keeps peak memory
    to one conversation and lets indexing start in seconds instead of after a full-split parse.
    Rows outside `indices` are never converted at all.
    """
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for idx in range(table.num_rows):
            if indices is not None and idx not in indices:
                continue
            yield idx, table.slice(idx, 1).to_pylist()[0]
        return
    # JSON fallback: no way to slice lazily, so this one is read whole.
    for idx, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        if indices is not None and idx not in indices:
            continue
        yield idx, row


#: Upper bound on a `probing_questions` cell before it reaches a recursive-descent parser. Real
#: blocks are a few hundred characters; this is ~250x the largest observed.
_MAX_PROBING_CHARS = 200_000


def _parse_probing(raw: Any) -> dict[str, Any]:
    """HuggingFace stores `probing_questions` as a Python-repr string; fall back to JSON.

    JSON is tried first and the Python-literal parser second. The literal parser accepts only
    literals (dicts, lists, strings, numbers) — it cannot call anything — so parsing this
    third-party field executes no code from the dataset.

    That is a statement about code execution, NOT a general safety clearance: CPython's own
    documentation warns `literal_eval` is unsafe on untrusted input, because a sufficiently
    nested literal exhausts the parser stack. This field arrives from a HuggingFace download, so
    the input is size-capped first and the failure modes that implies are caught below.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    if len(raw) > _MAX_PROBING_CHARS:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = _python_literal(raw)
        except (ValueError, SyntaxError, RecursionError, MemoryError):
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


def parse_conversation_indices(spec: str) -> list[int]:
    """`"0-14,20"` -> `[0, 1, ..., 14, 20]`, deduplicated and sorted.

    One implementation, because there were seven: two named `_parse_indices` and five inlined
    into a `main()`, in two variants that did not agree. The inline copies omitted the `.strip()`
    on each part, so `--conversations "0, 2"` raised `ValueError` there and worked in the other
    two; and they left deduplication to the call site, which `run.py` did and the probes did not.
    Seven copies of a CLI parser is how `--conversations "0-14, 20"` comes to mean different
    things in two probes that read each other's tables.
    """
    out: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi.strip()) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


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


#: Filename pattern for a sharded split. Zero-padded so lexical order is index order.
SHARD_NAME = "conv_{idx:03d}.json"
_SHARD_GLOB = "conv_*.json"


def shard_split(path: Path, out_dir: Path) -> int:
    """Split a BEAM parquet into one JSON file per conversation. Returns the count written.

    Why this exists: the 1M split is a SINGLE parquet row group, so there is no such thing as
    reading one row cheaply — any read decompresses all 35 conversations, measured at 626 MB
    resident that never returns to the OS. On a 12 GB machine that is a permanent tax for the
    whole run, on top of the embedder and the indexing peak, and it is what took the machine down.

    Sharding pays that cost ONCE, offline, and leaves 35 files of ~15 MB. A run then holds one
    conversation instead of thirty-five, and a re-run pays nothing at all.
    """
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(path)
    written = 0
    for idx in range(table.num_rows):
        row = table.slice(idx, 1).to_pylist()[0]
        (out_dir / SHARD_NAME.format(idx=idx)).write_text(
            json.dumps(row, ensure_ascii=False), encoding="utf-8"
        )
        written += 1
    return written


def _shard_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(_SHARD_GLOB))


def iter_conversations(
    path: Path, chat_size: str, indices: list[int] | None = None
) -> Iterator[Conversation]:
    """Yield `Conversation` objects one at a time, holding at most one in memory.

    This is what the runner uses: it ingests and scores a conversation before the next one is
    parsed, so peak memory is one conversation rather than the whole split.

    `path` may be a parquet, a converted JSON array, or a DIRECTORY of per-conversation shards
    written by `shard_split` — the last being the only one of the three that does not hold the
    whole split resident, and so the one to use on a memory-constrained machine.
    """
    wanted = set(indices) if indices is not None else None
    if path.is_dir():
        for shard in _shard_paths(path):
            idx = int(shard.stem.split("_")[-1])
            if wanted is not None and idx not in wanted:
                continue
            row = json.loads(shard.read_text(encoding="utf-8"))
            yield _conversation_of(row, chat_size, idx)
        return
    for idx, row in iter_rows(path, wanted):
        yield _conversation_of(row, chat_size, idx)


def _conversation_of(row: dict[str, Any], chat_size: str, idx: int) -> Conversation:
    return Conversation(
        chat_size=chat_size,
        index=idx,
        conversation_id=row.get("conversation_id") or f"{chat_size}_{idx}",
        turns=_flatten_turns(row.get("chat", [])),
        questions=_questions_of(
            _parse_probing(row.get("probing_questions", "{}")), chat_size, idx
        ),
    )


def count_conversations(path: Path, indices: list[int] | None = None) -> int:
    """How many conversations a run will cover, WITHOUT converting any of them to Python.

    Exists so the runner can print "35 conversations" up front without paying the parse cost that
    `iter_conversations` is designed to spread out.
    """
    if path.is_dir():
        total = len(_shard_paths(path))
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        total = pq.read_metadata(path).num_rows
    else:
        total = len(json.loads(path.read_text(encoding="utf-8")))
    if indices is None:
        return total
    return sum(1 for i in indices if i < total)


def load_conversations(
    path: Path, chat_size: str, indices: list[int] | None = None
) -> list[Conversation]:
    """Eager version of `iter_conversations`, for tests and small splits.

    `path` is either the HuggingFace parquet for the split or a JSON file previously converted
    from it. `indices` selects a subset by position — the same positions upstream's
    `--conversations` uses, so a subset run's question ids still join against the published
    artifact.
    """
    return list(iter_conversations(path, chat_size, indices))
