"""LOCOMO -> documents, questions and clusters, with ids namespaced per conversation.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Two deliberate exclusions, both of which would otherwise contaminate the ladder:

- **Category 5 is dropped.** Those 446 questions are unanswerable by ANNOTATION — an event
  attributed to the wrong speaker — not by excision. Mixing two constructions of "unanswerable"
  into one axis would make the axis mean two things. They remain valuable as an EXTERNAL check on
  H2 (RE-call scores 0.00/446 on them) and are scored separately, never as a rung.
- **Questions with no `evidence`** are dropped rather than kept with an empty gold set: a question
  with nothing to excise is answerable at every rung, which would flatten the very curve H1 tests.

`dia_id` is unique only within a conversation (see `recall/eval/locomo.py:415`), so every id is
namespaced `"{sample_id}/{dia_id}"`. Without this, "D1:3" from two conversations collide and the
builder excises the wrong turn while every count still looks right.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ADVERSARIAL_CATEGORY = 5

#: `D<int>:<int>`, no leading zeros on either integer — `D30:05` is a zero-padded annotation typo,
#: not a valid id, and must fail this match rather than be silently accepted as `D30:5`'s neighbor.
_TURN_ID = re.compile(r"^D(?:0|[1-9]\d*):(?:0|[1-9]\d*)$")


def _split_evidence(raw: str) -> list[str]:
    """LOCOMO packs several turn ids into one string in a handful of entries.

    `"D9:1 D4:4 D4:6"` is three ids that were never split, and the turns they name exist. Treating
    the whole string as one id makes the question structurally unanswerable — permanently a miss,
    for a parsing reason rather than a retrieval one.

    Only fragments matching `D<int>:<int>` survive. Deliberately NOT normalised: `D:11:26`,
    `D30:05` and a bare `D` are annotation typos, and guessing what an annotator meant would put
    our invention into a corpus chosen precisely because we did not make it.
    """
    return [frag for frag in re.split(r"[;\s]+", raw.strip()) if _TURN_ID.fullmatch(frag)]


@dataclass(frozen=True)
class SourceQuestion:
    question_id: str
    question: str
    gold_doc_ids: tuple[str, ...]
    cluster_id: str


@dataclass(frozen=True)
class SourceCorpus:
    documents: tuple[tuple[str, str], ...]
    questions: tuple[SourceQuestion, ...]
    cluster_members: dict[str, tuple[str, ...]]
    content_hash: str
    #: why questions were dropped -> how many. Reported, never silent: a corpus that quietly
    #: discards a third of its questions still produces a clean-looking curve.
    dropped: dict[str, int] = field(default_factory=dict)


def _turn_text(turn: dict[str, Any], session_date: str) -> str:
    speaker = turn.get("speaker", "unknown")
    text = turn.get("text", "")
    return f"{speaker} ({session_date}): {text}"


def _sessions(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )


def _hash(documents: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for doc_id, text in sorted(documents):
        h.update(doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_locomo(path: Path) -> SourceCorpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[tuple[str, str]] = []
    questions: list[SourceQuestion] = []
    cluster_members: dict[str, tuple[str, ...]] = {}
    dropped: dict[str, int] = {}

    for sample in data:
        sample_id = sample["sample_id"]
        conversation = sample.get("conversation", {})
        members: list[str] = []
        for key in _sessions(conversation):
            turns = conversation[key]
            if not isinstance(turns, list):
                continue
            date = conversation.get(f"{key}_date_time", "unknown date")
            for turn in turns:
                dia_id = turn.get("dia_id")
                if not dia_id:
                    continue
                doc_id = f"{sample_id}/{dia_id}"
                documents.append((doc_id, _turn_text(turn, date)))
                members.append(doc_id)
        cluster_members[sample_id] = tuple(members)

        for i, qa in enumerate(sample.get("qa", [])):
            if qa.get("category") == _ADVERSARIAL_CATEGORY:
                dropped["category_5_adversarial"] = dropped.get("category_5_adversarial", 0) + 1
                continue
            if not qa.get("question"):
                dropped["no_question_text"] = dropped.get("no_question_text", 0) + 1
                continue
            raw_evidence = [e for e in (qa.get("evidence") or []) if isinstance(e, str)]
            evidence = [frag for raw in raw_evidence for frag in _split_evidence(raw)]
            if not evidence:
                dropped["no_evidence"] = dropped.get("no_evidence", 0) + 1
                continue
            gold = tuple(f"{sample_id}/{e}" for e in evidence)
            # `rings.build_rings` REFUSES gold that is not in its own cluster — such a question is
            # broken upstream, not hard. Drop it HERE, where the data is, and count it: letting it
            # through would crash the builder on live LOCOMO, and dropping it silently would shrink
            # the corpus without anyone noticing.
            if not set(gold) <= set(members):
                dropped["evidence_not_in_conversation"] = (
                    dropped.get("evidence_not_in_conversation", 0) + 1
                )
                continue
            questions.append(
                SourceQuestion(
                    question_id=f"{sample_id}/qa{i}",
                    question=qa["question"],
                    gold_doc_ids=gold,
                    cluster_id=sample_id,
                )
            )

    return SourceCorpus(
        documents=tuple(documents),
        questions=tuple(questions),
        cluster_members=cluster_members,
        content_hash=_hash(documents),
        dropped=dropped,
    )
