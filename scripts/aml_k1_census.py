"""K-1 Stage 0 census: how often a returned window mixes speakers with no name to tell them apart.

Pre-registration: docs/preregistrations/2026-09-26-aml-c9-speaker-at-render-time.md. Free: no model,
no service, no key. For every top-10 text item of stored retrieval, the window is located in its own
session's content-only word sequence (``speaker_render.locate``) and the messages it overlaps are
read from ``speaker_render.message_word_ranges``. An item counts when it overlaps two or more
messages of different roles and at least one of the message starts inside it carries no speaker
name ("Name: ..."). A window found at no position, or at more than one, is counted as unmarked.

    python scripts/aml_k1_census.py locomo --collected collected-S.json.gz --data locomo10.json
    python scripts/aml_k1_census.py lme --stageb out/longmemeval_s.jsonl --data longmemeval_s_cleaned.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from recall_aml.speaker_render import locate, message_word_ranges  # noqa: E402

TOP_K = 10
#: The date header ``dated_search_content`` puts in front of a returned item.
DATE_HEADER = re.compile(r"^\[[^\]]*UTC\]\s*")
#: A message that opens with a speaker's name, as the LoCoMo collect writes every turn.
NAMED = re.compile(r"^[A-Z][\w.' -]{0,40}:\s")


def classify(window: str, messages: list[tuple[str, str]], words: list[str]) -> str:
    """``unmarked`` (not uniquely located), ``one_role``, ``named`` (mixed but every inner start is
    named), or ``mixed_unnamed`` (the case K-1 marks)."""
    start = locate(window, words)
    if start is None:
        return "unmarked"
    end = start + len(window.split())
    ranges = message_word_ranges(messages)
    overlapping = [(index, role, s) for index, (role, s, e) in enumerate(ranges) if s < end and e > start]
    if len({role for _, role, _ in overlapping}) < 2:
        return "one_role"
    inner = [index for index, _, s in overlapping if s > start]
    if all(NAMED.match(messages[index][1]) for index in inner):
        return "named"
    return "mixed_unnamed"


def speakers_spanned(window: str, messages: list[tuple[str, str]], words: list[str]) -> int:
    """Distinct ``Name:`` speakers among the messages a located window overlaps (LoCoMo context)."""
    start = locate(window, words)
    if start is None:
        return 0
    end = start + len(window.split())
    names = set()
    for (role, s, e), (_, content) in zip(message_word_ranges(messages), messages, strict=True):
        if s < end and e > start and (match := NAMED.match(content)):
            names.add(match.group(0))
    return len(names)


def tally(rows: list[tuple[str, str, int]]) -> dict[str, Any]:
    counts = Counter(label for _, label, _ in rows)
    items = len(rows)
    by_kind = Counter(kind for kind, _, _ in rows)
    mixed_by_kind = Counter(kind for kind, label, _ in rows if label == "mixed_unnamed")
    return {
        "items": items,
        "mixed_unnamed_share": round(counts["mixed_unnamed"] / items, 4) if items else None,
        "unmarked_share": round(counts["unmarked"] / items, 4) if items else None,
        "labels": dict(counts),
        "items_by_kind": dict(by_kind),
        "mixed_unnamed_by_kind": dict(mixed_by_kind),
        "two_or_more_named_speakers_share": round(sum(1 for *_, n in rows if n >= 2) / items, 4) if items else None,
    }


def text(item: dict[str, Any]) -> str:
    return DATE_HEADER.sub("", item.get("content") or "")


def locomo(args: argparse.Namespace) -> dict[str, Any]:
    from aml_locomo_route_compare import build_corpus

    data = json.loads(args.data.read_text(encoding="utf-8"))
    adds, _ = build_corpus(data, "census", None)
    sessions = {}
    for add in adds:
        messages = [(m["role"], m["content"]) for m in add["messages"]]
        sessions[add["session_id"]] = (messages, " ".join(c for _, c in messages).split())
    collected = json.loads(gzip.decompress(args.collected.read_bytes()))
    rows = []
    missing_session = 0
    for row in collected["rows"]:
        for item in [i for i in row["items"] if isinstance(i.get("content"), str)][:TOP_K]:
            session = sessions.get(str(item.get("session_id") or ""))
            if session is None:
                missing_session += 1
                continue
            window = text(item)
            rows.append((str(item.get("kind") or "raw"), classify(window, *session),
                         speakers_spanned(window, *session)))
    return {"set": "locomo", "questions": len(collected["rows"]), "items_without_a_known_session": missing_session,
            **tally(rows)}


def lme(args: argparse.Namespace) -> dict[str, Any]:
    data = {str(q["question_id"]): q for q in json.loads(args.data.read_text(encoding="utf-8"))}
    rows = []
    questions = 0
    missing_session = 0
    for line in args.stageb.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        question = data[record["tenant"]]
        sessions = {}
        for session_id, session in zip(question["haystack_session_ids"], question["haystack_sessions"], strict=True):
            messages = [(str(m["role"]), str(m["content"])) for m in session]
            sessions[str(session_id)] = (messages, " ".join(c for _, c in messages).split())
        for search in record["searches"]:
            questions += 1
            for item in [i for i in search["items"] if isinstance(i.get("content"), str)][:TOP_K]:
                session = sessions.get(str(item.get("session_id") or ""))
                if session is None:
                    missing_session += 1
                    continue
                window = text(item)
                rows.append((str(item.get("kind") or "raw"), classify(window, *session), 0))
    return {"set": "longmemeval_s", "questions": questions, "items_without_a_known_session": missing_session,
            **tally(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("locomo")
    a.add_argument("--collected", type=Path, required=True)
    a.add_argument("--data", type=Path, required=True)
    a.set_defaults(handler=locomo)
    b = sub.add_parser("lme")
    b.add_argument("--stageb", type=Path, required=True)
    b.add_argument("--data", type=Path, required=True)
    b.set_defaults(handler=lme)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), indent=2))


if __name__ == "__main__":
    main()
