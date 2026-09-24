"""An Add-time aggregation view for list questions, tested against C9's served evidence on LoCoMo.

Pre-registration: docs/preregistrations/2026-09-24-aml-c9-aggregation-view.md

    extract   gpt-4o-mini lists the concrete items each person has, does or owns, one session at a
              time, exactly as an Add would; entries whose turn id is not in that session, or whose
              person is not a speaker of it, are dropped.
    build     one aggregation record per (conversation, person, category), items oldest first,
              each with its date and the turn it came from.
    answer    AML's LoCoMo-Refined Answer prompt over C9's served list with up to two aggregation
              records inserted at ranks 6 and 7, the list cut back to 100 items.

Judging and paired comparison reuse ``aml_locomo_loss_diagnosis.py judge`` and ``compare``.
Nothing in this file is tuned on answers: the prompt, categories, selection and placement are
fixed by the pre-registration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aml_locomo_loss_diagnosis import (  # noqa: E402
    ANSWER_MODEL,
    OpenRouter,
    load_aml_pipeline,
    load_collected,
    qa_index,
    read_jsonl,
    render_memories,
    run_parallel,
    spent_so_far,
)
from aml_locomo_route_compare import SESSION_KEY, session_timestamp_ms  # noqa: E402

EXTRACT_MODEL = "openai/gpt-4o-mini"
CATEGORIES = (
    "activities and hobbies",
    "sports and exercise",
    "games",
    "books and reading",
    "films and shows",
    "music",
    "pets and animals",
    "family and relationships",
    "places visited and travel",
    "food and drink",
    "purchases and possessions",
    "things made or created",
    "events attended",
    "work and career",
    "education and courses",
    "health",
    "plans and goals",
    "causes and volunteering",
    "other",
)
EXTRACT_PROMPT = (
    "You read one conversation session and list the concrete items that a person in it has, "
    "does, owns, made, bought, visited, attended, plays, reads, watches, likes or plans, so that a "
    "later question such as \"what X has Y done\" can be answered completely.\n\n"
    "Rules:\n"
    "- Use only what a turn in this session states. Never infer, guess, or merge sessions.\n"
    "- One entry per item. Keep the item short and specific, copied or closely paraphrased from "
    "the turn.\n"
    "- person: the name of the person the item belongs to, exactly as the session names the "
    "speaker.\n"
    "- category: exactly one of: " + ", ".join(CATEGORIES) + ".\n"
    "- turn_id: the id of the turn that states the item, copied exactly.\n\n"
    'Return JSON only: {"items": [{"person": "", "category": "", "item": "", "turn_id": ""}]}. '
    'Return {"items": []} when there is nothing.'
)
INSERT_AFTER_RANK = 5
MAX_RECORDS = 2
TOP_K = 100
TOKEN = re.compile(r"[a-z0-9]+")


def sessions_of(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every LoCoMo session in chronological order, as one Add would carry it."""
    sessions = []
    for sample in data:
        conversation = sample["conversation"]
        keys = sorted(
            (k for k in conversation if SESSION_KEY.match(k)),
            key=lambda k: int(SESSION_KEY.match(k).group(1)),  # type: ignore[union-attr]
        )
        for key in keys:
            turns = conversation[key]
            if not turns:
                continue
            sessions.append(
                {
                    "id": f"{sample['sample_id']}:{key}",
                    "sample_id": sample["sample_id"],
                    "date": conversation[f"{key}_date_time"],
                    "timestamp_ms": session_timestamp_ms(conversation[f"{key}_date_time"]),
                    "speakers": (conversation["speaker_a"], conversation["speaker_b"]),
                    "turns": [
                        {"id": turn["dia_id"], "speaker": turn["speaker"], "text": turn["text"]}
                        for turn in turns
                    ],
                }
            )
    return sessions


def session_prompt(session: dict[str, Any]) -> str:
    lines = [f"Session date: {session['date']}"]
    lines += [f"[{t['id']}] {t['speaker']}: {t['text']}" for t in session["turns"]]
    return EXTRACT_PROMPT + "\n\nSession:\n" + "\n".join(lines)


def grounded_items(session: dict[str, Any], reply: str) -> tuple[list[dict[str, Any]], int]:
    """Keep only entries whose turn is in this session and whose person speaks in it."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        proposed = json.loads(match.group(0))["items"] if match else []
    except (json.JSONDecodeError, KeyError, TypeError):
        proposed = []
    if not isinstance(proposed, list):
        proposed = []
    turns = {t["id"]: t for t in session["turns"]}
    speakers = {name.casefold(): name for name in session["speakers"]}
    kept = []
    for entry in proposed:
        if not isinstance(entry, dict):
            continue
        # The prompt shows turns as "[D1:3] speaker: text", so a model copying the id exactly
        # returns it bracketed. Strip the brackets, never anything else.
        turn = turns.get(str(entry.get("turn_id", "")).strip().strip("[]").strip())
        person = speakers.get(str(entry.get("person", "")).strip().casefold())
        item = str(entry.get("item", "")).strip()
        if turn is None or person is None or not item:
            continue
        category = str(entry.get("category", "")).strip().casefold()
        kept.append(
            {
                "person": person,
                "category": category if category in CATEGORIES else "other",
                "item": item,
                "turn_id": turn["id"],
                "turn_text": turn["text"],
            }
        )
    return kept, len(proposed)


def extract(args: argparse.Namespace) -> None:
    sessions = {s["id"]: s for s in sessions_of(json.loads(args.data.read_bytes()))}
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.out))

    def work(ident: str) -> dict[str, Any]:
        session = sessions[ident]
        reply, usage = router.complete(EXTRACT_MODEL, session_prompt(session))
        kept, proposed = grounded_items(session, reply)
        return {"id": ident, "proposed": proposed, "items": kept, "reply": reply, "usage": usage}

    run_parallel([i for i in sessions if i not in done], work, args.out, "extracted")


def build_records(
    sessions: list[dict[str, Any]], extracted: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Per conversation, one record per (person, category), items oldest first and deduplicated."""
    grouped: dict[tuple[str, str, str], list[tuple[int, str, str, str]]] = {}
    order = {s["id"]: position for position, s in enumerate(sessions)}
    by_id = {s["id"]: s for s in sessions}
    for session_id in sorted(extracted, key=order.__getitem__):
        session = by_id[session_id]
        for entry in extracted[session_id]["items"]:
            key = (session["sample_id"], entry["person"], entry["category"])
            grouped.setdefault(key, []).append(
                (session["timestamp_ms"], session["date"], entry["item"], entry["turn_text"])
            )
    records: dict[str, list[dict[str, Any]]] = {}
    for (sample_id, person, category), entries in sorted(grouped.items()):
        seen: set[str] = set()
        lines = []
        for _, date, item, turn_text in entries:
            if item.casefold() in seen:
                continue
            seen.add(item.casefold())
            quote = " ".join(turn_text.split())[:160]
            lines.append(f'- {date}: {item} (from "{quote}")')
        content = (
            f"Aggregated memory for {person}, category {category}, every item mentioned across "
            "sessions, oldest first:\n" + "\n".join(lines)
        )
        latest_ms = max(entry[0] for entry in entries)
        records.setdefault(sample_id, []).append(
            {
                "id": "agg:" + hashlib.sha256(f"{sample_id}|{person}|{category}".encode()).hexdigest()[:24],
                "person": person,
                "category": category,
                "content": content,
                "created_at": _iso(latest_ms),
                "kind": "aggregate",
                "items": len(lines),
            }
        )
    return records


def _iso(timestamp_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def build(args: argparse.Namespace) -> None:
    sessions = sessions_of(json.loads(args.data.read_bytes()))
    records = build_records(sessions, read_jsonl(args.extracted))
    args.out.write_text(json.dumps(records, indent=1, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "conversations": len(records),
                "records": sum(len(r) for r in records.values()),
                "items": sum(x["items"] for r in records.values() for x in r),
            }
        )
    )


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def select_records(
    question: str, records: list[dict[str, Any]], speakers: tuple[str, str]
) -> list[dict[str, Any]]:
    """Top two records by Okapi BM25, limited to named people when the question names any."""
    named = {s for s in speakers if s.casefold() in set(_tokens(question))}
    pool = [r for r in records if not named or r["person"] in named]
    if not pool:
        return []
    documents = [_tokens(r["content"]) for r in pool]
    average = sum(len(d) for d in documents) / len(documents)
    frequency: Counter[str] = Counter(t for d in documents for t in set(d))
    query = set(_tokens(question))
    scored = []
    for record, document in zip(pool, documents, strict=True):
        counts = Counter(document)
        score = 0.0
        for term in query:
            if term not in counts:
                continue
            idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(document) / average))
        if score > 0:
            scored.append((score, record["id"], record))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [record for _, _, record in scored[:MAX_RECORDS]]


def with_records(items: list[dict[str, Any]], chosen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert after the protected top five, then cut back to the contract's 100 items."""
    return (items[:INSERT_AFTER_RANK] + chosen + items[INSERT_AFTER_RANK:])[:TOP_K]


def answer(args: argparse.Namespace) -> None:
    pipeline = load_aml_pipeline(args.aml_repo)
    rows = {row["id"]: row for row in load_collected(args.collected)["rows"]}
    qas = qa_index(json.loads(args.data.read_bytes()))
    records = json.loads(args.records.read_text(encoding="utf-8"))
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.out))

    def work(ident: str) -> dict[str, Any]:
        qa = qas[ident]
        speaker_a, speaker_b = qa["speakers"]
        chosen = select_records(qa["question"], records.get(qa["sample_id"], []), qa["speakers"])
        prompt = pipeline.render_answer_prompt(
            {
                "question": qa["question"],
                "speaker_1_name": f"{speaker_a} and {speaker_b}",
                "speaker_1_memories": render_memories(with_records(rows[ident]["items"], chosen)),
                "speaker_2_name": "(none)",
                "speaker_2_memories": "(all memories are listed above)",
            }
        )
        generated, usage = router.complete(ANSWER_MODEL, prompt)
        return {
            "id": ident,
            "generated_answer": generated,
            "inserted": [r["id"] for r in chosen],
            "prompt_chars": len(prompt),
            "usage": usage,
        }

    run_parallel([i for i in rows if i not in done], work, args.out, "answered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function, inputs in (
        ("extract", extract, ("data",)),
        ("build", build, ("data", "extracted")),
        ("answer", answer, ("collected", "data", "records", "aml_repo")),
    ):
        stage = commands.add_parser(name)
        for option in inputs:
            stage.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
        stage.add_argument("--out", type=Path, required=True)
        stage.set_defaults(run=function)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
