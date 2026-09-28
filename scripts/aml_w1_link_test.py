"""W1 key-link test: do conversation facts put an update's two statements on one key?

Pre-registration: kept in the maintainer's private research log (W1 key-link test, 2026-09-28).

Prior work: K-2 v1 to v3 (``recall_aml/conflict_order.py`` and its records) linked same-subject
statements by similarity and reached 0.16 of real pairs at best; ``benchmarks/beam/dataset.py``
parses BEAM (its size-capped ``_parse_probing`` is reused here) but drops the turn ids this test
needs, so turns are read here with their ids.

Each BEAM 100K conversation is added as AML adds it (at most 20 messages and 2,000 words per Add,
in order), the W1 extractor runs on each Add with the keys earlier Adds produced, and every
``knowledge_update`` and ``contradiction_resolution`` row's two source chats are checked for a
shared key. No embedding anywhere; gpt-4o-mini through OpenRouter, exactly as served.

    python scripts/aml_w1_link_test.py run --data 100K.parquet --out facts.jsonl
    python scripts/aml_w1_link_test.py score --data 100K.parquet --facts facts.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beam.dataset import _parse_probing  # noqa: E402

MAX_MESSAGES = 20
MAX_WORDS = 2_000
PAIR_TYPES = {
    "knowledge_update": ("original_info", "updated_info"),
    "contradiction_resolution": ("first_statement", "second_statement"),
}


def turns_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Every turn in order with its chat id, role, text and the date its batch is anchored on."""
    out: list[dict[str, Any]] = []
    anchor: str | None = None
    for batch in row["chat"]:
        for turn in batch:
            anchor = turn.get("time_anchor") or anchor
            out.append({"id": int(turn["id"]), "role": str(turn["role"]), "content": str(turn["content"]), "date": anchor})
    return out


def chunk_adds(turns: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cut turns into Adds of at most `MAX_MESSAGES` messages and `MAX_WORDS` words, in order."""
    adds: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    words = 0
    for turn in turns:
        count = len(turn["content"].split())
        if current and (len(current) >= MAX_MESSAGES or words + count > MAX_WORDS):
            adds.append(current)
            current, words = [], 0
        current.append(turn)
        words += count
    if current:
        adds.append(current)
    return adds


def _stamp(date: str | None) -> datetime | None:
    if not date:
        return None
    try:
        return datetime.strptime(date.strip(), "%B-%d-%Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pairs_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    probing = _parse_probing(row["probing_questions"])
    for kind, (left, right) in PAIR_TYPES.items():
        for index, entry in enumerate(probing.get(kind, [])):
            ids = _parse_probing(entry.get("source_chat_ids") or "{}")
            out.append({"type": kind, "index": index, "left": [int(i) for i in ids.get(left, [])],
                        "right": [int(i) for i in ids.get(right, [])]})
    return out


def pair_metrics(pairs: Sequence[dict[str, Any]], keys_by_turn: dict[int, set[str]]) -> dict[str, Any]:
    """Coverage and link recall of ``pairs`` given which keys each turn's facts carry."""
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        left: set[str] = set().union(*(keys_by_turn.get(t, set()) for t in pair["left"]))
        right: set[str] = set().union(*(keys_by_turn.get(t, set()) for t in pair["right"]))
        counts = by_type[pair["type"]]
        counts["pairs"] += 1
        counts["covered"] += int(bool(left) and bool(right))
        counts["linked"] += int(bool(left & right))
    out = {}
    for kind, c in by_type.items():
        out[kind] = {
            "pairs": c["pairs"], "covered": c["covered"], "linked": c["linked"],
            "coverage": round(c["covered"] / c["pairs"], 3) if c["pairs"] else None,
            "link_recall": round(c["linked"] / c["pairs"], 3) if c["pairs"] else None,
            "link_recall_given_coverage": round(c["linked"] / c["covered"], 3) if c["covered"] else None,
        }
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return list(pq.read_table(path).to_pylist())


def _balance() -> float:
    import httpx

    body = httpx.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=30,
    ).json()["data"]
    return float(body["total_credits"]) - float(body["total_usage"])


def run(args: argparse.Namespace) -> None:
    from recall_aml.__main__ import build_openrouter_client
    from recall_aml.compiler import OpenAICompiler
    from recall_aml.conversation_records import extract_conversation_facts
    from recall_aml.models import Message

    compiler = OpenAICompiler(build_openrouter_client(os.environ["OPENROUTER_API_KEY"]))
    start = _balance()
    print(json.dumps({"balance_start": round(start, 3)}), flush=True)
    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(line)["add"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    lock = threading.Lock()
    stop = threading.Event()

    def conversation(index_row: tuple[int, dict[str, Any]]) -> None:
        index, row = index_row
        known: list[str] = []
        for position, add in enumerate(chunk_adds(turns_of(row))):
            if stop.is_set():
                return
            name = f"c{index}:a{position}"
            if name in done:
                continue
            messages = [Message(role=t["role"], content=t["content"], timestamp=_stamp(t["date"])) for t in add]
            facts: list[dict[str, Any]] = []
            dropped: dict[str, int] = {}
            error = None
            try:
                result = extract_conversation_facts(compiler, messages, f"beam-{index}-{position}", known)
                facts = [
                    {"key": f.key, "value": f.value, "relation": f.relation, "speaker": f.speaker,
                     "turns": [add[o]["id"] for o in f.message_ordinals]}
                    for f in result.facts
                ]
                dropped = result.dropped
            except Exception as exc:  # BROAD-CATCH: one failed Add is recorded and the run goes on
                error = type(exc).__name__
            known.extend(f["key"] for f in facts)
            record = {"add": name, "conversation": index, "turns": [t["id"] for t in add], "facts": facts,
                      "dropped": dropped, "error": error}
            with lock, args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        with lock:
            now = _balance()
            print(json.dumps({"conversation_done": index, "spent_usd": round(start - now, 3), "balance": round(now, 3)}), flush=True)
            if start - now > args.max_usd or now < args.floor_usd:
                stop.set()
                print("STOP: budget or balance floor", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(conversation, enumerate(_rows(args.data))))
    print(json.dumps({"spent_usd": round(start - _balance(), 3), "stopped": stop.is_set()}), flush=True)


def score(rows: Sequence[dict[str, Any]], facts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    keys_by_turn: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    adds_per_key: dict[int, Counter[str]] = defaultdict(Counter)
    dropped: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    total_facts = 0
    for record in facts:
        conversation = int(record["conversation"])
        if record.get("error"):
            errors[record["error"]] += 1
        dropped.update(record.get("dropped") or {})
        for fact in record["facts"]:
            total_facts += 1
            adds_per_key[conversation][fact["key"]] += 1
            for turn in fact["turns"]:
                keys_by_turn[conversation][int(turn)].add(fact["key"])
    all_pairs: list[dict[str, Any]] = []
    merged: dict[int, set[str]] = {}
    for index, row in enumerate(rows):
        offset = index * 1_000_000
        for pair in pairs_of(row):
            all_pairs.append({**pair, "left": [offset + t for t in pair["left"]], "right": [offset + t for t in pair["right"]]})
        for turn, keys in keys_by_turn[index].items():
            merged[offset + turn] = keys
    keys = [k for counter in adds_per_key.values() for k in counter]
    reused = sum(1 for counter in adds_per_key.values() for n in counter.values() if n > 1)
    return {
        "facts": total_facts,
        "distinct_keys": len(keys),
        "share_keys_used_by_more_than_one_add": round(reused / len(keys), 3) if keys else None,
        "dropped": dict(dropped),
        "errors": dict(errors),
        "pairs": pair_metrics(all_pairs, merged),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    ru = sub.add_parser("run")
    ru.add_argument("--data", type=Path, required=True)
    ru.add_argument("--out", type=Path, required=True)
    ru.add_argument("--workers", type=int, default=4)
    ru.add_argument("--max-usd", type=float, default=3.0)
    ru.add_argument("--floor-usd", type=float, default=20.0)
    sc = sub.add_parser("score")
    sc.add_argument("--data", type=Path, required=True)
    sc.add_argument("--facts", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "run":
        run(args)
    else:
        facts = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(_rows(args.data), facts)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
