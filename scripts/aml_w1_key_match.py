"""W1 key matcher: replay an extraction's Adds, mapping each Add's new keys with gpt-4o-mini.

Pre-registration: kept in the maintainer's private research log (W1 model-side key linking,
2026-09-28).

Arm B of the model-side linking test. The facts the W1 link test already extracted are replayed per
conversation in Add order; for each Add, the keys not seen before go to `recall_aml.key_matching`
with the conversation's existing keys and their latest values, exactly as an Add-time step would,
and every later occurrence of a matched key takes the existing key. Arm A (closed-list
extraction) is `aml_w1_link_test.py run --closed-keys`; both are scored here the same way.

    python scripts/aml_w1_key_match.py match --facts facts.jsonl --out matched.jsonl
    python scripts/aml_w1_key_match.py score --data 100K.parquet --facts matched.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_w1_key_resolution import _add_index  # noqa: E402
from aml_w1_key_resolution_v2 import arm  # noqa: E402
from aml_w1_link_test import _balance, _rows  # noqa: E402

Matcher = Callable[[Sequence[tuple[str, str]], Sequence[tuple[str, str]]], dict[str, str]]


def replay(records: Sequence[dict[str, Any]], matcher: Matcher) -> list[dict[str, Any]]:
    """One conversation's records in Add order with every key replaced by its canonical key.

    Only keys never seen before in the conversation go to ``matcher``, with the existing canonical
    keys and their latest values (most recently stated last). A key the matcher maps takes that
    existing key from then on; one it does not becomes canonical itself. Each fact keeps its
    extracted key as ``raw_key``.
    """
    canonical: dict[str, str] = {}
    latest: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for record in sorted(records, key=_add_index):
        new: list[tuple[str, str]] = []
        for fact in record["facts"]:
            if fact["key"] not in canonical and all(fact["key"] != k for k, _ in new):
                new.append((fact["key"], fact["value"]))
        mapping = matcher(new, list(latest.items())) if new and latest else {}
        for key, _ in new:
            canonical[key] = mapping.get(key, key)
        facts = [{**f, "raw_key": f["key"], "key": canonical[f["key"]]} for f in record["facts"]]
        for fact in facts:
            latest.pop(fact["key"], None)
            latest[fact["key"]] = fact["value"]
        out.append({**record, "facts": facts})
    return out


def cluster_width(records: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """(largest number of distinct extracted keys under one key in any conversation, extracted
    keys merged into another), from ``raw_key``; (1, 0) for records never matched."""
    raws: dict[tuple[int, str], set[str]] = defaultdict(set)
    for record in records:
        for fact in record["facts"]:
            raws[(int(record["conversation"]), fact["key"])].add(fact.get("raw_key", fact["key"]))
    return max((len(v) for v in raws.values()), default=0), sum(len(v) - 1 for v in raws.values())


def match(args: argparse.Namespace) -> None:
    from recall_aml.__main__ import build_openrouter_client
    from recall_aml.compiler import OpenAICompiler
    from recall_aml.key_matching import match_new_keys

    compiler = OpenAICompiler(build_openrouter_client(os.environ["OPENROUTER_API_KEY"]))
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_conversation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_conversation[int(record["conversation"])].append(record)
    start = _balance()
    print(json.dumps({"balance_start": round(start, 3)}), flush=True)
    lock, stop = threading.Lock(), threading.Event()
    errors: list[str] = []

    def matcher(new: Sequence[tuple[str, str]], existing: Sequence[tuple[str, str]]) -> dict[str, str]:
        try:
            return match_new_keys(compiler, new, existing)
        except Exception as exc:  # BROAD-CATCH: one failed call leaves that Add's keys new, and is counted
            errors.append(type(exc).__name__)
            return {}

    def conversation(index: int) -> None:
        if stop.is_set():
            return
        matched = replay(by_conversation[index], matcher)
        with lock:
            with args.out.open("a", encoding="utf-8") as sink:
                for record in matched:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            now = _balance()
            print(json.dumps({"conversation_done": index, "spent_usd": round(start - now, 3)}), flush=True)
            if start - now > args.max_usd or now < args.floor_usd:
                stop.set()
                print("STOP: budget or balance floor", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(conversation, sorted(by_conversation)))
    print(json.dumps({"spent_usd": round(start - _balance(), 3), "stopped": stop.is_set(),
                      "matcher_errors": len(errors)}), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    ma = sub.add_parser("match")
    ma.add_argument("--facts", type=Path, required=True)
    ma.add_argument("--out", type=Path, required=True)
    ma.add_argument("--workers", type=int, default=6)
    ma.add_argument("--max-usd", type=float, default=2.0)
    ma.add_argument("--floor-usd", type=float, default=20.0)
    sc = sub.add_parser("score")
    sc.add_argument("--data", type=Path, required=True)
    sc.add_argument("--facts", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "match":
        match(args)
        return 0
    rows = _rows(args.data)
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    every = list(range(len(rows)))
    largest, merged = cluster_width(records)
    result = {
        "event_updates": arm(rows, records, None, every, event_updates=True),
        "as_recorded": arm(rows, records, None, every),
        "largest_cluster": largest, "keys_merged": merged,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"largest_cluster": largest, "keys_merged": merged}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
