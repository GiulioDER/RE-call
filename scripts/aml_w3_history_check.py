"""W3 history check: does code-ordered key history name the right current value and spot conflicts?

Pre-registration: kept in the maintainer's private research log (W3 history check, 2026-09-28).

Prior work: the W1 key-link test (``scripts/aml_w1_link_test.py``) writes the facts this reads and
supplies BEAM's pair labels (``pairs_of``); ``recall_aml.state_history`` is the code under test. Free:
no model call.

    python scripts/aml_w3_history_check.py --data 100K.parquet --facts facts.jsonl --out check.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_w1_link_test import _rows, pairs_of  # noqa: E402

from recall_aml.conversation_records import ConversationFact  # noqa: E402
from recall_aml.state_history import KeyHistory, key_histories  # noqa: E402


def facts_by_conversation(records: Iterable[dict[str, Any]]) -> dict[int, list[tuple[ConversationFact, tuple[int, ...]]]]:
    """Each conversation's facts, rebuilt from the link test's records, with the turns they cite."""
    out: dict[int, list[tuple[ConversationFact, tuple[int, ...]]]] = defaultdict(list)
    for record in records:
        for fact in record["facts"]:
            subject, _, attribute = fact["key"].partition("|")
            turns = tuple(int(t) for t in fact["turns"])
            out[int(record["conversation"])].append((
                ConversationFact(
                    key=fact["key"], subject=subject, attribute=attribute, value=fact["value"],
                    relation=fact["relation"], speaker=fact.get("speaker", ""), event_date=None,
                    mention_time=None, anchor_ids=(), message_ordinals=(), sensitive=False,
                ),
                turns,
            ))
    return out


def check(
    rows: Sequence[dict[str, Any]], records: Iterable[dict[str, Any]], *, event_updates: bool = False
) -> dict[str, Any]:
    by_conversation = facts_by_conversation(records)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows):
        entries = by_conversation.get(index, [])
        turns_of_fact = {id(fact): turns for fact, turns in entries}
        facts = [fact for fact, _ in entries]
        histories = key_histories(
            facts, order=lambda f: min(turns_of_fact[id(f)], default=0), event_updates=event_updates
        )
        keys_by_turn: dict[int, set[str]] = defaultdict(set)
        for fact, turns in entries:
            for turn in turns:
                keys_by_turn[turn].add(fact.key)
        for pair in pairs_of(row):
            left = set().union(*(keys_by_turn.get(t, set()) for t in pair["left"]))
            right = set().union(*(keys_by_turn.get(t, set()) for t in pair["right"]))
            shared = left & right
            if not shared:
                continue
            chosen: list[KeyHistory] = [histories[k] for k in sorted(shared)]
            c = counts[pair["type"]]
            c["linked"] += 1
            if pair["type"] == "knowledge_update":
                c["update_correct"] += int(any(
                    h.kind == "update" and h.latest is not None
                    and set(turns_of_fact[id(h.latest)]) & set(pair["right"]) for h in chosen
                ))
                c["update_read_as_conflict"] += int(all(h.kind == "conflict" for h in chosen))
            else:
                c["conflict_detected"] += int(any(h.kind == "conflict" for h in chosen))
                c["conflict_read_as_update"] += int(all(h.kind == "update" for h in chosen))
    out: dict[str, Any] = {}
    for kind, c in counts.items():
        linked = c["linked"]
        out[kind] = {"linked": linked, **{k: round(v / linked, 3) for k, v in c.items() if k != "linked"}}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = check(_rows(args.data), records)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
