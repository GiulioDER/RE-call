"""W1 key resolution: re-score the key-link test's facts with drifted keys merged in code.

Pre-registration: kept in the maintainer's private research log (W1 key resolution, 2026-09-28).

Free: reads the facts ``scripts/aml_w1_link_test.py run`` wrote and BEAM's pair labels, and calls
no model. Each conversation's facts are replayed in Add order through
``recall_aml.key_resolution.KeyResolver``; pairs are scored exactly as the link test scores them
(`pairs_of`, `pair_metrics`), per half: conversations with an even index tune the resolver's
thresholds, odd ones judge the chosen configuration once.

The over-merging guard is the distractor link rate: within one conversation, one pair's first side
against a DIFFERENT pair's side, both extracted, turn sets disjoint; the share sharing a key.
Merging everything drives it to 1.

    python scripts/aml_w1_key_resolution.py --data 100K.parquet --facts facts.jsonl --out res.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import itertools
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_w1_link_test import _rows, pair_metrics, pairs_of  # noqa: E402

from recall_aml.key_resolution import KeyResolver, ResolverConfig  # noqa: E402

GRID = [
    ResolverConfig(ts, ta, syn)
    for ts in (0.34, 0.5, 0.67)
    for ta in (0.34, 0.5, 1.0)
    for syn in (True, False)
]
#: Reported for the trade-off it shows; never selectable.
SAME_SUBJECT_ANY_ATTRIBUTE = ResolverConfig(0.5, 0.0, False)
DISTRACTOR_WEIGHT = 2.0


def _add_index(record: dict[str, Any]) -> int:
    return int(str(record["add"]).rsplit(":a", 1)[1])


def resolved_keys(
    records: Iterable[dict[str, Any]], config: ResolverConfig | None
) -> tuple[dict[int, dict[int, set[str]]], dict[int, tuple[int, int]]]:
    """Keys per conversation per turn (canonical when ``config`` is set, raw otherwise), replayed
    in Add order, and each conversation's (largest cluster, raw keys merged into another)."""
    by_conversation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_conversation[int(record["conversation"])].append(record)
    keys: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    largest: dict[int, tuple[int, int]] = {}
    for conversation, adds in by_conversation.items():
        resolver = KeyResolver(config) if config else None
        for record in sorted(adds, key=_add_index):
            for fact in record["facts"]:
                key = resolver.resolve(fact["key"]) if resolver else fact["key"]
                for turn in fact["turns"]:
                    keys[conversation][int(turn)].add(key)
        if resolver:
            merged = sum(len(names) - 1 for names in resolver.aliases.values())
            largest[conversation] = (resolver.largest_cluster(), merged)
        else:
            largest[conversation] = (1, 0)
    return keys, largest


def _side(turns: Sequence[int], keys: dict[int, set[str]]) -> set[str]:
    return set().union(*(keys.get(t, set()) for t in turns))


def distractor_counts(pairs: Sequence[dict[str, Any]], keys: dict[int, set[str]]) -> tuple[int, int]:
    """(linked, combinations) over one pair's first side against a different pair's either side."""
    linked = total = 0
    for first, other in itertools.permutations(pairs, 2):
        for side in (other["left"], other["right"]):
            if set(first["left"]) & set(side):
                continue
            a, b = _side(first["left"], keys), _side(side, keys)
            if not a or not b:
                continue
            total += 1
            linked += int(bool(a & b))
    return linked, total


def evaluate(
    rows: Sequence[dict[str, Any]], keys: dict[int, dict[int, set[str]]],
    largest: dict[int, tuple[int, int]], conversations: Sequence[int],
) -> dict[str, Any]:
    all_pairs: list[dict[str, Any]] = []
    merged: dict[int, set[str]] = {}
    linked = combos = 0
    for index in conversations:
        offset = index * 1_000_000
        pairs = pairs_of(rows[index])
        all_pairs += [{**p, "left": [offset + t for t in p["left"]], "right": [offset + t for t in p["right"]]}
                      for p in pairs]
        merged.update({offset + t: k for t, k in keys[index].items()})
        dl, dt = distractor_counts(pairs, keys[index])
        linked, combos = linked + dl, combos + dt
    metrics = pair_metrics(all_pairs, merged)
    return {
        "pairs": metrics,
        "distractor": {"linked": linked, "combinations": combos,
                       "rate": round(linked / combos, 4) if combos else None},
        "largest_cluster": max((largest.get(i, (0, 0))[0] for i in conversations), default=0),
        "keys_merged": sum(largest.get(i, (0, 0))[1] for i in conversations),
    }


def _lgc(result: dict[str, Any]) -> float:
    return result["pairs"].get("knowledge_update", {}).get("link_recall_given_coverage") or 0.0


def objective(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Gain in update link given coverage, less twice the gain in distractor rate."""
    gain = _lgc(result) - _lgc(baseline)
    return gain - DISTRACTOR_WEIGHT * ((result["distractor"]["rate"] or 0.0) - (baseline["distractor"]["rate"] or 0.0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _rows(args.data)
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    even = [i for i in range(len(rows)) if i % 2 == 0]
    odd = [i for i in range(len(rows)) if i % 2 == 1]

    def run(config: ResolverConfig | None, half: Sequence[int]) -> dict[str, Any]:
        keys, largest = resolved_keys(records, config)
        return evaluate(rows, keys, largest, half)

    baseline_even = run(None, even)
    tuning = []
    for config in GRID:
        result = run(config, even)
        tuning.append({"config": config.name, "objective": round(objective(result, baseline_even), 4),
                       "largest_cluster": result["largest_cluster"], "keys_merged": result["keys_merged"],
                       "result": result})
    # Ties on the objective go to the configuration that merges fewer keys.
    best_index = max(range(len(GRID)), key=lambda i: (tuning[i]["objective"], -tuning[i]["keys_merged"]))
    chosen = GRID[best_index]
    out = {
        "facts_records": len(records),
        "chosen": chosen.name,
        "even": {"exact_keys": baseline_even, "chosen": tuning[best_index]["result"],
                 "same_subject_any_attribute": run(SAME_SUBJECT_ANY_ATTRIBUTE, even)},
        "odd": {"exact_keys": run(None, odd), "chosen": run(chosen, odd),
                "same_subject_any_attribute": run(SAME_SUBJECT_ANY_ATTRIBUTE, odd)},
        "tuning_even": [{k: v for k, v in t.items() if k != "result"} for t in tuning],
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("chosen", "odd")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
