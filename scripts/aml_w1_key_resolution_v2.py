"""W1 key resolution v2 (generic-subject guard) and the W3 history check on resolved keys.

Pre-registration: kept in the maintainer's private research log (W1 key resolution v2 and W3,
2026-09-28).

Free, like v1 (``scripts/aml_w1_key_resolution.py``, whose replay and scoring this reuses). v1's
selection could not see a cluster that was too wide: its distractor rate stayed near zero while a
bare "budget" absorbed every budget. v2 selects, on the even half, by current-value recall given
coverage: the share of covered knowledge-update pairs whose sides share a canonical key whose
turn-ordered history (``recall_aml.state_history.key_histories``, through the unchanged W3 check)
is an update whose latest fact cites the updated turn. That rewards a link and punishes a cluster
whose latest value belongs to another slot.

    python scripts/aml_w1_key_resolution_v2.py --data 100K.parquet --facts facts.jsonl --out v2.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_w1_key_resolution import _add_index, evaluate, resolved_keys  # noqa: E402
from aml_w1_link_test import _rows  # noqa: E402
from aml_w3_history_check import check  # noqa: E402

from recall_aml.key_resolution import KeyResolver, ResolverConfig  # noqa: E402

V1_CHOSEN = ResolverConfig(subject_jaccard=0.5, attribute_jaccard=1.0, synonyms=True)
GRID_V2 = [
    ResolverConfig(ts, ta, True, guard, True)
    for guard in (1, 2)
    for ts in (0.34, 0.5, 0.67)
    for ta in (0.5, 1.0)
]


def resolved_records(records: Iterable[dict[str, Any]], config: ResolverConfig | None) -> list[dict[str, Any]]:
    """The records with every fact's key replaced by its canonical key, replayed in Add order."""
    by_conversation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_conversation[int(record["conversation"])].append(record)
    out: list[dict[str, Any]] = []
    for adds in by_conversation.values():
        resolver = KeyResolver(config) if config else None
        for record in sorted(adds, key=_add_index):
            facts = [{**f, "key": resolver.resolve(f["key"]) if resolver else f["key"]} for f in record["facts"]]
            out.append({**record, "facts": facts})
    return out


def current_value_recall(resolution: dict[str, Any], history: dict[str, Any]) -> float:
    """Covered update pairs whose shared key's history names the updated turn as latest."""
    covered = resolution["pairs"].get("knowledge_update", {}).get("covered") or 0
    update = history.get("knowledge_update") or {}
    correct = round((update.get("update_correct") or 0.0) * (update.get("linked") or 0))
    return correct / covered if covered else 0.0


def arm(
    rows: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]], config: ResolverConfig | None,
    half: Sequence[int],
) -> dict[str, Any]:
    keys, largest = resolved_keys(records, config)
    result = evaluate(rows, keys, largest, half)
    members = set(half)
    history = check(rows, [r for r in resolved_records(records, config) if int(r["conversation"]) in members])
    result["w3"] = history
    result["current_value_recall_given_coverage"] = round(current_value_recall(result, history), 4)
    return result


def selection_key(result: dict[str, Any]) -> tuple[float, float, int]:
    """Higher current-value recall; then lower distractor rate; then fewer merged keys."""
    return (result["current_value_recall_given_coverage"], -(result["distractor"]["rate"] or 0.0),
            -result["keys_merged"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _rows(args.data)
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    halves = {
        "even": [i for i in range(len(rows)) if i % 2 == 0],
        "odd": [i for i in range(len(rows)) if i % 2 == 1],
        "all": list(range(len(rows))),
    }
    tuning = [(config, arm(rows, records, config, halves["even"])) for config in GRID_V2]
    chosen = max(tuning, key=lambda item: selection_key(item[1]))[0]
    out: dict[str, Any] = {"chosen": chosen.name, "v1_chosen": V1_CHOSEN.name}
    for name, half in halves.items():
        out[name] = {
            "exact_keys": arm(rows, records, None, half),
            "v1_chosen": arm(rows, records, V1_CHOSEN, half),
            "v2_chosen": arm(rows, records, chosen, half),
        }
    out["tuning_even"] = [
        {"config": c.name, "current_value_recall_given_coverage": r["current_value_recall_given_coverage"],
         "link_given_coverage": r["pairs"].get("knowledge_update", {}).get("link_recall_given_coverage"),
         "distractor_rate": r["distractor"]["rate"], "largest_cluster": r["largest_cluster"],
         "keys_merged": r["keys_merged"]}
        for c, r in tuning
    ]
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"chosen": out["chosen"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
