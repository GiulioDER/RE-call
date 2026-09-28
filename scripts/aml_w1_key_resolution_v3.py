"""W1 key resolution v3 and W3 with event updates: tune on BEAM 100K, confirm on fresh conversations.

Pre-registration: kept in the maintainer's private research log (W1 and W3 fresh confirmation,
2026-09-28; v3 is frozen in its Amendment 1).

Free. v2 (``scripts/aml_w1_key_resolution_v2.py``, whose scoring this reuses) failed on BEAM 100K
for two named reasons: W3's classifier read "call on April 21" (event) then "call April 22" (state)
as no change, and clusters stayed too wide (12 keys). v3 turns on
``state_history``'s ``event_updates`` and tunes the resolver's width rules (the generic-subject
guard and ``same_add_values``) on all 20 BEAM 100K conversations, which every earlier design has
already seen; the fresh conversations cut from BEAM 500K judge the frozen result once.

    python scripts/aml_w1_key_resolution_v3.py tune --data 100K.parquet --facts facts.jsonl --out tune.json
    python scripts/aml_w1_key_resolution_v3.py confirm --data fresh.parquet --facts fresh.jsonl --out confirm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_w1_key_resolution_v2 import V1_CHOSEN, arm  # noqa: E402
from aml_w1_link_test import _rows  # noqa: E402

from recall_aml.key_resolution import ResolverConfig  # noqa: E402

V2_CHOSEN = ResolverConfig(0.67, 1.0, True, 1, True)
GRID_V3 = [
    ResolverConfig(ts, 1.0, True, guard, True, addvals, specific_containment=True)
    for guard in (1, 2)
    for addvals in (False, True)
    for ts in (0.5, 0.67)
]
#: The width bar the confirmation gates on; tuning selects only among configurations meeting it.
MAX_CLUSTER = 8
#: Frozen by `tune` on BEAM 100K and recorded in Amendment 1 before `confirm` reads fresh facts.
V3: ResolverConfig | None = None


def select(tuning: Sequence[tuple[ResolverConfig, dict[str, Any]]]) -> tuple[ResolverConfig, bool]:
    """Highest current-value recall among configurations whose largest cluster meets the bar;
    ties to fewer merged keys. The flag says whether any configuration met the bar (if none did,
    the narrowest is taken and the result must say so)."""
    eligible = [(c, r) for c, r in tuning if r["largest_cluster"] <= MAX_CLUSTER]
    if not eligible:
        return min(tuning, key=lambda item: (item[1]["largest_cluster"], item[1]["keys_merged"]))[0], False
    best = max(eligible, key=lambda item: (item[1]["current_value_recall_given_coverage"], -item[1]["keys_merged"]))
    return best[0], True


def _summary(config: ResolverConfig | None, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": config.name if config else "exact",
        "current_value_recall_given_coverage": result["current_value_recall_given_coverage"],
        "link_given_coverage": result["pairs"].get("knowledge_update", {}).get("link_recall_given_coverage"),
        "distractor_rate": result["distractor"]["rate"], "largest_cluster": result["largest_cluster"],
        "keys_merged": result["keys_merged"], "w3": result["w3"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tune", "confirm"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _rows(args.data)
    records = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines() if line.strip()]
    every = list(range(len(rows)))
    if args.mode == "tune":
        tuning = [(c, arm(rows, records, c, every, event_updates=True)) for c in GRID_V3]
        chosen, met = select(tuning)
        out: dict[str, Any] = {"chosen": chosen.name, "width_bar_met": met,
                               "tuning": [_summary(c, r) for c, r in tuning]}
    else:
        if V3 is None:
            raise SystemExit("V3 is not frozen: run `tune`, record it in Amendment 1, set V3, commit")
        out = {"v3": V3.name, "arms": {
            "exact_keys": arm(rows, records, None, every),
            "v1_chosen": arm(rows, records, V1_CHOSEN, every),
            "v2_chosen": arm(rows, records, V2_CHOSEN, every),
            "v3": arm(rows, records, V3, every, event_updates=True),
        }, "secondary": {
            "v1_chosen_event_updates": arm(rows, records, V1_CHOSEN, every, event_updates=True),
            "v2_chosen_event_updates": arm(rows, records, V2_CHOSEN, every, event_updates=True),
            "exact_keys_event_updates": arm(rows, records, None, every, event_updates=True),
        }}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "tuning" and k != "arms" and k != "secondary"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
