"""Summarize the frozen strict spare-slot rule on private development features."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.source_conditioning import (  # noqa: E402
    STRICT_SPARE_SLOT_CROSS_LEG_FLOOR,
    STRICT_SPARE_SLOT_POLICY,
    STRICT_SPARE_SLOT_SOURCE_MARGIN_FLOOR,
    STRICT_SPARE_SLOT_SPARSE_MAX_RANK,
    strict_source_candidate_eligible,
)


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("decision") != "READY_FOR_POLICY_FIT":
        raise ValueError("development feature artifact is not ready for policy fit")

    accepted_gold = 0
    accepted_answerable_non_gold = 0
    accepted_controls = 0
    later_gold_not_considered = 0
    for row in payload["rows"]:
        additions = list(row["additions"])
        later_gold_not_considered += sum(
            bool(addition["is_gold_source"]) for addition in additions[1:]
        )
        if not additions:
            continue
        first = additions[0]
        if not strict_source_candidate_eligible(first, first["features"]):
            continue
        if row["expected_answerability"] == "unanswerable":
            accepted_controls += 1
        elif first["is_gold_source"]:
            accepted_gold += 1
        else:
            accepted_answerable_non_gold += 1

    accepted_total = accepted_gold + accepted_answerable_non_gold + accepted_controls
    return {
        "protocol": "2026-09-14-guarded-spare-slot-strict-development",
        "measured_at": datetime.now(UTC).isoformat(),
        "policy": STRICT_SPARE_SLOT_POLICY,
        "rule": {
            "proposal_index": 1,
            "lane": "dual_leg",
            "sparse_rank_max": STRICT_SPARE_SLOT_SPARSE_MAX_RANK,
            "source_margin_min": STRICT_SPARE_SLOT_SOURCE_MARGIN_FLOOR,
            "cross_leg_fraction_min": STRICT_SPARE_SLOT_CROSS_LEG_FLOOR,
        },
        "parity_rows": int(payload["parity_rows"]),
        "selection_parity_mismatches": int(payload["selection_parity_mismatches"]),
        "observed_gold_additions_all_positions": int(payload["gold_source_additions"]),
        "accepted_total": accepted_total,
        "accepted_gold": accepted_gold,
        "accepted_answerable_non_gold": accepted_answerable_non_gold,
        "accepted_controls": accepted_controls,
        "later_gold_not_considered": later_gold_not_considered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(json.loads(args.features.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
