"""Score Stage 2 of the MM-1/MM-3 pre-registration: the registered contrasts, with paired CIs.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md``.
Reads the Stage 2 JSONL (``scripts/aml_mm_scope_stage2.py``) and the MemEye cache. A question's
score in an arm is its debiased exact match: the mean over its four option rotations. Every
contrast is paired by question, with a 10,000-resample bootstrap at seed 20260925, as registered.

Usage::

    python scripts/aml_mm_scope_score.py --answers S2.jsonl --cache-dir DIR --out SCORE.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aml_mm_scope_report import load_questions  # noqa: E402

RESAMPLES = 10_000
SEED = 20260925
ROTATIONS = 4


def paired(a: dict[Any, float], b: dict[Any, float], keys: list[Any]) -> dict[str, Any]:
    """Mean of a minus b over ``keys``, with a percentile bootstrap over questions."""
    if not keys:
        return {"n": 0}
    diffs = [a[key] - b[key] for key in keys]
    rng = random.Random(SEED)
    boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(RESAMPLES))
    return {
        "n": len(keys),
        "a": mean(a[key] for key in keys),
        "b": mean(b[key] for key in keys),
        "diff": mean(diffs),
        "ci95": [boots[int(0.025 * RESAMPLES)], boots[int(0.975 * RESAMPLES) - 1]],
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


def score(answers: Path, cache_dir: Path) -> dict[str, Any]:
    questions = load_questions(cache_dir)
    rotations: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    routes: dict[tuple[str, str], str] = {}
    spend = 0.0
    with answers.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row["scenario"], row["question_id"])
            rotations[(*key, row["arm"])][row["rotation"]] = row
            routes[key] = row["route"]
            spend += float(row.get("cost_usd") or 0.0)

    by_arm: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    valid: dict[str, list[bool]] = defaultdict(list)
    incomplete: dict[str, int] = defaultdict(int)
    for (scenario, question_id, arm), rows in rotations.items():
        for row in rows.values():
            valid[arm].append(bool(row["valid"]))
        if len(rows) != ROTATIONS:
            incomplete[arm] += 1
            continue
        by_arm[arm][(scenario, question_id)] = mean(float(row["em"]) for row in rows.values())

    def common(*arms: str, where=lambda key: True) -> list[tuple[str, str]]:
        keys = set.intersection(*(set(by_arm[arm]) for arm in arms))
        return sorted(key for key in keys if where(key))

    off_route = lambda key: routes[key] != "multimodal"  # noqa: E731
    fine = lambda key: bool({"X3", "X4"} & set(questions[key]["axes"]))  # noqa: E731
    coarse = lambda key: bool({"X1", "X2"} & set(questions[key]["axes"])) and not fine(key)  # noqa: E731
    y3 = lambda key: "Y3" in questions[key]["axes"]  # noqa: E731

    d_minus_b_fine = paired(by_arm["D"], by_arm["B"], common("D", "B", where=lambda k: off_route(k) and fine(k)))
    d_minus_b_coarse = paired(by_arm["D"], by_arm["B"], common("D", "B", where=lambda k: off_route(k) and coarse(k)))
    result = {
        "contrasts": {
            "D_minus_B_off_route": paired(by_arm["D"], by_arm["B"], common("D", "B", where=off_route)),
            "P_minus_B_off_route": paired(by_arm["P"], by_arm["B"], common("P", "B", where=off_route)),
            "P_minus_D_off_route": paired(by_arm["P"], by_arm["D"], common("P", "D", where=off_route)),
            "B2_minus_B_off_route": paired(by_arm["B2"], by_arm["B"], common("B2", "B", where=off_route)),
            "D_minus_B_off_route_X3_X4": d_minus_b_fine,
            "D_minus_B_off_route_X1_X2_only": d_minus_b_coarse,
            "Dt_minus_D_Y3": paired(by_arm["Dt"], by_arm["D"], common("Dt", "D", where=y3)),
            "Dt_minus_D_all": paired(by_arm["Dt"], by_arm["D"], common("Dt", "D")),
        },
        "mechanism_gap_fine_minus_coarse": (
            d_minus_b_fine.get("diff", 0.0) - d_minus_b_coarse.get("diff", 0.0)
            if d_minus_b_fine.get("n") and d_minus_b_coarse.get("n")
            else None
        ),
        "check_6_valid_choice_rate": {arm: mean(values) for arm, values in sorted(valid.items())},
        "incomplete_questions": dict(incomplete),
        "questions_per_arm": {arm: len(values) for arm, values in sorted(by_arm.items())},
        "spend_usd": round(spend, 4),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.answers, args.cache_dir)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
