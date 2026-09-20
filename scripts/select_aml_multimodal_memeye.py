"""Apply the frozen MemEye Brand admission selector to three arm artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("MM0_caption", "MM1_preserve", "MM2_dual")
EXPERIMENT_COST_CEILING_USD = 32.0
IDENTITY_KEYS = (
    "dataset_revision",
    "dataset_json",
    "dataset_sha256",
    "memeye_commit",
    "prompt_path",
    "prompt_sha256",
    "answer_model",
    "answer_temperature",
    "answer_token_budget",
    "top_k",
)


def decide(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if set(arms) != set(ARMS):
        reasons.append("the three registered arms are not present exactly once")
    for name in ARMS:
        payload = arms.get(name, {})
        if payload.get("arm") != name:
            reasons.append(f"{name} artifact identity mismatch")
        if payload.get("complete") is not True:
            reasons.append(f"{name} is incomplete")
        if payload.get("cleanup", {}).get("passed") is not True:
            reasons.append(f"{name} cleanup failed")
        if payload.get("add", {}).get("count") != 72:
            reasons.append(f"{name} did not add the registered 72 rounds")
        if payload.get("add", {}).get("replay_passed") is not True:
            reasons.append(f"{name} idempotent replay failed")
        aggregate = payload.get("aggregate", {})
        if aggregate.get("question_count") != 29:
            reasons.append(f"{name} did not score the registered 29 questions")
        if aggregate.get("rotation_count") != 116:
            reasons.append(f"{name} did not score the registered 116 rotations")
        cost = payload.get("provider_spend_usd")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or float(cost) < 0
        ):
            reasons.append(f"{name} provider cost is missing or invalid")
    if all(name in arms for name in ARMS):
        baseline = arms[ARMS[0]].get("identity", {})
        for name in ARMS[1:]:
            identity = arms[name].get("identity", {})
            for key in IDENTITY_KEYS:
                if identity.get(key) != baseline.get(key):
                    reasons.append(f"{name} differs on frozen identity field {key}")
        costs = [arms[name].get("provider_spend_usd") for name in ARMS]
        numeric_costs = [
            float(cost)
            for cost in costs
            if isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and float(cost) >= 0
        ]
        if len(numeric_costs) == len(ARMS):
            total_spend = sum(numeric_costs)
            if total_spend > EXPERIMENT_COST_CEILING_USD:
                reasons.append("the experiment exceeded the registered provider cost ceiling")
    if reasons:
        return {"verdict": "INVALID", "reasons": reasons, "conditions": {}}

    mm0 = arms["MM0_caption"]["aggregate"]
    mm1 = arms["MM1_preserve"]["aggregate"]
    mm2 = arms["MM2_dual"]["aggregate"]
    conditions = {
        "preservation_answer_gain": (
            float(mm1["mean_debiased_em"]) - float(mm0["mean_debiased_em"]) >= 0.05
        ),
        "dual_retrieval_gain": (
            float(mm2["any_recall_at_10"]) - float(mm1["any_recall_at_10"]) >= 0.05
        ),
        "dual_answer_gain": (
            float(mm2["mean_debiased_em"]) - float(mm1["mean_debiased_em"]) >= 0.02
        ),
        "dual_recall_100_noninferior": (
            float(mm2["any_recall_at_100"]) >= float(mm1["any_recall_at_100"])
        ),
        "no_contract_regression": True,
    }
    if all(conditions.values()):
        verdict = "ADMIT_FULL_MEMEYE"
    elif (
        conditions["dual_retrieval_gain"]
        and conditions["dual_recall_100_noninferior"]
        and conditions["no_contract_regression"]
    ):
        verdict = "RETRIEVAL_ONLY"
    elif (
        conditions["preservation_answer_gain"]
        and conditions["no_contract_regression"]
        and not conditions["dual_retrieval_gain"]
    ):
        verdict = "PRESERVATION_ONLY"
    else:
        verdict = "NO_GAIN"
    deltas = {
        "mm1_minus_mm0_mean_debiased_em": float(mm1["mean_debiased_em"])
        - float(mm0["mean_debiased_em"]),
        "mm2_minus_mm1_any_recall_at_10": float(mm2["any_recall_at_10"])
        - float(mm1["any_recall_at_10"]),
        "mm2_minus_mm1_mean_debiased_em": float(mm2["mean_debiased_em"])
        - float(mm1["mean_debiased_em"]),
        "mm2_minus_mm1_any_recall_at_100": float(mm2["any_recall_at_100"])
        - float(mm1["any_recall_at_100"]),
    }
    return {"verdict": verdict, "reasons": [], "conditions": conditions, "deltas": deltas}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        name: json.loads((args.result_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in ARMS
    }
    selection = decide(arms)
    selection["arm_sha256"] = {
        name: _sha256(args.result_dir / f"{name}.json") for name in ARMS
    }
    target = args.result_dir / "selection.json"
    target.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))
    if selection["verdict"] == "INVALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
