"""Independently recompute MemEye arm aggregates and verify immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARMS = ("MM0_caption", "MM1_preserve", "MM2_dual")


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def recompute(payload: dict[str, Any]) -> dict[str, float | int]:
    questions = payload["questions"]
    rotations = [rotation for question in questions for rotation in question["rotations"]]
    return {
        "question_count": len(questions),
        "rotation_count": len(rotations),
        "mean_debiased_em": _mean(
            _mean(float(rotation["em"]) for rotation in question["rotations"])
            for question in questions
        ),
        "strict_question_accuracy": _mean(
            float(all(float(rotation["em"]) == 1.0 for rotation in question["rotations"]))
            for question in questions
        ),
        "valid_choice_rate": _mean(float(rotation["valid_choice"]) for rotation in rotations),
        "any_recall_at_10": _mean(
            float(rotation["retrieval"]["any_recall_at_10"]) for rotation in rotations
        ),
        "any_recall_at_100": _mean(
            float(rotation["retrieval"]["any_recall_at_100"]) for rotation in rotations
        ),
        "complete_recall_at_10": _mean(
            float(rotation["retrieval"]["complete_recall_at_10"]) for rotation in rotations
        ),
        "clue_fraction_at_10": _mean(
            float(rotation["retrieval"]["clue_fraction_at_10"]) for rotation in rotations
        ),
        "mrr": _mean(float(rotation["retrieval"]["mrr"]) for rotation in rotations),
    }


def audit(result_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    checked: dict[str, Any] = {}
    selection_path = result_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for arm in ARMS:
        path = result_dir / f"{arm}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if selection.get("arm_sha256", {}).get(arm) != digest:
            failures.append(f"{arm} hash does not match selection")
        payload = json.loads(path.read_text(encoding="utf-8"))
        measured = recompute(payload)
        checked[arm] = measured
        aggregate = payload.get("aggregate", {})
        for key, value in measured.items():
            actual = aggregate.get(key)
            if isinstance(value, float):
                if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), value, rel_tol=0, abs_tol=1e-12
                ):
                    failures.append(f"{arm} aggregate mismatch for {key}")
            elif actual != value:
                failures.append(f"{arm} aggregate mismatch for {key}")
        if payload.get("complete") is not True:
            failures.append(f"{arm} is not complete")
        if payload.get("cleanup", {}).get("passed") is not True:
            failures.append(f"{arm} cleanup is not verified")
    return {"passed": not failures, "failures": failures, "recomputed": checked}


def write_sha256s(result_dir: Path) -> None:
    files = sorted(
        path for path in result_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in files]
    (result_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.result_dir)
    (args.result_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_sha256s(args.result_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
