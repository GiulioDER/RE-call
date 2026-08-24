"""Score preregistered selective-depth thresholds after non-gold feature capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["question_id"])] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    questions = _rows(args.questions)
    baseline = _rows(args.baseline)
    features = json.loads(args.features.read_text(encoding="utf-8"))["rows"]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    feature_names = ["max_dense_score", "eighth_hit_dense_score"]

    def metrics(chosen: dict[str, set[str]]) -> dict[str, float]:
        recalls = []
        exact = []
        extras = []
        for row in features:
            expected = {str(value) for value in questions[row["question_id"]].get("expected_doc_ids", [])}
            selected = chosen[row["question_id"]]
            recalls.append(len(expected & selected) / len(expected) if expected else 0.0)
            exact.append(float(bool(expected) and expected <= selected))
            extras.append(float(len(selected - expected)))
        return {
            "document_recall": sum(recalls) / len(recalls),
            "exact_coverage": sum(exact) / len(exact),
            "mean_extra_docs": sum(extras) / len(extras),
        }

    baseline_sets = {
        question_id: {str(value) for value in row.get("document_ids", [])}
        for question_id, row in baseline.items()
    }
    base = metrics(baseline_sets)
    candidates = []
    for feature in feature_names:
        for threshold in thresholds:
            chosen = {
                row["question_id"]: (
                    {str(value) for value in row["k12_document_ids"]}
                    if row.get(feature) is not None and row[feature] < threshold
                    else {str(value) for value in row["k8_document_ids"]}
                )
                for row in features
            }
            measured = metrics(chosen)
            candidates.append(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "expanded_questions": sum(
                        row.get(feature) is not None and row[feature] < threshold for row in features
                    ),
                    **measured,
                    "delta_recall_pp": (measured["document_recall"] - base["document_recall"]) * 100,
                    "delta_exact_pp": (measured["exact_coverage"] - base["exact_coverage"]) * 100,
                    "delta_extra_docs": measured["mean_extra_docs"] - base["mean_extra_docs"],
                }
            )
    passing = [
        row
        for row in candidates
        if row["delta_recall_pp"] >= 1.0
        and row["delta_exact_pp"] >= 0.0
        and row["delta_extra_docs"] <= 2.0
    ]
    selected = max(
        passing,
        key=lambda row: (
            row["delta_recall_pp"],
            -row["delta_extra_docs"],
            -row["threshold"],
        ),
    ) if passing else None
    report = {
        "phase": "development_posthoc_threshold_screen",
        "baseline": base,
        "candidates": candidates,
        "passing_candidates": passing,
        "selected": selected,
        "selection_rule": "highest development recall among candidates passing the preregistered gate",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": base, "selected": selected}, indent=2))
    print(f"wrote selective-depth screen to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
