"""Summarize only downloaded official EnterpriseRAG leaderboard artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _display_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    pattern = re.compile(r'^([a-z0-9_]+):\s*\n\s+display_name:\s+"([^"]+)"', re.MULTILINE)
    for key, display in pattern.findall(path.read_text(encoding="utf-8")):
        names[key] = display
    return names


def _leaderboard(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected object")
    return payload


def analyze(leaderboard_path: Path, systems_path: Path, results_dir: Path) -> dict[str, Any]:
    display_names = _display_names(systems_path)
    board = _leaderboard(leaderboard_path)
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("results_*.json")):
        key = path.stem.removeprefix("results_")
        results[key] = _result(path)

    rows: list[dict[str, Any]] = []
    for board_row in board:
        display = str(board_row["model"])
        key = next((candidate for candidate, name in display_names.items() if name == display), None)
        payload = results.get(key or "")
        if payload is None:
            continue
        aggregate = payload.get("aggregate_stats", {})
        rows.append(
            {
                "key": key,
                "model": display,
                "score": float(board_row["overall_score"]),
                "correctness": float(board_row["correctness"]),
                "completeness": float(board_row["completeness"]),
                "recall": float(board_row["recall"]),
                "invalid_extra_docs": float(board_row["invalid_extra_docs"]),
                "aggregate_artifact": aggregate,
                "question_type_stats": payload.get("question_type_stats", {}),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    top_five = rows[:5]

    def dimension_ranks(category: str) -> dict[str, list[dict[str, Any]]]:
        dimensions = {
            "recall": ("average_recall_pct", True),
            "completeness": ("average_completeness_pct", True),
            "correctness": ("average_correctness_pct", True),
            "invalid_extra_docs": ("average_invalid_extra_docs", False),
        }
        ranked: dict[str, list[dict[str, Any]]] = {}
        for dimension, (field, descending) in dimensions.items():
            values = []
            for row in rows:
                category_row = row["question_type_stats"].get(category)
                if isinstance(category_row, dict) and field in category_row:
                    values.append({"model": row["model"], "value": category_row[field]})
            ranked[dimension] = sorted(values, key=lambda item: float(item["value"]), reverse=descending)
        return ranked

    categories = sorted(
        {
            category
            for row in rows
            for category in row["question_type_stats"]
        }
    )
    return {
        "phase": "official_downloaded_artifact_analysis",
        "leaderboard": str(leaderboard_path),
        "systems": str(systems_path),
        "results_dir": str(results_dir),
        "official_rows_loaded": len(rows),
        "top_five": top_five,
        "category_dimension_ranks": {category: dimension_ranks(category) for category in categories},
        "inference_boundary": "These artifacts show outcome associations only. They do not document the systems' internal retrieval or reranking mechanisms.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", required=True, type=Path)
    parser.add_argument("--systems", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.leaderboard, args.systems, args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"loaded {report['official_rows_loaded']} official artifact rows")
    print(f"wrote competitor analysis to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
