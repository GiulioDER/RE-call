"""Choose the smallest preregistered evidence budget within one point of best coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_budget(coverage: dict[int, float]) -> int:
    if set(coverage) != {5_000, 7_000, 9_000}:
        raise ValueError("coverage must contain exactly the 5000, 7000, and 9000 character arms")
    if any(value < 0 or value > 1 for value in coverage.values()):
        raise ValueError("coverage values must be proportions in [0, 1]")
    threshold = max(coverage.values()) - 0.01
    return min(budget for budget, value in coverage.items() if value >= threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    coverage = {int(key): float(value) for key, value in raw.items()}
    print(json.dumps({"selected_context_chars": select_budget(coverage)}))


if __name__ == "__main__":
    main()
