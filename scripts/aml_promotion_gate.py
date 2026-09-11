"""Mechanical Full quota gate for preregistered AML Hosted result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide(payload: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required_true = (
        "contract_passed",
        "isolation_passed",
        "persistence_passed",
        "idempotency_passed",
        "release_frozen",
        "external_https_ready",
        "stable_30_days_committed",
        "paired_identity_passed",
    )
    for key in required_true:
        if payload.get(key) is not True:
            failures.append(f"{key} is not true")
    if int(payload.get("net_successful_cells", -1)) < 8:
        failures.append("net_successful_cells is below 8")
    if int(payload.get("task_count", 0)) != 34:
        failures.append("task_count is not the registered 34 task population")
    if int(payload.get("paired_cells", 0)) != 102:
        failures.append("paired_cells is not the registered 102 cell population")
    if payload.get("seeds") != [0, 1, 2]:
        failures.append("seeds are not the registered 0, 1, and 2 sequence")
    if int(payload.get("invalid_cells", 1)) != 0:
        failures.append("invalid_cells is not zero")
    if int(payload.get("new_control_failures", 999)) > 2:
        failures.append("new_control_failures exceeds 2")
    if int(payload.get("add_concurrency", 0)) < 16 or int(payload.get("add_errors", 1)) != 0:
        failures.append("Add did not sustain 16 concurrent requests without errors")
    if int(payload.get("search_concurrency", 0)) < 16 or int(payload.get("search_errors", 1)) != 0:
        failures.append("Search did not sustain 16 concurrent requests without errors")
    if float(payload.get("add_p95_seconds", 999)) >= 30:
        failures.append("Add p95 is not below 30 seconds")
    if float(payload.get("search_p95_seconds", 999)) >= 5:
        failures.append("Search p95 is not below 5 seconds")
    if float(payload.get("provider_spend_usd", 999)) > 450:
        failures.append("provider_spend_usd exceeds the 450 dollar promotion ceiling")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    promoted, failures = decide(payload)
    print(json.dumps({"promote_to_full": promoted, "failures": failures}, indent=2))
    if not promoted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
