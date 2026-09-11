"""Secret-safe preparation and launch preflight for RE-call Hosted 1.0."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Literal


Phase = Literal["prepare", "launch"]

ALWAYS_REQUIRED = (
    "RECALL_AML_GIT_COMMIT",
    "OPENROUTER_API_KEY",
    "VOYAGE_API_KEY",
)
ADMISSION_REQUIRED = (
    "RECALL_AML_DATABASE_URL",
    "RECALL_AML_API_KEY",
)
PLACEHOLDERS = frozenset({"", "CHANGE_ME", "ADMISSION_VALUE_REQUIRED"})


def _present(value: str | None) -> bool:
    return bool(value and value.strip() not in PLACEHOLDERS)


def inspect_environment(env: Mapping[str, str], phase: Phase) -> dict[str, object]:
    """Return readiness states without returning any configuration value."""
    if phase not in ("prepare", "launch"):
        raise ValueError(f"unsupported preflight phase: {phase}")

    checks: dict[str, dict[str, object]] = {}
    for name in ALWAYS_REQUIRED + ADMISSION_REQUIRED:
        value = env.get(name)
        present = _present(value)
        required = name in ALWAYS_REQUIRED or phase == "launch"
        if present:
            state = "present"
        elif required:
            state = "missing"
        else:
            state = "pending_admission"
        checks[name] = {"required": required, "state": state}

    return {
        "phase": phase,
        "ready": all(
            check["state"] != "missing" for check in checks.values()
        ),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "launch"), required=True)
    args = parser.parse_args()
    result = inspect_environment(os.environ, args.phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
