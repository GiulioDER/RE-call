#!/usr/bin/env python3
"""Break the shipped write-time hook seven ways and watch `tests/test_write_time_hook.py` go red.

    python scripts/write_time_prod_mutations.py

This hook runs on every tool call of every session, so its guards are the difference between a
memory feature and a tax on the client. Each mutation is a plausible wrong version: a cooldown that
never engages, a cooldown that never expires, a hook that denies, a hook that reads the wrong field.

The file is mutated IN PLACE and restored in a `finally`, because the tests import the installed
package and pointing them at a copy would silently test the original. The original bytes are held
in memory for the whole run and written back even on a keyboard interrupt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "recall_hooks" / "write_time.py"
TESTS = REPO_ROOT / "tests" / "test_write_time_hook.py"

MUTATIONS = [
    (
        "the cooldown never engages, so an unreachable corpus costs ~3s on EVERY tool call",
        "        _start_cooldown(options[\"cooldown_seconds\"])",
        "        pass",
    ),
    (
        "the cooldown is never checked, so it exists but does nothing",
        "    if _in_cooldown():\n        return 0",
        "    if False:\n        return 0",
    ),
    (
        "the cooldown never expires, so one outage disables the feature forever",
        "        return time.time() < float(raw)",
        "        return True",
    ),
    (
        "a success never clears the cooldown",
        "    _clear_cooldown()",
        "    pass",
    ),
    (
        "additionalContext becomes a deny decision",
        '            "additionalContext": render(hits),',
        '            "permissionDecision": "deny",',
    ),
    (
        "the min_chars early return is dropped, so `ls` pays a database round trip",
        "    if len(query) < options[\"min_chars\"]:\n        return 0",
        "    if False:\n        return 0",
    ),
    (
        "`enabled: false` is ignored",
        "    if not options[\"enabled\"]:\n        return 0",
        "    if False:\n        return 0",
    ),
]


def run_tests() -> tuple[int, str]:
    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    survivors: list[str] = []
    try:
        code, output = run_tests()
        if code != 0:
            print("BASELINE IS RED; mutation results would be meaningless.\n")
            print(output[-2000:])
            return 1
        print("baseline: green\n")

        for label, find, replace in MUTATIONS:
            if find not in original:
                print(f"  SKIP      {label}\n            anchor not found; the mutation is stale")
                survivors.append(label)
                continue
            TARGET.write_text(original.replace(find, replace, 1), encoding="utf-8", newline="\n")
            code, output = run_tests()
            if code == 0:
                print(f"  SURVIVED  {label}")
                survivors.append(label)
            else:
                reds = [line.split("::")[-1].split()[0]
                        for line in output.splitlines() if line.startswith("FAILED")]
                print(f"  killed    {label}")
                for red in dict.fromkeys(reds) or ["(see output)"]:
                    print(f"              red: {red}")
    finally:
        # Restored whatever happened. A mutation left on disk is a shipped defect.
        TARGET.write_text(original, encoding="utf-8", newline="\n")

    print()
    if survivors:
        print(f"{len(survivors)} MUTATION(S) SURVIVED; the tests are not evidence for these:")
        for label in survivors:
            print(f"  - {label}")
        return 1
    print(f"all {len(MUTATIONS)} mutations killed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
