#!/usr/bin/env python3
"""Break the isolation guard four ways and watch the named test go red.

    python scripts/agent_ab_isolation_mutations.py

The guard is the only thing standing between a hooked run and a silently contaminated one, so its
tests have to be evidence rather than decoration. Each mutation is a plausible wrong version, not a
syntax error: a string-prefix comparison, a warning instead of a refusal, an inverted test, and a
probe that asks about only one of the two leaks.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "scripts" / "agent_ab_isolation.py"
TESTS = REPO_ROOT / "scripts" / "agent_ab_isolation_tests.py"

MUTATIONS = [
    (
        "a string-prefix comparison, which refuses an isolated sibling",
        "    return not (candidate == profile or profile in candidate.parents)",
        "    return not str(candidate).startswith(str(profile))",
    ),
    (
        "the refusal becomes a warning, so the run proceeds contaminated",
        "    raise SystemExit(",
        "    print(",
    ),
    (
        "the isolation test is inverted",
        "    if is_outside_user_profile(root):\n        return",
        "    if not is_outside_user_profile(root):\n        return",
    ),
    (
        "the probe stops asking about project memory",
        '    "USER is yes if your context contains a document with the heading \'User-level notes\' "',
        '    "USER is yes if your context contains nothing at all. PROJECT_SKIPPED "',
    ),
]


def run_suite(directory: Path) -> tuple[int, str]:
    """Run the COPIED test file so it imports the COPIED module.

    The test computes its own path and inserts `scripts/` on `sys.path`, which beats `PYTHONPATH`,
    so pointing the original at a mutant silently tests the real module and reports every mutation
    as surviving.
    """

    import os

    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, str(directory / "scripts" / TESTS.name)],
        capture_output=True, text=True, cwd=str(directory), timeout=180,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scripts").mkdir()
        shutil.copy(TESTS, root / "scripts" / TESTS.name)
        shutil.copy(TARGET, root / "scripts" / TARGET.name)

        code, output = run_suite(root)
        if code != 0:
            print("BASELINE IS RED against the copy; mutation results would be meaningless.\n")
            print(output[-2500:])
            return 1
        print("baseline: green\n")

        survivors = []
        for label, find, replace in MUTATIONS:
            if find not in original:
                print(f"  SKIP      {label}\n            anchor not found; the mutation is stale")
                survivors.append(label)
                continue
            (root / "scripts" / TARGET.name).write_text(
                original.replace(find, replace, 1), encoding="utf-8", newline="\n"
            )
            code, output = run_suite(root)
            if code == 0:
                print(f"  SURVIVED  {label}")
                survivors.append(label)
            else:
                reds = [ln.strip()[5:].split(":", 1)[0].strip()
                        for ln in output.splitlines() if ln.strip().startswith("FAIL ")]
                print(f"  killed    {label}")
                for red in dict.fromkeys(reds) or ["RAISED"]:
                    print(f"              red: {red}")

        print()
        if survivors:
            print(f"{len(survivors)} MUTATION(S) SURVIVED; the test file's claim is not evidence:")
            for label in survivors:
                print(f"  - {label}")
            return 1
        print(f"all {len(MUTATIONS)} mutations killed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
