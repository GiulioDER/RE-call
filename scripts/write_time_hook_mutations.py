#!/usr/bin/env python3
"""Break the write-time hook six ways and watch the named test go red.

    python scripts/write_time_hook_mutations.py

"A guard nobody has watched fail has not been tested." This applies each mutation to a COPY of the
hook, runs the suite against that copy, and reports which tests died. It restores nothing because
it never touches the original: the copy lives in a temp directory and the suite is pointed at it
through `sys.path`.

The output is the evidence for the mutation table in the test file's own docstring. If a mutation
survives, that table is a lie and the row must be removed or the test strengthened.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "write_time_recall_hook.py"
TESTS = REPO_ROOT / "scripts" / "write_time_recall_hook_tests.py"

#: (label, find, replace). Each is a plausible wrong version of the real thing, not a syntax error.
MUTATIONS = [
    (
        "payload_of returns the command for a Write",
        '    if tool_name in WRITE_TOOLS:\n'
        '        return str(tool_input.get("content") or tool_input.get("new_string") or "")',
        '    if tool_name in WRITE_TOOLS:\n'
        '        return str(tool_input.get("command") or "")',
    ),
    (
        "MIN_QUERY_CHARS ignored, so it fires on any length",
        "    if len(query) < MIN_QUERY_CHARS:\n        return 0",
        "    if False:\n        return 0",
    ),
    (
        "the unconfigured-DSN early return is removed",
        '    if not dsn:\n',
        '    if False:\n',
    ),
    (
        "the retrieval try/except is removed",
        "    try:\n        hits = search(query, dsn)\n"
        "    except Exception as error:  # noqa: BLE001 - a retrieval failure must not break "
        "the session\n"
        '        trace({"tool": tool_name, "chars": len(query), '
        '"error": f"{type(error).__name__}: {error}"})\n'
        "        return 0",
        "    hits = search(query, dsn)",
    ),
    (
        "additionalContext swapped for a deny decision",
        '            "additionalContext": render(hits),',
        '            "permissionDecision": "deny",\n'
        '            "permissionDecisionReason": render(hits),',
    ),
    (
        "trace lets OSError escape",
        "    except OSError:\n        pass",
        "    except ValueError:\n        pass",
    ),
]


def run_suite(directory: Path) -> tuple[int, str]:
    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, str(directory / TESTS.name)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(directory), "PYTHONUTF8": "1"},
    )
    return result.returncode, result.stdout + result.stderr


def failed_tests(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAIL "):
            names.append(stripped.split(":", 1)[0][5:].strip())
        elif "Traceback" in stripped:
            names.append("RAISED")
    return names


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        shutil.copy(TESTS, directory / TESTS.name)
        original = HOOK.read_text(encoding="utf-8")

        shutil.copy(HOOK, directory / HOOK.name)
        code, output = run_suite(directory)
        if code != 0:
            print("BASELINE IS RED; mutation results would be meaningless.\n")
            print(output[-2000:])
            return 1
        print("baseline: green\n")

        survivors = []
        for label, find, replace in MUTATIONS:
            if find not in original:
                print(f"  SKIP  {label}\n        anchor not found; the mutation is stale")
                survivors.append(label)
                continue
            (directory / HOOK.name).write_text(
                original.replace(find, replace, 1), encoding="utf-8", newline="\n"
            )
            code, output = run_suite(directory)
            names = failed_tests(output)
            if code == 0:
                print(f"  SURVIVED  {label}")
                survivors.append(label)
            else:
                print(f"  killed    {label}")
                for name in dict.fromkeys(names):
                    print(f"              red: {name}")

        print()
        if survivors:
            print(f"{len(survivors)} MUTATION(S) SURVIVED. The docstring's table is not evidence "
                  "until every row is killed:")
            for label in survivors:
                print(f"  - {label}")
            return 1
        print(f"all {len(MUTATIONS)} mutations killed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
