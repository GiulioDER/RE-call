#!/usr/bin/env python3
"""Break the config-dir isolation four ways and watch the named test go red.

    python scripts/agent_ab_config_dir_mutations.py

The isolation boundary is the component every other result in this lane depends on, so its tests
have to be evidence rather than decoration. This copies `claude_exec.py`, mutates the copy, runs
the suite against it, and reports any survivor. It never touches the original.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "benchmarks" / "agent_ab" / "claude_exec.py"
TESTS = REPO_ROOT / "scripts" / "agent_ab_config_dir_tests.py"

MUTATIONS = [
    (
        "the bare + config_dir refusal is removed",
        "        if self.config_dir is not None and self.bare:",
        "        if False:",
    ),
    (
        "bare defaults to False, silently changing every existing arm",
        "    bare: bool = True",
        "    bare: bool = False",
    ),
    (
        "CLAUDE_CONFIG_DIR is never set, so the field is inert",
        '        environment["CLAUDE_CONFIG_DIR"] = str(config.config_dir)',
        "        pass",
    ),
    (
        "the isolation is set BEFORE config.env, so a caller can override it",
        '    environment = dict(os.environ)\n'
        '    environment.update({str(key): str(value) for key, value in config.env.items()})\n'
        '    if config.config_dir is not None:',
        '    environment = dict(os.environ)\n'
        '    if config.config_dir is not None:',
    ),
]


def run_suite(directory: Path) -> tuple[int, str]:
    """Run the COPIED test file, so it imports the COPIED package.

    ⛔ Running the original test file cannot work, and it silently reports every mutation as
    surviving: the test does `sys.path.insert(0, REPO_ROOT)` computed from its OWN location, which
    beats `PYTHONPATH` and re-imports the real module. The first version of this runner did that
    and printed "4 MUTATIONS SURVIVED" — a mutation that never ran against the mutant looks exactly
    like a mutation the tests failed to catch. Copying the test into the mirror makes
    `parents[1]` resolve to the temp root instead.
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
        # Mirror the package so `benchmarks.agent_ab.claude_exec` resolves from the copy first.
        # The WHOLE package, not just the file under mutation: `benchmarks/agent_ab/__init__.py`
        # imports `.runner`, so a partial mirror fails to import and every mutation would read as
        # a red baseline. Caught by the baseline check below, which is why it exists.
        (root / "benchmarks").mkdir()
        (root / "benchmarks" / "__init__.py").write_text("", encoding="utf-8")
        package = root / "benchmarks" / "agent_ab"
        shutil.copytree(REPO_ROOT / "benchmarks" / "agent_ab", package,
                        ignore=shutil.ignore_patterns("__pycache__", "workspaces", "artifacts"))
        # The test computes REPO_ROOT from its own path, so it has to live in the mirror too.
        (root / "scripts").mkdir()
        shutil.copy(TESTS, root / "scripts" / TESTS.name)

        shutil.copy(TARGET, package / TARGET.name)
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
            (package / TARGET.name).write_text(
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
                for red in dict.fromkeys(reds):
                    print(f"              red: {red}")

        print()
        if survivors:
            print(f"{len(survivors)} MUTATION(S) SURVIVED; the test file's table is not evidence:")
            for label in survivors:
                print(f"  - {label}")
            return 1
        print(f"all {len(MUTATIONS)} mutations killed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
