#!/usr/bin/env python3
"""Break the config-dir isolation and the arm guard eight ways, and watch the named test go red.

    python scripts/agent_ab_config_dir_mutations.py

Two targets, because two different things decide whether a run means anything: the isolation that
keeps the arms comparable, and the guard that decides whether an A/B with identical arms is the
registered design or an experiment that measures nothing. This copies both files, mutates the
copies, runs the suite against them, and reports any survivor. It never touches the originals.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "benchmarks" / "agent_ab" / "claude_exec.py"
ARMS = REPO_ROOT / "benchmarks" / "agent_ab" / "arms.py"
TESTS = REPO_ROOT / "scripts" / "agent_ab_config_dir_tests.py"

#: (target, label, find, replace). Each is a plausible wrong version of the real thing, not a
#: syntax error.
MUTATIONS = [
    (
        TARGET,
        "the bare + config_dir refusal is removed",
        "        if self.config_dir is not None and self.bare:",
        "        if False:",
    ),
    (
        TARGET,
        "bare defaults to False, silently changing every existing arm",
        "    bare: bool = True",
        "    bare: bool = False",
    ),
    (
        TARGET,
        "CLAUDE_CONFIG_DIR is never set, so the field is inert",
        '        environment["CLAUDE_CONFIG_DIR"] = str(config.config_dir)',
        "        pass",
    ),
    (
        TARGET,
        "the isolation is set BEFORE config.env, so a caller can override it",
        '    environment = dict(os.environ)\n'
        '    environment.update({str(key): str(value) for key, value in config.env.items()})\n'
        '    if config.config_dir is not None:',
        '    environment = dict(os.environ)\n'
        '    if config.config_dir is not None:',
    ),
    (
        ARMS,
        "an empty reason turns the arm guard off, so a boolean would do",
        "        if not identical_arms.strip():",
        "        if False:",
    ),
    (
        ARMS,
        "identical arms are allowed with no config_dirs, leaving nothing different at all",
        "        if config_dirs is None:\n            raise ValueError(\n"
        '                "identical_arms without config_dirs leaves NOTHING different between the arms. "',
        "        if False:\n            raise ValueError(\n"
        '                "identical_arms without config_dirs leaves NOTHING different between the arms. "',
    ),
    (
        ARMS,
        "the OFF-arm profile guard is dropped for everyone, not just a stated reason",
        "    if identical_arms is None and specs[RECALL_OFF].profile not in OFF_ARM_PROFILES:",
        "    if False:",
    ),
    (
        ARMS,
        "the hook is installed in BOTH arms, which erases the treatment",
        "        if script is not None and variant == RECALL_ON:",
        "        if script is not None:",
    ),
]


def run_suite(directory: Path) -> tuple[int, str]:
    """Run the COPIED test file, so it imports the COPIED package.

    Running the original cannot work and fails silently: the test does `sys.path.insert(0, ...)`
    computed from its OWN location, which beats `PYTHONPATH` and re-imports the real module. The
    first version of this runner did that and printed "4 MUTATIONS SURVIVED", because a mutation
    that never ran against the mutant looks exactly like one the tests failed to catch.
    """

    import os

    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, str(directory / "scripts" / TESTS.name)],
        capture_output=True, text=True, cwd=str(directory), timeout=300,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    originals = {path: path.read_text(encoding="utf-8") for path in (TARGET, ARMS)}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Mirror the WHOLE package: `benchmarks/agent_ab/__init__.py` imports `.runner`, so a
        # partial mirror fails to import and every mutation would read as a red baseline. Caught
        # by the baseline check below, which is why it exists.
        (root / "benchmarks").mkdir()
        (root / "benchmarks" / "__init__.py").write_text("", encoding="utf-8")
        package = root / "benchmarks" / "agent_ab"
        shutil.copytree(REPO_ROOT / "benchmarks" / "agent_ab", package,
                        ignore=shutil.ignore_patterns("__pycache__", "workspaces", "artifacts"))
        # The test computes REPO_ROOT from its own path, so it has to live in the mirror too.
        (root / "scripts").mkdir()
        shutil.copy(TESTS, root / "scripts" / TESTS.name)

        code, output = run_suite(root)
        if code != 0:
            print("BASELINE IS RED against the copy; mutation results would be meaningless.\n")
            print(output[-2500:])
            return 1
        print("baseline: green\n")

        survivors = []
        for target, label, find, replace in MUTATIONS:
            source = originals[target]
            if find not in source:
                print(f"  SKIP      {label}\n            anchor not found; the mutation is stale")
                survivors.append(label)
                continue
            mutant = package / target.name
            mutant.write_text(source.replace(find, replace, 1), encoding="utf-8", newline="\n")
            code, output = run_suite(root)
            # Restored before the next mutation, or they compound and a later one is measured
            # against a file already broken in some other way.
            mutant.write_text(source, encoding="utf-8", newline="\n")
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
            print(f"{len(survivors)} MUTATION(S) SURVIVED; the test file's table is not evidence:")
            for label in survivors:
                print(f"  - {label}")
            return 1
        print(f"all {len(MUTATIONS)} mutations killed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
