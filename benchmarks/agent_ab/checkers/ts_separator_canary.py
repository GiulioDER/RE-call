"""Is the new test a guard, or a test whose separator was flattened on the way into the file?

Writing a literal U+2028 into a source file through an editing tool is unreliable: the same pasted
character survived in one test and was silently turned into an ordinary space in two others written
minutes later. The consequence is worse than a broken test, because a flattened separator makes the
test pass against code that no longer handles the separator at all. Nothing in the pytest output
looks wrong. `chr(0x2028)` is pure ASCII in the file and cannot be normalised.

So the checker does what the memo says the fixture should do for itself: it **mutates the thing the
test claims to pin and watches the test go red.** `recall/normalise.py` is replaced with a copy that
differs only by dropping U+2028 from the collapse set, and the suite is run again in a sibling
directory.

Both runs are required and they are required in opposite directions:

- green as written, or the agent broke the file;
- **red under the mutation**, or the test does not pin what it says it pins.

The existing tests do not touch U+2028, so they stay green under the mutation. A failure in the
second run therefore comes from the test the agent added, and from nothing else.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_bounded
import sys

TESTS = "tests/test_normalise.py"
MODULE = "recall/normalise.py"


def _pytest(cwd: Path) -> tuple[bool, str]:
    result = run_bounded(
        [sys.executable, "-m", "pytest", TESTS, "-q", "-p", "no:cacheprovider"],
        cwd=cwd,
        timeout_s=180,
    )
    return (result.returncode == 0 and not result.timed_out), (result.stdout + result.stderr)[-700:]


def check(workdir: Path) -> CheckResult:
    from ..sandbox import oracle

    tests = workdir / TESTS
    if not tests.is_file():
        return CheckResult(False, f"{TESTS} is missing")
    source = tests.read_text(encoding="utf-8", errors="replace")

    green, green_output = _pytest(workdir)
    detail: dict[str, object] = {
        "green_as_written": green,
        "green_output": green_output,
        "mentions_chr_2028": "0x2028" in source or "\\u2028" in source.lower(),
        "contains_literal_2028": chr(0x2028) in source,
        "test_count_in_file": source.count("def test_"),
    }
    if not green:
        return CheckResult(False, "the test file does not pass as written", detail)

    # A sibling copy, so the agent's tree is left exactly as the session ended it.
    mutant_dir = workdir.parent / f"{workdir.name}__mutant"
    if mutant_dir.exists():
        shutil.rmtree(mutant_dir, ignore_errors=True)
    shutil.copytree(workdir, mutant_dir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copyfile(oracle("ts-separator-canary") / "normalise_broken.py", mutant_dir / MODULE)

    survived, mutant_output = _pytest(mutant_dir)
    detail.update({"passed_under_mutation": survived, "mutant_output": mutant_output})
    if survived:
        return CheckResult(
            False,
            "the suite still passes with U+2028 handling removed: the test pins nothing",
            detail,
        )
    return CheckResult(True, "green as written and red under the mutation", detail)
