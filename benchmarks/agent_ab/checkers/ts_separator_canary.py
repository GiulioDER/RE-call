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

Prior work: the existing executable task-success framework in
`benchmarks/agent_ab/tasksuccess.py` and its separator-normalization mutation task.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_bounded

TESTS = "tests/test_normalise.py"
MODULE = "recall/normalise.py"


@dataclass(frozen=True)
class PytestRun:
    passed: bool
    behavioral_red: bool
    nodes: frozenset[tuple[str, str]]
    output: str


def _pytest(cwd: Path) -> PytestRun:
    with tempfile.TemporaryDirectory(prefix="recall-canary-") as directory:
        report = Path(directory) / "pytest.xml"
        result = run_bounded(
            [sys.executable, "-m", "pytest", TESTS, "-q", "-p", "no:cacheprovider",
             f"--junitxml={report}"],
            cwd=cwd,
            timeout_s=180,
        )
        output = (result.stdout + result.stderr)[-700:]
        try:
            root = ET.parse(report).getroot()  # noqa: S314 - local pytest report
        except (OSError, ET.ParseError):
            return PytestRun(False, False, frozenset(), output)
    cases = list(root.iter("testcase"))
    nodes = frozenset((case.get("classname", ""), case.get("name", "")) for case in cases)
    failures = list(root.iter("failure"))
    valid = bool(cases) and len(nodes) == len(cases) and not list(root.iter("error")) and not list(root.iter("skipped"))
    assertions = all(
        failure.get("message", "") == "AssertionError"
        or failure.get("message", "").startswith(("assert ", "AssertionError:"))
        for failure in failures
    )
    return PytestRun(
        passed=valid and result.ok and not failures,
        behavioral_red=valid and result.returncode == 1 and not result.timed_out and bool(failures) and assertions,
        nodes=nodes,
        output=output,
    )


def check(workdir: Path) -> CheckResult:
    from ..sandbox import oracle

    tests = workdir / TESTS
    if not tests.is_file():
        return CheckResult(False, f"{TESTS} is missing")
    source = tests.read_text(encoding="utf-8", errors="replace")

    green = _pytest(workdir)
    detail: dict[str, object] = {
        "green_as_written": green.passed,
        "green_output": green.output,
        "mentions_chr_2028": "0x2028" in source or "\\u2028" in source.lower(),
        "contains_literal_2028": chr(0x2028) in source,
        "test_count_in_file": source.count("def test_"),
    }
    if not green.passed:
        return CheckResult(False, "the test file does not pass as written", detail)

    # A sibling copy, so the agent's tree is left exactly as the session ended it.
    mutant_dir = workdir.parent / f"{workdir.name}__mutant"
    if mutant_dir.exists():
        shutil.rmtree(mutant_dir, ignore_errors=True)
    shutil.copytree(workdir, mutant_dir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copyfile(oracle("ts-separator-canary") / "normalise_broken.py", mutant_dir / MODULE)

    mutant = _pytest(mutant_dir)
    detail.update({"passed_under_mutation": mutant.passed, "mutant_output": mutant.output})
    if mutant.passed:
        return CheckResult(
            False,
            "the suite still passes with U+2028 handling removed: the test pins nothing",
            detail,
        )
    if not mutant.behavioral_red or mutant.nodes != green.nodes:
        detail["checker_error"] = True
        return CheckResult(False, "mutation run is inconclusive: no matching behavioral assertion proof", detail)
    return CheckResult(True, "green as written and red under the mutation", detail)
