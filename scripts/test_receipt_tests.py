"""Regression tests for the test receipt.

The defect class this guards against is a receipt that says something reassuring without having
observed anything, so the tests that matter are the ones asserting it stays SILENT or reports
ABSENCE. A receipt that always prints "tests passed" would satisfy a naive suite.

Every test names what it pins. State goes to a temp HOME so no real session's receipt is read or
written.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent / "test_receipt.py")
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def load(tmp_state: str):
    spec = importlib.util.spec_from_file_location("receipt", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.STATE_DIR = tmp_state
    return m


REAL_PYTEST_TAIL = """
tests/test_store.py ....................                                 [ 84%]
tests/test_index.py .....s..s...                                         [100%]

============ 411 passed, 22 skipped, 3 warnings in 705.19s (0:11:45) ============
"""

NO_DB_TAIL = """
=========== 199 passed, 514 skipped, 1 warning in 313.02s (0:05:13) ============
"""

FAILING_TAIL = """
=================== 2 failed, 409 passed, 22 skipped in 700.10s ================
"""


def test_it_records_only_test_commands():
    """Pins the scope: an ordinary command must not create a receipt row."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        for cmd in ["git status", "ls tests/", "python -m ruff check .",
                    'git commit -m "run pytest later"', "head -5 tests/conftest.py"]:
            m.record({"session_id": "s1", "tool_input": {"command": cmd},
                      "tool_response": {"stdout": REAL_PYTEST_TAIL}})
        rows = m.read_rows("s1")
        check("it records only test commands", rows == [], f"rows={rows}")


def test_it_recognises_the_shapes_that_run_tests():
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        shapes = ["python -m pytest tests/ -q", "pytest tests/test_x.py",
                  "python scripts/session_mcp_approve_tests.py",
                  "timeout 900 python -m pytest tests/",
                  "RECALL_TEST_DSN=x python -m pytest tests/"]
        missed = [s for s in shapes if not m.is_test_run(s)]
        check("it recognises the shapes that run tests", not missed, f"missed={missed}")


def test_counts_come_from_the_summary_line_only():
    """Pins the lying-receipt bug: numbers scraped from arbitrary output are not a verdict."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        noisy = ("some test printed: 9999 passed in a log line\n"
                 "E   AssertionError: expected 5 passed\n" + REAL_PYTEST_TAIL)
        counts = m.parse_counts(noisy)
        check("counts come from the summary line only",
              counts == {"passed": 411, "skipped": 22}, f"counts={counts}")


def test_a_run_with_no_summary_line_is_not_a_pass():
    """A crashed or interrupted run must not be reported as green."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        m.record({"session_id": "s1", "tool_input": {"command": "python -m pytest tests/"},
                  "tool_response": {"stdout": "Killed\n", "exit_code": 137}})
        msg = m.describe(m.read_rows("s1"), time.time())
        check("a run with no summary line is not a pass",
              "no readable summary line" in msg and "not a pass" in msg, f"msg={msg[:160]}")


def test_absence_is_reported_as_absence():
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        msg = m.describe(m.read_rows("never-ran"), time.time())
        check("absence is reported as absence",
              "no test run was recorded" in msg, f"msg={msg[:120]}")


def test_the_false_green_skip_signature_is_called_out():
    """The documented failure: 514 skips means the DB tests never ran."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        m.record({"session_id": "s1", "tool_input": {"command": "python -m pytest tests/ -q"},
                  "tool_response": {"stdout": NO_DB_TAIL}})
        msg = m.describe(m.read_rows("s1"), time.time())
        check("the false-green skip signature is called out",
              "false-green" in msg and "514" in msg, f"msg={msg[:200]}")


def test_control_a_healthy_run_is_not_alarmed_about():
    """CONTROL: warning on every run would make the warning worthless."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        m.record({"session_id": "s1", "tool_input": {"command": "python -m pytest tests/ -q"},
                  "tool_response": {"stdout": REAL_PYTEST_TAIL}})
        msg = m.describe(m.read_rows("s1"), time.time())
        check("CONTROL a healthy run is not alarmed about",
              "false-green" not in msg and "411 passed" in msg.replace(", ", ", "),
              f"msg={msg[:200]}")


def test_failures_are_stated_plainly():
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        m.record({"session_id": "s1", "tool_input": {"command": "python -m pytest tests/"},
                  "tool_response": {"stdout": FAILING_TAIL}})
        msg = m.describe(m.read_rows("s1"), time.time())
        check("failures are stated plainly",
              "NOT green" in msg, f"msg={msg[:200]}")


def test_another_sessions_receipt_is_not_cited():
    """A run you did not start is not your evidence."""
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        m.record({"session_id": "other", "tool_input": {"command": "python -m pytest tests/"},
                  "tool_response": {"stdout": REAL_PYTEST_TAIL}})
        msg = m.describe(m.read_rows("mine"), time.time())
        check("another session's receipt is not cited",
              "no test run was recorded" in msg, f"msg={msg[:120]}")


def test_push_detection():
    with tempfile.TemporaryDirectory() as td:
        m = load(td)
        pushes = ["git push", "git push origin HEAD", "git push --force-with-lease origin b",
                  "cd /tmp && git push"]
        not_pushes = ["git status", 'git commit -m "push later"', "git pushd", "echo git push"]
        bad = [c for c in pushes if not m.is_push(c)] + [c for c in not_pushes if m.is_push(c)]
        check("push detection", not bad, f"wrong={bad}")


def test_it_prints_nothing_for_a_non_push():
    """The hook runs on every Bash call. Silence on irrelevant commands is the requirement."""
    import subprocess
    payload = json.dumps({"session_id": "s1", "tool_input": {"command": "ls -la"}})
    p = subprocess.run([sys.executable, SCRIPT], input=payload,
                       capture_output=True, text=True, timeout=60)
    check("it prints nothing for a non-push",
          p.returncode == 0 and p.stdout.strip() == "",
          f"rc={p.returncode} out={p.stdout[:120]}")


def test_it_fails_open_on_junk():
    import subprocess
    for junk in ["not json", "[]", "{}"]:
        p = subprocess.run([sys.executable, SCRIPT], input=junk,
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            check("it fails open on junk", False, f"junk={junk!r} rc={p.returncode}")
            return
    check("it fails open on junk", True)


def test_the_deployed_copy_matches_this_source():
    """A hook edited here and never redeployed is a hook that is not running."""
    deployed = Path(os.path.expanduser("~/.claude/hooks/test_receipt.py"))
    if not deployed.exists():
        check("the deployed copy matches this source", False, f"not deployed at {deployed}")
        return
    same = deployed.read_text(encoding="utf-8") == Path(SCRIPT).read_text(encoding="utf-8")
    check("the deployed copy matches this source", same,
          "" if same else f"redeploy: cp {SCRIPT} {deployed}")


if __name__ == "__main__":
    for fn in (test_it_records_only_test_commands,
               test_it_recognises_the_shapes_that_run_tests,
               test_counts_come_from_the_summary_line_only,
               test_a_run_with_no_summary_line_is_not_a_pass,
               test_absence_is_reported_as_absence,
               test_the_false_green_skip_signature_is_called_out,
               test_control_a_healthy_run_is_not_alarmed_about,
               test_failures_are_stated_plainly,
               test_another_sessions_receipt_is_not_cited,
               test_push_detection,
               test_it_prints_nothing_for_a_non_push,
               test_it_fails_open_on_junk,
               test_the_deployed_copy_matches_this_source):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
