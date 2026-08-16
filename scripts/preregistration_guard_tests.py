"""Regression tests for the pre-registration guard.

Every test names the defect it pins. **Half this file is ALLOW cases, deliberately.** A guard that
denies everything would pass a file of deny tests, and the more expensive failure here is not a
measurement that slips through: it is blocking `head` on a benchmark script, or a commit message
that mentions one. That guard gets switched off within the hour, and a switched-off guard protects
nothing.

Each case builds a real throwaway git repository and runs the real hook over stdin, because the
guard's answer depends on `git status`, and a stubbed git would be testing the stub.

Test commits pass `-c commit.gpgsign=false`: these are disposable repositories in a temp
directory, and the machine's global signing config would otherwise fail them for a reason that has
nothing to do with what is being tested. It does not weaken the signing rule on any real branch.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = str(HERE / "preregistration_guard.py")
DEPLOYED = Path(os.path.expanduser("~/.claude/hooks/preregistration_guard.py"))

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, timeout=60)


def make_repo(dirty: bool) -> Path:
    root = Path(tempfile.mkdtemp(prefix="prereg-guard-"))
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    prereg = root / "docs" / "preregistrations"
    prereg.mkdir(parents=True)
    (prereg / "2026-01-01-example.md").write_text("# committed prediction\n", encoding="utf-8")
    (root / "README.md").write_text("x\n", encoding="utf-8")
    git(root, "add", "docs/preregistrations/2026-01-01-example.md", "README.md")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    if dirty:
        # The violation this guard exists for: a prediction written but not committed.
        (prereg / "2026-01-02-unregistered.md").write_text("# not committed\n", encoding="utf-8")
    return root


def run(cmd: str, root: Path):
    payload = json.dumps({"tool_input": {"command": cmd}, "cwd": str(root)})
    p = subprocess.run([sys.executable, GUARD], input=payload,
                       capture_output=True, text=True, timeout=90)
    out = {}
    if p.stdout.strip():
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            out = {"_unparseable": p.stdout}
    decision = out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")
    return decision, out, p


def denies(cmd, root):
    return run(cmd, root)[0] == "deny"


# ---------------------------------------------------------------- deny cases


def test_a_measurement_with_an_uncommitted_prereg_is_denied():
    root = make_repo(dirty=True)
    d, out, _ = run("bash scripts/run_locomo_arms.sh", root)
    reason = out.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")
    check("a measurement with an uncommitted prereg is denied",
          d == "deny" and "2026-01-02-unregistered.md" in reason,
          f"decision={d} reason={reason[:120]}")
    shutil.rmtree(root, ignore_errors=True)


def test_every_launch_shape_is_denied():
    """The shapes that actually appear in this tree. Missing one is a silent hole."""
    root = make_repo(dirty=True)
    shapes = [
        "python -m recall.eval.locomo --arm a",
        "python scripts/ablate_store_latency_guards.py",
        "./scripts/run_peps_arms.sh",
        "nohup python scripts/score_pairs.py &",
        "timeout 600 bash scripts/run_gap_parallel.sh",
        "python -m recall.cli calibration calibrate --generation G --publish",
        "python -m pytest benchmarks/test_x.py",
        "cd /tmp && python -m benchmarks.analyze",
    ]
    missed = [s for s in shapes if not denies(s, root)]
    check("every launch shape is denied", not missed, f"allowed through: {missed}")
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------- allow cases


def test_control_the_same_command_is_allowed_when_the_prereg_is_committed():
    """THE control. Without it, a guard that denies unconditionally passes every deny test."""
    root = make_repo(dirty=False)
    d, out, _ = run("bash scripts/run_locomo_arms.sh", root)
    msg = out.get("systemMessage", "")
    check("CONTROL the same command is allowed when the prereg is committed",
          d == "allow" and "clean" in msg,
          f"decision={d} msg={msg[:120]}")
    shutil.rmtree(root, ignore_errors=True)


def test_ordinary_work_is_never_touched():
    """The commands a session runs all day. One false positive here kills the guard."""
    root = make_repo(dirty=True)
    ordinary = [
        "python -m pytest tests/ -q",
        "python -m ruff check .",
        "git status --short",
        "python scripts/session_mcp_approve_tests.py",
        "head -40 scripts/run_locomo_arms.sh",
        "cat scripts/ablate_store_latency_guards.py",
        "grep -rn benchmark docs/",
        "ls benchmarks/",
        "git log --oneline -- benchmarks/",
    ]
    blocked = [c for c in ordinary if denies(c, root)]
    check("ordinary work is never touched", not blocked, f"wrongly denied: {blocked}")
    shutil.rmtree(root, ignore_errors=True)


def test_a_commit_message_naming_a_benchmark_is_data():
    """Pins the classic over-block: writing ABOUT the thing is not running it."""
    root = make_repo(dirty=True)
    cmds = [
        'git commit -m "fix run_gap_parallel.sh arg handling"',
        'echo "run_locomo_arms.sh is the entry point"',
    ]
    blocked = [c for c in cmds if denies(c, root)]
    check("a commit message naming a benchmark is data", not blocked, f"wrongly denied: {blocked}")
    shutil.rmtree(root, ignore_errors=True)


def test_a_quoted_argument_is_not_a_measurement_target():
    """Pins what quote-stripping is actually FOR.

    Found by a surviving mutant: removing the quote-stripping broke nothing in the two obvious
    cases, because `git commit -m "..."` and `echo "..."` never reach the argument scan at all
    (their command word is not an interpreter). The place it matters is an interpreter's own
    arguments, where shlex hands back the quoted text as an ordinary token and a `-k` filter
    naming a benchmark would be read as running one.
    """
    root = make_repo(dirty=True)
    cmds = [
        'python -m pytest tests/ -q -k "not run_locomo_arms"',
        'python scripts/session_mcp_approve_tests.py --label "benchmark rerun"',
    ]
    blocked = [c for c in cmds if denies(c, root)]
    check("a quoted argument is not a measurement target", not blocked,
          f"wrongly denied: {blocked}")
    shutil.rmtree(root, ignore_errors=True)


def test_a_heredoc_body_is_data():
    root = make_repo(dirty=True)
    cmd = "cat >> notes.md <<'EOF'\npython -m recall.eval.locomo is how you run it\nEOF"
    check("a heredoc body is data", not denies(cmd, root))
    shutil.rmtree(root, ignore_errors=True)


def test_the_escape_hatch_works_and_is_reported():
    root = make_repo(dirty=True)
    d, out, _ = run("bash scripts/run_locomo_arms.sh ALLOW_UNREGISTERED_MEASUREMENT", root)
    check("the escape hatch works and is reported",
          d == "allow" and "ALLOWED via" in out.get("systemMessage", ""),
          f"decision={d} msg={out.get('systemMessage', '')[:120]}")
    shutil.rmtree(root, ignore_errors=True)


def test_a_quoted_escape_does_not_disarm_the_guard():
    """Pins the bug the sibling guard had: a message must not switch the guard off."""
    root = make_repo(dirty=True)
    cmd = 'bash scripts/run_locomo_arms.sh -m "ALLOW_UNREGISTERED_MEASUREMENT"'
    check("a quoted escape does not disarm the guard", denies(cmd, root))
    shutil.rmtree(root, ignore_errors=True)


def test_it_fails_open():
    root = make_repo(dirty=True)
    cases = {
        "unparseable payload": "not json at all",
        "empty object": "{}",
        "null tool_input": json.dumps({"tool_input": None, "cwd": str(root)}),
    }
    bad = []
    for name, payload in cases.items():
        p = subprocess.run([sys.executable, GUARD], input=payload,
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0 or '"deny"' in p.stdout:
            bad.append(f"{name}: rc={p.returncode} out={p.stdout[:80]}")
    check("it fails open", not bad, "; ".join(bad))
    shutil.rmtree(root, ignore_errors=True)


def test_outside_a_repository_it_allows():
    outside = Path(tempfile.mkdtemp(prefix="prereg-guard-norepo-"))
    d, _, _ = run("bash scripts/run_locomo_arms.sh", outside)
    check("outside a repository it allows", d == "allow", f"decision={d}")
    shutil.rmtree(outside, ignore_errors=True)


def test_a_repo_with_no_preregistrations_allows():
    """A project that does not use them must not be blocked by a rule it never adopted."""
    root = Path(tempfile.mkdtemp(prefix="prereg-guard-bare-"))
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    (root / "README.md").write_text("x\n", encoding="utf-8")
    git(root, "add", "README.md")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    d, _, _ = run("bash scripts/run_locomo_arms.sh", root)
    check("a repo with no preregistrations allows", d == "allow", f"decision={d}")
    shutil.rmtree(root, ignore_errors=True)


def test_an_unrelated_dirty_file_does_not_block():
    """Only the pre-registration paths count. Any-dirty-file would block all real work."""
    root = make_repo(dirty=False)
    (root / "recall_src.py").write_text("x = 1\n", encoding="utf-8")
    d, _, _ = run("bash scripts/run_locomo_arms.sh", root)
    check("an unrelated dirty file does not block", d == "allow", f"decision={d}")
    shutil.rmtree(root, ignore_errors=True)


def test_the_deployed_copy_matches_this_source():
    """The SessionStart hook's deployed copy has already drifted 57 lines from its source.

    A guard edited in the repo and never redeployed is a guard that is not running, and nothing
    else in the system reports that.
    """
    if not DEPLOYED.exists():
        check("the deployed copy matches this source", False, f"not deployed at {DEPLOYED}")
        return
    same = DEPLOYED.read_text(encoding="utf-8") == Path(GUARD).read_text(encoding="utf-8")
    check("the deployed copy matches this source", same,
          "" if same else f"redeploy: cp {GUARD} {DEPLOYED}")


if __name__ == "__main__":
    for fn in (test_a_measurement_with_an_uncommitted_prereg_is_denied,
               test_every_launch_shape_is_denied,
               test_control_the_same_command_is_allowed_when_the_prereg_is_committed,
               test_ordinary_work_is_never_touched,
               test_a_commit_message_naming_a_benchmark_is_data,
               test_a_quoted_argument_is_not_a_measurement_target,
               test_a_heredoc_body_is_data,
               test_the_escape_hatch_works_and_is_reported,
               test_a_quoted_escape_does_not_disarm_the_guard,
               test_it_fails_open,
               test_outside_a_repository_it_allows,
               test_a_repo_with_no_preregistrations_allows,
               test_an_unrelated_dirty_file_does_not_block,
               test_the_deployed_copy_matches_this_source):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
