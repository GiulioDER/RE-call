"""Tests for `git_clone_race_guard.py`, driven the way the hook is actually driven.

Each test builds a REAL git repository in a temp directory, feeds the hook a real PreToolUse
payload on stdin, and reads the decision off stdout. Nothing here fakes git: the defects this file
was written for are all about what `git rev-parse` returns for a particular kind of ref, and a fake
that answers on git's behalf would have agreed with whatever the guard assumed.

⛔ **The bug this file exists for: a TAG IS NOT A BRANCH, and the guard denied every tag push.**
The guard refuses `git push <remote> <name>` when `<name>` is not the currently checked out branch.
That is right for a branch and wrong for a tag: pushing `v1.2.3` from `master` is how every release
is made, and there is no working tree the tag could disagree with. Measured 2026-08-22 while
releasing v0.9.8, where it fired twice and both overrides were mine.

The second defect is in the same code path and made the refusal read as evidence: the message
printed `git rev-parse --short <name>`, which for an ANNOTATED tag is the hash of the tag OBJECT,
and compared it against HEAD's COMMIT hash. Those can never be equal for an annotated tag, so the
message showed a mismatch that did not exist.

Run: `python scripts/git_clone_race_guard_tests.py`. CI runs it in the `session-hooks` job, which
deliberately does not `pip install` this package: these hooks import nothing from `recall` and must
keep working in a checkout where it is absent, because that is how `~/.claude/hooks/` runs them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GUARD = Path(__file__).resolve().with_name("git_clone_race_guard.py")
DEPLOYED = Path(os.path.expanduser("~/.claude/hooks/git_clone_race_guard.py"))

_failures: list[str] = []
_ran = 0


def _git(args: list[str], cwd: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _repo(tmp: str) -> str:
    """A repository with one commit on `master`, and a committer git will accept."""
    _git(["init", "-q", "-b", "master"], tmp)
    _git(["config", "user.email", "t@example.com"], tmp)
    _git(["config", "user.name", "test"], tmp)
    # ⚠️ Signing off, deliberately: the machine running these tests signs commits by default and a
    # test that needs a key is a test that fails for a reason unrelated to the guard.
    _git(["config", "commit.gpgsign", "false"], tmp)
    _git(["config", "tag.gpgsign", "false"], tmp)
    Path(tmp, "f.txt").write_text("x", encoding="utf-8")
    _git(["add", "f.txt"], tmp)
    _git(["commit", "-qm", "one"], tmp)
    return tmp


def _run(command: str, cwd: str) -> dict:
    """Feed the hook a payload and return its parsed decision, or `{}` for "allowed"."""
    payload = {"tool_input": {"command": command}, "cwd": cwd}
    out = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if not out.stdout.strip():
        return {}
    return json.loads(out.stdout)


def _decision(result: dict) -> str:
    return (
        result.get("hookSpecificOutput", {}).get("permissionDecision", "allow")
        if result
        else "allow"
    )


def _reason(result: dict) -> str:
    return result.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


def check(name: str, condition: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{(': ' + detail) if detail else ''}")
        _failures.append(name)


# ── the regression ───────────────────────────────────────────────────────────────────────────


def test_pushing_an_annotated_tag_from_another_branch_is_allowed() -> None:
    """⛔ The release case. This is red against the guard as it stood.

    An annotated tag is what `git tag -s`/`git tag -a` produces, and it is what every signed
    release in these repositories uses.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["tag", "-a", "v1.2.3", "-m", "release"], cwd)
        result = _run("git push origin v1.2.3", cwd)
        check(
            "annotated tag push is allowed",
            _decision(result) == "allow",
            f"denied with: {_reason(result)[:160]}",
        )


def test_pushing_a_lightweight_tag_from_another_branch_is_allowed() -> None:
    """The other kind of tag. Kept separate because it exercises a different object shape.

    ⚠️ A lightweight tag points straight at the commit, so `rev-parse` returns the same hash as
    HEAD and the guard's comparison happened to pass. A test written only against this kind would
    have gone green before the fix and pinned nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["tag", "v1.2.4"], cwd)
        result = _run("git push origin v1.2.4", cwd)
        check(
            "lightweight tag push is allowed",
            _decision(result) == "allow",
            f"denied with: {_reason(result)[:160]}",
        )


def test_pushing_the_tags_flag_is_allowed() -> None:
    """`git push origin --tags` names no branch at all."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["tag", "-a", "v1.2.5", "-m", "release"], cwd)
        result = _run("git push origin --tags", cwd)
        check("--tags is allowed", _decision(result) == "allow", _reason(result)[:160])


# ── what must STILL be refused, or the fix has removed the guard ─────────────────────────────


def test_pushing_another_branch_is_still_refused() -> None:
    """⛔ The behaviour the guard exists for. If this goes green-to-allow, the fix broke it."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["branch", "feature/other"], cwd)
        result = _run("git push origin feature/other", cwd)
        check(
            "a different branch is still refused",
            _decision(result) == "deny",
            f"decision was {_decision(result)}",
        )


def test_a_name_that_is_both_a_tag_and_a_branch_is_still_refused() -> None:
    """Ambiguous, so the guard stays conservative rather than reading it as a tag.

    git itself refuses an ambiguous `push origin <name>`, and a guard that waved it through on the
    strength of the tag alone would be asserting something git does not agree with.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["branch", "ambiguous"], cwd)
        _git(["tag", "-a", "ambiguous", "-m", "confusing"], cwd)
        result = _run("git push origin ambiguous", cwd)
        check(
            "an ambiguous name is still refused",
            _decision(result) == "deny",
            f"decision was {_decision(result)}",
        )


def test_committing_on_the_default_branch_is_still_refused() -> None:
    """The other half of the guard, untouched by this fix and asserted so it stays that way."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        result = _run("git commit -m x", cwd)
        check(
            "commit on the default branch is still refused",
            _decision(result) == "deny",
            f"decision was {_decision(result)}",
        )


def test_the_escape_hatch_still_works() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["branch", "feature/other"], cwd)
        result = _run("git push origin feature/other # RACE_GUARD_OK", cwd)
        check("the escape hatch still allows", _decision(result) == "allow")


# ── the message, which is what the reader acts on ────────────────────────────────────────────


def test_the_refusal_prints_comparable_hashes() -> None:
    """⚠️ The second defect: the message compared a TAG OBJECT hash against a COMMIT hash.

    Only reachable now through the ambiguous case, which is exactly why that case is worth having:
    it is the one path where a tag can still reach the refusal, and the message must not claim a
    mismatch that is an artefact of dereferencing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _repo(tmp)
        _git(["branch", "ambiguous"], cwd)
        _git(["tag", "-a", "ambiguous", "-m", "confusing"], cwd)
        head = _git(["rev-parse", "--short", "HEAD"], cwd)
        result = _run("git push origin ambiguous", cwd)
        reason = _reason(result)
        tag_object = _git(["rev-parse", "--short", "refs/tags/ambiguous"], cwd)
        check(
            "the refusal does not quote the tag object hash",
            tag_object not in reason or tag_object == head,
            f"quoted the tag object {tag_object} rather than the commit {head}",
        )
        check(
            "the refusal quotes the commit the ref resolves to",
            head in reason,
            f"expected {head} in: {reason[:200]}",
        )


def test_the_deployed_copy_matches_this_source() -> None:
    """⛔ A guard edited here and never redeployed is a guard that is not running.

    This one had NO source at all until 2026-08-22: it existed only in `~/.claude/hooks/`, with no
    version control and no tests, while denying commits and pushes in every repository on the
    machine. The source now lives here, which only helps if the two stay the same file.

    ⚠️ **Three outcomes, not two, because a CI runner deploys no hooks at all.** Failing there
    would make this file unrunnable in CI, and dropping the check entirely would lose it on the
    machines where it means something. So the hooks DIRECTORY decides: no directory means this is
    not a machine that deploys hooks, which is reported as a SKIP naming the reason. A directory
    that exists with this file missing IS a failure, because that machine deploys hooks and is
    missing this one.

    A skip must never read as a pass, so the skip prints its own line either way.
    """
    if not DEPLOYED.parent.is_dir():
        print(f"  skip the deployed copy matches this source: no hooks directory at "
              f"{DEPLOYED.parent}, so nothing is deployed on this machine")
        return
    if not DEPLOYED.exists():
        check(
            "the deployed copy matches this source",
            False,
            f"{DEPLOYED.parent} exists but this guard is not in it; deploy with: "
            f"cp {GUARD} {DEPLOYED}",
        )
        return
    same = DEPLOYED.read_text(encoding="utf-8") == GUARD.read_text(encoding="utf-8")
    check(
        "the deployed copy matches this source",
        same,
        "" if same else f"redeploy: cp {GUARD} {DEPLOYED}",
    )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"git_clone_race_guard: {len(tests)} tests\n")
    for test in tests:
        print(test.__name__)
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - a crashing test is a failing test
            print(f"  FAIL {test.__name__} raised {type(exc).__name__}: {exc}")
            _failures.append(test.__name__)
    print(f"\n{_ran - len(_failures)}/{_ran} checks passed")
    if _failures:
        print("failed: " + ", ".join(_failures))
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    raise SystemExit(main())
