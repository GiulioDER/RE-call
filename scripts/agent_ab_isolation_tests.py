#!/usr/bin/env python3
"""Tests for the sandbox isolation guard.

The guard decides whether a hooked run may proceed. It exists because the failure it prevents is
SILENT: a run whose sandboxes sit under the user profile completes normally, produces artifacts
indistinguishable from a clean run, and has quietly fed this machine's `CLAUDE.md` into every
session of both arms. Measured 2026-08-27: ~3,000 input tokens clean against 47,676 to 66,167
leaked.

The path logic is pure, so these need no API, no database and no CLI. What they cannot cover is
whether a clean path really yields a clean session, which is why `verify_isolation` runs a real
session at run time and is not stubbed away here.

Mutation-tested 2026-08-27 by `scripts/agent_ab_isolation_mutations.py`, four ways, all four
killed. A surviving mutation means this file is not evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_ab_isolation import (  # noqa: E402
    ISOLATION_PROMPT,
    assert_sandbox_isolated,
    is_outside_user_profile,
    user_profile_root,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def test_a_path_under_the_profile_is_not_isolated() -> None:
    """The repository itself lives under the profile here, which is exactly the case that shipped
    a contaminated run, so it is the case asserted."""

    profile = user_profile_root()
    check("the profile itself is not outside itself", not is_outside_user_profile(profile))
    check("a child of the profile is not outside",
          not is_outside_user_profile(profile / "Documents" / "recall"))
    check("a deep descendant is not outside",
          not is_outside_user_profile(profile / "a" / "b" / "c" / "d"))


def test_a_path_elsewhere_is_isolated() -> None:
    root = Path(user_profile_root().anchor or "C:/")
    check("a root-level directory is outside", is_outside_user_profile(root / "recall-ab-sandbox"))


def test_a_sibling_with_a_shared_prefix_is_not_confused_for_a_child() -> None:
    """`C:\\Users\\gde00-backup` starts with the profile's STRING but is not inside it. A prefix
    comparison would refuse a perfectly isolated directory, and the cost of that is somebody
    working around the guard."""

    profile = user_profile_root()
    sibling = profile.parent / (profile.name + "-backup")
    check("a name-prefix sibling is outside", is_outside_user_profile(sibling),
          f"{sibling} was called inside {profile}")


def test_the_guard_raises_rather_than_warning() -> None:
    """A warning would be read after the spend, which is the same as not having a guard."""

    try:
        assert_sandbox_isolated(user_profile_root() / "Documents" / "recall" / "work")
    except SystemExit as error:
        message = str(error)
        check("it raises SystemExit", True)
        check("the message says what to pass", "--work-root" in message, message[:120])
        check("the message says why", "walking up from cwd" in message, message[:200])
    else:
        check("it raises SystemExit", False, "a run inside the profile was allowed")

    root = Path(user_profile_root().anchor or "C:/") / "recall-ab-sandbox"
    try:
        assert_sandbox_isolated(root)
    except SystemExit as error:
        check("an isolated root is allowed", False, str(error)[:120])
    else:
        check("an isolated root is allowed", True)


def test_the_probe_asks_for_both_leaks() -> None:
    """Two leaks with two causes. A prompt that asks about only one would pass a run that still
    carried the other, and the project half is the one the repository's own CLAUDE.md supplies."""

    # The answer SHAPE alone is not the question. A prompt can carry "USER=" and "PROJECT="
    # in its output format while asking about neither, which is exactly what a mutation of this
    # file did while these two checks stayed green. Assert the identifying content instead.
    check("it names a user-memory document", "User-level notes" in ISOLATION_PROMPT)
    check("it names a second user-memory marker", "No dash as punctuation" in ISOLATION_PROMPT)
    check("it names a project-memory document", "recall: working rules" in ISOLATION_PROMPT)
    check("it names a second project-memory marker",
          "One session, one workspace" in ISOLATION_PROMPT)
    check("it still fixes the answer shape",
          "USER=" in ISOLATION_PROMPT and "PROJECT=" in ISOLATION_PROMPT)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"agent_ab sandbox isolation: {len(tests)} test groups\n")
    for test in tests:
        print(test.__name__)
        test()
        print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
