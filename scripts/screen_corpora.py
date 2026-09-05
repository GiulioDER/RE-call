"""Run the hosted screen over every memory corpus on this machine and print counts.

Read-only. Never modifies, moves or uploads anything.

⛔ **This never prints matched text.** `screening.secrets_in` is built not to return it, and this
harness must not reintroduce it: a measurement that copies a corpus's credentials into a terminal,
a log, or a markdown file in a public repository is the exact disclosure the feature exists to
prevent. Only file names, line numbers and rule names are printed.

Pre-registration: `docs/preregistrations/2026-09-02-hosted-screen-firing-rate.md`.

    python scripts/screen_corpora.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from recall_hooks import screening  # noqa: E402


def apparatus_check() -> bool:
    """Verify the harness on a case whose answer is already known, before trusting any corpus number.

    🔑 Predicting the outcome does not reveal a broken harness, and exit code 0 is not a
    measurement. The fixtures come from the test module rather than being retyped here, so this
    checks the same strings the unit tests do and this file holds no credential-shaped literal.
    """
    try:
        from tests.test_hosted_screening import LIVE
    except ImportError as exc:
        print(f"APPARATUS FAILED: cannot load the known-answer fixtures ({exc})")
        return False

    found = [rule for rule in sorted(LIVE) if screening.secrets_in(f"k = {LIVE[rule]}\n")]
    expected = len(LIVE)
    print(f"apparatus: {len(found)}/{expected} known fixtures detected")
    if len(found) != expected:
        missed = sorted(set(LIVE) - set(found))
        print(f"APPARATUS FAILED: these known credentials were NOT detected: {missed}")
        return False
    clean = screening.secrets_in("Use `postgresql://user:pw@localhost/recall` as the example.\n")
    if clean:
        print("APPARATUS FAILED: a known-clean line fired; the numbers below would be noise")
        return False
    print("apparatus: a known-clean line did not fire")
    return True


def main() -> int:
    if not apparatus_check():
        return 2

    projects = Path.home() / ".claude" / "projects"
    corpora = sorted(p for p in projects.glob("*/memory") if p.is_dir())

    total_files = 0
    total_withheld = 0
    rules: Counter[str] = Counter()
    detail: list[tuple[str, str, str]] = []

    print()
    print(f"{'corpus':<52} {'files':>6} {'withheld':>9} {'rate':>7}")
    print("-" * 78)
    for corpus in corpora:
        files = sorted(corpus.glob("*.md"))
        if not files:
            continue
        withheld = 0
        for path in files:
            findings = screening.screen_file(path)
            if findings:
                withheld += 1
                for finding in findings:
                    rules[finding.rule] += 1
                    detail.append((corpus.parent.name, path.name, str(finding)))
        total_files += len(files)
        total_withheld += withheld
        rate = withheld / len(files) if files else 0.0
        print(f"{corpus.parent.name:<52} {len(files):>6} {withheld:>9} {rate:>6.2%}")

    print("-" * 78)
    overall = total_withheld / total_files if total_files else 0.0
    print(f"{'ALL':<52} {total_files:>6} {total_withheld:>9} {overall:>6.2%}")

    if rules:
        print("\nrule distribution (findings, not files):")
        for rule, count in rules.most_common():
            print(f"  {count:>4}  {rule}")
        print("\nwithheld files, for hand classification:")
        for project, name, finding in detail:
            print(f"  {project} :: {name} :: {finding}")
    else:
        print("\nnothing withheld in any corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
