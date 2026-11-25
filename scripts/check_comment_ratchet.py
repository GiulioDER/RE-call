"""Refuse new escape comments in tracked Python: deferral notes and suppressions without a reason.

An agent's cheapest way past a problem is a comment that makes the defect look reviewed: a
``TODO`` instead of the work, or a ``# noqa`` / ``# type: ignore`` instead of the fix. Few people
read either, nothing checks them, and the next agent copies the pattern. This check counts both.

Rules (Python comments only, read with ``tokenize``, so text inside strings and docstrings is not a
comment and is not counted):

* ``escape-note``: ``TODO``, ``FIXME``, ``XXX`` or ``HACK`` as a word, or ``for now``.
* ``bare-suppression``: a suppression marker (``noqa``, ``type: ignore``, ``pyright: ignore``,
  ``pragma: no cover`` / ``no branch``, ``nosec``, ``ast-grep-ignore``) that lacks its specific code
  where the tool takes one, or lacks a reason. The reason is any words left on the same comment once
  the markers are removed, or a comment line directly above (``ast-grep-ignore`` cannot carry text
  on its own line, so its reasons sit there).

Both are RATCHETED: ``scripts/comment_ratchet_baseline.json`` records, per file, how many of each
the tree held when the check was added (2026-11-25), and a file over its count fails.
``--write-baseline`` rewrites the record from the current tree; its diff is the review, so a count
may fall freely and a rise has to be argued for in the pull request that makes it.

    python scripts/check_comment_ratchet.py                   # every tracked .py file
    python scripts/check_comment_ratchet.py --write-baseline  # record the current counts
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("comment_ratchet_baseline.json")

ESCAPE_NOTE = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b|\b[Ff]or now\b")

# Each alternative captures the tool's code list, so a marker without one can be told apart.
MARKERS = (
    (re.compile(r"\bnoqa\b(?::\s*([A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*))?"), True),
    (re.compile(r"\btype:\s*ignore\b(\[[^\]]+\])?"), True),
    (re.compile(r"\bpyright:\s*ignore\b(\[[^\]]+\])?"), True),
    (re.compile(r"\bnosec\b(?:\s+(B\d+(?:\s*,\s*B\d+)*))?"), True),
    (re.compile(r"\bast-grep-ignore\b(?::\s*([\w-]+(?:\s*,\s*[\w-]+)*))?"), True),
    (re.compile(r"\bpragma:\s*no\s+(?:cover|branch)\b()"), False),
)
REASON = re.compile(r"[A-Za-z]{3,}")

Finding = tuple[str, int, str, str]


def _markers(comment: str) -> list[tuple[re.Match[str], bool]]:
    return [(m, needs_code) for pattern, needs_code in MARKERS for m in pattern.finditer(comment)]


def findings(path: str, text: str) -> list[Finding]:
    """Every escape note and bare suppression in one file's comments.

    Raises ``tokenize.TokenError`` or ``SyntaxError`` for a file that does not tokenize; the caller
    reports it rather than counting nothing for it.
    """
    lines = text.splitlines()
    out: list[Finding] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, comment = token.start[0], token.string
        if ESCAPE_NOTE.search(comment):
            out.append((path, row, "escape-note", comment))
        found = _markers(comment)
        if not found:
            continue
        rest = comment
        for match, _ in found:
            rest = rest.replace(match.group(0), " ")
        above = lines[row - 2].lstrip() if row >= 2 else ""
        reasoned = bool(REASON.search(rest)) or (above.startswith("#") and not _markers(above))
        coded = all(match.group(1) is not None or not needs_code for match, needs_code in found)
        if not (reasoned and coded):
            out.append((path, row, "bare-suppression", comment))
    return out


def tracked() -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, capture_output=True, check=True)
    return [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]


def scan(paths: Iterable[str], base: Path = ROOT) -> tuple[list[Finding], list[str]]:
    """Findings, and the files that could not be read or tokenized."""
    out: list[Finding] = []
    unread: list[str] = []
    for path in paths:
        try:
            out.extend(findings(path, (base / path).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, SyntaxError, tokenize.TokenError):
            unread.append(path)
    return out, unread


def counts(found: Iterable[Finding]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for path, _, rule, _ in found:
        table.setdefault(path, {}).setdefault(rule, 0)
        table[path][rule] += 1
    return table


def over_baseline(found: list[Finding], baseline: dict[str, dict[str, int]]) -> tuple[list[Finding], int]:
    """Findings in files over their recorded count, and how far the baseline could fall.

    Every finding of a rule is reported once its file holds more than the recorded count, because
    the reviewer has to find the new one among them.
    """
    now = counts(found)
    over = {
        (path, rule)
        for path, rules in now.items()
        for rule, n in rules.items()
        if n > baseline.get(path, {}).get(rule, 0)
    }
    slack = sum(
        max(0, recorded - now.get(path, {}).get(rule, 0))
        for path, rules in baseline.items()
        for rule, recorded in rules.items()
    )
    return [f for f in found if (f[0], f[2]) in over], slack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="record the current per-file counts")
    args = parser.parse_args()
    found, unread = scan(tracked(), ROOT)
    if unread:
        for path in unread:
            print(f"{path}: could not be read or tokenized", file=sys.stderr)
        print(f"{len(unread)} file(s) not checked; refusing a clean verdict.", file=sys.stderr)
        return 2
    if args.write_baseline:
        table = counts(found)
        BASELINE.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.name}: {len(found)} finding(s) in {len(table)} file(s)")
        return 0
    baseline: dict[str, dict[str, int]] = json.loads(BASELINE.read_text(encoding="utf-8"))
    report, slack = over_baseline(found, baseline)
    for path, row, rule, comment in report:
        print(f"{path}:{row}: {rule}: {comment.strip()}")
    if report:
        print(
            f"\n{len(report)} finding(s) in file(s) over {BASELINE.name}. Do the work instead of a TODO; "
            "fix the cause instead of suppressing it; or give the suppression its code and a reason "
            "on the same line or the comment line above. A justified rise is recorded with "
            "--write-baseline and argued for in the pull request.",
            file=sys.stderr,
        )
        return 1
    note = f"; {BASELINE.name} could fall by {slack}, rerun with --write-baseline" if slack else ""
    print(f"no new escape comments: {len(found)} recorded finding(s) within the baseline{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
