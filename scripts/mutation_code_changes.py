"""Which changed files add executable code, and how much: the mutation job's file selection.

    python scripts/mutation_code_changes.py BASE path/one.py path/two.py ...

Prints one tab-separated line per path, files that add code first, most added code lines first:

    code    12    recall/_env.py
    nocode  0     recall/types.py

Why: mutmut mutates WHOLE files, and `scripts/mutation_changed.sh` mutates at most MAX_FILES of them
in a 20-minute job. A change that only deletes lines, or only rewords comments and docstrings, adds
no behaviour a mutant could test, yet each such file costs a whole-file mutation run. PR 849 changed
29 shipped files, most by deletion or a reworded comment; alphabetical selection filled the cap with
large modules whose only edit was a comment, and the job timed out at 20 minutes having reported
nothing. An added line counts as code unless it is blank, a comment-only line, or inside a module,
class or function docstring of the file as it stands at HEAD.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def added_lines(diff_text: str) -> set[int]:
    """Line numbers (in the new file) added by a unified diff with zero context."""
    out: set[int] = set()
    for line in diff_text.splitlines():
        match = _HUNK.match(line)
        if match:
            start, count = int(match.group(1)), int(match.group(2) or "1")
            out.update(range(start, start + count))
    return out


def code_lines(source: str) -> set[int]:
    """Lines that hold executable code: not blank, not comment-only, not inside a docstring."""
    docstrings: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str) and first.end_lineno is not None):
                docstrings.update(range(first.lineno, first.end_lineno + 1))
    out: set[int] = set()
    for number, line in enumerate(source.splitlines(), start=1):
        text = line.strip()
        if text and not text.startswith("#") and number not in docstrings:
            out.add(number)
    return out


def added_code_lines(base: str, path: str) -> int:
    diff = subprocess.run(["git", "diff", "-U0", f"{base}...HEAD", "--", path],
                          capture_output=True, text=True, check=True).stdout
    try:
        source = Path(path).read_text(encoding="utf-8")
        return len(added_lines(diff) & code_lines(source))
    except (OSError, SyntaxError, UnicodeDecodeError):
        # Unreadable or unparsable: count every added line, so the file is mutated rather than
        # silently skipped.
        return len(added_lines(diff))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    base, paths = argv[0], argv[1:]
    counts = {path: added_code_lines(base, path) for path in paths}
    for path in sorted(paths, key=lambda p: (-counts[p], p)):
        print(f"{'code' if counts[path] else 'nocode'}\t{counts[path]}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
