"""The mutation job's file selection: only files whose change adds executable code are mutated.

Invariant: an added line counts as code unless it is blank, a comment-only line, or inside a
module, class or function docstring; a diff's added line numbers are read from its hunk headers;
files that add code are listed first, most added code first. Failure mode caught: the job mutating
files whose only change is a deletion or a reworded comment (PR 849 filled its 20-file cap with
them and timed out), or worse, skipping a file that did add code.

Red proof, 2026-10-02, each against the named line of ``scripts/mutation_code_changes.py`` with this
file unchanged, each failing at the named test, then restored and green:
- M1 ``code_lines`` not excluding docstring lines (``number not in docstrings`` removed):
  ``test_docstrings_comments_and_blank_lines_are_not_code``.
- M2 ``code_lines`` not excluding comment-only lines (``not text.startswith("#")`` removed):
  ``test_docstrings_comments_and_blank_lines_are_not_code`` and the ordering test (the comment-only
  file is then listed as code).
- M3 ``added_lines`` ignoring a hunk's count (``range(start, start + 1)``):
  ``test_added_lines_come_from_hunk_headers`` and the ordering test.
- M4 ``main`` sorting by name only (``key=lambda p: p``):
  ``test_files_that_add_code_come_first_most_code_first``.
- M5 ``touched_functions`` without decorator lines, and M6 without methods:
  ``test_a_line_maps_to_its_top_level_function_or_method``.
- M7 ``mutmut_pattern`` with ``.`` for mutmut's class separator, and M8 keeping ``.__init__`` in the
  module name: ``test_patterns_use_mutmuts_own_names``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

_SPEC = importlib.util.spec_from_file_location(
    "mutation_code_changes", Path(__file__).resolve().parents[1] / "scripts" / "mutation_code_changes.py")
assert _SPEC is not None and _SPEC.loader is not None
mcc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcc)

SOURCE = '''"""Module docstring,
over two lines."""

import os


def f(x):
    """Function docstring."""
    # a comment
    return x + 1  # trailing comment on code

'''


def test_docstrings_comments_and_blank_lines_are_not_code() -> None:
    assert mcc.code_lines(SOURCE) == {4, 7, 10}


FUNCTIONS = '''X = 1


@decorator
def top(a):
    def inner(b):
        return b
    return inner(a)


class Box:
    def method(self):
        return 2

    def other(self):
        return 3
'''


def test_a_line_maps_to_its_top_level_function_or_method() -> None:
    # 1: module level, nothing; 4: a decorator belongs to its function; 7: a nested function
    # counts for the enclosing one; 13: a method of Box.
    assert mcc.touched_functions(FUNCTIONS, {1}) == []
    assert mcc.touched_functions(FUNCTIONS, {4}) == [(None, "top")]
    assert mcc.touched_functions(FUNCTIONS, {7}) == [(None, "top")]
    assert mcc.touched_functions(FUNCTIONS, {13}) == [("Box", "method")]


def test_patterns_use_mutmuts_own_names() -> None:
    assert mcc.mutmut_pattern("recall/_env.py", None, "f") == "recall._env.x_f__mutmut_*"
    assert mcc.mutmut_pattern("recall/__init__.py", None, "f") == "recall.x_f__mutmut_*"
    assert mcc.mutmut_pattern("recall_mcp/server.py", "Box", "m") == "recall_mcp.server.xǁBoxǁm__mutmut_*"


def test_added_lines_come_from_hunk_headers() -> None:
    diff = "diff --git a/x b/x\n@@ -3,0 +4,2 @@\n+a\n+b\n@@ -9 +11 @@\n-c\n+d\n@@ -20,3 +22,0 @@\n-e\n"
    assert mcc.added_lines(diff) == {4, 5, 11}


def test_files_that_add_code_come_first_most_code_first(tmp_path: Path, monkeypatch, capsys) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("z = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    git("tag", "base")
    (tmp_path / "a.py").write_text("x = 1\n# only a comment\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\ny2 = 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("z = 1\nz2 = 2\nz3 = 3\n", encoding="utf-8")
    git("commit", "-qam", "change")
    monkeypatch.chdir(tmp_path)
    assert mcc.main(["base", "a.py", "b.py", "c.py"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["code\t2\tc.py", "code\t1\tb.py", "nocode\t0\ta.py"]
