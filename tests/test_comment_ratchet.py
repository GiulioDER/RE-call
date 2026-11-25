"""The comment ratchet (``scripts/check_comment_ratchet.py``): escape notes and suppressions without
a reason may not grow in any tracked Python file.

Invariant: a ``TODO``-family word or ``for now`` in a comment, and a suppression marker missing its
code or its reason, are counted per file; a file whose count rises over the recorded baseline
fails the check, and text in strings is never counted. Failure mode caught: an agent adding a
``# type: ignore[...]`` or a ``TODO`` in place of a fix, with nothing in CI that notices.

Red proof, 2026-11-25, one mutation to ``scripts/check_comment_ratchet.py`` per test, run alone and
restored; each failed at its own assertion:

- ``ESCAPE_NOTE`` without the TODO family: ``test_deferral_words_are_escape_notes`` (``[] == [...]``).
- ``ESCAPE_NOTE`` without ``for now``: ``test_for_now_is_an_escape_note`` (``[] == [...]``).
- ``findings`` reading STRING tokens too: ``test_text_in_strings_and_docstrings_is_not_a_comment``.
- ``reasoned = True``: ``test_a_marker_without_a_reason_is_bare``.
- ``coded = True``: ``test_a_marker_without_its_code_is_bare_even_with_a_reason``.
- the comment-line-above clause removed:
  ``test_a_reason_on_the_same_line_or_the_comment_line_above_passes``.
- ``and not _markers(above)`` removed: ``test_a_suppression_line_above_is_not_a_reason``.
- only the first marker stripped from the comment: ``test_two_markers_with_nothing_else_are_bare``.
- ``scan`` dropping ``unread.append``: ``test_a_file_that_does_not_tokenize_is_not_checked``.
- ``over_baseline`` returning only the last finding:
  ``test_a_rise_over_the_baseline_reports_every_finding_of_that_file``.
- ``>`` turned into ``>=`` in ``over_baseline``:
  ``test_a_file_at_or_below_its_count_passes_and_the_slack_is_reported``.
- ``main`` returning 0 on a report:
  ``test_the_command_fails_on_a_rise_and_passes_once_it_is_recorded`` (``assert 0 == 1``).

The last test also caught a real defect before the mutations: ``main`` called ``scan`` with its
default ``base=ROOT``, bound at definition time, so the command read files from the wrong root.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_comment_ratchet", Path(__file__).parents[1] / "scripts" / "check_comment_ratchet.py"
)
assert SPEC and SPEC.loader
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def rules(source: str) -> list[tuple[int, str]]:
    return [(row, rule) for _, row, rule, _ in check.findings("x.py", textwrap.dedent(source))]


def test_deferral_words_are_escape_notes() -> None:
    assert rules("a = 1  # TODO handle the empty case\nb = 2  # FIXME\nc = 3  # XXX\nd = 4  # HACK\n") == [
        (1, "escape-note"), (2, "escape-note"), (3, "escape-note"), (4, "escape-note"),
    ]


def test_for_now_is_an_escape_note() -> None:
    assert rules("limit = 10  # lenient for now\n") == [(1, "escape-note")]


def test_text_in_strings_and_docstrings_is_not_a_comment() -> None:
    source = '''
        """TODO in a docstring, and a # noqa in prose."""
        text = "# type: ignore  TODO for now"
    '''
    assert rules(source) == []


def test_a_marker_without_a_reason_is_bare() -> None:
    assert rules("import os  # noqa: F401\nx = f()  # type: ignore[arg-type]\n") == [
        (1, "bare-suppression"), (2, "bare-suppression"),
    ]


def test_a_marker_without_its_code_is_bare_even_with_a_reason() -> None:
    assert rules("import os  # noqa because the import registers a plugin\n") == [(1, "bare-suppression")]
    assert rules("x = f()  # type: ignore - the stub is wrong\n") == [(1, "bare-suppression")]


def test_a_reason_on_the_same_line_or_the_comment_line_above_passes() -> None:
    source = """
        import os  # noqa: F401 - importing it registers the plugin
        x = f()  # pragma: no cover - only without the extra
        # The rule cannot see that the path was confined two lines up.
        # ast-grep-ignore: containment-via-confined
        y = g()
    """
    assert rules(source) == []


def test_a_suppression_line_above_is_not_a_reason() -> None:
    source = """
        # noqa: E402
        import os  # noqa: E402
    """
    assert rules(source) == [(2, "bare-suppression"), (3, "bare-suppression")]


def test_two_markers_with_nothing_else_are_bare() -> None:
    assert rules("x = f()  # type: ignore[arg-type]  # noqa: E501\n") == [(1, "bare-suppression")]


def test_a_file_that_does_not_tokenize_is_not_checked(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text('x = """never closed\n', encoding="utf-8")
    found, unread = check.scan(["broken.py"], tmp_path)
    assert (found, unread) == ([], ["broken.py"])


def test_a_rise_over_the_baseline_reports_every_finding_of_that_file() -> None:
    found = [("a.py", 1, "bare-suppression", "#"), ("a.py", 9, "bare-suppression", "#")]
    report, _ = check.over_baseline(found, {"a.py": {"bare-suppression": 1}})
    assert report == found


def test_a_file_at_or_below_its_count_passes_and_the_slack_is_reported() -> None:
    found = [("a.py", 1, "bare-suppression", "#")]
    assert check.over_baseline(found, {"a.py": {"bare-suppression": 1}}) == ([], 0)
    assert check.over_baseline(found, {"a.py": {"bare-suppression": 3}, "gone.py": {"escape-note": 1}}) == ([], 3)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_command_fails_on_a_rise_and_passes_once_it_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "mod.py").write_text("x = f()  # type: ignore[arg-type]\n", encoding="utf-8")
    _git(tmp_path, "add", "mod.py")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "BASELINE", baseline)
    monkeypatch.setattr("sys.argv", ["check_comment_ratchet.py"])

    assert check.main() == 1
    assert "mod.py:1: bare-suppression" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["check_comment_ratchet.py", "--write-baseline"])
    assert check.main() == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"mod.py": {"bare-suppression": 1}}

    monkeypatch.setattr("sys.argv", ["check_comment_ratchet.py"])
    assert check.main() == 0
