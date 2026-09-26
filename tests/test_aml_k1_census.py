"""The K-1 Stage 0 census labels a window by the speakers it mixes, on synthetic sessions.

Invariant: a window inside one role's messages is ``one_role``; one that crosses a user/assistant
boundary with no name at the inner start is ``mixed_unnamed``; one whose inner starts all carry a
``Name:`` is ``named``; one whose words occur twice is ``unmarked``, never guessed.

Red proof, 2026-09-26, each against ``scripts/aml_k1_census.py`` with this file unchanged:
- ``classify`` treating every mixed window as named (``if True:`` for the ``all(NAMED...)`` test):
  ``test_a_window_crossing_an_unnamed_role_boundary_is_mixed_unnamed`` failed on ``== "mixed_unnamed"``.
- ``classify`` ignoring names (the ``all(NAMED...)`` test made ``False``):
  ``test_named_inner_starts_make_a_mixed_window_named`` failed on ``== "named"``.
- ``tally`` counting ``unmarked`` into the mixed share: ``test_the_share_counts_only_mixed_unnamed``
  failed on ``== 0.25``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "aml_k1_census", Path(__file__).parents[1] / "scripts" / "aml_k1_census.py"
)
assert SPEC and SPEC.loader
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)

SESSION = [
    ("user", "I moved to Lisbon last spring and I love it"),
    ("assistant", "That sounds wonderful, how is the weather there"),
    ("user", "Warm and sunny most days"),
]
WORDS = " ".join(content for _, content in SESSION).split()


def test_a_window_inside_one_message_is_one_role() -> None:
    assert census.classify("moved to Lisbon last spring", SESSION, WORDS) == "one_role"


def test_a_window_crossing_an_unnamed_role_boundary_is_mixed_unnamed() -> None:
    assert census.classify("I love it That sounds wonderful,", SESSION, WORDS) == "mixed_unnamed"


def test_named_inner_starts_make_a_mixed_window_named() -> None:
    named = [("user", "Caroline: I love it"), ("assistant", "Melanie: That sounds wonderful")]
    words = " ".join(content for _, content in named).split()
    assert census.classify("love it Melanie: That sounds", named, words) == "named"


def test_a_window_found_twice_is_unmarked() -> None:
    repeated = [("user", "hello there"), ("assistant", "hello there")]
    words = " ".join(content for _, content in repeated).split()
    assert census.classify("hello there", repeated, words) == "unmarked"


def test_the_share_counts_only_mixed_unnamed() -> None:
    rows = [("raw", "mixed_unnamed", 0), ("raw", "one_role", 0), ("raw", "unmarked", 0), ("raw", "named", 2)]
    result = census.tally(rows)
    assert result["mixed_unnamed_share"] == 0.25
    assert result["unmarked_share"] == 0.25
    assert result["two_or_more_named_speakers_share"] == 0.25
