"""The concatenation rule is owned by RE-call, not by callers.

If each caller built its own history string, two installations would send different
concatenations and both would call it the measured configuration. The benchmark's `full` variant
is one specific rule and this reproduces it.
"""

from __future__ import annotations

import pytest

from recall.retriever import FUSED_HISTORY_MAX_CHARS, build_history_query


def test_turns_are_newline_joined_in_order() -> None:
    assert build_history_query(["first", "second"]) == "first\nsecond"


def test_speaker_prefixes_are_stripped() -> None:
    """MTRAG-human prefixes every turn with `|user|: `.

    Left in, the literal token reaches both encoders on every query and depresses the whole run
    with nothing failing. The benchmark stripped it; serving must strip it identically.
    """
    assert build_history_query(["|user|: what is x", "|user|: and y"]) == "what is x\nand y"


def test_a_colon_inside_a_turn_survives() -> None:
    """Only a leading speaker tag is removed. A colon in the question is content."""
    assert build_history_query(["note: this matters"]) == "note: this matters"


def test_blank_turns_are_dropped_not_joined_as_empty_lines() -> None:
    assert build_history_query(["a", "   ", "b"]) == "a\nb"


def test_the_budget_is_4096_matching_the_mcp_cap() -> None:
    assert FUSED_HISTORY_MAX_CHARS == 4096
