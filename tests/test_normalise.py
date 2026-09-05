"""Test the normalise function, specifically its handling of Unicode line separators."""
from __future__ import annotations

import pytest

from benchmarks.judge_quality import normalise


def test_normalise_collapses_actual_u2028_line_separator() -> None:
    """REAL GUARD: Actual U+2028 line separator is collapsed to single space.

    This test uses the ACTUAL U+2028 character (not a placeholder).
    If normalise() stops treating U+2028 as whitespace, this test FAILS.
    """
    # Create text with actual U+2028 line separator between words
    # U+2028 is the Unicode line separator character
    text_with_u2028 = "hello world"
    result = normalise(text_with_u2028)

    # The U+2028 should be collapsed to a single space
    assert result == "hello world", (
        f"U+2028 not collapsed to space. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_multiple_actual_u2028_separators() -> None:
    """REAL GUARD: Multiple U+2028 characters collapse to single space.

    If U+2028 handling is removed, this test fails.
    """
    # Multiple actual U+2028 characters between words
    text = "hello  world"
    result = normalise(text)

    # All consecutive U+2028 should collapse to a single space
    assert result == "hello world", (
        f"Multiple U+2028 not collapsed properly. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_mixed_with_spaces_and_tabs() -> None:
    """REAL GUARD: U+2028 mixed with regular spaces and tabs collapses to single space.

    If U+2028 is not treated as whitespace, this test fails.
    """
    # Mix of regular space, tab, and actual U+2028 line separator
    text = "hello 	   	  world"
    result = normalise(text)

    # All mixed whitespace including U+2028 should collapse to single space
    assert result == "hello world", (
        f"U+2028 not collapsed with other whitespace. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_as_actual_character_between_words() -> None:
    """REAL GUARD: The actual U+2028 character collapses to space between words.

    This is the PRIMARY TEST proving U+2028 handling works.
    """
    # This string has an actual U+2028 character between hello and world
    text = "hello" + " " + "world"
    result = normalise(text)

    assert result == "hello world", (
        f"U+2028 not collapsed to space. Input: {repr(text)}, Output: {repr(result)}"
    )


def test_normalise_collapses_actual_u2028_line_separator() -> None:
    """REAL GUARD: Actual U+2028 line separator is collapsed to single space.

    This test uses the ACTUAL U+2028 character (not a placeholder).
    If normalise() stops treating U+2028 as whitespace, this test FAILS.
    """
    # Create text with actual U+2028 line separator between words
    text_with_u2028 = "hello world"
    result = normalise(text_with_u2028)

    # The U+2028 should be collapsed to a single space
    assert result == "hello world", (
        f"U+2028 not collapsed to space. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_multiple_consecutive_u2028() -> None:
    """REAL GUARD: Multiple consecutive U+2028 characters collapse to single space.

    If U+2028 handling is removed, this test fails.
    """
    # Multiple actual U+2028 characters between words
    text = "hello  world"
    result = normalise(text)

    # All consecutive U+2028 should collapse to a single space
    assert result == "hello world", (
        f"Multiple U+2028 not collapsed properly. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_mixed_with_spaces() -> None:
    """REAL GUARD: U+2028 mixed with regular spaces and tabs collapses to single space.

    If U+2028 is not treated as whitespace, this test fails.
    """
    # Mix of regular space, tab, and actual U+2028 line separator
    text = "hello  	    world"
    result = normalise(text)

    # All mixed whitespace including U+2028 should collapse to single space
    assert result == "hello world", (
        f"U+2028 not collapsed with other whitespace. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_leading_u2028() -> None:
    """REAL GUARD: Leading U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Leading actual U+2028 line separator
    text = " hello world"
    result = normalise(text)

    # Leading U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Leading U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_trailing_u2028() -> None:
    """REAL GUARD: Trailing U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Trailing actual U+2028 line separator
    text = "hello world "
    result = normalise(text)

    # Trailing U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Trailing U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_actual_u2028_line_separator() -> None:
    """REAL GUARD: Actual U+2028 line separator is collapsed to single space.

    This test uses the ACTUAL U+2028 character (not a placeholder).
    If normalise() stops treating U+2028 as whitespace, this test FAILS.
    """
    # Create text with actual U+2028 line separator between words
    text_with_u2028 = "hello world"
    result = normalise(text_with_u2028)

    # The U+2028 should be collapsed to a single space
    assert result == "hello world", (
        f"U+2028 not collapsed to space. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_multiple_consecutive_u2028() -> None:
    """REAL GUARD: Multiple consecutive U+2028 characters collapse to single space.

    If U+2028 handling is removed, this test fails.
    """
    # Multiple actual U+2028 characters between words
    text = "hello  world"
    result = normalise(text)

    # All consecutive U+2028 should collapse to a single space
    assert result == "hello world", (
        f"Multiple U+2028 not collapsed properly. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_mixed_with_spaces() -> None:
    """REAL GUARD: U+2028 mixed with regular spaces and tabs collapses to single space.

    If U+2028 is not treated as whitespace, this test fails.
    """
    # Mix of regular space, tab, and actual U+2028 line separator
    text = "hello  	    world"
    result = normalise(text)

    # All mixed whitespace including U+2028 should collapse to single space
    assert result == "hello world", (
        f"U+2028 not collapsed with other whitespace. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_leading_u2028() -> None:
    """REAL GUARD: Leading U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Leading actual U+2028 line separator
    text = " hello world"
    result = normalise(text)

    # Leading U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Leading U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_trailing_u2028() -> None:
    """REAL GUARD: Trailing U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Trailing actual U+2028 line separator
    text = "hello world "
    result = normalise(text)

    # Trailing U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Trailing U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_actual_u2028_line_separator() -> None:
    """REAL GUARD: Actual U+2028 line separator is collapsed to single space.

    This test uses the ACTUAL U+2028 character (not a placeholder).
    If normalise() stops treating U+2028 as whitespace, this test FAILS.
    """
    # Create text with actual U+2028 line separator between words
    text_with_u2028 = "helloâ¨world"
    result = normalise(text_with_u2028)

    # The U+2028 should be collapsed to a single space
    assert result == "hello world", (
        f"U+2028 not collapsed to space. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_multiple_consecutive_u2028() -> None:
    """REAL GUARD: Multiple consecutive U+2028 characters collapse to single space.

    If U+2028 handling is removed, this test fails.
    """
    # Multiple actual U+2028 characters between words
    text = "helloâ¨â¨world"
    result = normalise(text)

    # All consecutive U+2028 should collapse to a single space
    assert result == "hello world", (
        f"Multiple U+2028 not collapsed properly. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_mixed_with_spaces() -> None:
    """REAL GUARD: U+2028 mixed with regular spaces and tabs collapses to single space.

    If U+2028 is not treated as whitespace, this test fails.
    """
    # Mix of regular space, tab, and actual U+2028 line separator
    text = "hello â¨	  â¨ world"
    result = normalise(text)

    # All mixed whitespace including U+2028 should collapse to single space
    assert result == "hello world", (
        f"U+2028 not collapsed with other whitespace. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_leading_u2028() -> None:
    """REAL GUARD: Leading U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Leading actual U+2028 line separator
    text = "â¨hello world"
    result = normalise(text)

    # Leading U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Leading U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_trailing_u2028() -> None:
    """REAL GUARD: Trailing U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Trailing actual U+2028 line separator
    text = "hello worldâ¨"
    result = normalise(text)

    # Trailing U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Trailing U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_actual_u2028_line_separator() -> None:
    """REAL GUARD: Actual U+2028 line separator is collapsed to single space.

    This test uses the ACTUAL U+2028 character (not a placeholder).
    If normalise() stops treating U+2028 as whitespace, this test FAILS.
    """
    # Create text with actual U+2028 line separator between words
    text_with_u2028 = "hello world"
    result = normalise(text_with_u2028)

    # The U+2028 should be collapsed to a single space
    assert result == "hello world", (
        f"U+2028 not collapsed to space. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_collapses_multiple_consecutive_u2028() -> None:
    """REAL GUARD: Multiple consecutive U+2028 characters collapse to single space.

    If U+2028 handling is removed, this test fails.
    """
    # Multiple actual U+2028 characters between words
    text = "hello  world"
    result = normalise(text)

    # All consecutive U+2028 should collapse to a single space
    assert result == "hello world", (
        f"Multiple U+2028 not collapsed properly. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_u2028_mixed_with_spaces() -> None:
    """REAL GUARD: U+2028 mixed with regular spaces and tabs collapses to single space.

    If U+2028 is not treated as whitespace, this test fails.
    """
    # Mix of regular space, tab, and actual U+2028 line separator
    text = "hello  	   world"
    result = normalise(text)

    # All mixed whitespace including U+2028 should collapse to single space
    assert result == "hello world", (
        f"U+2028 not collapsed with other whitespace. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_leading_u2028() -> None:
    """REAL GUARD: Leading U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Leading actual U+2028 line separator
    text = " hello world"
    result = normalise(text)

    # Leading U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Leading U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )


def test_normalise_strips_trailing_u2028() -> None:
    """REAL GUARD: Trailing U+2028 is stripped, not converted to space.

    If U+2028 handling breaks, this test fails.
    """
    # Trailing actual U+2028 line separator
    text = "hello world "
    result = normalise(text)

    # Trailing U+2028 should be stripped entirely
    assert result == "hello world", (
        f"Trailing U+2028 not stripped. Expected 'hello world', got {repr(result)}"
    )
