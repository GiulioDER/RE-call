"""One parser for `--conversations`, because there were seven.

Two were named `_parse_indices`; five were inlined into a `main()`. They did not agree: the
inline copies omitted the `.strip()` on each part, so `--conversations "0-14, 20"` raised
`ValueError` in five probes and worked in two — and the probes read each other's tables, so the
same flag meaning different things is a real hazard, not a tidiness complaint.
"""
from __future__ import annotations

import pytest

from benchmarks.beam.dataset import parse_conversation_indices as parse

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def test_a_single_index() -> None:
    assert parse("7") == [7]


def test_an_inclusive_range() -> None:
    assert parse("0-4") == [0, 1, 2, 3, 4]


def test_ranges_and_singletons_together() -> None:
    assert parse("0-2,7,10-11") == [0, 1, 2, 7, 10, 11]


@pytest.mark.parametrize("spec", ["0-2, 7", " 0-2 ,7 ", "0 - 2,7"])
def test_whitespace_is_tolerated_everywhere(spec: str) -> None:
    """The exact case the five inline copies raised ValueError on."""
    assert parse(spec) == [0, 1, 2, 7]


def test_duplicates_and_overlaps_collapse() -> None:
    # `run.py` deduplicated; the probes pushed `sorted(set(...))` to the call site and one forgot.
    assert parse("0-3,2-4,3") == [0, 1, 2, 3, 4]


def test_the_result_is_sorted_regardless_of_input_order() -> None:
    assert parse("9,0-1") == [0, 1, 9]


def test_empty_and_trailing_separators_yield_nothing_extra() -> None:
    assert parse("") == []
    assert parse("3,") == [3]
    assert parse(",,") == []


def test_a_non_numeric_part_is_an_error_rather_than_a_silent_skip() -> None:
    # A typo'd conversation id must not quietly narrow the run: `n` is what the study reports.
    with pytest.raises(ValueError):
        parse("0-2,abc")
