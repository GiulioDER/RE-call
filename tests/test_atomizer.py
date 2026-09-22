"""Window-aware atomizer invariants.

Red proofs (2026-09-22, each mutation applied to ``recall/atomizer.py`` and reverted):

* ``test_every_view_is_an_exact_word_range_inside_its_parent`` failed its equality assertion
  when ``window_views`` emitted ``" ".join(words[start:end + 1])``.
* ``test_bounds_match_the_hosted_word_windows`` failed when ``window_bounds`` dropped the
  ``min(..., word_count)`` clamp on the last window.
* ``test_a_view_in_the_overlap_goes_to_the_window_where_it_is_central`` failed when
  ``parent_window`` returned the first containing window (``margin > best[0]`` changed to
  ``best is None`` only).
* ``test_one_line_session_without_paragraphs_still_yields_bounded_views`` failed when
  ``_sentence_ranges`` stopped cutting runs longer than ``max_words``.
* ``test_duplicate_spans_in_one_session_emit_one_view`` failed when the ``seen`` check was removed.
"""

from __future__ import annotations

import pytest

from recall.atomizer import parent_window, window_bounds, window_views
from recall_aml.code4 import word_windows


def _session(sentences: int, words_per_sentence: int = 9) -> str:
    return " ".join(
        " ".join(f"s{index}w{position}" for position in range(words_per_sentence - 1))
        + f" end{index}."
        for index in range(sentences)
    )


@pytest.mark.parametrize("strategy", ["sentence", "micro"])
def test_every_view_is_an_exact_word_range_inside_its_parent(strategy: str) -> None:
    text = _session(60)
    words = text.split()
    windows = word_windows(text, size=160, stride=120)
    bounds = window_bounds(len(words), size=160, stride=120)
    views = window_views(text, window_size=160, window_stride=120, strategy=strategy)  # type: ignore[arg-type]
    assert views
    for view in views:
        assert view.text == " ".join(words[view.word_start : view.word_end])
        start, end = bounds[view.parent_segment]
        assert start <= view.word_start and view.word_end <= end
        assert view.text in windows[view.parent_segment]
    assert [view.view_ordinal for view in views] == list(range(len(views)))


@pytest.mark.parametrize("count", [1, 159, 160, 161, 280, 281, 1_000])
def test_bounds_match_the_hosted_word_windows(count: int) -> None:
    text = " ".join(f"w{index}" for index in range(count))
    words = text.split()
    expected = word_windows(text, size=160, stride=120)
    bounds = window_bounds(count, size=160, stride=120)
    assert [" ".join(words[start:end]) for start, end in bounds] == expected
    # Compare the ranges themselves too: slicing clamps an overlong end silently, so joined text
    # alone cannot see an unclamped bound, while parent_window's centrality margin can.
    starts = [index * 120 for index in range(len(expected))]
    assert bounds == [
        (start, start + len(window.split())) for start, window in zip(starts, expected, strict=True)
    ]


def test_a_view_in_the_overlap_goes_to_the_window_where_it_is_central() -> None:
    bounds = window_bounds(400, size=160, stride=120)
    # Words 150..160 lie in window 0 ([0,160), margin 0) and window 1 ([120,280), margin 30).
    assert parent_window(150, 160, bounds) == 1
    # Words 125..135 lie in window 0 (margin 25) and window 1 (margin 5).
    assert parent_window(125, 135, bounds) == 0
    assert parent_window(0, 200, bounds) is None


def test_one_line_session_without_paragraphs_still_yields_bounded_views() -> None:
    # The hosted C8 rendering: one line, no blank lines, one very long unpunctuated run.
    text = " ".join(f"token{index}" for index in range(500)) + " done."
    views = window_views(text, window_size=160, window_stride=120, max_words=40)
    assert len(views) >= 12
    assert all(view.word_end - view.word_start <= 40 for view in views)


def test_short_fragments_fold_into_the_following_sentence() -> None:
    text = "ok. yes. the migration ledger records the filename next to the version number."
    views = window_views(text, window_size=160, window_stride=120, min_words=6)
    assert [view.text for view in views] == [text]


def test_duplicate_spans_in_one_session_emit_one_view() -> None:
    repeated = "the build failed because the lock file was stale again."
    views = window_views(
        f"{repeated} {repeated} something else entirely happened after that retry.",
        window_size=160,
        window_stride=120,
    )
    assert [view.text for view in views].count(repeated) == 1


def test_views_that_cannot_fit_the_overlap_are_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        window_views("a b c", window_size=160, window_stride=120, max_words=41)
    with pytest.raises(ValueError, match="overlap"):
        window_views("a b c", window_size=160, window_stride=120, strategy="micro", micro_size=41)
