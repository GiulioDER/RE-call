"""K-1's Search-time speaker marks (``recall_aml.speaker_render``).

Pre-registration: docs/preregistrations/2026-09-26-aml-c9-speaker-at-render-time.md, apparatus
checks 1 and 2.

Red proof, 2026-09-26, each mutation to ``recall_aml/speaker_render.py`` alone, then restored:
- ``if len({role ...}) < 2`` changed to ``< 1`` (mark one-role windows too):
  ``test_a_one_role_window_is_unchanged`` failed on its equality assertion.
- ``marks[max(message_start, start) - start]`` changed to ``marks[message_start - start]`` (drop the
  clamp for the message the window opens in): ``test_a_two_role_window_is_marked_at_every_boundary``
  failed on its equality assertion.
- ``if found is not None: return None`` deleted (take the last match of a repeated window):
  ``test_a_window_found_twice_is_not_located`` failed on ``is None``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from recall_aml.code4 import word_windows
from recall_aml.models import SearchItem
from recall_aml.speaker_render import locate, mark_window, message_word_ranges, speaker_marked_items

MESSAGES = [
    ("user", "I moved to Lisbon in March and started a new job"),
    ("assistant", "Congratulations, a move and a new job at once is a lot"),
    ("user", "Yes and I adopted a cat named Miso"),
]
SESSION_WORDS = " ".join(content for _, content in MESSAGES).split()
RANGES = message_word_ranges(MESSAGES)


def _item(content: str) -> SearchItem:
    return SearchItem(
        id="w1", content=content, created_at=datetime(2024, 3, 1, tzinfo=UTC), source="raw",
        session_id="s1", kind="raw", score=0.5,
    )


def test_ranges_match_how_build_chunks_counts_words() -> None:
    assert RANGES == [("user", 0, 11), ("assistant", 11, 23), ("user", 23, 31)]
    # The same words, cut by the production windowing, land at the offsets the ranges describe.
    windows = word_windows(" ".join(content for _, content in MESSAGES), size=8, stride=6)
    assert locate(windows[1], SESSION_WORDS) == 6


def test_a_one_role_window_is_unchanged() -> None:
    window = " ".join(SESSION_WORDS[0:8])
    assert mark_window(window, 0, RANGES) == window


def test_a_two_role_window_is_marked_at_every_boundary() -> None:
    window = " ".join(SESSION_WORDS[6:26])
    marked = mark_window(window, 6, RANGES)
    assert marked == (
        "[user] and started a new job [assistant] Congratulations, a move and a new job at once is a lot "
        "[user] Yes and I"
    )


def test_a_window_found_twice_is_not_located() -> None:
    repeated = ["a", "b", "c", "x", "a", "b", "c"]
    assert locate("a b c", repeated) is None
    assert locate("b c x", repeated) == 1


def test_items_order_and_scores_are_unchanged_and_unknown_items_pass_through() -> None:
    marked_source = " ".join(SESSION_WORDS[6:26])
    items = [_item(marked_source), _item("words found nowhere"), _item(" ".join(SESSION_WORDS[0:8]))]
    positions = {marked_source: 6, " ".join(SESSION_WORDS[0:8]): 0}
    out = speaker_marked_items(
        items, lambda item: (positions[item.content], RANGES) if item.content in positions else None
    )
    assert [i.id for i in out] == [i.id for i in items]
    assert [i.score for i in out] == [i.score for i in items]
    assert out[0].content.startswith("[user] ")
    assert out[1] is items[1]
    assert out[2] is items[2]
