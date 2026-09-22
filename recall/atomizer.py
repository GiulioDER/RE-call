"""Deterministic atomic views over overlapping word windows.

The memo atomizer in ``scripts/audit_atomic_fact_auxiliary_view.py`` needs blank-line paragraphs
and one parent chunk per paragraph. Hosted Code4 windows have neither: a session is joined on
single spaces, split into 160-word windows with a 120-word stride, and stored as one line, so
neighbouring windows share 40 words. Measured 2026-09-22 on the frozen CAMBench coding corpus
(196 sessions, 1,220 windows), the memo atomizer produced one view.

This module segments the same word sequence the windows are cut from, so every view is an exact
word range of its session and maps to a parent window by offset rather than by substring search.
A view that fits inside two overlapping windows is assigned to the window where it sits most
centrally, which gives the rescued parent the most context on both sides of the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


ATOMIZER_STRATEGIES = ("sentence", "micro")
AtomizerStrategy = Literal["sentence", "micro"]

#: A word that ends a sentence: terminal punctuation, optionally followed by closing quotes or
#: brackets. Decimal points and dotted names do not match because the dot is not word-final.
_SENTENCE_END = re.compile(r"[.!?;][\"')\]]*$")
_ALNUM = re.compile(r"[0-9A-Za-z]")


@dataclass(frozen=True)
class WindowView:
    """One atomic view: an exact word range of a session and the window that holds it."""

    parent_segment: int
    view_ordinal: int
    word_start: int
    word_end: int
    text: str


def window_bounds(word_count: int, *, size: int, stride: int) -> list[tuple[int, int]]:
    """Return the ``[start, end)`` word range of every window ``word_windows`` would emit."""

    if size < 1 or stride < 1:
        raise ValueError("window size and stride must be positive")
    if word_count <= 0:
        return [(0, 0)]
    bounds: list[tuple[int, int]] = []
    for start in range(0, word_count, stride):
        bounds.append((start, min(start + size, word_count)))
        if start + size >= word_count:
            break
    return bounds


def parent_window(
    start: int, end: int, bounds: list[tuple[int, int]]
) -> int | None:
    """Return the window that contains ``[start, end)`` most centrally, or None."""

    best: tuple[int, int] | None = None
    for segment, (window_start, window_end) in enumerate(bounds):
        if window_start <= start and end <= window_end:
            margin = min(start - window_start, window_end - end)
            if best is None or margin > best[0]:
                best = (margin, segment)
    return None if best is None else best[1]


def _content_words(words: list[str]) -> int:
    return sum(1 for word in words if _ALNUM.search(word))


def _sentence_ranges(words: list[str], *, min_words: int, max_words: int) -> list[tuple[int, int]]:
    """Split at sentence ends, fold short fragments forward, and cut long runs."""

    raw: list[tuple[int, int]] = []
    start = 0
    for index, word in enumerate(words):
        if _SENTENCE_END.search(word):
            raw.append((start, index + 1))
            start = index + 1
    if start < len(words):
        raw.append((start, len(words)))

    folded: list[tuple[int, int]] = []
    pending: int | None = None
    for range_start, range_end in raw:
        begin = range_start if pending is None else pending
        if range_end - begin < min_words:
            pending = begin
            continue
        folded.append((begin, range_end))
        pending = None
    if pending is not None:
        if folded and len(words) - pending < min_words:
            folded[-1] = (folded[-1][0], len(words))
        else:
            folded.append((pending, len(words)))

    # Cut an overlong run into the fewest near-equal pieces. Folding a short tail into the previous
    # piece instead would exceed max_words, and a view longer than the window overlap can straddle
    # a boundary and be dropped without any error.
    ranges: list[tuple[int, int]] = []
    for range_start, range_end in folded:
        length = range_end - range_start
        pieces = -(-length // max_words)
        for piece in range(pieces):
            ranges.append(
                (
                    range_start + (length * piece) // pieces,
                    range_start + (length * (piece + 1)) // pieces,
                )
            )
    return ranges


def _micro_ranges(word_count: int, *, size: int, stride: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, word_count)) for start, _ in window_bounds(
        word_count, size=size, stride=stride
    )]


def window_views(
    session_text: str,
    *,
    window_size: int,
    window_stride: int,
    strategy: AtomizerStrategy = "sentence",
    min_words: int = 6,
    max_words: int = 40,
    micro_size: int = 24,
    micro_stride: int = 12,
    min_content_words: int = 4,
) -> list[WindowView]:
    """Build deterministic atomic views for one session's hosted word windows.

    Every returned view satisfies two invariants that callers may rely on: its text is exactly
    ``" ".join(words[word_start:word_end])``, and that range lies inside the parent window's
    range. Views whose text repeats an earlier view of the same session are dropped, because a
    duplicate row can only tie with itself in the rescue matrix.
    """

    if strategy not in ATOMIZER_STRATEGIES:
        raise ValueError(f"unknown atomizer strategy {strategy!r}")
    if not 1 <= min_words <= max_words:
        raise ValueError("atomizer word bounds are invalid")
    if max_words > window_size - window_stride and strategy == "sentence":
        # A view longer than the window overlap can straddle a boundary with no single window
        # containing it; it would then be dropped silently. Refuse the configuration instead.
        raise ValueError("sentence views must fit inside the window overlap")
    if strategy == "micro" and micro_size > window_size - window_stride:
        raise ValueError("micro views must fit inside the window overlap")
    words = session_text.split()
    bounds = window_bounds(len(words), size=window_size, stride=window_stride)
    ranges = (
        _sentence_ranges(words, min_words=min_words, max_words=max_words)
        if strategy == "sentence"
        else _micro_ranges(len(words), size=micro_size, stride=micro_stride)
    )
    views: list[WindowView] = []
    seen: set[str] = set()
    for start, end in ranges:
        span = words[start:end]
        if _content_words(span) < min_content_words:
            continue
        text = " ".join(span)
        if text in seen:
            continue
        segment = parent_window(start, end, bounds)
        if segment is None:
            continue
        seen.add(text)
        views.append(WindowView(segment, len(views), start, end, text))
    return views


__all__ = [
    "ATOMIZER_STRATEGIES",
    "AtomizerStrategy",
    "WindowView",
    "parent_window",
    "window_bounds",
    "window_views",
]
