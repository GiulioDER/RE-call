"""K-1: mark who is speaking inside a returned content-only window, at Search time.

Pre-registration: docs/preregistrations/2026-09-26-aml-c9-speaker-at-render-time.md. A content-only
window is 160 words cut from a session's messages joined end to end (``build_chunks``), so a window
that spans a user turn and an assistant turn reads as one voice. This inserts ``[user]`` or
``[assistant]`` where each message's words begin inside the window, and at the window's start for
the message it opens in, but only in a window that spans two or more roles. Items, order and scores
are unchanged; image items and items whose boundaries are unknown pass through untouched.

Boundaries are ``(role, start, end)`` word ranges over the session's content-only word sequence,
laid out exactly as ``build_chunks`` lays them (``message_word_ranges``). ``locate`` finds a window's
position in that sequence for retrieval already stored; a window found nowhere or at more than one
position is left unmarked, never guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

WordRange = tuple[str, int, int]


def message_word_ranges(messages: Sequence[tuple[str, str]]) -> list[WordRange]:
    """``(role, start, end)`` per message, as ``build_chunks`` counts words for content-only windows."""
    ranges: list[WordRange] = []
    cursor = 0
    for role, content in messages:
        count = len(content.split())
        ranges.append((role, cursor, cursor + count))
        cursor += count
    return ranges


def locate(window: str, session_words: Sequence[str]) -> int | None:
    """The unique word offset at which ``window``'s words occur contiguously, else ``None``."""
    words = window.split()
    if not words:
        return None
    width = len(words)
    found: int | None = None
    for start in range(len(session_words) - width + 1):
        if session_words[start] == words[0] and list(session_words[start : start + width]) == words:
            if found is not None:
                return None
            found = start
    return found


def mark_window(window: str, start: int, ranges: Sequence[WordRange]) -> str:
    """``window`` with a role marker at each message start inside it, if it spans two roles."""
    words = window.split()
    end = start + len(words)
    overlapping = [r for r in ranges if r[1] < end and r[2] > start]
    if len({role for role, _, _ in overlapping}) < 2:
        return window
    marks: dict[int, str] = {}
    for role, message_start, _ in overlapping:
        marks[max(message_start, start) - start] = f"[{role.lower()}]"
    out: list[str] = []
    for index, word in enumerate(words):
        if index in marks:
            out.append(marks[index])
        out.append(word)
    return " ".join(out)


def speaker_marked_items(
    items: Sequence[Any], boundaries: Callable[[Any], tuple[int, Sequence[WordRange]] | None]
) -> list[Any]:
    """Each text item re-rendered by ``mark_window`` where ``boundaries`` knows its position."""
    marked = []
    for item in items:
        content = getattr(item, "content", None)
        located = boundaries(item) if isinstance(content, str) else None
        if located is None:
            marked.append(item)
            continue
        start, ranges = located
        rendered = mark_window(content, start, ranges)
        marked.append(item if rendered == content else item.model_copy(update={"content": rendered}))
    return marked
