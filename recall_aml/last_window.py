"""LW-1: after the top ranks, return each retrieved session's last window.

A session's decisive text often sits in its closing turn ("decision: ..."), and ranking finds the
session's earlier windows but not that one. Measured offline on 2026-09-27 against the deploy
candidate's own Coding retrieval: appending each top-10 session's last window raised fact-term
coverage from 0.726 to 0.966, against 0.770 for the same number of next-ranked windows.

The rule, as pre-registered: take the top ``depth`` hits; for every session with a raw window
among them, in order of its first appearance, append that session's last raw window unless it is
already there; everything else keeps its order after them. "Last" is the window of the session's
latest Add (``event_time``), then its highest ``segment``, so a session sent over several Adds
still resolves to its final text. Nothing is removed: a caller that truncates to ``top_k`` drops
from the tail.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from recall.types import Chunk, ScoredChunk

LAST_WINDOW_DEPTH = 10


def last_window(chunks: Sequence[Chunk]) -> Chunk | None:
    """The session's final raw window, or ``None`` when it has none."""
    raws = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("kind") == "raw"
        and isinstance(chunk.metadata.get("segment"), int)
        and not isinstance(chunk.metadata.get("segment"), bool)
    ]
    if not raws:
        return None
    return max(raws, key=lambda c: (str(c.metadata.get("event_time") or ""), int(c.metadata["segment"]), c.id))


def with_last_windows(
    hits: Sequence[ScoredChunk],
    chunks_for_source: Callable[[str], Sequence[Chunk]],
    depth: int = LAST_WINDOW_DEPTH,
) -> tuple[list[ScoredChunk], int]:
    """``hits`` with each top-``depth`` session's last window placed right after the top block."""
    head = list(hits[:depth])
    seen = {hit.chunk.id for hit in head}
    sources = list(dict.fromkeys(hit.chunk.source for hit in head if hit.chunk.metadata.get("kind") == "raw"))
    floor = min((hit.score for hit in head), default=0.0)
    added: list[ScoredChunk] = []
    for source in sources:
        last = last_window(chunks_for_source(source))
        if last is not None and last.id not in seen:
            seen.add(last.id)
            added.append(ScoredChunk(last, floor))
    tail = [hit for hit in hits[depth:] if hit.chunk.id not in seen]
    return [*head, *added, *tail], len(added)
