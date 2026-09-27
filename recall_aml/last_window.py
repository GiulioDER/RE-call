"""LW-1: after the top ranks, return each retrieved session's last window.

A session's decisive text often sits in its closing turn ("decision: ..."), and ranking finds the
session's earlier windows but not that one. Measured offline on 2026-09-27 against the deploy
candidate's own Coding retrieval: appending each top-10 session's last window raised fact-term
coverage from 0.726 to 0.966, against 0.770 for the same number of next-ranked windows.

The rule, as pre-registered: take the top ``depth`` hits; for every session with a raw window
among them, in order of its first appearance, append that session's last raw window unless it is
already there; everything else keeps its order after them. Nothing is removed: a caller that
truncates to ``top_k`` drops from the tail.

"Last" is defined only for a session stored by ONE Add: its highest ``segment``. Every Add numbers
its windows from 0 and dates them by its own latest message, so a session split over several Adds
(AML's Textual track sends at most 20 messages per Add, and LoCoMo gives every turn of a session
the same timestamp) has no reliable last window, and it gets none. Owner decision 2026-09-27; a
Coding session arrives as one Add, so Coding is unchanged by it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from recall.types import Chunk, ScoredChunk

LAST_WINDOW_DEPTH = 10


def _segment(chunk: Chunk) -> int | None:
    segment = chunk.metadata.get("segment")
    return segment if isinstance(segment, int) and not isinstance(segment, bool) else None


def last_window(chunks: Sequence[Chunk]) -> Chunk | None:
    """The session's final raw window, or ``None`` when it has none or was stored by several Adds.

    Adds are counted over every raw row, not only text windows: an image-bearing Add stores its
    raw rows as ``kind="multimodal"``, each message's from segment 0, so a session mixing text and
    image Adds is several Adds and gets no window (audit cca789b, BUG-002).
    """
    stored = [
        chunk
        for chunk in chunks
        if (chunk.metadata.get("record_type") == "raw" or chunk.metadata.get("kind") == "raw")
        and _segment(chunk) is not None
    ]
    if sum(_segment(chunk) == 0 for chunk in stored) > 1:
        return None
    raws = [chunk for chunk in stored if chunk.metadata.get("kind") == "raw"]
    if not raws:
        return None
    return max(raws, key=lambda c: (int(c.metadata["segment"]), c.id))


def with_last_windows(
    hits: Sequence[ScoredChunk],
    chunks_for_source: Callable[[str], Sequence[Chunk]],
    depth: int = LAST_WINDOW_DEPTH,
) -> tuple[list[ScoredChunk], int]:
    """``hits`` with each top-``depth`` session's last window placed right after the top block.

    It keeps the first ``depth`` HITS, not the first ``depth`` items served. A renderer that drops
    a head hit (``render_full_evidence`` skips a superseded compiled record unless the query is
    historical) frees a place that an appended window then takes. That happens only where the
    store holds compiled records, which is the context route: C9's code-route store holds raw
    windows only. It is inside the LoCoMo LW-1 measurement (recall-lab 218db49), so it is recorded
    here rather than changed (audit cca789b, STAKES-003).
    """
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
