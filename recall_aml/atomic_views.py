"""Atomic rescue views built inside Add, stored beside the corpus they rescue.

The C8 atomic stage read a file artifact that ``scripts/build_aml_atomic_rescue_artifact.py`` built
by hand after ingest and bound to the served corpus fingerprint. AML never pauses between Add and
Search, every Add changes that fingerprint, and the contract requires each chunk to be searchable
before Add returns (checked 2026-09-23 on https://agentmemoryleaderboard.ai). Under the official
flow the stage therefore fell back on every query; the live C8 reference measured exactly that
with no artifact present: 187 of 187 queries attempted, 0 active, 187 fallback.

Here the views are a function of one Add request alone. The request's raw windows are exact slices
of one word sequence, so the sequence is rebuilt from them, cut into ``micro`` views with the
confirmed atomizer (``recall.atomizer.window_views``), and each view is stored with its parent
window's chunk id in an isolated tenant of its scope. Nothing depends on any other request, so a
retried Add rewrites byte-identical rows, and a streaming Search between two Adds sees every view
of every Add that has returned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

from recall.atomic_rescue import AtomicRescueSelection, AtomicRescueSelectionError
from recall.atomizer import ATOMIZER_STRATEGIES, window_views
from recall.types import Chunk, ScoredChunk
from recall_aml.identity import canonical_digest


#: Identity of the view derivation. Changing the atomizer, its parameters or the view metadata
#: must change this, because stored views are never recomputed.
ATOMIC_VIEW_PROFILE = "aml-add-micro-v1"
ATOMIC_VIEW_STRATEGY = "micro"
ATOMIC_VIEW_RECORD_TYPE = "atomic_view"
#: The dense prefix the rescue never replaces, as in ``recall.atomic_rescue``.
PROTECTED_PARENTS = 5


class BuildRefusal(RuntimeError):
    """The raw windows cannot yield trustworthy atomic views; nothing is written."""


@dataclass(frozen=True)
class RebuiltSession:
    session: str
    text: str
    window_size: int
    window_stride: int
    chunk_ids: tuple[str, ...]


def _int(metadata: Mapping[str, object], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildRefusal(f"raw window lacks a non-negative integer {name}")
    return value


def rebuild_sessions(chunks: Iterable[Chunk]) -> list[RebuiltSession]:
    """Reassemble each session's word sequence from its raw content-only windows."""

    by_session: dict[str, dict[int, Chunk]] = defaultdict(dict)
    for chunk in chunks:
        metadata = chunk.metadata
        if metadata.get("record_type") != "raw":
            continue
        session = metadata.get("source_session_id")
        if not isinstance(session, str) or not session:
            raise BuildRefusal("raw window lacks source_session_id")
        segment = _int(metadata, "segment")
        previous = by_session[session].get(segment)
        if previous is not None and previous.text != chunk.text:
            raise BuildRefusal(f"session {session!r} has two different windows at segment {segment}")
        by_session[session][segment] = chunk

    rebuilt: list[RebuiltSession] = []
    for session in sorted(by_session):
        windows = by_session[session]
        ordered = [windows[index] for index in sorted(windows)]
        counts = {_int(chunk.metadata, "segment_count") for chunk in ordered}
        sizes = {_int(chunk.metadata, "word_window_size") for chunk in ordered}
        strides = {_int(chunk.metadata, "word_window_stride") for chunk in ordered}
        if len(counts) != 1 or len(sizes) != 1 or len(strides) != 1:
            raise BuildRefusal(f"session {session!r} mixes window geometries")
        count, size, stride = counts.pop(), sizes.pop(), strides.pop()
        if sorted(windows) != list(range(count)):
            raise BuildRefusal(f"session {session!r} is missing window segments")
        if not 0 < stride <= size:
            raise BuildRefusal(f"session {session!r} has an invalid window stride")
        words: list[str] = []
        for chunk in ordered:
            start = _int(chunk.metadata, "word_start")
            end = _int(chunk.metadata, "word_end")
            tokens = chunk.text.split()
            if end - start != len(tokens) or start > len(words):
                raise BuildRefusal(f"session {session!r} window offsets are inconsistent")
            overlap = words[start:]
            if tokens[: len(overlap)] != overlap:
                raise BuildRefusal(f"session {session!r} windows disagree on their overlap")
            words.extend(tokens[len(overlap) :])
        rebuilt.append(
            RebuiltSession(
                session,
                " ".join(words),
                size,
                stride,
                tuple(chunk.id for chunk in ordered),
            )
        )
    return rebuilt


def plan_views(
    sessions: Sequence[RebuiltSession], strategy: str
) -> tuple[list[str], list[dict[str, object]]]:
    """Return the view texts and the artifact metadata rows, in one deterministic order."""

    if strategy not in ATOMIZER_STRATEGIES:
        raise BuildRefusal(f"unknown atomizer strategy {strategy!r}")
    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for rebuilt in sessions:
        for view in window_views(
            rebuilt.text,
            window_size=rebuilt.window_size,
            window_stride=rebuilt.window_stride,
            strategy=strategy,  # type: ignore[arg-type]
        ):
            texts.append(view.text)
            rows.append(
                {
                    "chunk_id": rebuilt.chunk_ids[view.parent_segment],
                    "source": rebuilt.session,
                    "parent_ordinal": view.parent_segment,
                    "view_ordinal": view.view_ordinal,
                }
            )
    if not rows:
        raise BuildRefusal("the tenant yields no atomic views")
    return texts, rows


def max_views_per_parent(window_size: int, micro_stride: int = 12) -> int:
    """An upper bound on the views any one window can own.

    A ``micro`` view starts on a multiple of the micro stride and is assigned to one window that
    contains it, so a window owns at most one view per stride step inside its span.
    """

    if window_size < 1 or micro_stride < 1:
        raise ValueError("window size and micro stride must be positive")
    return math.ceil(window_size / micro_stride) + 1


def view_query_width(window_size: int) -> int:
    """How many nearest views guarantee one outside the protected parents, if any exists.

    The protected parents own at most ``PROTECTED_PARENTS * max_views_per_parent`` views, so one
    more row than that must include the best view of some other parent whenever there is one.
    """

    return PROTECTED_PARENTS * max_views_per_parent(window_size) + 1


def build_view_chunks(raw_chunks: Sequence[Chunk]) -> list[Chunk]:
    """Build the atomic view rows of one Add request from its raw windows.

    Returns an empty list when the request has no raw windows, or when no span clears the
    atomizer's content floor (a request of a few words): neither is an error, the rescue simply
    has nothing new to offer for that request.
    """

    windows = [chunk for chunk in raw_chunks if chunk.metadata.get("record_type") == "raw"]
    if not windows:
        return []
    sources = {chunk.source for chunk in windows}
    if len(sources) != 1:
        raise BuildRefusal("one Add request must yield windows of one source")
    (source,) = sources
    sessions = rebuild_sessions(windows)
    if len(sessions) != 1:
        raise BuildRefusal("one Add request must yield one session")
    (rebuilt,) = sessions
    bound = max_views_per_parent(rebuilt.window_size)
    owned: dict[str, int] = defaultdict(int)
    views: list[Chunk] = []
    for view in window_views(
        rebuilt.text,
        window_size=rebuilt.window_size,
        window_stride=rebuilt.window_stride,
        strategy=ATOMIC_VIEW_STRATEGY,
    ):
        parent = rebuilt.chunk_ids[view.parent_segment]
        owned[parent] += 1
        if owned[parent] > bound:
            raise BuildRefusal("a window owns more atomic views than the query width assumes")
        identity = {
            "profile": ATOMIC_VIEW_PROFILE,
            "parent_chunk_id": parent,
            "word_start": view.word_start,
            "word_end": view.word_end,
            "text": view.text,
        }
        views.append(
            Chunk(
                id="view_" + canonical_digest(identity),
                source=source,
                text=view.text,
                metadata={
                    "record_type": ATOMIC_VIEW_RECORD_TYPE,
                    "kind": ATOMIC_VIEW_RECORD_TYPE,
                    "atomic_view_profile": ATOMIC_VIEW_PROFILE,
                    "parent_chunk_id": parent,
                    "source_session_id": rebuilt.session,
                    "parent_ordinal": view.parent_segment,
                    "view_ordinal": view.view_ordinal,
                    "word_start": view.word_start,
                    "word_end": view.word_end,
                },
            )
        )
    return views


def _view_key(hit: ScoredChunk) -> tuple[str, int, int, str, str]:
    metadata = hit.chunk.metadata
    return (
        hit.chunk.source,
        int(metadata["parent_ordinal"]),
        int(metadata["view_ordinal"]),
        str(metadata["parent_chunk_id"]),
        hit.chunk.id,
    )


def select_view_rescue(
    view_hits: Sequence[ScoredChunk], dense: Sequence[ScoredChunk]
) -> AtomicRescueSelection:
    """Select the best view whose parent is outside the dense top five.

    Mirrors ``recall.atomic_rescue._select_atomic_rescue_unlocked``: the same protected prefix,
    the same refusals, the highest score wins, and ties go to the lowest (source, parent ordinal,
    view ordinal), extended with the parent and view ids because ordinals restart per request.
    ``view_hits`` must be the nearest views by the same query vector, at least
    ``view_query_width`` of them when the scope holds that many.
    """

    if len(dense) < PROTECTED_PARENTS:
        raise AtomicRescueSelectionError("atomic rescue requires five dense candidates")
    protected = [hit.chunk.id for hit in dense[:PROTECTED_PARENTS]]
    if len(set(protected)) != PROTECTED_PARENTS:
        raise AtomicRescueSelectionError("atomic rescue dense prefix repeats a parent")
    excluded = set(protected)
    candidates: list[ScoredChunk] = []
    for hit in view_hits:
        metadata = hit.chunk.metadata
        if metadata.get("record_type") != ATOMIC_VIEW_RECORD_TYPE:
            raise AtomicRescueSelectionError("atomic view store returned a non-view row")
        if not math.isfinite(float(hit.score)):
            raise AtomicRescueSelectionError("atomic rescue produced nonfinite scores")
        if str(metadata.get("parent_chunk_id", "")) in excluded:
            continue
        candidates.append(hit)
    if not candidates:
        raise AtomicRescueSelectionError("atomic rescue has no parent outside dense top five")
    best_score = max(float(hit.score) for hit in candidates)
    winner = min((hit for hit in candidates if float(hit.score) == best_score), key=_view_key)
    metadata = winner.chunk.metadata
    return AtomicRescueSelection(
        str(metadata["parent_chunk_id"]),
        winner.chunk.source,
        int(metadata["parent_ordinal"]),
        int(metadata["view_ordinal"]),
        best_score,
    )


__all__ = [
    "ATOMIC_VIEW_PROFILE",
    "ATOMIC_VIEW_RECORD_TYPE",
    "ATOMIC_VIEW_STRATEGY",
    "BuildRefusal",
    "PROTECTED_PARENTS",
    "RebuiltSession",
    "build_view_chunks",
    "max_views_per_parent",
    "plan_views",
    "rebuild_sessions",
    "select_view_rescue",
    "view_query_width",
]
