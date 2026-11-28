"""Promotion may move at most two linked raw chunks into ranks 9 and 10, and nothing else.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.promotion``):

- ``test_edges_by_offset_overlap``: overlap made inclusive (``start <= chunk_end``) fails on the
  edge tuple of a span starting at a chunk's end (the touching chunk is linked).
- ``test_chunk_without_offsets_gets_no_edge``: missing offsets read as ``0`` and ``len(text)`` fails
  on ``()``.
- ``test_edges_follow_each_span_of_a_code_chunk`` (recorded 2026-11-27): ``placements`` ignoring
  ``TEXT_SPANS_KEY`` fails on ``() == ("C.m",)``; and with ``recall.index._line_spans`` making each
  piece one run, the class chunk covers its method and the test fails on ``("C", "C.m")``.
- ``test_malformed_text_spans_place_nothing`` (recorded 2026-11-28, CCA run 20261128T105336Z): the
  pre-fix ``placements`` (one comprehension unpacking every run as ``start, end, offset``, no
  shape check) fails inside ``placements`` with ``ValueError: not enough values to unpack``
  (``promotion.py:46``) on the first malformed value, which is the defect itself, a crash where an
  unplaced chunk belongs; the fixed code passes.
- ``test_text_spans_held_in_a_tuple_are_placed`` (recorded 2026-11-28, CCA FA-1-5): with
  ``placements``'s container check back to ``isinstance(runs, list)`` (commit 7206f0d5f), it fails
  on ``assert [] == [(0, 3, 0), (5, 7, 3)]``: a tuple of valid runs placed nothing.
- ``test_top_eight_never_move``: protecting only ``pool[:PROTECTED_PREFIX - 1]`` lets rank 8 be
  promoted again and fails on membership (a duplicate id).
- ``test_at_most_two_promoted_into_ranks_nine_and_ten``: ``MAX_PROMOTIONS = 3`` fails on
  ``promoted_count == 2``.
- ``test_repeated_chunk_id_keeps_every_copy``: the pre-fix id-keyed reorder (hits looked up by id,
  the last copy winning, and every copy of a promoted id dropped from the tail) fails on
  ``len(result.hits) == len(pool)``; ``setdefault`` replaced by plain assignment (the last copy's
  rank) fails on ``result.hits[PROTECTED_PREFIX] is pool[10]``; the tail emitted in reverse fails
  on ``result.hits[12] is pool[12]``.
- ``test_membership_and_scores_unchanged`` (membership half): keeping promoted hits in the tail
  (``and rank not in moved_ranks`` dropped from the tail filter) fails on the sorted ids.
- ``test_only_first_twenty_seeds_count``: ``seeds[:SEED_K]`` widened to ``seeds`` fails on
  ``relation_hits``.
- ``test_target_outside_pool_ignored``: admitting a target outside the pool as a synthetic hit
  fails on the returned ids.
- ``test_seeds_without_edges_take_seed_slots``: dropping edgeless seeds before ``[:SEED_K]``
  (``[seed for seed in seeds if seed[1]][:SEED_K]``) fails on ``promoted_count == 0``.
- ``test_earlier_seed_outweighs_later``: a constant weight in place of ``1 / (RRF_K + seed_rank)``
  fails on the promoted order (ties fall back to baseline rank).
- ``test_repeated_record_target_counted_once``: the ``seen`` check removed fails on
  ``relation_hits`` and the promoted order.
- ``test_membership_and_scores_unchanged`` (scores half): a promoted hit rebuilt with score 1.0
  fails on the score check.
- ``test_ties_break_by_baseline_rank_then_id``: sorting by id before baseline rank fails on the
  promoted order.
"""

from __future__ import annotations

from recall.compiled_graph.promotion import (
    MAX_PROMOTIONS,
    PROTECTED_PREFIX,
    SEED_K,
    placements,
    promote,
    record_edges,
)
from recall.compiled_graph.records import CompiledRecord
from recall.index import CODE_ANCHOR_KEY, code_pieces
from recall.types import TEXT_SPANS_KEY, Chunk, ScoredChunk


def _chunk(chunk_id: str, start: int | None = None, end: int | None = None) -> Chunk:
    metadata: dict[str, object] = {"file": "notes/a.md"}
    if start is not None:
        metadata["text_start"] = start
        metadata["text_end"] = end
    return Chunk(id=chunk_id, source="notes/a.md", text="t", metadata=metadata)


def _record(*spans: tuple[int, int]) -> CompiledRecord:
    return CompiledRecord(kind="fact", fields=("", "x"), entities=(), anchor_ids=("a000_x",), spans=spans)


def _pool(n: int = 30) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=_chunk(f"c{i:02d}"), score=1.0 - i / 100) for i in range(n)]


def _ids(hits: list[ScoredChunk]) -> list[str]:
    return [hit.chunk.id for hit in hits]


def test_edges_by_offset_overlap() -> None:
    chunks = [_chunk("a", 0, 500), _chunk("b", 500, 1_000), _chunk("c", 1_000, 1_500)]
    assert record_edges(_record((400, 1_000)), chunks) == ("a", "b")
    assert record_edges(_record((1_200, 1_300), (0, 10)), chunks) == ("a", "c")
    assert record_edges(_record((1_000, 1_100)), chunks) == ("c",)


def test_chunk_without_offsets_gets_no_edge() -> None:
    assert record_edges(_record((0, 100)), [_chunk("a")]) == ()


def test_edges_follow_each_span_of_a_code_chunk() -> None:
    """The class chunk holds the lines before and after its method, and only those: a record on
    the method links the method's chunk alone, one on ``y = 2`` the class's chunk alone."""
    source = "class C:\n    x = 1\n\n    def m(self):\n        return 1\n\n    y = 2\n"
    chunks = [
        Chunk(id=meta[CODE_ANCHOR_KEY]["symbol"], source="c.py", text=text, metadata=meta)
        for text, meta in code_pieces(source, file="c.py", repo=None)
    ]
    method = source.index("        return 1")
    after = source.index("    y = 2")
    assert record_edges(_record((method, method + 5)), chunks) == ("C.m",)
    assert record_edges(_record((after, after + 5)), chunks) == ("C",)


def test_malformed_text_spans_place_nothing() -> None:
    """A stored ``text_spans`` value of the wrong shape leaves the chunk unplaced, as a missing one
    does, and a well-formed run beside a malformed one is still placed."""
    for runs in ("abc", 5, [[1, 2]], [None], [[0, 3, 0, 1]]):
        chunk = Chunk(id="m", source="m.py", text="abc", metadata={TEXT_SPANS_KEY: runs})
        assert placements(chunk) == [], runs
        assert record_edges(_record((0, 3)), [chunk]) == (), runs
    mixed = Chunk(id="m", source="m.py", text="abc", metadata={TEXT_SPANS_KEY: [[0, 3, 0], [1, 2]]})
    assert placements(mixed) == [(0, 3, 0)]


def test_text_spans_held_in_a_tuple_are_placed() -> None:
    """A ``text_spans`` value held as a tuple of runs is placed as the same runs in a list are:
    the shape check accepts a tuple for the container as it does for each run."""
    runs = [(0, 3, 0), (5, 7, 3)]
    as_list = Chunk(id="m", source="m.py", text="abcde", metadata={TEXT_SPANS_KEY: runs})
    as_tuple = Chunk(id="m", source="m.py", text="abcde", metadata={TEXT_SPANS_KEY: tuple(runs)})
    assert placements(as_tuple) == placements(as_list) == runs


def test_top_eight_never_move() -> None:
    pool = _pool()
    result = promote(pool, [("r0", ("c03", "c07", "c20"))])
    assert PROTECTED_PREFIX == 8
    assert _ids(result.hits)[:8] == _ids(pool)[:8]
    assert sorted(_ids(result.hits)) == sorted(_ids(pool))
    assert _ids(result.hits)[8] == "c20"


def test_at_most_two_promoted_into_ranks_nine_and_ten() -> None:
    pool = _pool()
    result = promote(pool, [("r0", ("c20",)), ("r1", ("c21",)), ("r2", ("c22",))])
    assert MAX_PROMOTIONS == 2
    assert result.promoted_count == 2
    assert _ids(result.hits)[8:10] == ["c20", "c21"]
    assert _ids(result.hits)[10] == "c08"


def test_membership_and_scores_unchanged() -> None:
    pool = _pool()
    result = promote(pool, [("r0", ("c25",))])
    assert sorted(_ids(result.hits)) == sorted(_ids(pool))
    scores = {hit.chunk.id: hit.score for hit in pool}
    assert all(hit.score == scores[hit.chunk.id] for hit in result.hits)


def test_repeated_chunk_id_keeps_every_copy() -> None:
    pool = _pool()
    pool[12] = ScoredChunk(chunk=_chunk("c10"), score=0.05)
    result = promote(pool, [("r0", ("c10",))])
    assert len(result.hits) == len(pool)
    assert sorted(_ids(result.hits)) == sorted(_ids(pool))
    assert result.hits[PROTECTED_PREFIX] is pool[10]
    assert result.hits[12] is pool[12]


def test_only_first_twenty_seeds_count() -> None:
    pool = _pool()
    seeds = [(f"r{i}", ("c25",)) for i in range(25)]
    result = promote(pool, seeds)
    assert SEED_K == 20
    assert result.relation_hits == 20


def test_target_outside_pool_ignored() -> None:
    pool = _pool(12)
    result = promote(pool, [("r0", ("elsewhere",))])
    assert _ids(result.hits) == _ids(pool)
    assert result.promoted_count == 0


def test_ties_break_by_baseline_rank_then_id() -> None:
    pool = _pool()
    pool = pool[:8] + list(reversed(pool[8:]))
    result = promote(pool, [("r0", ("c22", "c25", "c27"))])
    assert _ids(result.hits)[8:10] == ["c27", "c25"]


def test_seeds_without_edges_take_seed_slots() -> None:
    pool = _pool()
    seeds = [(f"r{i}", ()) for i in range(SEED_K)] + [("late", ("c25",))]
    result = promote(pool, seeds)
    assert result.promoted_count == 0
    assert result.relation_hits == 0


def test_earlier_seed_outweighs_later() -> None:
    result = promote(_pool(), [("r0", ("c25",)), ("r1", ("c20",))])
    assert _ids(result.hits)[8:10] == ["c25", "c20"]


def test_repeated_record_target_counted_once() -> None:
    seeds = [("r0", ("c25", "c25")), ("r1", ("c20",)), ("r2", ("c20",))]
    result = promote(_pool(), seeds)
    assert result.relation_hits == 3
    assert _ids(result.hits)[8:10] == ["c20", "c25"]


def test_no_seeds_or_empty_pool_is_identity() -> None:
    pool = _pool()
    assert _ids(promote(pool, []).hits) == _ids(pool)
    assert promote([], [("r0", ("c01",))]).hits == []
