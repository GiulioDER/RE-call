"""Server-resolved edges from compiled records to raw chunks, and bounded raw promotion.

Ported from ``recall_aml/graph.py`` (``attach_grounded_relations``, ``promote_grounded_raw``). An
edge exists only where a cited anchor window overlaps a chunk's own offsets in the same source;
promotion never moves ranks 1 to 8, moves at most two, and changes no membership and no score.

Three differences from the AML code: a (record, target) pair counts once however many of the
record's spans touch the target, because anchors overlap by 160 characters and AML counted one
target twice through that overlap; a chunk id the baseline repeats is promoted by its first copy
with every copy kept, where AML moved the last copy and dropped the others; and edges are not
re-checked here at query time. Nothing in
this module checks where the edges came from: a caller must pass only edges built from the chunks
it is serving.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard

from recall.compiled_graph.records import CompiledRecord
from recall.types import TEXT_SPANS_KEY, Chunk, ScoredChunk

#: The AML grounded graph's constants (``recall_aml/graph.py``), unchanged.
SEED_K = 20
PROTECTED_PREFIX = 8
MAX_PROMOTIONS = 2
RRF_K = 60


def _is_offset(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def placements(chunk: Chunk) -> list[tuple[int, int, int]]:
    """Where ``chunk``'s text sits in its source, as ``(source_start, source_end, text_offset)``.

    One per run `TEXT_SPANS_KEY` records, which the code chunker writes; otherwise the one
    ``text_start``/``text_end`` span at text offset 0. Empty when the chunk records neither. A run
    that is not three non-negative integers with its end past its start is dropped, so malformed
    stored metadata leaves a chunk unplaced rather than failing the build.
    """
    runs = chunk.metadata.get(TEXT_SPANS_KEY)
    if runs is None:
        runs = [[chunk.metadata.get("text_start"), chunk.metadata.get("text_end"), 0]]
    if not isinstance(runs, list):
        return []
    placed: list[tuple[int, int, int]] = []
    for run in runs:
        if not isinstance(run, (list, tuple)) or len(run) != 3:
            continue
        start, end, offset = run
        if _is_offset(start) and _is_offset(end) and _is_offset(offset) and end > start:
            placed.append((start, end, offset))
    return placed


def record_edges(record: CompiledRecord, chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Ids of ``chunks`` (one source's, in order) that overlap any span of ``record``."""
    return tuple(
        chunk.id
        for chunk in chunks
        if any(
            start < chunk_end and end > chunk_start
            for chunk_start, chunk_end, _ in placements(chunk)
            for start, end in record.spans
        )
    )


@dataclass(frozen=True)
class Promotion:
    hits: list[ScoredChunk]
    relation_hits: int = 0
    candidate_count: int = 0
    promoted_count: int = 0


def promote(
    baseline: Sequence[ScoredChunk], seeds: Sequence[tuple[str, tuple[str, ...]]]
) -> Promotion:
    """Promote raw hits linked from the first ``SEED_K`` seeds into ranks 9 and 10.

    ``seeds`` are ``(record_key, edge_chunk_ids)`` in seed rank order, and must include records
    with no edges: every seed takes a rank and one of the ``SEED_K`` slots, as on the AML service.
    Each distinct (record, target) adds ``1 / (RRF_K + seed_rank)`` to the target; ties break by
    baseline rank, then id. A chunk id the baseline holds twice is promoted by moving its first
    copy, and every other copy keeps its place in the order of the tail.
    """
    pool = list(baseline)
    if not pool or not seeds:
        return Promotion(pool)
    baseline_rank: dict[str, int] = {}
    for rank, hit in enumerate(pool):
        baseline_rank.setdefault(hit.chunk.id, rank)
    scores: dict[str, float] = {}
    seen: set[tuple[str, str]] = set()
    relation_hits = 0
    for seed_rank, (record_key, targets) in enumerate(seeds[:SEED_K], start=1):
        for target in targets:
            if target not in baseline_rank or (record_key, target) in seen:
                continue
            seen.add((record_key, target))
            relation_hits += 1
            scores[target] = scores.get(target, 0.0) + 1.0 / (RRF_K + seed_rank)
    protected = {hit.chunk.id for hit in pool[:PROTECTED_PREFIX]}
    candidates = sorted(
        (chunk_id for chunk_id in scores if chunk_id not in protected),
        key=lambda chunk_id: (-scores[chunk_id], baseline_rank[chunk_id], chunk_id),
    )
    promoted = candidates[:MAX_PROMOTIONS]
    if not promoted:
        return Promotion(pool, relation_hits=relation_hits, candidate_count=len(candidates))
    moved_ranks = [baseline_rank[chunk_id] for chunk_id in promoted]
    reordered = [
        *pool[:PROTECTED_PREFIX],
        *(pool[rank] for rank in moved_ranks),
        *(
            hit
            for rank, hit in enumerate(pool)
            if rank >= PROTECTED_PREFIX and rank not in moved_ranks
        ),
    ]
    return Promotion(
        reordered,
        relation_hits=relation_hits,
        candidate_count=len(candidates),
        promoted_count=len(promoted),
    )
