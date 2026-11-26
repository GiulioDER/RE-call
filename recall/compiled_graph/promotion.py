"""Server-resolved edges from compiled records to raw chunks, and bounded raw promotion.

Ported from ``recall_aml/graph.py`` (``attach_grounded_relations``, ``promote_grounded_raw``). An
edge exists only where a cited anchor window overlaps a chunk's own offsets in the same source;
promotion never moves ranks 1 to 8, moves at most two, and changes no membership and no score.

Two differences from the AML code: a (record, target) pair counts once however many of the
record's spans touch the target, because anchors overlap by 160 characters and AML counted one
target twice through that overlap; and edges are not re-checked here at query time, because the
artifact that carries them is refused unless its lineage is the serving one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from recall.compiled_graph.records import CompiledRecord
from recall.types import Chunk, ScoredChunk

#: The AML grounded graph's constants (``recall_aml/graph.py``), unchanged.
SEED_K = 20
PROTECTED_PREFIX = 8
MAX_PROMOTIONS = 2
RRF_K = 60


def _offsets(chunk: Chunk) -> tuple[int, int] | None:
    start = chunk.metadata.get("text_start")
    end = chunk.metadata.get("text_end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return None
    return start, end


def record_edges(record: CompiledRecord, chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Ids of ``chunks`` (one source's, in order) that overlap any span of ``record``."""
    linked: list[str] = []
    for chunk in chunks:
        offsets = _offsets(chunk)
        if offsets is None:
            continue
        chunk_start, chunk_end = offsets
        if any(start < chunk_end and end > chunk_start for start, end in record.spans):
            linked.append(chunk.id)
    return tuple(linked)


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
    baseline rank, then id.
    """
    pool = list(baseline)
    if not pool or not seeds:
        return Promotion(pool)
    raw_by_id = {hit.chunk.id: hit for hit in pool}
    baseline_rank = {hit.chunk.id: rank for rank, hit in enumerate(pool)}
    scores: dict[str, float] = {}
    seen: set[tuple[str, str]] = set()
    relation_hits = 0
    for seed_rank, (record_key, targets) in enumerate(seeds[:SEED_K], start=1):
        for target in targets:
            if target not in raw_by_id or (record_key, target) in seen:
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
    promoted_set = set(promoted)
    reordered = [
        *pool[:PROTECTED_PREFIX],
        *(raw_by_id[chunk_id] for chunk_id in promoted),
        *(hit for hit in pool[PROTECTED_PREFIX:] if hit.chunk.id not in promoted_set),
    ]
    return Promotion(
        reordered,
        relation_hits=relation_hits,
        candidate_count=len(candidates),
        promoted_count=len(promoted),
    )
