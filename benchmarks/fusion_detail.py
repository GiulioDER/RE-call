"""Recover the two fusion quantities `search()` computes and then throws away.

`HybridRetriever.search` orders the pool by the **RRF fused score** and then overwrites every
hit's `score` with the **dense cosine** (`retriever.py`, `_rescored`). A fixture built from its
output therefore records a curve that is not the ranking criterion, which is exactly how
`ratio_8_over_1` came to predict a retrieval miss with no available explanation: see
`results/enterprise_rag/PREREGISTRATION-triage-mechanism.md`.

This module reads the fused score and each leg's rank off the shared `_Legs` seam, so a benchmark
can capture them without a second retrieval pass. Retrieval is the expensive half (about 52
seconds per question on the EnterpriseRAG index, dominated by SPLADE on CPU), and it is paid once.

⚠️ **This restates the handful of `search()` lines that turn legs into a ranked list, and that is
a genuine drift hazard** — the same one that motivated extracting `_Legs`. It is held in check by
`tests/test_bench_fusion_detail.py`, whose first test pins this ranked output to `search()`'s hit
for hit and score for score. Change the fusion in `search()` and that test goes red. Nothing here
may be "simplified" in a way that makes it stop mirroring; the mirroring IS the contract.

Reranking is deliberately not modelled. A reranked run reorders the pool afterwards, so `fuse()`
describes the fused ordering only, which is what the mechanism question is about.
"""

from __future__ import annotations

from dataclasses import dataclass

from recall.guards import DEFAULT_GAP_THRESHOLD, gap_warning
from recall.retriever import _Legs, _rescored, _rrf
from recall.types import ScoredChunk

#: Leg order, fixed. `ranks` is a tuple, so its meaning is positional and a reader needs this.
#: It matches the order `search()` passes the rankings to `_rrf`, which also fixes tie-breaking:
#: `sorted` is stable and `_rrf` inserts in leg order, so equal fused scores fall out dense first.
LEG_NAMES = ("dense", "lexical", "learned")


@dataclass(frozen=True)
class FusedHit:
    """One pool member, with the score that ranked it and the legs that found it."""

    #: The hit exactly as `search()` would report it, carrying the DENSE cosine as its score.
    hit: ScoredChunk
    #: The RRF score that determined this chunk's position: sum of 1/(60 + rank + 1) over legs.
    fused_score: float
    #: Zero-based rank within each leg, positionally per `LEG_NAMES`. `None` means that leg did
    #: not return the chunk at all, which is not the same as returning it first.
    ranks: tuple[int | None, int | None, int | None]

    @property
    def legs_hit(self) -> int:
        """How many of the three legs found this chunk. The mechanism hypothesis in one number."""
        return sum(rank is not None for rank in self.ranks)


def fuse(legs: _Legs) -> list[FusedHit]:
    """Fuse `legs` into the ranked list `search()` would produce, keeping the fusion detail.

    Returns best-first over the whole fused candidate set, untruncated: the caller decides its
    own pool depth, and `search()` truncates only after reranking anyway.
    """
    dense, sparse, learned = legs.dense, legs.sparse, legs.learned
    fused = _rrf([[h.chunk.id for h in dense], [h.chunk.id for h in sparse],
                  [h.chunk.id for h in learned]])

    by_id = {h.chunk.id: h for h in dense}
    for hit in sparse:
        by_id.setdefault(hit.chunk.id, hit)  # sparse hits carry their true cosine (vec=qvec)
    for hit in learned:
        by_id.setdefault(hit.chunk.id, hit)
    dense_score = {h.chunk.id: h.score for h in dense}

    # Per-leg rank maps. A leg can return the same id twice only if the store does, which would
    # be a store bug; `enumerate` order means the FIRST occurrence wins, matching `_rrf`, which
    # accumulates both. Recording the first is the honest reading of "where the leg put it".
    positions = [
        {h.chunk.id: rank for rank, h in reversed(list(enumerate(leg)))}
        for leg in (dense, sparse, learned)
    ]

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [
        FusedHit(
            hit=_rescored(by_id[cid], dense_score.get(cid, by_id[cid].score)),
            fused_score=fused[cid],
            ranks=(positions[0].get(cid), positions[1].get(cid), positions[2].get(cid)),
        )
        for cid in ranked_ids
    ]


def dense_gap_warning(legs: _Legs, threshold: float = DEFAULT_GAP_THRESHOLD) -> bool:
    """The `gap_warning` `search()` would report for these legs.

    ⚠️ It is computed over the DEDUPLICATED dense scores, because `search()` reads them off a
    `{chunk_id: score}` dict. Passing the raw leg instead disagrees whenever the dense leg returns
    one id twice: `[0.9, 0.1]` for the same chunk keeps the 0.9 and reports no gap, where
    `search()` keeps the 0.1 and reports one. That needs a store bug to reach, which is exactly
    why it belongs in the one place that claims to mirror `search()` rather than being restated
    at each call site with a comment asserting equivalence.
    """
    return gap_warning(list({h.chunk.id: h.score for h in legs.dense}.values()), threshold)
