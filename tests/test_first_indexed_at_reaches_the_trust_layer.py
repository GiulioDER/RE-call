"""`first_indexed_at` must survive the retriever, or the point-in-time fix is inert.

Prior work searched 2026-08-01: `docs_search` unavailable (VPS2 down, no `docs_chunks` on the
VPS3 mirror), so grep/Read over the repo. `docs/REFERENCE_TIME_DESIGN.md` carries the design.

This file exists because of a specific failure. `first_indexed_at` was added to the schema, to
`ScoredChunk`, to all four SELECTs and to `_verdict`, with four `requires_db` tests. Every one of
them passed, and the feature was completely inert in production: `HybridRetriever.search` rebuilt
each hit by LISTING the fields to carry, that list was written before the column existed, so the
field was dropped and `_verdict`'s fallback quietly restored the pre-fix behaviour.

Nothing caught it because no test built a hit through a retriever. The DB tests read the column
with raw SQL; the pure tests constructed `ScoredChunk` directly. The gap was between the two
layers, which is exactly where a hand-maintained field list breaks.

So these assert the SEAM, not the ends: a value set on the store's hit has to still be there when
the trust layer reads it. DB-free, via a stub store, so it runs everywhere.
"""
from __future__ import annotations

from datetime import datetime, timezone

from recall.calibration import Calibration
from recall.retriever import HybridRetriever
from recall.trust import _verdict, trusted_search
from recall.types import Chunk, ScoredChunk

JAN = datetime(2026, 1, 1, tzinfo=timezone.utc)
MAR = datetime(2026, 3, 1, tzinfo=timezone.utc)
AUG = datetime(2026, 8, 1, tzinfo=timezone.utc)

CAL = Calibration(embedder="constant", threshold=0.5, scale=0.05)


class _Store:
    """Serves one hit written in JANUARY and last re-indexed in AUGUST."""

    def __init__(self, first=JAN, last=AUG):
        self._hit = ScoredChunk(
            chunk=Chunk(id="1", source="/c/a.md", text="body", metadata={"file": "a.md"}),
            score=0.99,
            indexed_at=last,
            first_indexed_at=first,
        )

    def query_dense(self, vector, k, source=None):
        return [self._hit]

    def query_sparse(self, query, k, source=None, vec=None):
        return []

    def newest_indexed_at(self):
        return AUG

    def supersession(self):
        return {}, frozenset()

    def supersession_all(self):
        # Present, so these run the PRODUCTION branch. Without it `trusted_search` takes the
        # degraded `getattr` fallback, warns, and leaves this class in the process-global
        # `_WARNED_NO_EDGE_DATES` — a test billed as end-to-end exercising a path production
        # never takes, and polluting a global on the way out.
        return {}, frozenset(), {}


class _Embedder:
    dim = 2
    name = "constant"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_the_retriever_does_not_drop_first_indexed_at():
    """The seam. Listing carried fields by hand dropped it; copying the hit cannot."""
    result = HybridRetriever(_Store(), _Embedder(), gap_threshold=0.5).search("q", k=1)
    assert result.hits[0].first_indexed_at == JAN
    assert result.hits[0].indexed_at == AUG, "the last write must survive too"


def test_a_reindexed_memory_is_visible_at_an_instant_before_the_reindex():
    """The capability, end to end. The memo was first written in January and re-indexed in
    August. Asked as of March it EXISTED, and answering `not_yet_known` would be the store
    claiming it never held a document it had held for two months."""
    res = trusted_search(
        _Store(), _Embedder(), "q", k=1, calibration=CAL, known_as_of=MAR
    )
    assert res.hits[0].verdict == "ok"


def test_the_same_memory_is_invisible_before_its_FIRST_write():
    """The control. Without it the test above passes on a build that ignores both columns."""
    res = trusted_search(
        _Store(), _Embedder(), "q", k=1, calibration=CAL,
        known_as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    assert res.hits[0].verdict == "not_yet_known"


def test_first_indexed_at_wins_over_indexed_at_in_the_verdict():
    """The rule itself, isolated from the retriever."""
    hit = ScoredChunk(
        chunk=Chunk(id="1", source="s", text="t", metadata={"file": "a.md"}),
        score=0.9, indexed_at=AUG, first_indexed_at=JAN,
    )
    assert _verdict(hit, {}, 0.5, AUG, frozenset(), MAR)[0] == "ok"


def test_a_hit_with_no_first_write_falls_back_to_its_last_write():
    """Rows predating the column carry NULL, and `indexed_at` is then the only evidence there
    is. This is the branch the migration deliberately leaves reachable."""
    hit = ScoredChunk(
        chunk=Chunk(id="1", source="s", text="t", metadata={"file": "a.md"}),
        score=0.9, indexed_at=AUG, first_indexed_at=None,
    )
    assert _verdict(hit, {}, 0.5, AUG, frozenset(), MAR)[0] == "not_yet_known"
