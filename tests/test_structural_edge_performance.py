from __future__ import annotations

from benchmarks.structural_edge_performance import _contexts
from recall.types import Chunk, ScoredChunk


def _hit(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(Chunk(chunk_id, "memory", chunk_id), score)


def test_selective_graph_gate_admits_only_candidates_above_direct_tail() -> None:
    """The score gate protects the direct prefix and fills rejected graph slots.

    Red proof target: ``_contexts``. Mutating the strict ``< min_score`` rejection to an
    unconditional admission makes ``graph-weak`` appear in the context and fails the assertion.
    Node id: selective-gate-admission. Target symbol: ``_contexts``. Failure reason: weak graph
    candidates must not displace the direct tail after the calibrated margin is applied.
    """
    seed_hits = [_hit(f"direct-{index}", score) for index, score in enumerate((0.99, 0.95, 0.90, 0.85, 0.82, 0.80, 0.78, 0.76))]
    direct_fallback = [_hit("direct-8", 0.70), _hit("direct-9", 0.68)]
    chunks = {hit.chunk.id: hit.chunk for hit in [*seed_hits, *direct_fallback, _hit("graph-strong", 0.76), _hit("graph-weak", 0.71)]}
    neighbors = {
        "direct-0": (
            ("graph-strong", "speaker", "same"),
            ("graph-weak", "speaker", "same"),
        )
    }
    scores = {chunk_id: hit.score for chunk_id, hit in [(hit.chunk.id, hit) for hit in [*seed_hits, *direct_fallback, _hit("graph-strong", 0.76), _hit("graph-weak", 0.71)]]}

    selected, additions = _contexts(
        seed_hits,
        neighbors,
        chunks,
        context_k=10,
        edge_budget=2,
        neighbor_order="retrieval",
        retrieval_scores=scores,
        direct_fallback=direct_fallback,
        min_score=0.73,
    )

    assert [hit.chunk.id for hit in selected] == [*[(f"direct-{index}") for index in range(8)], "graph-strong", "direct-8"]
    assert [item["chunk_id"] for item in additions] == ["graph-strong"]
    assert additions[0]["retrieval_score"] == 0.76


def test_selective_graph_gate_keeps_full_direct_context_when_no_candidate_clears_margin() -> None:
    """A selective miss retains ten direct items instead of shrinking the evidence budget.

    Red proof target: ``_contexts``. Mutating the direct fallback loop to skip fallback makes the
    result eight items and fails the context length assertion. Node id: selective-gate-fallback.
    Target symbol: ``_contexts``. Failure reason: a rejected graph candidate must not reduce the
    fixed context budget or alter the protected direct prefix.
    """
    seed_hits = [_hit(f"direct-{index}", 0.90 - index * 0.01) for index in range(8)]
    direct_fallback = [_hit("direct-8", 0.70), _hit("direct-9", 0.68)]
    chunks = {hit.chunk.id: hit.chunk for hit in [*seed_hits, *direct_fallback, _hit("graph-weak", 0.71)]}
    selected, additions = _contexts(
        seed_hits,
        {"direct-0": (("graph-weak", "speaker", "same"),)},
        chunks,
        context_k=10,
        edge_budget=2,
        neighbor_order="retrieval",
        retrieval_scores={"graph-weak": 0.71},
        direct_fallback=direct_fallback,
        min_score=0.73,
    )

    assert additions == []
    assert [hit.chunk.id for hit in selected] == [*[(f"direct-{index}") for index in range(10)]]
