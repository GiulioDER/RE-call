from __future__ import annotations

import argparse

from benchmarks.structural_edge_performance import _configurations, _contexts
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


def test_category_confirmation_freezes_existing_and_strict_margins() -> None:
    """The category confirmation exposes both preregistered margins on one slice.

    Red proof target: ``_configurations``. Mutating the strict arm to reuse 0.05 makes the two
    margins equal and fails this assertion. Node id: category4-margin-confirmation. Target symbol:
    ``_configurations``. Failure reason: the confirmation must distinguish the existing gate from
    the stricter gate and must not activate graph expansion outside category 4.
    """
    args = argparse.Namespace(
        relation_types="all",
        seed_k=8,
        edge_budget=2,
        context_k=10,
        retrieval_k=20,
        neighbor_order="retrieval",
        direct_fallback=False,
        activation_categories=None,
        graph_score_margin=None,
        sweep=False,
        selective_gate=True,
        selective_margin=0.10,
        selective_category=4,
    )

    configs = _configurations(args)

    assert [name for name, _ in configs] == ["selective_margin_005", "selective_margin_strict"]
    assert [config["graph_score_margin"] for _, config in configs] == [0.05, 0.10]
    assert all(config["activation_categories"] == frozenset({4}) for _, config in configs)
