from __future__ import annotations

import argparse

from benchmarks.structural_edge_performance import (
    _configurations,
    _controlled_neighbors,
    _contexts,
    _production_linked_tail_context,
)
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport

from datetime import timedelta


def _hit(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(Chunk(chunk_id, "memory", chunk_id), score)


def test_selective_graph_gate_admits_only_candidates_above_direct_tail() -> None:
    """The score gate protects the direct prefix and fills rejected graph slots.

    Red proof target: ``_contexts``. Mutating the strict ``< min_score`` rejection to an
    unconditional admission makes ``graph-weak`` appear in the context and fails the assertion.
    Node id: selective-gate-admission. Target symbol: ``_contexts``. Failure reason: weak graph
    candidates must not displace the direct tail after the calibrated margin is applied.
    """
    seed_hits = [
        _hit(f"direct-{index}", score)
        for index, score in enumerate((0.99, 0.95, 0.90, 0.85, 0.82, 0.80, 0.78, 0.76))
    ]
    direct_fallback = [_hit("direct-8", 0.70), _hit("direct-9", 0.68)]
    chunks = {
        hit.chunk.id: hit.chunk
        for hit in [
            *seed_hits,
            *direct_fallback,
            _hit("graph-strong", 0.76),
            _hit("graph-weak", 0.71),
        ]
    }
    neighbors = {
        "direct-0": (
            ("graph-strong", "speaker", "same"),
            ("graph-weak", "speaker", "same"),
        )
    }
    scores = {
        chunk_id: hit.score
        for chunk_id, hit in [
            (hit.chunk.id, hit)
            for hit in [
                *seed_hits,
                *direct_fallback,
                _hit("graph-strong", 0.76),
                _hit("graph-weak", 0.71),
            ]
        ]
    }

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

    assert [hit.chunk.id for hit in selected] == [
        *[(f"direct-{index}") for index in range(8)],
        "graph-strong",
        "direct-8",
    ]
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
    chunks = {
        hit.chunk.id: hit.chunk for hit in [*seed_hits, *direct_fallback, _hit("graph-weak", 0.71)]
    }
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


def test_production_linked_tail_matches_selective_context_order() -> None:
    """The production helper reproduces the selective benchmark's ordered context.

    Invariant: both implementations compare graph candidates with the weakest original direct
    tail score. Red proof node ``production-linked-tail-parity-01`` mutates the production call to
    compare with the first direct tail score. The production context then keeps ``direct-9`` and
    fails the equality assertion against the benchmark reference.
    """
    direct = [
        _hit(f"direct-{index}", score)
        for index, score in enumerate(
            (0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.70, 0.60, 0.68)
        )
    ]
    chunks = {hit.chunk.id: hit.chunk for hit in direct}
    neighbors = {"direct-0": (("direct-10", "speaker", "same"),)}
    scores = {hit.chunk.id: hit.score for hit in direct}
    reference, _reference_additions = _contexts(
        direct[:8],
        neighbors,
        chunks,
        context_k=10,
        edge_budget=2,
        neighbor_order="retrieval",
        retrieval_scores=scores,
        direct_fallback=direct[8:10],
        min_score=0.65,
    )
    retrieval = RetrievalResult(
        "q",
        direct,
        False,
        StalenessReport(False, None, None, timedelta(days=1)),
    )

    production, additions = _production_linked_tail_context(
        retrieval,
        neighbors,
        chunks,
        margin=0.05,
    )

    assert [hit.chunk.id for hit in production] == [hit.chunk.id for hit in reference]
    assert [item["chunk_id"] for item in additions] == ["direct-10"]


def test_production_parity_configuration_freezes_both_implementations() -> None:
    args = argparse.Namespace(
        relation_types="all",
        seed_k=8,
        edge_budget=2,
        context_k=10,
        retrieval_k=20,
        neighbor_order="retrieval",
        sweep=False,
        selective_gate=False,
        production_parity=True,
    )

    configs = _configurations(args)

    assert [name for name, _config in configs] == [
        "selective_margin_005",
        "production_linked_tail",
    ]
    assert configs[1][1]["implementation"] == "production_linked_tail"


def test_shuffled_relation_control_is_deterministic_and_degree_preserving() -> None:
    """The null changes endpoints without changing source or target opportunity counts.

    Red proof node ``production-topology-control-01`` used a unique endpoint permutation. The
    target occurrence assertion failed because nodes with repeated target occurrences changed
    frequency, so the null did not preserve the frozen opportunity count.
    """
    neighbors = {
        "a": (("b", "speaker", "one"), ("c", "temporal", "two")),
        "b": (("a", "speaker", "one"),),
        "c": (("a", "temporal", "two"),),
    }

    shuffled = _controlled_neighbors(neighbors, control="shuffled", seed=20260912)
    repeated = _controlled_neighbors(neighbors, control="shuffled", seed=20260912)

    assert shuffled == repeated
    assert shuffled != neighbors
    assert {source: len(rows) for source, rows in shuffled.items()} == {
        source: len(rows) for source, rows in neighbors.items()
    }
    assert sorted(target for rows in shuffled.values() for target, _, _ in rows) == sorted(
        target for rows in neighbors.values() for target, _, _ in rows
    )
    assert _controlled_neighbors(neighbors, control="removed", seed=20260912) == {}
