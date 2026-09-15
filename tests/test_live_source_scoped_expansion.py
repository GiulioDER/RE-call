from __future__ import annotations

from scripts.run_live_source_scoped_expansion import (
    _decision,
    _rank_sources,
    _scoped_select,
    _source_scoped_candidates,
)


def _item(
    chunk_id: str,
    source: str,
    cosine: float,
    pool_rank: int,
    verdict: str = "ok",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source": source,
        "ordinal": pool_rank,
        "pool_rank": pool_rank,
        "text": chunk_id,
        "cosine": cosine,
        "confidence": 0.8,
        "verdict": verdict,
    }


def test_source_ranking_uses_support_then_original_pool_rank() -> None:
    """Support ties are deterministic and preserve the stronger fused source.

    Red proof node ``source-scoped-rank-01`` reverses the pool-rank tie break. The
    ordered source assertion then fails at its intended assertion.
    """
    row = {
        "source_admission_audit": {
            "items": [
                _item("b", "later", 0.8, 2),
                _item("a", "earlier", 0.7, 1),
                _item("c", "strong", 0.6, 3),
            ]
        }
    }

    ranked = _rank_sources(row, {"earlier": 0.5, "later": 0.5, "strong": 0.9})

    assert ranked == ["strong", "earlier", "later"]


def test_source_scoped_candidates_cap_each_source_and_keep_rank_metadata() -> None:
    """Each chosen source contributes no more than the registered eight chunks.

    Red proof node ``source-scoped-cap-01`` raises the per source slice by one. The
    candidate count assertion then fails at its intended assertion.
    """
    first = [_item(f"a-{index}", "a", 0.90 - index / 100, index + 1) for index in range(10)]
    second = [_item(f"b-{index}", "b", 0.80 - index / 100, index + 1) for index in range(3)]
    candidates = _source_scoped_candidates(
        {
            "a": {"items": list(reversed(first))},
            "b": {"items": second},
        },
        ["a", "b"],
        {"a": 0.8, "b": 0.7},
        2,
    )

    assert len(candidates) == 11
    assert [item["chunk_id"] for item in candidates[:8]] == [f"a-{index}" for index in range(8)]
    assert {item["source_rank"] for item in candidates[:8]} == {1}
    assert {item["source_rank"] for item in candidates[8:]} == {2}


def test_scoped_selection_keeps_trust_and_raw_cosine_boundaries() -> None:
    """Source support cannot override a non-admissible trust verdict or raw floor.

    Red proof node ``source-scoped-trust-01`` disables the verdict check. The exact
    selected chunk assertion then fails at its intended assertion.
    """
    candidates = [
        {
            **_item("invalid", "a", 0.8, 1, "superseded"),
            "source_support": 1.0,
            "source_rank": 1,
            "source_scoped_pool_rank": 1,
        },
        {
            **_item("too-low", "a", 0.29, 2, "low_confidence"),
            "source_support": 1.0,
            "source_rank": 1,
            "source_scoped_pool_rank": 2,
        },
        {
            **_item("promoted", "a", 0.48, 3, "low_confidence"),
            "source_support": 0.9,
            "source_rank": 1,
            "source_scoped_pool_rank": 3,
        },
        {
            **_item("already-ok", "b", 0.53, 1),
            "source_support": 0.4,
            "source_rank": 2,
            "source_scoped_pool_rank": 1,
        },
    ]

    selected = _scoped_select(candidates, 0.509, 0.08)

    assert [item["chunk_id"] for item in selected] == ["already-ok", "promoted"]


def test_decision_selects_smallest_build_budget() -> None:
    """The decision rule chooses the lowest source budget that clears every gate.

    Red proof node ``source-scoped-decision-01`` evaluates budgets in reverse. The
    selected arm assertion then fails at its intended assertion.
    """
    control = {
        "complete_queries": 14,
        "covered_facts": 16,
        "unanswerable_answers": 2,
        "context_precision": 0.50,
    }
    arms: dict[str, dict[str, object]] = {
        "global_alpha_0.08": dict(control),
        "global_alpha_0.15": dict(control),
    }
    for budget in (1, 2, 3):
        for alpha in (0.08, 0.15):
            arms[f"scoped_sources_{budget}_alpha_{alpha:.2f}"] = dict(control)
    arms["scoped_sources_2_alpha_0.08"] = {
        **control,
        "complete_queries": 15,
    }
    arms["scoped_sources_3_alpha_0.08"] = {
        **control,
        "complete_queries": 16,
    }

    assert _decision({"arms": arms}) == {
        "verdict": "BUILD SOURCE SCOPED EXPANSION",
        "selected_arm": "scoped_sources_2_alpha_0.08",
    }
