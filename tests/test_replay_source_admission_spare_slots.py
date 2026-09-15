from __future__ import annotations

from recall.source_conditioning import SourceConditioningArtifact
from scripts.replay_source_admission_spare_slots import (
    EXPECTED_ROWS,
    _decision,
    _select_spare_slot_rescue,
)


def _model() -> SourceConditioningArtifact:
    return SourceConditioningArtifact(
        schema_version=1,
        model_id="source-logistic-v1",
        feature_names=(
            "max_cosine",
            "best_dense_rr",
            "best_sparse_rr",
            "rrf_mass",
            "cross_leg_fraction",
            "log_chunk_count",
            "source_margin",
        ),
        means=(0.0,) * 7,
        scales=(1.0,) * 7,
        intercept=0.0,
        coefficients=(0.0,) * 7,
        alpha=0.08,
        raw_cosine_floor=0.30,
        rrf_k=60,
        candidate_k=20,
        item_budget=5,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile="fast",
        training_generation_id="generation",
        training_calibration_id="calibration",
        training_pipeline_fingerprint="pipeline",
        training_corpus_fingerprint="corpus",
        training_query_set_digest="queries",
        training_fact_labels_digest="facts",
        training_artifact_sha256="training",
        training_source_rows=1,
        artifact_fingerprint="fingerprint",
    )


def _item(source: str, rank: int, cosine: float) -> dict[str, object]:
    return {
        "chunk_id": f"{source}-{rank}",
        "source": source,
        "ordinal": rank,
        "pool_rank": rank,
        "rank": rank,
        "text": "evidence",
        "cosine": cosine,
        "confidence": 0.5,
        "verdict": "ok",
    }


def test_spare_slot_rescue_never_exceeds_budget_or_displaces_base() -> None:
    """Rescue fills one spare slot without displacing accepted alpha 0.08 items.

    Red proof on 2026-09-14: I plausibly mutated the inner budget guard from ``>=`` to ``>``. The
    intended five-item selection assertion failed. The production symbol under proof is
    ``_select_spare_slot_rescue``.
    """
    base = [_item("base", rank, 0.5) for rank in range(1, 5)]
    rescue_one = _item("rescue", 1, 0.4)
    rescue_two = _item("rescue-two", 2, 0.39)
    row = {
        "alpha008_items": base,
        "trace": {
            "pool": [rescue_one, rescue_two],
            "dense": [rescue_one, rescue_two],
            "sparse": [rescue_one, rescue_two],
        },
    }

    selected, receipts = _select_spare_slot_rescue(_model(), row)

    assert selected == [*base, rescue_one]
    assert len(receipts) == 1


def test_dual_and_lexical_lanes_fill_only_spare_slots() -> None:
    dual = _item("dual", 1, 0.36)
    lexical_one = _item("lexical", 1, 0.29)
    lexical_two = _item("lexical", 2, 0.28)
    row = {
        "alpha008_items": [],
        "trace": {
            "pool": [dual, lexical_one, lexical_two],
            "dense": [dual],
            "sparse": [dual, lexical_one, lexical_two],
        },
    }

    selected, receipts = _select_spare_slot_rescue(_model(), row)

    assert [item["source"] for item in selected] == ["dual", "lexical", "lexical"]
    assert [value["lane"] for value in receipts] == [
        "dual_leg",
        "lexical_dominant",
        "lexical_dominant",
    ]


def test_decision_requires_fact_headroom_and_all_guardrails() -> None:
    arms = {
        "alpha008": {
            "covered_facts": 34,
            "context_precision": 0.78,
            "unanswerable_answers": 1,
        },
        "candidate": {
            "covered_facts": 36,
            "context_precision": 0.79,
            "unanswerable_answers": 1,
        },
    }
    comparison = {
        "complete_gain_ids": ["buried-015", "buried-017"],
        "complete_loss_ids": [],
        "fact_gain_ids": ["buried-015", "buried-017"],
        "fact_loss_ids": [],
    }
    integrity = {
        "rows": EXPECTED_ROWS,
        "capture_fixed_hash_parity": EXPECTED_ROWS,
        "recomputed_fixed_parity": EXPECTED_ROWS,
        "base_prefix_preserved": EXPECTED_ROWS,
    }
    assert _decision(arms, comparison, integrity) == "HEADROOM_FOR_FRESH_VALIDATION"

    regression = {**comparison, "fact_loss_ids": ["holdout-003"]}
    assert _decision(arms, regression, integrity) == "NO_HEADROOM"

    unsafe = {**arms, "candidate": {**arms["candidate"], "unanswerable_answers": 2}}
    assert _decision(unsafe, comparison, integrity) == "NO_HEADROOM"

    broken = {**integrity, "recomputed_fixed_parity": EXPECTED_ROWS - 1}
    assert _decision(arms, comparison, broken) == "REPAIR"
