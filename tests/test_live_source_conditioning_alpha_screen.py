from __future__ import annotations

from recall.source_conditioning import SourceConditioningArtifact
from scripts.run_live_source_conditioning_alpha_screen import (
    EXPECTED_REQUESTS,
    _compact_leg,
    _decision,
    _screen_model,
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


def _arm(*, facts: int, precision: float, false_answers: int) -> dict[str, object]:
    return {
        "complete_queries": facts,
        "covered_facts": facts,
        "source_hit_queries": facts,
        "gold_context_items": facts,
        "total_context_items": facts,
        "context_precision": precision,
        "unanswerable_answers": false_answers,
    }


def _summary(
    *,
    fixed_facts: int = 34,
    screen_facts: int = 35,
    fixed_precision: float = 0.70,
    screen_precision: float = 0.69,
    fixed_false: int = 1,
    screen_false: int = 1,
) -> dict[str, object]:
    return {
        "arms": {
            "alpha008": _arm(
                facts=fixed_facts,
                precision=fixed_precision,
                false_answers=fixed_false,
            ),
            "alpha015": _arm(
                facts=screen_facts,
                precision=screen_precision,
                false_answers=screen_false,
            ),
        }
    }


def _integrity() -> dict[str, int]:
    return {
        "requests": EXPECTED_REQUESTS,
        "fixed_hash_parity": EXPECTED_REQUESTS,
        "baseline_hash_parity": EXPECTED_REQUESTS,
        "shadow_ok": EXPECTED_REQUESTS,
        "timing_receipts": EXPECTED_REQUESTS,
        "alpha_selection_differences": 2,
    }


def _comparison() -> dict[str, list[str]]:
    return {
        "complete_gain_ids": ["buried-017"],
        "complete_loss_ids": [],
        "fact_gain_ids": ["buried-017"],
        "fact_loss_ids": [],
    }


def test_screen_model_changes_only_alpha() -> None:
    """The screen cannot silently evaluate the production alpha again.

    Red proof on 2026-09-13: I plausibly mutated ``_screen_model`` to retain alpha ``0.08``. The
    intended alpha assertion failed. The production symbol under proof is ``_screen_model``.
    """
    fixed = _model()
    screen = _screen_model(fixed)

    assert fixed.alpha == 0.08
    assert screen.alpha == 0.15
    assert screen == SourceConditioningArtifact(**{**fixed.__dict__, "alpha": 0.15})


def test_compact_leg_retains_model_inputs_without_text() -> None:
    """The replay trace keeps rank identity but does not duplicate private text.

    Red proof on 2026-09-13: I plausibly mutated ``_compact_leg`` to omit ``rank``. The intended
    rank assertion failed. The production symbol under proof is ``_compact_leg``.
    """
    result = _compact_leg(
        [
            {
                "chunk_id": "chunk-1",
                "source": "memo.md",
                "ordinal": 3,
                "rank": 2,
                "cosine": 0.51,
                "text": "not duplicated in the leg trace",
                "extra": "ignored",
            }
        ]
    )

    assert result == [
        {
            "chunk_id": "chunk-1",
            "source": "memo.md",
            "ordinal": 3,
            "rank": 2,
            "cosine": 0.51,
        }
    ]


def test_decision_requires_gain_and_every_safety_guardrail() -> None:
    holdouts = {"independent": _summary(), "buried": _summary()}
    assert (
        _decision(_summary(), holdouts, _comparison(), _integrity())
        == "PROMISING_FOR_FRESH_VALIDATION"
    )

    no_gain = _summary(screen_facts=34)
    assert (
        _decision(no_gain, {"independent": no_gain}, _comparison(), _integrity())
        == "CLOSE_GLOBAL_ALPHA_INCREASE"
    )

    unsafe_holdout = _summary(screen_false=2)
    assert (
        _decision(
            _summary(),
            {"independent": unsafe_holdout},
            _comparison(),
            _integrity(),
        )
        == "CLOSE_GLOBAL_ALPHA_INCREASE"
    )

    regression = _comparison()
    regression["fact_loss_ids"] = ["holdout-003"]
    assert (
        _decision(_summary(), holdouts, regression, _integrity())
        == "CLOSE_GLOBAL_ALPHA_INCREASE"
    )

    broken = _integrity()
    broken["fixed_hash_parity"] -= 1
    assert _decision(_summary(), holdouts, _comparison(), broken) == "REPAIR"
