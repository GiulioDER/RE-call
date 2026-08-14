from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.eval import reasoning_session3
from recall.eval.metrics import nan_to_null
from recall.eval.reasoning_session3 import _artifact_json, reasoning_session3_artifact


def test_reasoning_session3_artifact_matches_generator() -> None:
    expected = json.loads(
        Path("results/reasoning_session3_proposals.json").read_text(encoding="utf-8-sig")
    )

    assert reasoning_session3_artifact() == expected


def test_reasoning_session3_artifact_never_emits_a_bare_nan_token() -> None:
    """`proposal_precision_recall` can now return NaN, and `NaN` is not valid JSON.

    The artifact is only finite today because the supersedes fixture is populated. Scoring an
    empty relation, which the docs now tell readers to do, produces NaN, so the writer has to
    sanitise rather than depend on the fixture staying lucky.
    """

    encoded = _artifact_json()

    json.loads(encoded)
    assert "NaN" not in encoded


def test_artifact_serialiser_renders_a_nan_score_as_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercised with a NaN actually present, so the sanitiser cannot be a no-op that passes."""

    monkeypatch.setattr(
        reasoning_session3,
        "reasoning_session3_artifact",
        lambda: {"precision_recall": {"precision": float("nan"), "recall": 1.0}},
    )

    encoded = _artifact_json()

    assert "NaN" not in encoded
    assert json.loads(encoded) == {"precision_recall": {"precision": None, "recall": 1.0}}


def test_nan_to_null_sanitises_nested_non_finite_values() -> None:
    assert nan_to_null({"precision": float("nan"), "counts": [1, float("-inf")]}) == {
        "precision": None,
        "counts": [1, None],
    }


def test_reasoning_session3_records_control_artifacts() -> None:
    artifact = reasoning_session3_artifact()

    assert artifact["protocol_spec"] == "docs/INFERENCE_PROPOSALS.md"
    assert artifact["precision_recall"]["recall"] == 1.0
    assert artifact["precision_recall"]["precision"] >= 0.5
    # Referrals are reported beside precision, never folded into it: the three asserted
    # candidates are what precision is computed over, the four requires_review proposals are not.
    assert artifact["precision_recall"]["asserted_proposals"] == 3
    assert artifact["precision_recall"]["referred_proposals"] == 4
    assert artifact["rejected_proposal_examples"]
    assert artifact["provider_failure_matrix"] == {
        "malformed_output": 1,
        "provider_error": 1,
        "timeout": 1,
        "wrong_cardinality": 1,
    }
    assert artifact["audit"] == {
        "all_proposals_trace_to_evidence": True,
        "authored_edges_unchanged": True,
        "proposal_output_promotes_trust": False,
        "side_effect_free": True,
    }
