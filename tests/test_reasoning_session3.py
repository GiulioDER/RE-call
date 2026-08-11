from __future__ import annotations

import json
from pathlib import Path

from recall.eval.reasoning_session3 import reasoning_session3_artifact


def test_reasoning_session3_artifact_matches_generator() -> None:
    expected = json.loads(
        Path("results/reasoning_session3_proposals.json").read_text(encoding="utf-8-sig")
    )

    assert reasoning_session3_artifact() == expected


def test_reasoning_session3_records_control_artifacts() -> None:
    artifact = reasoning_session3_artifact()

    assert artifact["protocol_spec"] == "docs/INFERENCE_PROPOSALS.md"
    assert artifact["precision_recall"]["recall"] == 1.0
    assert artifact["precision_recall"]["precision"] >= 0.5
    # Referrals are reported beside precision, never folded into it: the three asserted
    # candidates are what precision is computed over, the four requires_review proposals are not.
    assert artifact["precision_recall"]["asserted"] == 3
    assert artifact["precision_recall"]["referred"] == 4
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
