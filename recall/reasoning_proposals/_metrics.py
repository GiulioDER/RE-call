"""Evaluation metrics for inference proposal sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from recall.reasoning_proposals.types import InferenceProposal

def proposal_precision_recall(
    proposals: Sequence[InferenceProposal], expected_pairs: set[tuple[str, str]]
) -> Mapping[str, float | int]:
    predicted = {
        (proposal.subject_id, proposal.object_id)
        for proposal in proposals
        if proposal.proposed_relation == "supersedes" and proposal.status != "rejected"
    }
    true_positive = len(predicted & expected_pairs)
    false_positive = len(predicted - expected_pairs)
    false_negative = len(expected_pairs - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected_pairs) if expected_pairs else 1.0
    return MappingProxyType(
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        }
    )
