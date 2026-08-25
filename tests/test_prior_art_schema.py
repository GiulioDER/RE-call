from __future__ import annotations

from datetime import date

from tools.prior_art.loader import load_dataset
from tools.prior_art.validate import validate_dataset


def test_checked_in_prior_art_corpus_is_valid() -> None:
    assert validate_dataset(load_dataset(), today=date(2026, 8, 24)) == []


def test_taxonomy_contains_target_hypothesis_capabilities() -> None:
    dataset = load_dataset()
    capabilities = {
        capability["capability_id"]
        for group in dataset["taxonomy"]["groups"]
        for capability in group["capabilities"]
    }
    assert {
        "provenance.transformation_lineage",
        "deletion_and_forgetting.derived_propagation",
        "action_feedback.outcome_linked_revision",
    } <= capabilities
