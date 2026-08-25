from __future__ import annotations

from copy import deepcopy
from datetime import date

from tools.prior_art.loader import load_dataset
from tools.prior_art.validate import validate_dataset


def test_verified_claim_requires_primary_source() -> None:
    dataset = deepcopy(load_dataset())
    source = next(source for source in dataset["sources"] if source["source_id"] == "src_langmem_docs")
    source["source_tier"] = "secondary"
    errors = validate_dataset(dataset, today=date(2026, 8, 24))
    assert any("verified claims require a primary source" in error for error in errors)


def test_missing_review_is_rejected() -> None:
    dataset = deepcopy(load_dataset())
    dataset["claims"][0]["review_id"] = "missing_review"
    errors = validate_dataset(dataset, today=date(2026, 8, 24))
    assert any("review_id references a missing review" in error for error in errors)


def test_unknown_and_not_evidenced_are_valid_distinct_values() -> None:
    dataset = deepcopy(load_dataset())
    values = {claim["value"] for claim in dataset["claims"]}
    assert "unknown" in values
    assert "not_evidenced" in values


def test_future_access_date_is_rejected() -> None:
    dataset = deepcopy(load_dataset())
    dataset["sources"][0]["accessed_at"] = "2026-08-25"
    errors = validate_dataset(dataset, today=date(2026, 8, 24))
    assert any("accessed_at is in the future" in error for error in errors)


def test_long_evidence_excerpt_is_rejected() -> None:
    dataset = deepcopy(load_dataset())
    dataset["claims"][0]["evidence_excerpt"] = "word " * 26
    errors = validate_dataset(dataset, today=date(2026, 8, 24))
    assert any("evidence_excerpt exceeds" in error for error in errors)


def test_unknown_target_hypothesis_capability_is_rejected() -> None:
    dataset = deepcopy(load_dataset())
    dataset["report_config"]["target_hypothesis"].append("missing.capability")
    errors = validate_dataset(dataset, today=date(2026, 8, 24))
    assert any("target_hypothesis references a missing capability" in error for error in errors)
