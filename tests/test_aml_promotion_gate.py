import pytest

from scripts.aml_select_context import select_budget
from scripts.aml_promotion_gate import decide


def passing_artifact():
    return {
        "contract_passed": True,
        "isolation_passed": True,
        "persistence_passed": True,
        "idempotency_passed": True,
        "release_frozen": True,
        "external_https_ready": True,
        "stable_30_days_committed": True,
        "paired_identity_passed": True,
        "task_count": 34,
        "paired_cells": 102,
        "seeds": [0, 1, 2],
        "invalid_cells": 0,
        "net_successful_cells": 8,
        "new_control_failures": 2,
        "add_concurrency": 16,
        "add_errors": 0,
        "search_concurrency": 16,
        "search_errors": 0,
        "add_p95_seconds": 29.99,
        "search_p95_seconds": 4.99,
        "provider_spend_usd": 450,
    }


def test_promotion_gate_accepts_every_boundary_value_that_passes():
    promoted, failures = decide(passing_artifact())
    assert promoted is True
    assert failures == []


def test_promotion_gate_fails_closed_for_each_missing_or_bad_gate():
    for key, bad in (
        ("contract_passed", False),
        ("paired_identity_passed", False),
        ("task_count", 12),
        ("paired_cells", 101),
        ("seeds", [0, 1]),
        ("invalid_cells", 1),
        ("net_successful_cells", 7),
        ("new_control_failures", 3),
        ("add_concurrency", 15),
        ("add_errors", 1),
        ("search_concurrency", 15),
        ("search_errors", 1),
        ("add_p95_seconds", 30),
        ("search_p95_seconds", 5),
        ("provider_spend_usd", 450.01),
    ):
        artifact = passing_artifact()
        artifact[key] = bad
        promoted, failures = decide(artifact)
        assert promoted is False, key
        assert failures, key


def test_context_selector_chooses_smallest_arm_within_one_point_of_best():
    assert select_budget({5_000: 0.796, 7_000: 0.80, 9_000: 0.805}) == 5_000
    assert select_budget({5_000: 0.70, 7_000: 0.80, 9_000: 0.82}) == 9_000
    with pytest.raises(ValueError):
        select_budget({5_000: 0.8, 7_000: 0.81})
