from __future__ import annotations

from copy import deepcopy

from scripts.prepare_atomic_fact_pilot import pilot_construction_gate
from scripts.run_production_atomic_fact_fresh_audit import INPUT_HASHES


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    pool: dict[str, object] = {
        "selected_sources": 22,
        "input_hashes": dict(sorted(INPUT_HASHES.items())),
    }
    audit: dict[str, object] = {
        "metrics": {
            "integrity": {"missing_sources": 0, "source_hash_mismatches": 0},
            "size": {
                "zero_view_sources": 0,
                "auxiliary_views": 60,
                "auxiliary_row_growth": 0.8,
                "rendered_character_growth": 0.5,
            },
            "distribution": {"views_per_parent_chunk": {"p95": 2, "max": 4}},
            "coverage": {"oversized_views": 0},
            "duplicates": {
                "within_source_fact_excess": 0,
                "cross_source_rendered_excess": 0,
            },
            "gold": {
                "unique_view": 22,
                "parent_ordinal_match": 22,
                "family_coverage": {
                    "extractive_field": {"rate": 1.0},
                    "extractive_heading": {"rate": 1.0},
                },
            },
        }
    }
    return pool, audit


def test_pilot_gate_requires_every_gold_view() -> None:
    """Red proof weakens equality to 21, which incorrectly proceeds at the intended assertion."""
    pool, audit = _fixture()
    missing = deepcopy(audit)
    missing["metrics"]["gold"]["unique_view"] = 21  # type: ignore[index]

    good_decision, _ = pilot_construction_gate(pool, audit)
    bad_decision, checks = pilot_construction_gate(pool, missing)

    assert good_decision == "PROCEED_ATOMIC_FACT_22_EMBEDDING"
    assert bad_decision == "STOP_ATOMIC_FACT_22_CONSTRUCTION"
    assert checks["unique_gold_view"] is False
