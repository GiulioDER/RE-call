from __future__ import annotations

from copy import deepcopy

import pytest

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


def test_pilot_gate_stops_rather_than_raises_when_no_view_was_built() -> None:
    """A pool whose every source failed integrity must yield a recorded STOP, not a traceback.

    Then the audit reports zero views, no growth ratios and no parent percentile. The gate divided
    by the view count and called int() on the percentile, so main() died after writing the pool
    and never wrote the result (CCA deferred run 20261029T222748Z, BUG-002). Red proof: against
    prepare_atomic_fact_pilot.py at 8c296fd31 this fails at `pytest.fail` with ZeroDivisionError;
    restored, it passes.
    """
    pool, audit = _fixture()
    empty = deepcopy(audit)
    empty["metrics"]["integrity"]["missing_sources"] = 22  # type: ignore[index]
    empty["metrics"]["size"].update(  # type: ignore[index]
        auxiliary_views=0, auxiliary_row_growth=None, rendered_character_growth=None
    )
    empty["metrics"]["distribution"]["views_per_parent_chunk"] = {"p95": None, "max": None}  # type: ignore[index]

    try:
        decision, checks = pilot_construction_gate(pool, empty)
    except (ZeroDivisionError, TypeError) as exc:
        pytest.fail(f"the gate raised {exc!r} instead of returning STOP")

    assert decision == "STOP_ATOMIC_FACT_22_CONSTRUCTION"
    for name in (
        "source_integrity",
        "within_source_fact_duplicates",
        "cross_source_rendered_collisions",
        "row_growth",
        "character_growth",
        "parent_crowding",
    ):
        assert checks[name] is False, name
