"""Build the exhaustive 22 source atomic fact pilot and gate construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_production_atomic_fact_fresh_audit import (
    DEFAULT_SEED,
    INPUT_HASHES,
    _root,
    audit_fresh_pool,
    build_fresh_pool,
)


EXPECTED_COUNT = 22


def pilot_construction_gate(
    pool: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[str, dict[str, bool]]:
    metrics = audit["metrics"]
    family_coverage = metrics["gold"]["family_coverage"]
    checks = {
        "exhaustive_pool": (
            int(pool.get("selected_sources", 0)) == EXPECTED_COUNT
            and pool.get("input_hashes") == dict(sorted(INPUT_HASHES.items()))
        ),
        "source_integrity": (
            int(metrics["integrity"]["missing_sources"]) == 0
            and int(metrics["integrity"]["source_hash_mismatches"]) == 0
        ),
        "unique_gold_view": int(metrics["gold"]["unique_view"]) == EXPECTED_COUNT,
        "gold_parent_matches": (
            int(metrics["gold"]["parent_ordinal_match"]) == EXPECTED_COUNT
        ),
        "gold_family_coverage": (
            len(family_coverage) >= 2
            and all(float(value["rate"]) == 1.0 for value in family_coverage.values())
        ),
        "no_zero_view_sources": int(metrics["size"]["zero_view_sources"]) == 0,
        "view_length": int(metrics["coverage"]["oversized_views"]) == 0,
        "within_source_fact_duplicates": (
            int(metrics["duplicates"]["within_source_fact_excess"])
            / int(metrics["size"]["auxiliary_views"])
            <= 0.01
        ),
        "cross_source_rendered_collisions": (
            int(metrics["duplicates"]["cross_source_rendered_excess"])
            / int(metrics["size"]["auxiliary_views"])
            <= 0.01
        ),
        "row_growth": float(metrics["size"]["auxiliary_row_growth"]) <= 2.0,
        "character_growth": float(metrics["size"]["rendered_character_growth"]) <= 1.5,
        "parent_crowding": (
            int(metrics["distribution"]["views_per_parent_chunk"]["p95"]) <= 4
            and int(metrics["distribution"]["views_per_parent_chunk"]["max"]) <= 8
        ),
    }
    decision = (
        "PROCEED_ATOMIC_FACT_22_EMBEDDING"
        if all(checks.values())
        else "STOP_ATOMIC_FACT_22_CONSTRUCTION"
    )
    return decision, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_root, required=True)
    parser.add_argument("--exclusion-input", action="append", type=Path, required=True)
    parser.add_argument("--pool-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()

    roots = list(args.source_root)
    pool = build_fresh_pool(
        roots,
        args.exclusion_input,
        count=EXPECTED_COUNT,
        seed=DEFAULT_SEED,
    )
    args.pool_output.parent.mkdir(parents=True, exist_ok=True)
    args.pool_output.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit = audit_fresh_pool(args.pool_output, dict(roots))
    decision, checks = pilot_construction_gate(pool, audit)
    result = {
        **audit,
        "protocol": "2026-09-16-atomic-fact-exhaustive-22-pilot-construction",
        "decision": decision,
        "failed_gates": sorted(name for name, passed in checks.items() if not passed),
        "gates": checks,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "pool_sha256": result["pool_sha256"],
                "eligible_sources": pool["eligible_sources"],
                "selected_sources": pool["selected_sources"],
                "decision": decision,
            }
        )
    )


if __name__ == "__main__":
    main()
