"""Summarize the frozen query-anchor rule on private development features."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.query_anchor_admission import (  # noqa: E402
    QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR,
    QUERY_ANCHOR_LIMIT,
    QUERY_ANCHOR_POLICY,
    query_anchor_candidate_eligible,
)


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload["rows"])
    parity_rows = [row for row in rows if not row.get("parity_mismatch", False)]
    mismatches = len(rows) - len(parity_rows)
    proposals = [row for row in parity_rows if isinstance(row.get("proposal"), dict)]
    exact_proposals = [
        row
        for row in proposals
        if row["proposal"]["chunk_contains_exact_span"]
        and not row["base_exact_span_covered"]
    ]
    if len(parity_rows) < 450 or len(exact_proposals) < 5:
        decision = "INSUFFICIENT_DEVELOPMENT_PARITY"
        accepted: list[dict[str, Any]] = []
    else:
        decision = "POLICY_FROZEN"
        accepted = [
            row
            for row in proposals
            if query_anchor_candidate_eligible(
                int(row["base_count"]), row["proposal"]["anchor_features"]
            )
        ]

    accepted_exact = sum(
        row["proposal"]["chunk_contains_exact_span"]
        and not row["base_exact_span_covered"]
        for row in accepted
    )
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-development-summary",
        "measured_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "policy": QUERY_ANCHOR_POLICY,
        "rule": {
            "proposal_index": 1,
            "base_count": 0,
            "anchor_count": QUERY_ANCHOR_LIMIT,
            "zero_document_frequency_anchors": 0,
            "chunk_coverage_fraction_min": QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR,
            "maximum_additions": 1,
        },
        "query_pool_sha256": payload["query_pool_sha256"],
        "generation_id": payload["generation_id"],
        "calibration_id": payload["calibration_id"],
        "pipeline_fingerprint": payload["pipeline_fingerprint"],
        "corpus_fingerprint": payload["corpus_fingerprint"],
        "total_rows": len(rows),
        "parity_rows": len(parity_rows),
        "parity_mismatches": mismatches,
        "first_proposals": len(proposals),
        "first_proposal_exact_span_gains": len(exact_proposals),
        "accepted_total": len(accepted),
        "accepted_exact_span_gains": accepted_exact,
        "accepted_nonexact_additions": len(accepted) - accepted_exact,
        "accepted_gold_sources": sum(row["proposal"]["is_gold_source"] for row in accepted),
        "accepted_source_pool_exact_spans": sum(
            row["proposal"]["source_pool_contains_exact_span"] for row in accepted
        ),
        "accepted_controls": sum(
            row["expected_answerability"] != "answerable" for row in accepted
        ),
        "accepted_exact_span_precision": (
            accepted_exact / len(accepted) if accepted else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.features.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
