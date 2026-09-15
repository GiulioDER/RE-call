"""Score the retired guarded spare-slot review against frozen source labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def score_source_audit(
    pool_path: Path, capture_path: Path, review_path: Path
) -> dict[str, Any]:
    pool = _load(pool_path)
    capture = _load(capture_path)
    review = _load(review_path)

    pool_sha256 = _sha256(pool_path)
    capture_sha256 = _sha256(capture_path)
    if capture.get("query_pool_sha256") != pool_sha256:
        raise ValueError("capture query pool digest mismatch")
    if review.get("query_pool_sha256") != pool_sha256:
        raise ValueError("review query pool digest mismatch")
    if review.get("capture_sha256") != capture_sha256:
        raise ValueError("review capture digest mismatch")

    pool_by_id = {str(item["id"]): item for item in pool["queries"]}
    review_by_id = {str(item["query_id"]): item for item in review["queries"]}
    triggered_rows = [row for row in capture["rows"] if row["triggered"]]
    if {str(row["query_id"]) for row in triggered_rows} != set(review_by_id):
        raise ValueError("capture and review triggered query sets differ")

    counts = {
        "answerable_triggered": 0,
        "answerable_base_source_hits": 0,
        "answerable_candidate_source_hits": 0,
        "answerable_source_hit_gains": 0,
        "answerable_source_hit_losses": 0,
        "answerable_added_gold_items": 0,
        "answerable_added_non_gold_items": 0,
        "unanswerable_triggered": 0,
        "unanswerable_added_non_gold_items": 0,
        "total_added_items": 0,
    }
    prefix_preserved = True

    for row in triggered_rows:
        query_id = str(row["query_id"])
        pool_item = pool_by_id[query_id]
        review_item = review_by_id[query_id]
        if row["expected_answerability"] != pool_item["expected_answerability"]:
            raise ValueError(f"answerability mismatch for {query_id}")
        evidence_by_id = {
            str(item["evidence_id"]): item for item in review_item["evidence_items"]
        }
        evidence_id_by_hash = row["review_evidence_ids_by_hash"]
        base_hashes = list(row["base_hashes"])
        candidate_hashes = list(row["candidate_hashes"])
        if candidate_hashes[: len(base_hashes)] != base_hashes:
            prefix_preserved = False
            raise ValueError(f"candidate changed the base prefix for {query_id}")

        def sources(hashes: list[str]) -> list[str]:
            return [
                str(evidence_by_id[str(evidence_id_by_hash[value])]["source"])
                for value in hashes
            ]

        base_sources = sources(base_hashes)
        candidate_sources = sources(candidate_hashes)
        added_sources = sources(candidate_hashes[len(base_hashes) :])
        counts["total_added_items"] += len(added_sources)

        answerability = str(pool_item["expected_answerability"])
        gold_sources = {str(value) for value in pool_item["gold_sources"]}
        if answerability == "answerable":
            counts["answerable_triggered"] += 1
            base_hit = any(value in gold_sources for value in base_sources)
            candidate_hit = any(value in gold_sources for value in candidate_sources)
            counts["answerable_base_source_hits"] += int(base_hit)
            counts["answerable_candidate_source_hits"] += int(candidate_hit)
            counts["answerable_source_hit_gains"] += int(candidate_hit and not base_hit)
            counts["answerable_source_hit_losses"] += int(base_hit and not candidate_hit)
            counts["answerable_added_gold_items"] += sum(
                value in gold_sources for value in added_sources
            )
            counts["answerable_added_non_gold_items"] += sum(
                value not in gold_sources for value in added_sources
            )
        elif answerability == "unanswerable":
            counts["unanswerable_triggered"] += 1
            if gold_sources:
                raise ValueError(f"unanswerable query has gold sources: {query_id}")
            counts["unanswerable_added_non_gold_items"] += len(added_sources)
        else:
            raise ValueError(f"unknown answerability for {query_id}: {answerability}")

    added_gold = counts["answerable_added_gold_items"]
    total_added = counts["total_added_items"]
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-source-audit",
        "decision": "INVALID_REVIEW_INSTRUMENT",
        "promotion_eligible": False,
        "scope": "descriptive_source_audit_only",
        "integrity": {
            "query_pool_sha256": pool_sha256,
            "capture_sha256": capture_sha256,
            "capture_review_query_sets_match": True,
            "base_prefix_preserved": prefix_preserved,
            "human_labels_consumed": False,
        },
        "metrics": {
            **counts,
            "added_gold_source_precision": (
                added_gold / total_added if total_added else None
            ),
        },
        "limitations": [
            "No required-fact labels existed before retrieval.",
            "This audit does not measure answer sufficiency or false-answer rate.",
            "The result cannot promote or reject the guarded rescue.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_source_audit(args.pool, args.capture, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
