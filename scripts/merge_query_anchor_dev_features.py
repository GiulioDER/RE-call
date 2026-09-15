"""Merge complete private query-anchor feature shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "query_pool_sha256",
    "artifact_sha256",
    "generation_id",
    "calibration_id",
    "pipeline_fingerprint",
    "corpus_fingerprint",
    "shard_count",
)


def merge(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one shard is required")
    first = payloads[0]
    for payload in payloads[1:]:
        if any(payload[field] != first[field] for field in IDENTITY_FIELDS):
            raise ValueError("shard lineage differs")
    expected_shards = set(range(int(first["shard_count"])))
    observed_shards = {int(payload["shard_index"]) for payload in payloads}
    if observed_shards != expected_shards or len(payloads) != len(expected_shards):
        raise ValueError("shard set is incomplete or duplicated")
    rows = [row for payload in payloads for row in payload["rows"]]
    rows.sort(key=lambda row: int(row["query_index"]))
    if [int(row["query_index"]) for row in rows] != list(range(500)):
        raise ValueError("merged development query indexes are incomplete or duplicated")
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-development-features-merged",
        **{field: first[field] for field in IDENTITY_FIELDS if field != "shard_count"},
        "source_commits": sorted({str(payload["source_commit"]) for payload in payloads}),
        "shards": len(payloads),
        "elapsed_ms_sum": round(sum(float(payload["elapsed_ms"]) for payload in payloads), 3),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(result["rows"]),
                "shards": result["shards"],
            }
        )
    )


if __name__ == "__main__":
    main()
