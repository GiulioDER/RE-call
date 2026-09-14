"""Validate frozen query-anchor source digests against one immutable generation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import psycopg


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inventory(
    pool: dict[str, Any], observed: Iterable[tuple[str, str]]
) -> dict[str, int | str | bool]:
    expected = {
        str(query["gold_sources"][0]): str(query["source_sha256"])
        for query in pool["queries"]
        if query["expected_answerability"] == "answerable"
    }
    live = {str(source): str(digest) for source, digest in observed}
    matched = sum(source in live for source in expected)
    mismatches = sum(
        source in live and live[source] != digest for source, digest in expected.items()
    )
    missing = len(expected) - matched
    return {
        "frozen_sources": len(expected),
        "matched_sources": matched,
        "missing_sources": missing,
        "digest_mismatches": mismatches,
        "decision": (
            "HOLDOUT_LINEAGE_VALID"
            if matched == len(expected) and mismatches == 0
            else "HOLDOUT_LINEAGE_INVALID"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("RECALL_SERVING_DSN") or os.environ.get("RECALL_DSN"),
    )
    args = parser.parse_args()
    if not args.dsn:
        raise RuntimeError("RECALL_SERVING_DSN or RECALL_DSN is required")

    pool = json.loads(args.query_pool.read_text(encoding="utf-8"))
    with psycopg.connect(args.dsn, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('recall.tenant_id', %s, false)", (args.tenant,)
            )
            cursor.execute(
                "SELECT DISTINCT source_uri, source_sha256 FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s",
                (args.tenant, args.generation_id),
            )
            observed = list(cursor.fetchall())
    summary = validate_inventory(pool, observed)
    result = {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-inventory",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "policy_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "generation_id": args.generation_id,
        "inventory_truncated": False,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
