"""Recapture numeric guarded spare-slot features for the retired development cohort."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.source_conditioning import (  # noqa: E402
    SOURCE_FEATURE_NAMES,
    SourceConditioningArtifact,
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
    source_features,
    source_support,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_query(
    query: dict[str, Any],
    capture_row: dict[str, Any],
    pool_audit: dict[str, Any],
    leg_audit: dict[str, Any],
    artifact: SourceConditioningArtifact,
) -> dict[str, Any]:
    pool = list(pool_audit["items"])
    dense = list(leg_audit["dense"])
    sparse = list(leg_audit["sparse"])
    by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in pool}
    base_hashes = list(capture_row["base_hashes"])
    expected_hashes = list(capture_row["candidate_hashes"])
    missing = [value for value in {*base_hashes, *expected_hashes} if value not in by_hash]
    if missing:
        raise ValueError(f"recapture cannot resolve {len(missing)} frozen chunk hashes")
    base = [by_hash[value] for value in base_hashes]
    selected, receipts = fill_source_conditioned_spare_slots(
        artifact, base, pool, dense, sparse
    )
    observed_hashes = [chunk_identifier_hash(item["chunk_id"]) for item in selected]
    if observed_hashes != expected_hashes:
        raise ValueError("recaptured guarded selection differs from frozen candidate")

    features = source_features(
        pool,
        dense,
        sparse,
        candidate_k=artifact.candidate_k,
        rrf_k=artifact.rrf_k,
    )
    supports = source_support(artifact, features)
    gold_sources = {str(value) for value in query.get("gold_sources", [])}
    base_sources = {str(item["source"]) for item in base}
    additions: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts, start=1):
        source = str(receipt["source"])
        item = by_hash[str(receipt["chunk_hash"])]
        source_values = features[source]
        additions.append(
            {
                "addition_index": index,
                "chunk_hash": receipt["chunk_hash"],
                "source": source,
                "ordinal": receipt["ordinal"],
                "lane": receipt["lane"],
                "cosine": receipt["cosine"],
                "confidence": item["confidence"],
                "verdict": item["verdict"],
                "pool_rank": item["pool_rank"],
                "source_support": supports[source],
                "dense_rank": receipt["dense_rank"],
                "sparse_rank": receipt["sparse_rank"],
                "features": dict(zip(SOURCE_FEATURE_NAMES, source_values, strict=True)),
                "is_gold_source": source in gold_sources,
            }
        )
    return {
        "query_id": str(query["id"]),
        "expected_answerability": str(query["expected_answerability"]),
        "base_count": len(base),
        "base_gold_source_hit": bool(gold_sources & base_sources),
        "additions": additions,
    }


def try_analyze_query(
    query: dict[str, Any],
    capture_row: dict[str, Any],
    pool_audit: dict[str, Any],
    leg_audit: dict[str, Any],
    artifact: SourceConditioningArtifact,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a parity row or an aggregate-safe mismatch reason."""
    try:
        return analyze_query(query, capture_row, pool_audit, leg_audit, artifact), None
    except ValueError as exc:
        if str(exc) == "recaptured guarded selection differs from frozen candidate":
            return None, "selection_parity"
        raise


def _identity(payload: dict[str, Any], args: argparse.Namespace) -> None:
    observed = (
        payload.get("generation_id"),
        payload.get("calibration_id"),
        payload.get("pipeline_fingerprint"),
        payload.get("corpus_fingerprint"),
    )
    expected = (
        args.generation_id,
        args.calibration_id,
        args.pipeline_fingerprint,
        args.corpus_fingerprint,
    )
    if observed != expected:
        raise RuntimeError(f"serving lineage mismatch: expected {expected}, got {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--pipeline-fingerprint", required=True)
    parser.add_argument("--corpus-fingerprint", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    pool_payload = json.loads(args.query_pool.read_text(encoding="utf-8"))
    capture_payload = json.loads(args.capture.read_text(encoding="utf-8"))
    if capture_payload["query_pool_sha256"] != _sha256(args.query_pool):
        raise ValueError("capture query pool digest mismatch")
    capture_rows = {
        str(row["query_id"]): row for row in capture_payload["rows"] if row["triggered"]
    }
    queries = {
        str(query["id"]): query
        for query in pool_payload["queries"]
        if str(query["id"]) in capture_rows
    }
    if set(queries) != set(capture_rows):
        raise ValueError("query pool does not cover the frozen triggered cohort")
    artifact = load_source_conditioning_artifact(args.artifact)
    artifact.assert_compatible(
        pipeline_fingerprint=args.pipeline_fingerprint,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile=args.profile,
        candidate_k=20,
    )

    command = _command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        "combined",
        "none",
        20260825,
        32,
        0.10,
        args.generation_id,
        benchmark_retrieval_leg_audit=True,
        benchmark_source_admission_audit=True,
        source_conditioning_mode="shadow",
        source_conditioning_artifact=str(args.artifact).replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
        source_conditioning_policy="guarded_spare_slot",
    )
    rows: list[dict[str, Any]] = []
    mismatches = 0
    client = TTYMCP(command, args.timeout)
    started = time.perf_counter()
    try:
        request_id = _initialize(client)
        for index, query_id in enumerate(capture_rows, start=1):
            payload = _call_query(client, request_id, str(queries[query_id]["query"]))
            _identity(payload, args)
            row, mismatch = try_analyze_query(
                queries[query_id],
                capture_rows[query_id],
                _audit(payload, "source_admission_benchmark_audit"),
                _audit(payload, "retrieval_leg_benchmark_audit"),
                artifact,
            )
            if row is not None:
                rows.append(row)
            elif mismatch == "selection_parity":
                mismatches += 1
            else:
                raise RuntimeError("development recapture returned an unknown mismatch")
            print(f"recaptured {index}/{len(capture_rows)}", flush=True)
            request_id += 1
    finally:
        client.close()

    gold_source_additions = sum(
        addition["is_gold_source"] for row in rows for addition in row["additions"]
    )
    decision = (
        "READY_FOR_POLICY_FIT"
        if len(rows) >= 30 and gold_source_additions >= 5
        else "INSUFFICIENT_DEVELOPMENT_PARITY"
    )
    output = {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-development-features",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "capture_sha256": _sha256(args.capture),
        "artifact_sha256": _sha256(args.artifact),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "decision": decision,
        "attempted_rows": len(capture_rows),
        "parity_rows": len(rows),
        "selection_parity_mismatches": mismatches,
        "gold_source_additions": gold_source_additions,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": decision,
                "parity_rows": len(rows),
                "selection_parity_mismatches": mismatches,
                "gold_source_additions": gold_source_additions,
            }
        )
    )


if __name__ == "__main__":
    main()
