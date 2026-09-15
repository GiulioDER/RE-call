"""Compare reused source conditioning traces with the former duplicate query path."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.observability import percentile  # noqa: E402
from recall.source_conditioning import load_source_conditioning_artifact  # noqa: E402
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import (  # noqa: E402
    EXPECTED_FACT_SHA256,
    EXPECTED_PIPELINE,
    EXPECTED_QUERY_SHA256,
    _call_query,
    _performance,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


EXPECTED_GENERATION_ID = "gen_ff9076737d834e21b0205c3a360bea99"
EXPECTED_CALIBRATION_ID = "cal_473e73efffe243d097742d652b596eea"
EXPECTED_CORPUS = "c1161c68125b45e6ff4fd8b673bc1a78e8b65ef4e5f229026b546298a74aa965"


def _identity(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = (
        str(payload.get("generation_id")),
        str(payload.get("calibration_id")),
        str(payload.get("pipeline_fingerprint")),
        str(payload.get("corpus_fingerprint")),
    )
    expected = (
        EXPECTED_GENERATION_ID,
        EXPECTED_CALIBRATION_ID,
        EXPECTED_PIPELINE,
        EXPECTED_CORPUS,
    )
    if identity != expected:
        raise RuntimeError(f"serving lineage mismatch: expected {expected}, got {identity}")
    return identity


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [
        row
        for row in rows
        if isinstance(row.get("reuse_ms"), (int, float))
        and not isinstance(row.get("reuse_ms"), bool)
        and isinstance(row.get("duplicate_ms"), (int, float))
        and not isinstance(row.get("duplicate_ms"), bool)
    ]
    reuse = [float(row["reuse_ms"]) for row in paired]
    duplicate = [float(row["duplicate_ms"]) for row in paired]
    reuse_p95 = percentile(sorted(reuse), 0.95, ndigits=None) if reuse else None
    duplicate_p95 = percentile(sorted(duplicate), 0.95, ndigits=None) if duplicate else None
    reuse_median = statistics.median(reuse) if reuse else None
    duplicate_median = statistics.median(duplicate) if duplicate else None
    return {
        "requests": len(rows),
        "hash_parity_count": sum(bool(row["selected_hash_parity"]) for row in rows),
        "shadow_ok_count": sum(row["shadow_status"] == "ok" for row in rows),
        "audit_ok_count": sum(row["audit_status"] == "ok" for row in rows),
        "timing_receipt_count": len(paired),
        "reuse_median_ms": None if reuse_median is None else round(reuse_median, 3),
        "reuse_p95_ms": None if reuse_p95 is None else round(reuse_p95, 3),
        "duplicate_median_ms": (None if duplicate_median is None else round(duplicate_median, 3)),
        "duplicate_p95_ms": None if duplicate_p95 is None else round(duplicate_p95, 3),
        "median_ratio": (
            None
            if reuse_median is None or duplicate_median in {None, 0.0}
            else round(reuse_median / duplicate_median, 6)
        ),
        "p95_ratio": (
            None
            if reuse_p95 is None or duplicate_p95 in {None, 0.0}
            else round(reuse_p95 / duplicate_p95, 6)
        ),
    }


def _decision(summary: dict[str, Any]) -> str:
    if (
        summary["requests"] != 50
        or summary["hash_parity_count"] != 50
        or summary["shadow_ok_count"] != 50
        or summary["audit_ok_count"] != 50
        or summary["timing_receipt_count"] != 50
    ):
        return "REPAIR"
    if (
        float(summary["median_ratio"]) <= 0.10
        and float(summary["p95_ratio"]) <= 0.20
        and float(summary["reuse_p95_ms"]) <= 15.0
    ):
        return "SHIP LOW RATE SHADOW"
    return "KEEP LOW SAMPLE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-set",
        default="docs/preregistrations/2026-09-13-memory-queries-source-gold.json",
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.generation_id != EXPECTED_GENERATION_ID:
        raise ValueError("generation differs from the registered validation generation")

    query_bytes = Path(args.query_set).read_bytes()
    artifact_bytes = Path(args.artifact).read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != EXPECTED_QUERY_SHA256:
        raise RuntimeError("query set differs from the registered digest")
    queries = json.loads(query_bytes.decode("utf-8"))
    if not isinstance(queries, list) or len(queries) != 50:
        raise RuntimeError("registered query set must contain exactly 50 queries")
    model = load_source_conditioning_artifact(args.artifact)
    model.assert_compatible(
        pipeline_fingerprint=EXPECTED_PIPELINE,
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
        benchmark_source_conditioning_reuse_audit=True,
        source_conditioning_mode="shadow",
        source_conditioning_artifact=args.artifact.replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
    )
    client = TTYMCP(command, args.timeout)
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"trace-reuse {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            public = _call_query(client, request_id, str(query["query"]))
            client_ms = (time.perf_counter() - started) * 1000.0
            _identity(public)
            shadow = _audit(public, "source_conditioning_shadow")
            audit = _audit(public, "source_conditioning_trace_reuse_audit")
            performance = _performance(public)
            spans = performance.get("spans_ms", {})
            if not isinstance(spans, dict):
                raise RuntimeError("performance spans are missing")
            reuse_ms = audit.get("reuse_ms")
            duplicate_ms = audit.get("duplicate_ms")
            rows.append(
                {
                    "query_index": query_index,
                    "query_id": query["id"],
                    "query": query["query"],
                    "client_observed_ms": round(client_ms, 3),
                    "shadow_status": shadow.get("status"),
                    "audit_status": audit.get("status"),
                    "selected_hash_parity": audit.get("selected_hash_parity") is True,
                    "reuse_ms": (
                        float(reuse_ms)
                        if isinstance(reuse_ms, (int, float)) and not isinstance(reuse_ms, bool)
                        else None
                    ),
                    "duplicate_ms": (
                        float(duplicate_ms)
                        if isinstance(duplicate_ms, (int, float))
                        and not isinstance(duplicate_ms, bool)
                        else None
                    ),
                    "shadow_span_ms": spans.get("source_conditioning_shadow_ms"),
                    "duplicate_span_ms": spans.get("source_conditioning_duplicate_audit_ms"),
                    "selected_chunk_hashes": shadow.get("selected_chunk_hashes"),
                    "baseline_chunk_hashes": shadow.get("baseline_chunk_hashes"),
                }
            )
            request_id += 1
    finally:
        client.close()

    summary = _summary(rows)
    decision = _decision(summary)
    result = {
        "schema_version": 1,
        "protocol": "2026-09-13-source-conditioning-trace-reuse",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels_sha256": EXPECTED_FACT_SHA256,
        "model_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "model_artifact_fingerprint": model.artifact_fingerprint,
        "generation_id": EXPECTED_GENERATION_ID,
        "calibration_id": EXPECTED_CALIBRATION_ID,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "client_observed_total_ms": round((time.perf_counter() - started_run) * 1000.0, 3),
        "decision": decision,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(output), "decision": decision, "summary": summary}))


if __name__ == "__main__":
    main()
