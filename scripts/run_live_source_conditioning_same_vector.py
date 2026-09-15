"""Validate source conditioning against the public baseline from the same query vector."""

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
    chunk_identifier_hash,
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import (  # noqa: E402
    _audit,
    _score_selection,
)
from scripts.run_live_source_conditioning_shadow import (  # noqa: E402
    EXPECTED_CALIBRATION_ID,
    EXPECTED_CORPUS,
    EXPECTED_FACT_SHA256,
    EXPECTED_GENERATION_ID,
    EXPECTED_PIPELINE,
    EXPECTED_QUERY_SHA256,
    _call_query,
    _identity,
    _performance,
    _shadow_internal_ms,
    _summarize,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


def _decision(
    summary: dict[str, Any],
    *,
    candidate_hash_parity_count: int,
    baseline_hash_parity_count: int,
    errors: int,
    timing_receipts: int,
    elapsed_ms: float,
) -> str:
    if (
        candidate_hash_parity_count != 50
        or baseline_hash_parity_count != 50
        or errors
        or timing_receipts != 50
    ):
        return "REPAIR"
    baseline = summary["arms"]["baseline"]
    candidate = summary["arms"]["candidate"]
    if (
        int(candidate["complete_queries"]) < int(baseline["complete_queries"])
        or int(candidate["covered_facts"]) < int(baseline["covered_facts"])
        or int(candidate["unanswerable_answers"])
        > int(baseline["unanswerable_answers"])
        or int(candidate["unanswerable_answers"]) > 2
        or float(candidate["context_precision"] or 0.0) < 0.55
    ):
        return "CLOSE"
    gain = (
        int(candidate["complete_queries"]) > int(baseline["complete_queries"])
        or int(candidate["covered_facts"]) > int(baseline["covered_facts"])
    )
    if gain and elapsed_ms < 12 * 60 * 1000:
        return "BUILD SAMPLED SHADOW"
    return "GATE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-set",
        default="docs/preregistrations/2026-09-13-memory-queries-source-gold.json",
    )
    parser.add_argument(
        "--fact-labels",
        default="docs/preregistrations/2026-09-13-memory-essential-facts.json",
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
    label_bytes = Path(args.fact_labels).read_bytes()
    artifact_bytes = Path(args.artifact).read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != EXPECTED_QUERY_SHA256:
        raise RuntimeError("query set differs from the registered digest")
    if hashlib.sha256(label_bytes).hexdigest() != EXPECTED_FACT_SHA256:
        raise RuntimeError("fact labels differ from the registered digest")
    model = load_source_conditioning_artifact(args.artifact)
    model.assert_compatible(
        pipeline_fingerprint=EXPECTED_PIPELINE,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile=args.profile,
        candidate_k=20,
    )
    queries = json.loads(query_bytes.decode("utf-8"))
    labels = json.loads(label_bytes.decode("utf-8"))
    labels_by_id = {str(value["query_id"]): value for value in labels}

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
        source_conditioning_artifact=args.artifact.replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
    )
    client = TTYMCP(command, args.timeout)
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"same-vector {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            public = _call_query(client, request_id, str(query["query"]))
            observed_ms = (time.perf_counter() - started) * 1000.0
            _identity(public)
            shadow = _audit(public, "source_conditioning_shadow")
            leg_audit = _audit(public, "retrieval_leg_benchmark_audit")
            pool_audit = _audit(public, "source_admission_benchmark_audit")
            selected = select_source_conditioned(
                model,
                pool_audit["items"],
                leg_audit["dense"],
                leg_audit["sparse"],
                threshold=float(pool_audit["threshold"]),
            )
            expected_candidate_hashes = [
                chunk_identifier_hash(value["chunk_id"]) for value in selected
            ]
            candidate_hash_parity = (
                shadow.get("selected_chunk_hashes") == expected_candidate_hashes
                and int(shadow.get("selected_count", -1)) == len(selected)
            )
            public_items = list(public.get("trusted_evidence", {}).get("items", []))
            expected_baseline_hashes = [
                chunk_identifier_hash(value["chunk_id"]) for value in public_items
            ]
            baseline_hash_parity = (
                shadow.get("baseline_chunk_hashes") == expected_baseline_hashes
            )
            internal_ms = _shadow_internal_ms(_performance(public))
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": labels_by_id.get(str(query["id"])),
                    "client_observed_ms": round(observed_ms, 3),
                    "shadow_internal_ms": internal_ms,
                    "candidate_hash_parity": candidate_hash_parity,
                    "baseline_hash_parity": baseline_hash_parity,
                    "shadow_diagnostic": shadow,
                    "baseline_items": public_items,
                    "candidate_items": selected,
                }
            )
            request_id += 1
    finally:
        client.close()

    elapsed_ms = round((time.perf_counter() - started_run) * 1000.0, 3)
    for row in rows:
        row["scores"] = {
            "baseline": _score_selection(row["baseline_items"], row["label"]),
            "candidate": _score_selection(row["candidate_items"], row["label"]),
        }
    summary = _summarize(rows)
    candidate_hash_parity_count = sum(
        bool(row["candidate_hash_parity"]) for row in rows
    )
    baseline_hash_parity_count = sum(
        bool(row["baseline_hash_parity"]) for row in rows
    )
    error_count = sum(row["shadow_diagnostic"].get("status") != "ok" for row in rows)
    timing_receipts = sum(row["shadow_internal_ms"] is not None for row in rows)
    decision = _decision(
        summary,
        candidate_hash_parity_count=candidate_hash_parity_count,
        baseline_hash_parity_count=baseline_hash_parity_count,
        errors=error_count,
        timing_receipts=timing_receipts,
        elapsed_ms=elapsed_ms,
    )
    result = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-source-conditioning-same-vector",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "model_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "model_artifact_fingerprint": model.artifact_fingerprint,
        "generation_id": EXPECTED_GENERATION_ID,
        "calibration_id": EXPECTED_CALIBRATION_ID,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "public_request_count": len(rows),
        "client_observed_total_ms": elapsed_ms,
        "candidate_hash_parity_count": candidate_hash_parity_count,
        "baseline_hash_parity_count": baseline_hash_parity_count,
        "shadow_error_count": error_count,
        "timing_receipt_count": timing_receipts,
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
