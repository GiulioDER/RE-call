"""Diagnose frozen query-anchor blockers on consumed empty-base holdout rows."""

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

from recall.query_anchor_admission import (  # noqa: E402
    QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR,
    QUERY_ANCHOR_LIMIT,
    query_anchor_candidate_eligible,
    query_anchor_features,
)
from recall.source_conditioning import chunk_identifier_hash  # noqa: E402
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    proposals = [row for row in rows if row["has_proposal"]]
    return {
        "empty_base_rows": len(rows),
        "empty_base_answerable": sum(
            row["expected_answerability"] == "answerable" for row in rows
        ),
        "empty_base_controls": sum(
            row["expected_answerability"] == "unanswerable" for row in rows
        ),
        "guarded_proposals": len(proposals),
        "no_guarded_proposal": len(rows) - len(proposals),
        "three_anchor_proposals": sum(
            row["features"]["anchor_count"] == QUERY_ANCHOR_LIMIT for row in proposals
        ),
        "zero_df_clear_proposals": sum(
            row["features"]["zero_document_frequency_anchors"] == 0
            for row in proposals
        ),
        "chunk_coverage_clear_proposals": sum(
            row["features"]["chunk_coverage_fraction"]
            >= QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR
            for row in proposals
        ),
        "eligible_proposals": sum(
            query_anchor_candidate_eligible(0, row["features"]) for row in proposals
        ),
        "answerable_proposals": sum(
            row["expected_answerability"] == "answerable" for row in proposals
        ),
        "control_proposals": sum(
            row["expected_answerability"] == "unanswerable" for row in proposals
        ),
        "proposal_exact_spans": sum(row["chunk_contains_exact_span"] for row in proposals),
        "proposal_gold_sources": sum(row["is_gold_source"] for row in proposals),
        "proposal_source_pool_exact_spans": sum(
            row["source_pool_contains_exact_span"] for row in proposals
        ),
    }


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
    parser.add_argument("--holdout-result", type=Path, required=True)
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
    queries = list(pool_payload["queries"])
    holdout = json.loads(args.holdout_result.read_text(encoding="utf-8"))
    if holdout.get("status") != "COMPLETE" or holdout.get("decision") != "FAIL_NO_EXACT_SPAN_GAIN":
        raise RuntimeError("diagnostic requires the completed failed holdout")
    indexes = [
        int(row["query_index"]) for row in holdout["rows"] if int(row["base_count"]) == 0
    ]
    if len(indexes) != 30:
        raise RuntimeError("registered diagnostic expected exactly 30 empty-base rows")

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
        benchmark_source_admission_audit=True,
        source_conditioning_mode="shadow",
        source_conditioning_artifact=str(args.artifact).replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
        source_conditioning_policy="guarded_spare_slot",
    )
    rows: list[dict[str, Any]] = []
    client = TTYMCP(command, args.timeout)
    started = time.perf_counter()
    try:
        request_id = _initialize(client)
        for completed, query_index in enumerate(indexes, start=1):
            query = queries[query_index]
            payload = _call_query(client, request_id, str(query["query"]))
            _identity(payload, args)
            shadow = _audit(payload, "source_conditioning_shadow")
            pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
            base_hashes = list(shadow.get("alpha008_chunk_hashes", []))
            selected_hashes = list(shadow.get("selected_chunk_hashes", []))
            if base_hashes:
                raise RuntimeError("repeated diagnostic no longer has an empty base")
            by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in pool}
            missing = [value for value in selected_hashes if value not in by_hash]
            if missing:
                raise RuntimeError("benchmark pool cannot resolve guarded hashes")
            row: dict[str, Any] = {
                "query_index": query_index,
                "expected_answerability": str(query["expected_answerability"]),
                "has_proposal": bool(selected_hashes),
            }
            if selected_hashes:
                proposal = by_hash[selected_hashes[0]]
                answerable = query["expected_answerability"] == "answerable"
                answer_span = " ".join(str(query["answer_span"]).split())
                source_items = [item for item in pool if item["source"] == proposal["source"]]
                row.update(
                    {
                        "features": query_anchor_features(
                            str(query["query"]), proposal, pool
                        ),
                        "chunk_contains_exact_span": answerable
                        and answer_span in " ".join(str(proposal["text"]).split()),
                        "is_gold_source": str(proposal["source"])
                        in {str(value) for value in query.get("gold_sources", [])},
                        "source_pool_contains_exact_span": answerable
                        and answer_span
                        in " ".join(
                            "\n".join(str(item["text"]) for item in source_items).split()
                        ),
                    }
                )
            rows.append(row)
            print(f"diagnostic {completed}/{len(indexes)}", flush=True)
            request_id += 1
    finally:
        client.close()

    result = {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-empty-base-diagnostic",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "policy_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "holdout_result_sha256": _sha256(args.holdout_result),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
