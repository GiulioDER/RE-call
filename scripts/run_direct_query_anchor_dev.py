"""Measure direct query-anchor selection on consumed empty-base rows."""

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

from recall.query_anchor_admission import direct_query_anchor_candidate  # noqa: E402
from recall.source_conditioning import chunk_identifier_hash  # noqa: E402
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(str(value).split())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    selected = [row for row in rows if row["selected"]]
    exact = sum(row["selected_exact_span"] for row in selected)
    precision = exact / len(selected) if selected else None
    proceed = exact >= 2 and not any(row["selected"] for row in controls) and precision is not None and precision >= 0.5
    return {
        "decision": "PROCEED_FRESH_HOLDOUT" if proceed else "STOP_DIRECT_POOL_RANKING",
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(controls),
        "pool_gold_source_reachable": sum(row["pool_gold_source"] for row in answerable),
        "pool_exact_span_reachable": sum(row["pool_exact_span"] for row in answerable),
        "dense_gold_source_reachable": sum(row["dense_gold_source"] for row in answerable),
        "dense_exact_span_reachable": sum(row["dense_exact_span"] for row in answerable),
        "sparse_gold_source_reachable": sum(row["sparse_gold_source"] for row in answerable),
        "sparse_exact_span_reachable": sum(row["sparse_exact_span"] for row in answerable),
        "selected_total": len(selected),
        "selected_answerable": sum(row["expected_answerability"] == "answerable" for row in selected),
        "selected_controls": sum(row["expected_answerability"] == "unanswerable" for row in selected),
        "selected_exact_spans": exact,
        "selected_gold_sources": sum(row["selected_gold_source"] for row in selected),
        "selected_source_pool_exact_spans": sum(
            row["selected_source_pool_exact_span"] for row in selected
        ),
        "selected_exact_span_precision": precision,
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

    queries = list(json.loads(args.query_pool.read_text(encoding="utf-8"))["queries"])
    holdout = json.loads(args.holdout_result.read_text(encoding="utf-8"))
    indexes = [int(row["query_index"]) for row in holdout["rows"] if int(row["base_count"]) == 0]
    if holdout.get("status") != "COMPLETE" or len(indexes) != 30:
        raise RuntimeError("development screen requires the completed 30-row empty-base cohort")

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
    client = TTYMCP(command, args.timeout)
    started = time.perf_counter()
    try:
        request_id = _initialize(client)
        for completed, query_index in enumerate(indexes, start=1):
            query = queries[query_index]
            payload = _call_query(client, request_id, str(query["query"]))
            _identity(payload, args)
            shadow = _audit(payload, "source_conditioning_shadow")
            if shadow.get("alpha008_chunk_hashes"):
                raise RuntimeError("repeated development row no longer has an empty base")
            pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
            legs = _audit(payload, "retrieval_leg_benchmark_audit")
            by_id = {str(item["chunk_id"]): item for item in pool}
            by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in pool}

            def leg_items(name: str) -> list[dict[str, Any]]:
                return [
                    by_id[str(item["chunk_id"])]
                    for item in legs[name]
                    if str(item["chunk_id"]) in by_id
                ]

            dense = leg_items("dense")
            sparse = leg_items("sparse")
            answerable = query["expected_answerability"] == "answerable"
            gold = {str(value) for value in query.get("gold_sources", [])}
            span = _normalize(query["answer_span"])

            def source_hit(items: list[dict[str, Any]]) -> bool:
                return bool(gold & {str(item["source"]) for item in items})

            def exact_hit(items: list[dict[str, Any]]) -> bool:
                return answerable and any(span in _normalize(item["text"]) for item in items)

            selected = direct_query_anchor_candidate(str(query["query"]), pool)
            if selected is not None and chunk_identifier_hash(selected["chunk_id"]) not in by_hash:
                raise RuntimeError("direct selection escaped the audited pool")
            source_items = (
                [item for item in pool if item["source"] == selected["source"]]
                if selected is not None
                else []
            )
            rows.append(
                {
                    "query_index": query_index,
                    "expected_answerability": str(query["expected_answerability"]),
                    "pool_gold_source": source_hit(pool) if answerable else False,
                    "pool_exact_span": exact_hit(pool),
                    "dense_gold_source": source_hit(dense) if answerable else False,
                    "dense_exact_span": exact_hit(dense),
                    "sparse_gold_source": source_hit(sparse) if answerable else False,
                    "sparse_exact_span": exact_hit(sparse),
                    "selected": selected is not None,
                    "selected_gold_source": selected is not None
                    and str(selected["source"]) in gold,
                    "selected_exact_span": selected is not None
                    and exact_hit([dict(selected)]),
                    "selected_source_pool_exact_span": selected is not None
                    and exact_hit(source_items),
                }
            )
            print(f"direct development {completed}/{len(indexes)}", flush=True)
            request_id += 1
    finally:
        client.close()

    result = {
        "schema_version": 1,
        "protocol": "2026-09-14-direct-query-anchor-development",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
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
