"""Measure dense-ranked empty-base rescue behind the query anchor safety gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.query_anchor_admission import (  # noqa: E402
    QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR,
    QUERY_ANCHOR_LIMIT,
    query_anchor_features,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


RULES = ("dense_first_anchor_safe", "dense_first_anchor_compatible")
CUTOFFS = (1, 3, 5, 10, 20)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(str(value).split())


def choose_dense_candidates(
    query: str,
    pool: Sequence[Mapping[str, object]],
    dense: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object] | None]:
    if not pool or not dense:
        return {rule: None for rule in RULES}
    query_features = query_anchor_features(query, pool[0], pool)
    safe = (
        int(query_features["anchor_count"]) == QUERY_ANCHOR_LIMIT
        and int(query_features["zero_document_frequency_anchors"]) == 0
    )
    if not safe:
        return {rule: None for rule in RULES}
    compatible = next(
        (
            item
            for item in dense
            if float(query_anchor_features(query, item, pool)["chunk_coverage_fraction"])
            >= QUERY_ANCHOR_CHUNK_COVERAGE_FLOOR
        ),
        None,
    )
    return {
        "dense_first_anchor_safe": dense[0],
        "dense_first_anchor_compatible": compatible,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    result: dict[str, Any] = {
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(rows) - len(answerable),
        "dense_gold_source_by_cutoff": {
            str(cutoff): sum(
                row["min_dense_gold_rank"] is not None
                and row["min_dense_gold_rank"] <= cutoff
                for row in answerable
            )
            for cutoff in CUTOFFS
        },
        "dense_exact_span_by_cutoff": {
            str(cutoff): sum(
                row["min_dense_exact_rank"] is not None
                and row["min_dense_exact_rank"] <= cutoff
                for row in answerable
            )
            for cutoff in CUTOFFS
        },
        "rules": {},
    }
    for rule in RULES:
        selected = [row for row in rows if row[rule]["selected"]]
        exact = sum(row[rule]["exact_span"] for row in selected)
        result["rules"][rule] = {
            "selected_total": len(selected),
            "selected_answerable": sum(
                row["expected_answerability"] == "answerable" for row in selected
            ),
            "selected_controls": sum(
                row["expected_answerability"] == "unanswerable" for row in selected
            ),
            "exact_span_gains": exact,
            "gold_source_selections": sum(row[rule]["gold_source"] for row in selected),
            "source_pool_exact_spans": sum(
                row[rule]["source_pool_exact_span"] for row in selected
            ),
            "exact_span_precision": exact / len(selected) if selected else None,
        }
    ranked = sorted(
        RULES,
        key=lambda rule: (
            result["rules"][rule]["exact_span_precision"] or -1.0,
            result["rules"][rule]["exact_span_gains"],
            -result["rules"][rule]["selected_total"],
        ),
        reverse=True,
    )
    winner = ranked[0]
    metrics = result["rules"][winner]
    proceed = (
        metrics["exact_span_gains"] >= 2
        and metrics["selected_controls"] == 0
        and metrics["exact_span_precision"] is not None
        and metrics["exact_span_precision"] >= 0.5
    )
    result["selected_rule"] = winner
    result["decision"] = "PROCEED_FRESH_HOLDOUT" if proceed else "STOP_DENSE_RANK_RESCUE"
    return result


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
            if _audit(payload, "source_conditioning_shadow").get("alpha008_chunk_hashes"):
                raise RuntimeError("repeated development row no longer has an empty base")
            pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
            dense_audit = sorted(
                _audit(payload, "retrieval_leg_benchmark_audit")["dense"],
                key=lambda item: int(item["rank"]),
            )
            by_id = {str(item["chunk_id"]): item for item in pool}
            dense = [
                by_id[str(item["chunk_id"])]
                for item in dense_audit
                if str(item["chunk_id"]) in by_id
            ]
            answerable = query["expected_answerability"] == "answerable"
            gold = {str(value) for value in query.get("gold_sources", [])}
            span = _normalize(query["answer_span"])

            def exact(item: Mapping[str, object]) -> bool:
                return answerable and span in _normalize(item["text"])

            candidates = choose_dense_candidates(str(query["query"]), pool, dense)
            row: dict[str, Any] = {
                "query_index": query_index,
                "expected_answerability": str(query["expected_answerability"]),
                "min_dense_gold_rank": next(
                    (rank for rank, item in enumerate(dense, start=1) if str(item["source"]) in gold),
                    None,
                ),
                "min_dense_exact_rank": next(
                    (rank for rank, item in enumerate(dense, start=1) if exact(item)), None
                ),
            }
            for rule, selected in candidates.items():
                source_items = (
                    [item for item in pool if item["source"] == selected["source"]]
                    if selected is not None
                    else []
                )
                row[rule] = {
                    "selected": selected is not None,
                    "exact_span": selected is not None and exact(selected),
                    "gold_source": selected is not None and str(selected["source"]) in gold,
                    "source_pool_exact_span": selected is not None
                    and any(exact(item) for item in source_items),
                }
            rows.append(row)
            print(f"dense development {completed}/{len(indexes)}", flush=True)
            request_id += 1
    finally:
        client.close()

    result = {
        "schema_version": 1,
        "protocol": "2026-09-14-dense-rank-anchor-development",
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
