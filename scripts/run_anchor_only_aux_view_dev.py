"""Measure one anchor only auxiliary retrieval view on consumed empty base rows."""

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

from recall.query_anchor_admission import QUERY_ANCHOR_LIMIT  # noqa: E402
from scripts.run_anchor_boosted_query_dev import (  # noqa: E402
    _cutoff_counts,
    _dense_items,
    _identity,
    _minimum_rank,
    _normalize,
    selected_supported_anchors,
    validate_baseline,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


UNION_CUTOFF = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_only_query(
    query: str,
    pool: Sequence[Mapping[str, object]],
) -> tuple[str | None, tuple[str, ...]]:
    """Return only the three safe anchors in original question order."""

    anchors = selected_supported_anchors(query, pool)
    if len(anchors) != QUERY_ANCHOR_LIMIT:
        return None, anchors
    return " ".join(anchors), anchors


def deduplicated_union(
    original: Sequence[Mapping[str, object]],
    auxiliary: Sequence[Mapping[str, object]],
    *,
    cutoff: int = UNION_CUTOFF,
) -> list[Mapping[str, object]]:
    """Preserve the original prefix and append unique auxiliary candidates."""

    combined: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for item in (*original[:cutoff], *auxiliary[:cutoff]):
        chunk_id = str(item["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        combined.append(item)
    return combined


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    summary: dict[str, Any] = {
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(controls),
        "eligible_answerable": sum(row["eligible"] for row in answerable),
        "eligible_controls": sum(row["eligible"] for row in controls),
        "original_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "original_min_gold_rank"
        ),
        "original_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "original_min_exact_rank"
        ),
        "auxiliary_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "auxiliary_min_gold_rank"
        ),
        "auxiliary_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "auxiliary_min_exact_rank"
        ),
        "union_top20_gold_source_reachable": sum(
            row["union_top20_gold_source"] for row in answerable
        ),
        "union_top20_exact_span_reachable": sum(
            row["union_top20_exact_span"] for row in answerable
        ),
        "incremental_gold_source_rows": sum(
            row["incremental_gold_source"] for row in answerable
        ),
        "incremental_exact_span_rows": sum(
            row["incremental_exact_span"] for row in answerable
        ),
        "unique_auxiliary_chunks_added": sum(
            int(row["unique_auxiliary_chunks_added"]) for row in rows
        ),
    }
    validate_baseline(summary)
    proceed = (
        summary["incremental_exact_span_rows"] >= 2
        and summary["eligible_controls"] == 0
    )
    summary["decision"] = (
        "PROCEED_ONE_SLOT_SELECTOR"
        if proceed
        else "STOP_LEXICAL_ANCHOR_QUERY_CONSTRUCTION"
    )
    return summary


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
    indexes = [
        int(row["query_index"])
        for row in holdout["rows"]
        if int(row["base_count"]) == 0
    ]
    if holdout.get("status") != "COMPLETE" or len(indexes) != 30:
        raise RuntimeError("development screen requires the completed 30 row cohort")

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
            query_text = str(query["query"])
            original_payload = _call_query(client, request_id, query_text)
            request_id += 1
            _identity(original_payload, args)
            if _audit(original_payload, "source_conditioning_shadow").get(
                "alpha008_chunk_hashes"
            ):
                raise RuntimeError("development row no longer has an empty base")
            original_pool = list(
                _audit(original_payload, "source_admission_benchmark_audit")["items"]
            )
            original_dense = _dense_items(original_payload)
            auxiliary_query, anchors = anchor_only_query(query_text, original_pool)
            auxiliary_dense: list[dict[str, Any]] = []
            if auxiliary_query is not None:
                auxiliary_payload = _call_query(client, request_id, auxiliary_query)
                request_id += 1
                _identity(auxiliary_payload, args)
                auxiliary_dense = _dense_items(auxiliary_payload)

            answerable = query["expected_answerability"] == "answerable"
            gold = {str(value) for value in query.get("gold_sources", [])}
            span = _normalize(query["answer_span"])

            def exact(item: Mapping[str, object]) -> bool:
                return answerable and span in _normalize(item["text"])

            def gold_source(item: Mapping[str, object]) -> bool:
                return answerable and str(item["source"]) in gold

            original_top20 = original_dense[:UNION_CUTOFF]
            auxiliary_top20 = auxiliary_dense[:UNION_CUTOFF]
            union_top20 = deduplicated_union(original_dense, auxiliary_dense)
            original_ids = {str(item["chunk_id"]) for item in original_top20}
            original_exact = any(exact(item) for item in original_top20)
            original_gold = any(gold_source(item) for item in original_top20)
            auxiliary_exact = any(exact(item) for item in auxiliary_top20)
            auxiliary_gold = any(gold_source(item) for item in auxiliary_top20)
            rows.append(
                {
                    "query_index": query_index,
                    "expected_answerability": str(query["expected_answerability"]),
                    "anchors": list(anchors),
                    "auxiliary_query": auxiliary_query,
                    "eligible": auxiliary_query is not None,
                    "original_min_gold_rank": _minimum_rank(
                        original_dense, gold_source
                    ),
                    "original_min_exact_rank": _minimum_rank(original_dense, exact),
                    "auxiliary_min_gold_rank": _minimum_rank(
                        auxiliary_dense, gold_source
                    ),
                    "auxiliary_min_exact_rank": _minimum_rank(
                        auxiliary_dense, exact
                    ),
                    "original_top20_chunk_ids": [
                        str(item["chunk_id"]) for item in original_top20
                    ],
                    "auxiliary_top20": [
                        {
                            "chunk_id": str(item["chunk_id"]),
                            "source": str(item["source"]),
                            "rank": rank,
                            "gold_source": gold_source(item),
                            "exact_span": exact(item),
                        }
                        for rank, item in enumerate(auxiliary_top20, start=1)
                    ],
                    "union_top20_chunk_ids": [
                        str(item["chunk_id"]) for item in union_top20
                    ],
                    "union_top20_gold_source": any(
                        gold_source(item) for item in union_top20
                    ),
                    "union_top20_exact_span": any(exact(item) for item in union_top20),
                    "incremental_gold_source": not original_gold and auxiliary_gold,
                    "incremental_exact_span": not original_exact and auxiliary_exact,
                    "unique_auxiliary_chunks_added": sum(
                        str(item["chunk_id"]) not in original_ids
                        for item in auxiliary_top20
                    ),
                }
            )
            print(
                f"anchor auxiliary development {completed}/{len(indexes)}",
                flush=True,
            )
    finally:
        client.close()

    result = {
        "schema_version": 1,
        "protocol": "2026-09-15-anchor-only-auxiliary-view-development",
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
