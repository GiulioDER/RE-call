"""Measure one deterministic anchor boosted query on consumed empty base rows."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.query_anchor_admission import (  # noqa: E402
    QUERY_ANCHOR_LIMIT,
    query_anchor_tokens,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


CUTOFFS = (1, 3, 5, 10, 20)
EXPECTED_GOLD_SOURCE = {"1": 6, "3": 6, "5": 7, "10": 8, "20": 10}
EXPECTED_EXACT_SPAN = {"1": 3, "3": 3, "5": 5, "10": 8, "20": 10}
_TOKEN = re.compile(r"[a-z0-9]+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(str(value).split())


def selected_supported_anchors(
    query: str,
    pool: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the three frozen rare anchors in their original query order."""

    if not pool:
        return ()
    source_tokens: dict[str, set[str]] = {}
    for item in pool:
        source_tokens.setdefault(str(item["source"]), set()).update(
            query_anchor_tokens(item["text"])
        )
    query_tokens = query_anchor_tokens(query)
    document_frequency = {
        token: sum(token in tokens for tokens in source_tokens.values())
        for token in query_tokens
    }
    ranked = sorted(
        query_tokens,
        key=lambda token: (
            document_frequency[token],
            hashlib.sha256(token.encode()).hexdigest(),
        ),
    )[:QUERY_ANCHOR_LIMIT]
    if len(ranked) != QUERY_ANCHOR_LIMIT or any(
        document_frequency[token] == 0 for token in ranked
    ):
        return ()
    selected = set(ranked)
    ordered: list[str] = []
    for token in _TOKEN.findall(query.casefold()):
        if token in selected and token not in ordered:
            ordered.append(token)
    return tuple(ordered)


def anchor_boosted_query(
    query: str,
    pool: Sequence[Mapping[str, object]],
) -> tuple[str | None, tuple[str, ...]]:
    """Append one duplicate of each safe anchor without adding new vocabulary."""

    anchors = selected_supported_anchors(query, pool)
    if len(anchors) != QUERY_ANCHOR_LIMIT:
        return None, anchors
    return f"{_normalize(query)} {' '.join(anchors)}", anchors


def changed_rank_one_candidate(
    original_dense: Sequence[Mapping[str, object]],
    transformed_dense: Sequence[Mapping[str, object]],
    *,
    eligible: bool,
) -> Mapping[str, object] | None:
    """Select transformed rank one only when the safe query changed rank one."""

    if not eligible or not original_dense or not transformed_dense:
        return None
    if str(original_dense[0]["chunk_id"]) == str(transformed_dense[0]["chunk_id"]):
        return None
    return transformed_dense[0]


def _cutoff_counts(
    rows: Sequence[Mapping[str, Any]],
    rank_key: str,
) -> dict[str, int]:
    return {
        str(cutoff): sum(
            row[rank_key] is not None and int(row[rank_key]) <= cutoff for row in rows
        )
        for cutoff in CUTOFFS
    }


def validate_baseline(summary: Mapping[str, Any]) -> None:
    """Refuse scoring if the original retrieval snapshot no longer reproduces."""

    observed = (
        summary["original_dense_gold_source_by_cutoff"],
        summary["original_dense_exact_span_by_cutoff"],
        summary["eligible_answerable"],
        summary["eligible_controls"],
    )
    expected = (EXPECTED_GOLD_SOURCE, EXPECTED_EXACT_SPAN, 14, 0)
    if observed != expected:
        raise RuntimeError(
            f"original dense apparatus mismatch: expected {expected}, got {observed}"
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    selected = [row for row in rows if row["selected"]]
    selected_exact = sum(row["selected_exact_span"] for row in selected)
    precision = selected_exact / len(selected) if selected else None

    rank_movements = {"improved": 0, "tied": 0, "worsened": 0}
    for row in answerable:
        original = row["original_min_exact_rank"]
        transformed = row["transformed_min_exact_rank"]
        original_order = int(original) if original is not None else 10**9
        transformed_order = int(transformed) if transformed is not None else 10**9
        if transformed_order < original_order:
            rank_movements["improved"] += 1
        elif transformed_order > original_order:
            rank_movements["worsened"] += 1
        else:
            rank_movements["tied"] += 1

    summary: dict[str, Any] = {
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(rows) - len(answerable),
        "eligible_answerable": sum(row["eligible"] for row in answerable),
        "eligible_controls": sum(
            row["eligible"]
            for row in rows
            if row["expected_answerability"] == "unanswerable"
        ),
        "original_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "original_min_gold_rank"
        ),
        "original_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "original_min_exact_rank"
        ),
        "transformed_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "transformed_min_gold_rank"
        ),
        "transformed_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "transformed_min_exact_rank"
        ),
        "exact_rank_movements": rank_movements,
        "changed_rank_one_total": len(selected),
        "changed_rank_one_answerable": sum(
            row["expected_answerability"] == "answerable" for row in selected
        ),
        "changed_rank_one_controls": sum(
            row["expected_answerability"] == "unanswerable" for row in selected
        ),
        "changed_rank_one_exact_spans": selected_exact,
        "changed_rank_one_gold_sources": sum(
            row["selected_gold_source"] for row in selected
        ),
        "changed_rank_one_exact_precision": precision,
    }
    validate_baseline(summary)
    proceed = (
        selected_exact >= 2
        and summary["changed_rank_one_controls"] == 0
        and precision is not None
        and precision >= 0.5
        and summary["transformed_dense_exact_span_by_cutoff"]["5"]
        >= summary["original_dense_exact_span_by_cutoff"]["5"]
    )
    summary["decision"] = (
        "PROCEED_FRESH_HOLDOUT" if proceed else "STOP_SINGLE_VIEW_ANCHOR_BOOST"
    )
    return summary


def _identity(payload: Mapping[str, Any], args: argparse.Namespace) -> None:
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


def _dense_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
    by_id = {str(item["chunk_id"]): item for item in pool}
    dense_audit = sorted(
        _audit(payload, "retrieval_leg_benchmark_audit")["dense"],
        key=lambda item: int(item["rank"]),
    )
    return [
        by_id[str(item["chunk_id"])]
        for item in dense_audit
        if str(item["chunk_id"]) in by_id
    ]


def _minimum_rank(
    items: Sequence[Mapping[str, object]],
    predicate: Any,
) -> int | None:
    return next(
        (rank for rank, item in enumerate(items, start=1) if predicate(item)),
        None,
    )


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
            boosted_query, anchors = anchor_boosted_query(query_text, original_pool)
            transformed_dense: list[dict[str, Any]] = []
            if boosted_query is not None:
                transformed_payload = _call_query(client, request_id, boosted_query)
                request_id += 1
                _identity(transformed_payload, args)
                transformed_dense = _dense_items(transformed_payload)

            answerable = query["expected_answerability"] == "answerable"
            gold = {str(value) for value in query.get("gold_sources", [])}
            span = _normalize(query["answer_span"])

            def exact(item: Mapping[str, object]) -> bool:
                return answerable and span in _normalize(item["text"])

            def gold_source(item: Mapping[str, object]) -> bool:
                return answerable and str(item["source"]) in gold

            eligible = boosted_query is not None
            selected = changed_rank_one_candidate(
                original_dense,
                transformed_dense,
                eligible=eligible,
            )
            rows.append(
                {
                    "query_index": query_index,
                    "expected_answerability": str(query["expected_answerability"]),
                    "anchors": list(anchors),
                    "eligible": eligible,
                    "original_min_gold_rank": _minimum_rank(
                        original_dense, gold_source
                    ),
                    "original_min_exact_rank": _minimum_rank(original_dense, exact),
                    "transformed_min_gold_rank": _minimum_rank(
                        transformed_dense, gold_source
                    ),
                    "transformed_min_exact_rank": _minimum_rank(
                        transformed_dense, exact
                    ),
                    "selected": selected is not None,
                    "selected_gold_source": selected is not None
                    and gold_source(selected),
                    "selected_exact_span": selected is not None and exact(selected),
                }
            )
            print(f"anchor boost development {completed}/{len(indexes)}", flush=True)
    finally:
        client.close()

    result = {
        "schema_version": 1,
        "protocol": "2026-09-15-anchor-boosted-query-development",
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
