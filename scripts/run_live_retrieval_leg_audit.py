"""Audit dense, lexical, union, and fused source-gold reachability on VPS2."""

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

from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


DEPTHS = (5, 10, 20, 50, 100)
RRF_K = 60


def _audit(payload: dict[str, Any]) -> dict[str, Any]:
    value = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("values", {})
        .get("retrieval_leg_benchmark_audit")
    )
    if not isinstance(value, dict):
        raise ValueError("retrieval leg benchmark audit is missing")
    if value.get("depth") != max(DEPTHS):
        raise ValueError(f"retrieval leg audit depth must be {max(DEPTHS)}")
    return value


def _fused(legs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for leg in legs:
        for rank, item in enumerate(leg, start=1):
            chunk_id = str(item["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(chunk_id, item)
    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [by_id[chunk_id] for chunk_id in ranked_ids]


def _source_hit(items: list[dict[str, Any]], gold: set[str], depth: int) -> bool:
    return any(str(item.get("source")) in gold for item in items[:depth])


def _rate(count: int, denominator: int) -> float | None:
    return round(count / denominator, 4) if denominator else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if bool(row["query"].get("answerable"))]
    unanswerable = [row for row in rows if not bool(row["query"].get("answerable"))]
    counts = {
        arm: {str(depth): 0 for depth in DEPTHS}
        for arm in ("dense", "sparse", "union", "fused_k20", "fused_k100")
    }
    classifications: dict[str, list[int]] = {
        "reachable_fused_top10": [],
        "selection_loss_fused_11_to_20": [],
        "fusion_loss_from_leg_top20": [],
        "deep_candidate_only_21_to_100": [],
        "candidate_generation_miss_at_100": [],
    }
    served_source_hits: list[int] = []
    dense_only_top20: list[int] = []
    sparse_only_top20: list[int] = []
    both_top20: list[int] = []

    for row in answerable:
        query_index = int(row["query_index"])
        gold = {str(value) for value in row["query"].get("relevant_files", [])}
        if not gold:
            raise ValueError(f"answerable query {query_index} has no relevant_files")
        audit = row["audit"]
        dense = list(audit["dense"])
        sparse = list(audit["sparse"])
        fused_k20 = _fused([dense[:20], sparse[:20]])
        fused_k100 = _fused([dense, sparse])
        for depth in DEPTHS:
            dense_hit = _source_hit(dense, gold, depth)
            sparse_hit = _source_hit(sparse, gold, depth)
            counts["dense"][str(depth)] += int(dense_hit)
            counts["sparse"][str(depth)] += int(sparse_hit)
            counts["union"][str(depth)] += int(dense_hit or sparse_hit)
            counts["fused_k20"][str(depth)] += int(_source_hit(fused_k20, gold, depth))
            counts["fused_k100"][str(depth)] += int(_source_hit(fused_k100, gold, depth))

        dense20 = _source_hit(dense, gold, 20)
        sparse20 = _source_hit(sparse, gold, 20)
        if dense20 and sparse20:
            both_top20.append(query_index)
        elif dense20:
            dense_only_top20.append(query_index)
        elif sparse20:
            sparse_only_top20.append(query_index)

        served = row.get("trusted_evidence", [])
        if _source_hit(served, gold, len(served)):
            served_source_hits.append(query_index)

        if _source_hit(fused_k20, gold, 10):
            classifications["reachable_fused_top10"].append(query_index)
        elif _source_hit(fused_k20, gold, 20):
            classifications["selection_loss_fused_11_to_20"].append(query_index)
        elif _source_hit(dense, gold, 20) or _source_hit(sparse, gold, 20):
            classifications["fusion_loss_from_leg_top20"].append(query_index)
        elif _source_hit(dense, gold, 100) or _source_hit(sparse, gold, 100):
            classifications["deep_candidate_only_21_to_100"].append(query_index)
        else:
            classifications["candidate_generation_miss_at_100"].append(query_index)

    denominator = len(answerable)
    return {
        "queries": len(rows),
        "answerable_queries": denominator,
        "unanswerable_queries": len(unanswerable),
        "source_gold_hit_rates": {
            arm: {
                depth: {
                    "hits": value,
                    "rate": _rate(value, denominator),
                }
                for depth, value in depth_counts.items()
            }
            for arm, depth_counts in counts.items()
        },
        "served_source_hit_queries": served_source_hits,
        "served_source_hit_rate": _rate(len(served_source_hits), denominator),
        "served_unanswerable_abstentions": sum(
            not row.get("trusted_evidence") for row in unanswerable
        ),
        "served_unanswerable_answers": sum(
            bool(row.get("trusted_evidence")) for row in unanswerable
        ),
        "top20_leg_coverage": {
            "dense_only": dense_only_top20,
            "sparse_only": sparse_only_top20,
            "both": both_top20,
        },
        "classification": classifications,
    }


def _call_query(
    client: TTYMCP,
    request_id: int,
    query: str,
    *,
    max_steps: int,
    max_evidence_tokens: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.call(
        request_id,
        "tools/call",
        {
            "name": "recall_reasoning_query",
            "arguments": {
                "query": query,
                "k": 5,
                "mode": "evidence_assembly",
                "max_steps": max_steps,
                "max_graph_nodes": 1,
                "max_evidence_tokens": max_evidence_tokens,
                "graph_expansion": "off",
            },
        },
    )
    payload = json.loads(_extract_payload(response))
    return payload, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-set",
        default="docs/preregistrations/2026-09-13-memory-queries-source-gold.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-evidence-tokens", type=int, default=2048)
    args = parser.parse_args()

    query_path = Path(args.query_set)
    query_bytes = query_path.read_bytes()
    queries = json.loads(query_bytes.decode("utf-8"))
    if not isinstance(queries, list) or not all(isinstance(item, dict) for item in queries):
        raise ValueError("query set must be a JSON list of objects")

    client = TTYMCP(
        _command(
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
        ),
        args.timeout,
    )
    rows: list[dict[str, Any]] = []
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"recorded {query_index + 1}/{len(queries)}", flush=True)
            payload, client_ms = _call_query(
                client,
                request_id,
                str(query["query"]),
                max_steps=args.max_steps,
                max_evidence_tokens=args.max_evidence_tokens,
            )
            if payload.get("generation_id") != args.generation_id:
                raise RuntimeError(
                    f"pinned generation mismatch: expected {args.generation_id}, "
                    f"got {payload.get('generation_id')}"
                )
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "client_observed_ms": round(client_ms, 3),
                    "outcome": payload.get("outcome"),
                    "refusal_reason": payload.get("refusal_reason"),
                    "trust_state": payload.get("trust_state"),
                    "generation_id": payload.get("generation_id"),
                    "pipeline_fingerprint": payload.get("pipeline_fingerprint"),
                    "corpus_fingerprint": payload.get("corpus_fingerprint"),
                    "calibration_id": payload.get("calibration_id"),
                    "trusted_evidence": payload.get("trusted_evidence", {}).get("items", []),
                    "audit": _audit(payload),
                }
            )
            request_id += 1
    finally:
        client.close()

    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-retrieval-leg-source-gold-audit",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": len(queries),
        "generation_id": args.generation_id,
        "candidate_depth": max(DEPTHS),
        "summary": _summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": artifact["summary"], "output": str(output)}))


if __name__ == "__main__":
    main()
