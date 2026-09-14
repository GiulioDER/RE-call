"""Capture private query-anchor features on the consumed extractive cohort."""

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.query_anchor_admission import query_anchor_features  # noqa: E402
from recall.source_conditioning import (  # noqa: E402
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def analyze_query(
    query: dict[str, Any],
    shadow: dict[str, Any],
    pool_audit: dict[str, Any],
    leg_audit: dict[str, Any],
    artifact: Any,
) -> dict[str, Any]:
    pool = list(pool_audit["items"])
    dense = list(leg_audit["dense"])
    sparse = list(leg_audit["sparse"])
    by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in pool}
    base_hashes = list(shadow.get("alpha008_chunk_hashes", []))
    expected_hashes = list(shadow.get("selected_chunk_hashes", []))
    if expected_hashes[: len(base_hashes)] != base_hashes:
        raise ValueError("guarded proposal changed the base prefix")
    missing = [value for value in {*base_hashes, *expected_hashes} if value not in by_hash]
    if missing:
        raise ValueError("benchmark pool cannot resolve guarded hashes")
    base = [by_hash[value] for value in base_hashes]
    guarded, receipts = fill_source_conditioned_spare_slots(
        artifact, base, pool, dense, sparse
    )
    observed_hashes = [chunk_identifier_hash(item["chunk_id"]) for item in guarded]
    if observed_hashes != expected_hashes:
        raise ValueError("local guarded proposal differs from live shadow")

    answerability = str(query["expected_answerability"])
    answer_span = _normalize(query["answer_span"])
    gold_sources = {str(value) for value in query.get("gold_sources", [])}
    base_span = (
        any(answer_span in _normalize(item["text"]) for item in base)
        if answerability == "answerable"
        else False
    )
    base_source = bool(gold_sources & {str(item["source"]) for item in base})
    result: dict[str, Any] = {
        "parity_mismatch": False,
        "expected_answerability": answerability,
        "base_count": len(base),
        "base_exact_span_covered": base_span,
        "base_source_hit": base_source,
        "has_proposal": bool(receipts),
        "proposal": None,
    }
    if not receipts:
        return result

    proposal = guarded[len(base)]
    source_items = [item for item in pool if item["source"] == proposal["source"]]
    source_text = "\n".join(str(item["text"]) for item in source_items)
    result["proposal"] = {
        "position": 1,
        "lane": receipts[0]["lane"],
        "is_gold_source": str(proposal["source"]) in gold_sources,
        "chunk_contains_exact_span": (
            answerability == "answerable"
            and answer_span in _normalize(proposal["text"])
        ),
        "source_pool_contains_exact_span": (
            answerability == "answerable" and answer_span in _normalize(source_text)
        ),
        "anchor_features": query_anchor_features(str(query["query"]), proposal, pool),
    }
    return result


def try_analyze_query(
    query: dict[str, Any],
    shadow: dict[str, Any],
    pool_audit: dict[str, Any],
    leg_audit: dict[str, Any],
    artifact: Any,
) -> dict[str, Any]:
    """Return an aggregate-safe row for the one registered parity mismatch."""

    try:
        return analyze_query(query, shadow, pool_audit, leg_audit, artifact)
    except ValueError as exc:
        if str(exc) != "local guarded proposal differs from live shadow":
            raise
        return {
            "parity_mismatch": True,
            "expected_answerability": str(query["expected_answerability"]),
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


def _output_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-development-features",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "artifact_sha256": _sha256(args.artifact),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "expected_rows": expected_rows,
        "attempted_rows": len(rows),
        "parity_mismatches": sum(bool(row["parity_mismatch"]) for row in rows),
        "elapsed_ms": round(elapsed_ms, 3),
        "rows": rows,
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--pipeline-fingerprint", required=True)
    parser.add_argument("--corpus-fingerprint", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard boundary")

    pool_payload = json.loads(args.query_pool.read_text(encoding="utf-8"))
    all_queries = list(pool_payload["queries"])
    indexed_queries = [
        (index, query)
        for index, query in enumerate(all_queries)
        if index % args.shard_count == args.shard_index
    ]
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
    elapsed_before_ms = 0.0
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        expected_identity = {
            "query_pool_sha256": _sha256(args.query_pool),
            "artifact_sha256": _sha256(args.artifact),
            "generation_id": args.generation_id,
            "calibration_id": args.calibration_id,
            "pipeline_fingerprint": args.pipeline_fingerprint,
            "corpus_fingerprint": args.corpus_fingerprint,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        }
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError("development shard resume lineage differs")
        rows = list(prior["rows"])
        expected_indexes = [index for index, _ in indexed_queries[: len(rows)]]
        if [int(row["query_index"]) for row in rows] != expected_indexes:
            raise RuntimeError("development shard checkpoint indexes differ")
        elapsed_before_ms = float(prior.get("elapsed_ms", 0.0))
    client = TTYMCP(command, args.timeout)
    started = time.perf_counter()
    try:
        request_id = _initialize(client)
        for completed, (query_index, query) in enumerate(
            indexed_queries[len(rows) :], start=len(rows) + 1
        ):
            payload = _call_query(client, request_id, str(query["query"]))
            _identity(payload, args)
            row = try_analyze_query(
                query,
                _audit(payload, "source_conditioning_shadow"),
                _audit(payload, "source_admission_benchmark_audit"),
                _audit(payload, "retrieval_leg_benchmark_audit"),
                artifact,
            )
            row["query_index"] = query_index
            rows.append(row)
            _write_output(
                args.output,
                _output_payload(
                    args,
                    rows,
                    expected_rows=len(indexed_queries),
                    elapsed_ms=elapsed_before_ms
                    + (time.perf_counter() - started) * 1000.0,
                ),
            )
            print(
                f"anchor shard {args.shard_index + 1}/{args.shard_count} "
                f"{completed}/{len(indexed_queries)}",
                flush=True,
            )
            request_id += 1
    finally:
        client.close()

    output = _output_payload(
        args,
        rows,
        expected_rows=len(indexed_queries),
        elapsed_ms=elapsed_before_ms + (time.perf_counter() - started) * 1000.0,
    )
    _write_output(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}))


if __name__ == "__main__":
    main()
