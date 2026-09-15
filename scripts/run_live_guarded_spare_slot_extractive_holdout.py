"""Run the sealed exact-span holdout for the frozen strict spare-slot policy."""

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

from recall.source_conditioning import (  # noqa: E402
    STRICT_SPARE_SLOT_POLICY,
    chunk_identifier_hash,
    fill_strict_source_conditioned_spare_slot,
    load_source_conditioning_artifact,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


HOLDOUT_SIZE = 500


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def score_query(
    query: dict[str, Any],
    base: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    base_ids = [str(item["chunk_id"]) for item in base]
    candidate_ids = [str(item["chunk_id"]) for item in candidate]
    prefix_preserved = candidate_ids[: len(base_ids)] == base_ids
    additions = candidate[len(base) :] if prefix_preserved else []
    answerability = str(query["expected_answerability"])
    gold_sources = {str(value) for value in query.get("gold_sources", [])}
    answer_span = _normalize(query["answer_span"])

    def contains_span(items: list[dict[str, Any]]) -> bool:
        return any(answer_span in _normalize(item["text"]) for item in items)

    def source_hit(items: list[dict[str, Any]]) -> bool:
        return bool(gold_sources & {str(item["source"]) for item in items})

    base_span = contains_span(base) if answerability == "answerable" else False
    candidate_span = contains_span(candidate) if answerability == "answerable" else False
    base_source = source_hit(base) if answerability == "answerable" else False
    candidate_source = source_hit(candidate) if answerability == "answerable" else False
    added_span_items = (
        sum(answer_span in _normalize(item["text"]) for item in additions)
        if answerability == "answerable"
        else 0
    )
    return {
        "expected_answerability": answerability,
        "base_count": len(base),
        "candidate_count": len(candidate),
        "added_count": len(additions),
        "base_prefix_preserved": prefix_preserved,
        "base_exact_span_covered": base_span,
        "candidate_exact_span_covered": candidate_span,
        "exact_span_gain": candidate_span and not base_span,
        "exact_span_loss": base_span and not candidate_span,
        "base_source_hit": base_source,
        "candidate_source_hit": candidate_source,
        "source_hit_gain": candidate_source and not base_source,
        "source_hit_loss": base_source and not candidate_source,
        "added_exact_span_items": added_span_items,
        "added_non_gold_items": len(additions) - added_span_items,
        "control_activation": answerability == "unanswerable" and bool(additions),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    metrics = {
        "processed_queries": len(rows),
        "answerable_queries": len(answerable),
        "control_queries": len(controls),
        "answerable_base_exact_span_coverage": sum(
            row["base_exact_span_covered"] for row in answerable
        ),
        "answerable_candidate_exact_span_coverage": sum(
            row["candidate_exact_span_covered"] for row in answerable
        ),
        "exact_span_gains": sum(row["exact_span_gain"] for row in answerable),
        "exact_span_losses": sum(row["exact_span_loss"] for row in answerable),
        "answerable_base_source_hits": sum(row["base_source_hit"] for row in answerable),
        "answerable_candidate_source_hits": sum(
            row["candidate_source_hit"] for row in answerable
        ),
        "source_hit_gains": sum(row["source_hit_gain"] for row in answerable),
        "source_hit_losses": sum(row["source_hit_loss"] for row in answerable),
        "added_exact_span_items": sum(row["added_exact_span_items"] for row in rows),
        "added_non_gold_items": sum(row["added_non_gold_items"] for row in rows),
        "total_added_items": sum(row["added_count"] for row in rows),
        "control_activations": sum(row["control_activation"] for row in controls),
        "control_added_items": sum(row["added_count"] for row in controls),
        "base_prefix_failures": sum(not row["base_prefix_preserved"] for row in rows),
        "max_added_items_per_query": max((row["added_count"] for row in rows), default=0),
    }
    total_added = metrics["total_added_items"]
    metrics["added_item_exact_span_precision"] = (
        metrics["added_exact_span_items"] / total_added if total_added else None
    )
    if metrics["base_prefix_failures"]:
        decision = "FAIL_BASE_PREFIX"
    elif metrics["max_added_items_per_query"] > 1:
        decision = "FAIL_ADDITION_BUDGET"
    elif metrics["control_activations"]:
        decision = "FAIL_CONTROL_ACTIVATION"
    elif metrics["exact_span_losses"]:
        decision = "FAIL_EXACT_SPAN_LOSS"
    elif metrics["source_hit_losses"]:
        decision = "FAIL_SOURCE_HIT_LOSS"
    elif metrics["exact_span_gains"] < 1:
        decision = "FAIL_NO_EXACT_SPAN_GAIN"
    else:
        decision = "PASS_RETRIEVAL_GATE"
    return {"decision": decision, "promotion_eligible": decision == "PASS_RETRIEVAL_GATE", "metrics": metrics}


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


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _output(
    args: argparse.Namespace,
    *,
    pool_sha256: str,
    artifact_sha256: str,
    inventory_sha256: str,
    rows: list[dict[str, Any]],
    elapsed_ms: float,
    status: str,
) -> dict[str, Any]:
    result = aggregate(rows)
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-extractive-holdout",
        "measured_at": datetime.now(UTC).isoformat(),
        "status": status,
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "policy_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "policy": STRICT_SPARE_SLOT_POLICY,
        "query_pool_sha256": pool_sha256,
        "artifact_sha256": artifact_sha256,
        "inventory_receipt_sha256": inventory_sha256,
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "elapsed_ms": round(elapsed_ms, 3),
        **result,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--inventory-receipt", type=Path, required=True)
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

    pool_sha256 = _sha256(args.query_pool)
    artifact_sha256 = _sha256(args.artifact)
    inventory_sha256 = _sha256(args.inventory_receipt)
    pool = json.loads(args.query_pool.read_text(encoding="utf-8"))
    queries = list(pool["queries"])
    if len(queries) != HOLDOUT_SIZE:
        raise ValueError(f"extractive holdout must contain exactly {HOLDOUT_SIZE} queries")
    inventory = json.loads(args.inventory_receipt.read_text(encoding="utf-8"))
    if (
        inventory.get("decision") != "HOLDOUT_LINEAGE_VALID"
        or inventory.get("query_pool_sha256") != pool_sha256
        or inventory.get("generation_id") != args.generation_id
        or inventory.get("matched_sources") != 250
        or inventory.get("missing_sources") != 0
        or inventory.get("digest_mismatches") != 0
        or inventory.get("inventory_truncated") is not False
    ):
        raise RuntimeError("HOLDOUT_LINEAGE_INVALID")

    model = load_source_conditioning_artifact(args.artifact)
    model.assert_compatible(
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
        if (
            prior.get("query_pool_sha256") != pool_sha256
            or prior.get("artifact_sha256") != artifact_sha256
            or prior.get("inventory_receipt_sha256") != inventory_sha256
            or prior.get("generation_id") != args.generation_id
        ):
            raise RuntimeError("holdout resume lineage differs")
        rows = list(prior["rows"])
        elapsed_before_ms = float(prior.get("elapsed_ms", 0.0))

    client = TTYMCP(command, args.timeout)
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for index, query in enumerate(queries[len(rows) :], start=len(rows) + 1):
            started = time.perf_counter()
            payload = _call_query(client, request_id, str(query["query"]))
            observed_ms = (time.perf_counter() - started) * 1000.0
            _identity(payload, args)
            shadow = _audit(payload, "source_conditioning_shadow")
            if shadow.get("policy") != "guarded_spare_slot":
                raise RuntimeError("guarded shadow receipt is missing")
            pool_audit = _audit(payload, "source_admission_benchmark_audit")
            leg_audit = _audit(payload, "retrieval_leg_benchmark_audit")
            by_hash = {
                chunk_identifier_hash(item["chunk_id"]): item for item in pool_audit["items"]
            }
            base_hashes = list(shadow.get("alpha008_chunk_hashes", []))
            missing = [value for value in base_hashes if value not in by_hash]
            if missing:
                raise RuntimeError("benchmark pool cannot resolve base hashes")
            base = [by_hash[value] for value in base_hashes]
            candidate, receipts = fill_strict_source_conditioned_spare_slot(
                model,
                base,
                list(pool_audit["items"]),
                list(leg_audit["dense"]),
                list(leg_audit["sparse"]),
            )
            row = score_query(query, base, candidate)
            if len(receipts) != row["added_count"]:
                raise RuntimeError("strict receipt count differs from candidate additions")
            row["query_id"] = str(query["id"])
            row["client_observed_ms"] = round(observed_ms, 3)
            rows.append(row)
            elapsed_ms = elapsed_before_ms + (time.perf_counter() - started_run) * 1000.0
            _write(
                args.output,
                _output(
                    args,
                    pool_sha256=pool_sha256,
                    artifact_sha256=artifact_sha256,
                    inventory_sha256=inventory_sha256,
                    rows=rows,
                    elapsed_ms=elapsed_ms,
                    status="RUNNING",
                ),
            )
            current = aggregate(rows)["metrics"]
            print(
                f"holdout {index}/{HOLDOUT_SIZE} additions={current['total_added_items']} "
                f"gains={current['exact_span_gains']} controls={current['control_activations']}",
                flush=True,
            )
            request_id += 1
    finally:
        client.close()

    result = _output(
        args,
        pool_sha256=pool_sha256,
        artifact_sha256=artifact_sha256,
        inventory_sha256=inventory_sha256,
        rows=rows,
        elapsed_ms=elapsed_before_ms + (time.perf_counter() - started_run) * 1000.0,
        status="COMPLETE",
    )
    _write(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
