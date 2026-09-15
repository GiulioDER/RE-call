"""Run the sealed query-anchor spare-slot holdout on VPS2."""

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
    QUERY_ANCHOR_POLICY,
    query_anchor_candidate_eligible,
    query_anchor_features,
)
from recall.source_conditioning import (  # noqa: E402
    chunk_identifier_hash,
    load_source_conditioning_artifact,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_guarded_spare_slot_extractive_holdout import (  # noqa: E402
    aggregate as base_aggregate,
    score_query,
)
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


HOLDOUT_SIZE = 160
FROZEN_SOURCE_COUNT = 80
PRECISION_FLOOR = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = base_aggregate(rows)
    precision = result["metrics"]["added_item_exact_span_precision"]
    if result["decision"] == "PASS_RETRIEVAL_GATE" and (
        precision is None or precision < PRECISION_FLOOR
    ):
        result["decision"] = "FAIL_EXACT_SPAN_PRECISION"
        result["promotion_eligible"] = False
    return result


def select_candidate(
    query_text: str,
    base: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the frozen policy to the first original guarded proposal."""

    if selected[: len(base)] != base or len(selected) == len(base):
        return list(base)
    proposal = selected[len(base)]
    features = query_anchor_features(query_text, proposal, pool)
    if query_anchor_candidate_eligible(len(base), features):
        return [*base, proposal]
    return list(base)


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
    return {
        "schema_version": 1,
        "protocol": "2026-09-14-query-anchor-spare-slot-holdout",
        "measured_at": datetime.now(UTC).isoformat(),
        "status": status,
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "policy_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "policy": QUERY_ANCHOR_POLICY,
        "query_pool_sha256": pool_sha256,
        "artifact_sha256": artifact_sha256,
        "inventory_receipt_sha256": inventory_sha256,
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "elapsed_ms": round(elapsed_ms, 3),
        **aggregate(rows),
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
    queries = list(json.loads(args.query_pool.read_text(encoding="utf-8"))["queries"])
    if len(queries) != HOLDOUT_SIZE:
        raise ValueError(f"query-anchor holdout must contain exactly {HOLDOUT_SIZE} queries")
    inventory = json.loads(args.inventory_receipt.read_text(encoding="utf-8"))
    if (
        inventory.get("decision") != "HOLDOUT_LINEAGE_VALID"
        or inventory.get("query_pool_sha256") != pool_sha256
        or inventory.get("generation_id") != args.generation_id
        or inventory.get("matched_sources") != FROZEN_SOURCE_COUNT
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
        expected = {
            "query_pool_sha256": pool_sha256,
            "artifact_sha256": artifact_sha256,
            "inventory_receipt_sha256": inventory_sha256,
            "generation_id": args.generation_id,
            "calibration_id": args.calibration_id,
            "pipeline_fingerprint": args.pipeline_fingerprint,
            "corpus_fingerprint": args.corpus_fingerprint,
        }
        if any(prior.get(key) != value for key, value in expected.items()):
            raise RuntimeError("holdout resume lineage differs")
        rows = list(prior["rows"])
        elapsed_before_ms = float(prior.get("elapsed_ms", 0.0))

    client = TTYMCP(command, args.timeout)
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries[len(rows) :], start=len(rows)):
            started = time.perf_counter()
            payload = _call_query(client, request_id, str(query["query"]))
            observed_ms = (time.perf_counter() - started) * 1000.0
            _identity(payload, args)
            shadow = _audit(payload, "source_conditioning_shadow")
            if shadow.get("policy") != "guarded_spare_slot":
                raise RuntimeError("guarded shadow receipt is missing")
            pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
            by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in pool}
            base_hashes = list(shadow.get("alpha008_chunk_hashes", []))
            selected_hashes = list(shadow.get("selected_chunk_hashes", []))
            if selected_hashes[: len(base_hashes)] != base_hashes:
                raise RuntimeError("guarded proposal changed the base prefix")
            missing = [
                value for value in {*base_hashes, *selected_hashes} if value not in by_hash
            ]
            if missing:
                raise RuntimeError("benchmark pool cannot resolve guarded hashes")
            base = [by_hash[value] for value in base_hashes]
            selected = [by_hash[value] for value in selected_hashes]
            candidate = select_candidate(str(query["query"]), base, selected, pool)
            row = score_query(query, base, candidate)
            row["query_index"] = query_index
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
            metrics = aggregate(rows)["metrics"]
            print(
                f"holdout {query_index + 1}/{HOLDOUT_SIZE} "
                f"additions={metrics['total_added_items']} "
                f"gains={metrics['exact_span_gains']} "
                f"controls={metrics['control_activations']}",
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
