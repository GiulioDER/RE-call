"""Collect the preregistered fresh guarded spare-slot cohort and a blind review packet."""

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
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import (  # noqa: E402
    _call_query,
    _performance,
    _shadow_internal_ms,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


ANSWERABLE_QUOTA = 20
UNANSWERABLE_QUOTA = 20
SCREEN_CAP = 500


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _hashes(items: list[dict[str, Any]]) -> list[str]:
    return [chunk_identifier_hash(item["chunk_id"]) for item in items]


def _review_items(
    query_id: str, base: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_hash = {chunk_identifier_hash(item["chunk_id"]): item for item in [*base, *candidate]}
    ordered = sorted(
        by_hash,
        key=lambda value: hashlib.sha256(f"{seed}\0{query_id}\0{value}".encode()).hexdigest(),
    )
    evidence_ids = {value: f"E{index:02d}" for index, value in enumerate(ordered, start=1)}
    return (
        [
            {
                "evidence_id": evidence_ids[value],
                "source": by_hash[value]["source"],
                "ordinal": by_hash[value]["ordinal"],
                "text": by_hash[value]["text"],
                "supports_question": None,
                "covered_facts": [],
            }
            for value in ordered
        ],
        evidence_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-output", required=True)
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

    pool_path = Path(args.query_pool)
    artifact_path = Path(args.artifact)
    pool_bytes = pool_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    pool = json.loads(pool_bytes.decode("utf-8"))
    queries = list(pool["queries"])
    if len(queries) != SCREEN_CAP:
        raise ValueError(f"fresh pool must contain exactly {SCREEN_CAP} queries")
    model = load_source_conditioning_artifact(artifact_path)
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
        source_conditioning_artifact=args.artifact.replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
        source_conditioning_policy="guarded_spare_slot",
    )
    client = TTYMCP(command, args.timeout)
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    triggered = {"answerable": 0, "unanswerable": 0}
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for index, query in enumerate(queries, start=1):
            started = time.perf_counter()
            payload = _call_query(client, request_id, str(query["query"]))
            observed_ms = (time.perf_counter() - started) * 1000.0
            _identity(payload, args)
            shadow = _audit(payload, "source_conditioning_shadow")
            legs = _audit(payload, "retrieval_leg_benchmark_audit")
            pool_audit = _audit(payload, "source_admission_benchmark_audit")
            base = select_source_conditioned(
                model,
                pool_audit["items"],
                legs["dense"],
                legs["sparse"],
                threshold=float(pool_audit["threshold"]),
            )
            candidate, receipts = fill_source_conditioned_spare_slots(
                model, base, pool_audit["items"], legs["dense"], legs["sparse"]
            )
            base_hashes = _hashes(base)
            candidate_hashes = _hashes(candidate)
            if shadow.get("policy") != "guarded_spare_slot":
                raise RuntimeError("guarded shadow policy receipt is missing")
            if shadow.get("alpha008_chunk_hashes") != base_hashes:
                raise RuntimeError("alpha008 shadow hash parity failed")
            if shadow.get("selected_chunk_hashes") != candidate_hashes:
                raise RuntimeError("candidate shadow hash parity failed")
            if candidate_hashes[: len(base_hashes)] != base_hashes:
                raise RuntimeError("candidate changed the alpha008 prefix")
            span_ms = _shadow_internal_ms(_performance(payload))
            if span_ms is None:
                raise RuntimeError("shadow timing receipt is missing")
            answerability = str(query["expected_answerability"])
            is_triggered = candidate_hashes != base_hashes
            items: list[dict[str, Any]] = []
            evidence_ids: dict[str, str] = {}
            if is_triggered:
                items, evidence_ids = _review_items(
                    str(query["id"]), base, candidate, str(pool["seed"])
                )
            rows.append(
                {
                    "query_id": query["id"],
                    "expected_answerability": answerability,
                    "triggered": is_triggered,
                    "base_count": len(base_hashes),
                    "candidate_count": len(candidate_hashes),
                    "added_count": len(receipts),
                    "lane_counts": shadow.get("lane_counts"),
                    "base_hashes": base_hashes if is_triggered else None,
                    "candidate_hashes": candidate_hashes if is_triggered else None,
                    "review_evidence_ids_by_hash": evidence_ids if is_triggered else None,
                    "client_observed_ms": round(observed_ms, 3),
                    "shadow_internal_ms": span_ms,
                }
            )
            if is_triggered:
                triggered[answerability] += 1
                review_rows.append(
                    {
                        "query_id": query["id"],
                        "query": query["query"],
                        "expected_answerability": answerability,
                        "required_facts": [],
                        "evidence_items": items,
                        "review_complete": False,
                    }
                )
            print(
                f"screen {index}/{SCREEN_CAP} triggered="
                f"{triggered['answerable']}/{triggered['unanswerable']}",
                flush=True,
            )
            request_id += 1
            if (
                triggered["answerable"] >= ANSWERABLE_QUOTA
                and triggered["unanswerable"] >= UNANSWERABLE_QUOTA
            ):
                break
    finally:
        client.close()

    quotas_met = (
        triggered["answerable"] >= ANSWERABLE_QUOTA
        and triggered["unanswerable"] >= UNANSWERABLE_QUOTA
    )
    capture = {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-fresh-screen",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "processed_queries": len(rows),
        "triggered": triggered,
        "quotas_met": quotas_met,
        "screen_decision": "READY_FOR_BLIND_REVIEW" if quotas_met else "INSUFFICIENT",
        "elapsed_ms": round((time.perf_counter() - started_run) * 1000.0, 3),
        "rows": rows,
    }
    review = {
        "schema_version": 1,
        "protocol": "2026-09-14-guarded-spare-slot-blind-review",
        "query_pool_sha256": capture["query_pool_sha256"],
        "capture_sha256": None,
        "instructions": (
            "For every evidence item, set supports_question and list the required fact names it "
            "covers. Add the required fact names once per query and set review_complete only "
            "after judging every item. Do not inspect the capture artifact while reviewing."
        ),
        "queries": review_rows,
    }
    output = Path(args.output)
    review_output = Path(args.review_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(capture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    review["capture_sha256"] = _sha256(output)
    review_output.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "review_output": str(review_output),
                "screen_decision": capture["screen_decision"],
                "processed_queries": len(rows),
                "triggered": triggered,
                "capture_sha256": _sha256(output),
                "review_sha256": _sha256(review_output),
            }
        )
    )


if __name__ == "__main__":
    main()
