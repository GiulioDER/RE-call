"""Validate the fitted source conditioning model through a production-shaped shadow."""

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
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import (  # noqa: E402
    ITEM_BUDGET,
    _audit,
    _score_selection,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


EXPECTED_QUERY_SHA256 = "06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f"
EXPECTED_FACT_SHA256 = "45c38731e138f6aed425635b78ee79b692e142f5ef040e0efffeb042ad47186b"
EXPECTED_GENERATION_ID = "gen_808e6c2592494eebabf144eabb21f838"
EXPECTED_CALIBRATION_ID = "cal_bc65614ff5ba4471850d53384381d7d5"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "079a63744f2a9ce44aae344a06f57c0f3d959499b674f66c95f3cfc16680e58a"


def _call_query(client: TTYMCP, request_id: int, query: str) -> dict[str, Any]:
    response = client.call(
        request_id,
        "tools/call",
        {
            "name": "recall_reasoning_query",
            "arguments": {
                "query": query,
                "k": ITEM_BUDGET,
                "mode": "evidence_assembly",
                "max_steps": 12,
                "max_graph_nodes": 1,
                "max_evidence_tokens": 2048,
                "graph_expansion": "off",
            },
        },
    )
    payload = json.loads(_extract_payload(response))
    if not isinstance(payload, dict):
        raise ValueError("reasoning query payload must be an object")
    return payload


def _identity(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = (
        str(payload.get("generation_id")),
        str(payload.get("calibration_id")),
        str(payload.get("pipeline_fingerprint")),
        str(payload.get("corpus_fingerprint")),
    )
    expected = (
        EXPECTED_GENERATION_ID,
        EXPECTED_CALIBRATION_ID,
        EXPECTED_PIPELINE,
        EXPECTED_CORPUS,
    )
    if identity != expected:
        raise RuntimeError(f"serving lineage mismatch: expected {expected}, got {identity}")
    return identity


def _public_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": payload.get("outcome"),
        "refusal_reason": payload.get("refusal_reason"),
        "trust_state": payload.get("trust_state"),
        "generation_id": payload.get("generation_id"),
        "calibration_id": payload.get("calibration_id"),
        "pipeline_fingerprint": payload.get("pipeline_fingerprint"),
        "corpus_fingerprint": payload.get("corpus_fingerprint"),
        "trusted_evidence": payload.get("trusted_evidence"),
    }


def _performance(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("diagnostics", {}).get("performance", {})
    if not isinstance(value, dict):
        raise ValueError("performance diagnostics are missing")
    return value


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["label"] is not None]
    unanswerable = [row for row in rows if row["label"] is None]
    total_facts = sum(len(row["label"]["facts"]) for row in answerable)
    arms: dict[str, Any] = {}
    for name in ("baseline", "candidate"):
        answer_scores = [row["scores"][name] for row in answerable]
        negative_scores = [row["scores"][name] for row in unanswerable]
        total_items = sum(int(value["items"]) for value in answer_scores)
        gold_items = sum(int(value["gold_items"]) for value in answer_scores)
        arms[name] = {
            "complete_queries": sum(bool(value["complete"]) for value in answer_scores),
            "covered_facts": sum(len(value["covered_facts"]) for value in answer_scores),
            "source_hit_queries": sum(bool(value["source_hit"]) for value in answer_scores),
            "gold_context_items": gold_items,
            "total_context_items": total_items,
            "context_precision": round(gold_items / total_items, 4) if total_items else None,
            "unanswerable_answers": sum(
                bool(value["answered_unanswerable"]) for value in negative_scores
            ),
        }
    candidate = arms["candidate"]
    baseline = arms["baseline"]
    candidate["delta_vs_baseline"] = {
        key: candidate[key] - baseline[key]
        for key in ("complete_queries", "covered_facts", "unanswerable_answers")
    }
    return {
        "queries": len(rows),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "essential_facts": total_facts,
        "arms": arms,
    }


def _decision(
    summary: dict[str, Any], *, parity_count: int, hash_parity_count: int, errors: int,
    elapsed_ms: float,
) -> str:
    if parity_count != 50 or hash_parity_count != 50 or errors:
        return "REPAIR"
    baseline = summary["arms"]["baseline"]
    candidate = summary["arms"]["candidate"]
    if (
        int(candidate["complete_queries"]) < int(baseline["complete_queries"])
        or int(candidate["covered_facts"]) < int(baseline["covered_facts"])
        or int(candidate["unanswerable_answers"]) > int(baseline["unanswerable_answers"])
        or int(candidate["unanswerable_answers"]) > 2
        or float(candidate["context_precision"] or 0.0) < 0.55
    ):
        return "CLOSE"
    gain = (
        int(candidate["complete_queries"]) > int(baseline["complete_queries"])
        or int(candidate["covered_facts"]) > int(baseline["covered_facts"])
    )
    if gain and elapsed_ms < 20 * 60 * 1000:
        return "BUILD SAMPLED SHADOW"
    return "GATE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-set",
        default="docs/preregistrations/2026-09-13-memory-queries-source-gold.json",
    )
    parser.add_argument(
        "--fact-labels",
        default="docs/preregistrations/2026-09-13-memory-essential-facts.json",
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.generation_id != EXPECTED_GENERATION_ID:
        raise ValueError("generation differs from the registered validation generation")

    query_path = Path(args.query_set)
    label_path = Path(args.fact_labels)
    artifact_path = Path(args.artifact)
    query_bytes = query_path.read_bytes()
    label_bytes = label_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != EXPECTED_QUERY_SHA256:
        raise RuntimeError("query set differs from the registered digest")
    if hashlib.sha256(label_bytes).hexdigest() != EXPECTED_FACT_SHA256:
        raise RuntimeError("fact labels differ from the registered digest")
    model = load_source_conditioning_artifact(artifact_path)
    model.assert_compatible(
        pipeline_fingerprint=EXPECTED_PIPELINE,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile=args.profile,
        candidate_k=20,
    )
    queries = json.loads(query_bytes.decode("utf-8"))
    labels = json.loads(label_bytes.decode("utf-8"))
    labels_by_id = {str(value["query_id"]): value for value in labels}

    common = (
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
    )
    off_client = TTYMCP(_command(*common), args.timeout)
    shadow_client = TTYMCP(
        _command(
            *common,
            benchmark_retrieval_leg_audit=True,
            benchmark_source_admission_audit=True,
            source_conditioning_mode="shadow",
            source_conditioning_artifact=args.artifact.replace("\\", "/"),
            source_conditioning_sample_rate=1.0,
        ),
        args.timeout,
    )
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    try:
        off_id = _initialize(off_client)
        shadow_id = _initialize(shadow_client)
        for query_index, query in enumerate(queries):
            print(f"paired {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            off = _call_query(off_client, off_id, str(query["query"]))
            off_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            shadow = _call_query(shadow_client, shadow_id, str(query["query"]))
            shadow_ms = (time.perf_counter() - started) * 1000.0
            _identity(off)
            _identity(shadow)
            public_parity = _public_signature(off) == _public_signature(shadow)
            shadow_diagnostic = _audit(shadow, "source_conditioning_shadow")
            leg_audit = _audit(shadow, "retrieval_leg_benchmark_audit")
            pool_audit = _audit(shadow, "source_admission_benchmark_audit")
            selected = select_source_conditioned(
                model,
                pool_audit["items"],
                leg_audit["dense"],
                leg_audit["sparse"],
                threshold=float(pool_audit["threshold"]),
            )
            expected_hashes = [
                chunk_identifier_hash(value["chunk_id"]) for value in selected
            ]
            hash_parity = (
                shadow_diagnostic.get("selected_chunk_hashes") == expected_hashes
                and int(shadow_diagnostic.get("selected_count", -1)) == len(selected)
            )
            performance = _performance(shadow)
            stage_ms = performance.get("stage_ms", {})
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": labels_by_id.get(str(query["id"])),
                    "off_client_observed_ms": round(off_ms, 3),
                    "shadow_client_observed_ms": round(shadow_ms, 3),
                    "shadow_internal_ms": stage_ms.get("source_conditioning_shadow_ms"),
                    "public_parity": public_parity,
                    "hash_parity": hash_parity,
                    "shadow_diagnostic": shadow_diagnostic,
                    "baseline_items": list(
                        off.get("trusted_evidence", {}).get("items", [])
                    ),
                    "candidate_items": selected,
                }
            )
            off_id += 1
            shadow_id += 1
    finally:
        off_client.close()
        shadow_client.close()

    elapsed_ms = round((time.perf_counter() - started_run) * 1000.0, 3)
    for row in rows:
        row["scores"] = {
            "baseline": _score_selection(row["baseline_items"], row["label"]),
            "candidate": _score_selection(row["candidate_items"], row["label"]),
        }
    summary = _summarize(rows)
    parity_count = sum(bool(row["public_parity"]) for row in rows)
    hash_parity_count = sum(bool(row["hash_parity"]) for row in rows)
    error_count = sum(row["shadow_diagnostic"].get("status") != "ok" for row in rows)
    decision = _decision(
        summary,
        parity_count=parity_count,
        hash_parity_count=hash_parity_count,
        errors=error_count,
        elapsed_ms=elapsed_ms,
    )
    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-source-conditioning-shadow",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "model_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "model_artifact_fingerprint": model.artifact_fingerprint,
        "generation_id": EXPECTED_GENERATION_ID,
        "calibration_id": EXPECTED_CALIBRATION_ID,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "public_request_count": len(rows) * 2,
        "client_observed_total_ms": elapsed_ms,
        "public_parity_count": parity_count,
        "hash_parity_count": hash_parity_count,
        "shadow_error_count": error_count,
        "decision": decision,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(output), "decision": decision, "summary": summary}))


if __name__ == "__main__":
    main()
