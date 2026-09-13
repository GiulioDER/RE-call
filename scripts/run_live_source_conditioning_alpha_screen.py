"""Screen alpha 0.15 against alpha 0.08 on two inspected memory holdouts."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    SourceConditioningArtifact,
    chunk_identifier_hash,
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import (  # noqa: E402
    _audit,
    _score_selection,
)
from scripts.run_live_source_conditioning_holdout import (  # noqa: E402
    EXPECTED_ARTIFACT_FINGERPRINT,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_CALIBRATION_ID,
    EXPECTED_CORPUS,
    EXPECTED_GENERATION_ID,
    EXPECTED_PIPELINE,
    _challenge_config,
    _identity,
    _validate_holdout_contract,
)
from scripts.run_live_source_conditioning_shadow import (  # noqa: E402
    _call_query,
    _performance,
    _shadow_internal_ms,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


SCREEN_ALPHA = 0.15
EXPECTED_REQUESTS = 72
MIN_SELECTION_DIFFERENCES = 2
PRECISION_TOLERANCE = 0.05


def _screen_model(model: SourceConditioningArtifact) -> SourceConditioningArtifact:
    """Return an in-memory evaluation copy with only alpha changed."""

    return replace(model, alpha=SCREEN_ALPHA)


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    answerable = [row for row in rows if row["label"] is not None]
    unanswerable = [row for row in rows if row["label"] is None]
    scores = [row["scores"][arm] for row in answerable]
    gold_items = sum(int(value["gold_items"]) for value in scores)
    total_items = sum(int(value["items"]) for value in scores)
    return {
        "complete_queries": sum(bool(value["complete"]) for value in scores),
        "covered_facts": sum(len(value["covered_facts"]) for value in scores),
        "source_hit_queries": sum(bool(value["source_hit"]) for value in scores),
        "gold_context_items": gold_items,
        "total_context_items": total_items,
        "context_precision": round(gold_items / total_items, 4) if total_items else None,
        "unanswerable_answers": sum(
            bool(row["scores"][arm]["answered_unanswerable"]) for row in unanswerable
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {
        arm: _arm_summary(rows, arm) for arm in ("baseline", "alpha008", "alpha015")
    }
    return {
        "queries": len(rows),
        "answerable_queries": sum(row["label"] is not None for row in rows),
        "unanswerable_queries": sum(row["label"] is None for row in rows),
        "arms": arms,
        "delta_alpha015_vs_alpha008": {
            "complete_queries": (
                int(arms["alpha015"]["complete_queries"])
                - int(arms["alpha008"]["complete_queries"])
            ),
            "covered_facts": (
                int(arms["alpha015"]["covered_facts"])
                - int(arms["alpha008"]["covered_facts"])
            ),
            "unanswerable_answers": (
                int(arms["alpha015"]["unanswerable_answers"])
                - int(arms["alpha008"]["unanswerable_answers"])
            ),
        },
    }


def _comparison(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    complete_gains: list[str] = []
    complete_losses: list[str] = []
    fact_gains: list[str] = []
    fact_losses: list[str] = []
    for row in rows:
        if row["label"] is None:
            continue
        query_id = str(row["query"]["id"])
        fixed = row["scores"]["alpha008"]
        screen = row["scores"]["alpha015"]
        if bool(screen["complete"]) and not bool(fixed["complete"]):
            complete_gains.append(query_id)
        if bool(fixed["complete"]) and not bool(screen["complete"]):
            complete_losses.append(query_id)
        if len(screen["covered_facts"]) > len(fixed["covered_facts"]):
            fact_gains.append(query_id)
        if len(screen["covered_facts"]) < len(fixed["covered_facts"]):
            fact_losses.append(query_id)
    return {
        "complete_gain_ids": complete_gains,
        "complete_loss_ids": complete_losses,
        "fact_gain_ids": fact_gains,
        "fact_loss_ids": fact_losses,
    }


def _decision(
    summary: dict[str, Any],
    holdout_summaries: dict[str, dict[str, Any]],
    comparison: dict[str, list[str]],
    integrity: dict[str, int],
) -> str:
    if (
        integrity["requests"] != EXPECTED_REQUESTS
        or integrity["fixed_hash_parity"] != EXPECTED_REQUESTS
        or integrity["baseline_hash_parity"] != EXPECTED_REQUESTS
        or integrity["shadow_ok"] != EXPECTED_REQUESTS
        or integrity["timing_receipts"] != EXPECTED_REQUESTS
        or integrity["alpha_selection_differences"] < MIN_SELECTION_DIFFERENCES
    ):
        return "REPAIR"
    fixed = summary["arms"]["alpha008"]
    screen = summary["arms"]["alpha015"]
    per_holdout_safe = all(
        int(value["arms"]["alpha015"]["unanswerable_answers"])
        <= int(value["arms"]["alpha008"]["unanswerable_answers"])
        for value in holdout_summaries.values()
    )
    precision_safe = float(screen["context_precision"] or 0.0) >= (
        float(fixed["context_precision"] or 0.0) - PRECISION_TOLERANCE
    )
    if (
        int(screen["covered_facts"]) > int(fixed["covered_facts"])
        and not comparison["complete_loss_ids"]
        and not comparison["fact_loss_ids"]
        and int(screen["unanswerable_answers"]) <= int(fixed["unanswerable_answers"])
        and per_holdout_safe
        and precision_safe
    ):
        return "PROMISING_FOR_FRESH_VALIDATION"
    return "CLOSE_GLOBAL_ALPHA_INCREASE"


def _load_challenge(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = _challenge_config(name)
    query_path = Path(str(config["query_set"]))
    fact_path = Path(str(config["fact_labels"]))
    query_bytes = query_path.read_bytes()
    fact_bytes = fact_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != str(config["query_sha256"]):
        raise RuntimeError(f"{name} query set differs from the registered digest")
    if hashlib.sha256(fact_bytes).hexdigest() != str(config["fact_sha256"]):
        raise RuntimeError(f"{name} fact labels differ from the registered digest")
    queries = json.loads(query_bytes.decode("utf-8"))
    labels = json.loads(fact_bytes.decode("utf-8"))
    prior_queries = [
        query
        for path in config["prior_query_sets"]
        for query in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
    contract = _validate_holdout_contract(queries, labels, prior_queries)
    return queries, labels, {
        **contract,
        "query_set": str(query_path),
        "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels": str(fact_path),
        "fact_sha256": hashlib.sha256(fact_bytes).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", default="docs/results/2026-09-13-source-conditioning-model.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.generation_id != EXPECTED_GENERATION_ID:
        raise ValueError("generation differs from the registered screen generation")

    artifact_path = Path(args.artifact)
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("source conditioning artifact differs from the registered digest")
    fixed_model = load_source_conditioning_artifact(artifact_path)
    if fixed_model.artifact_fingerprint != EXPECTED_ARTIFACT_FINGERPRINT:
        raise RuntimeError("source conditioning artifact fingerprint mismatch")
    fixed_model.assert_compatible(
        pipeline_fingerprint=EXPECTED_PIPELINE,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile=args.profile,
        candidate_k=20,
    )
    screen_model = _screen_model(fixed_model)

    loaded = {name: _load_challenge(name) for name in ("independent", "buried")}
    independent_sources = {str(value["source"]) for value in loaded["independent"][1]}
    buried_sources = {str(value["source"]) for value in loaded["buried"][1]}
    if independent_sources & buried_sources:
        raise RuntimeError("screen holdout source sets overlap")

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
    )
    client = TTYMCP(command, args.timeout)
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        query_number = 0
        for challenge, (queries, labels, _) in loaded.items():
            labels_by_id = {str(value["query_id"]): value for value in labels}
            for query in queries:
                query_number += 1
                print(f"screen {query_number}/{EXPECTED_REQUESTS}", flush=True)
                started = time.perf_counter()
                public = _call_query(client, request_id, str(query["query"]))
                observed_ms = (time.perf_counter() - started) * 1000.0
                _identity(public)
                shadow = _audit(public, "source_conditioning_shadow")
                leg_audit = _audit(public, "retrieval_leg_benchmark_audit")
                pool_audit = _audit(public, "source_admission_benchmark_audit")
                fixed = select_source_conditioned(
                    fixed_model,
                    pool_audit["items"],
                    leg_audit["dense"],
                    leg_audit["sparse"],
                    threshold=float(pool_audit["threshold"]),
                )
                screen = select_source_conditioned(
                    screen_model,
                    pool_audit["items"],
                    leg_audit["dense"],
                    leg_audit["sparse"],
                    threshold=float(pool_audit["threshold"]),
                )
                public_items = list(public.get("trusted_evidence", {}).get("items", []))
                fixed_hashes = [chunk_identifier_hash(value["chunk_id"]) for value in fixed]
                baseline_hashes = [
                    chunk_identifier_hash(value["chunk_id"]) for value in public_items
                ]
                screen_hashes = [chunk_identifier_hash(value["chunk_id"]) for value in screen]
                rows.append(
                    {
                        "challenge": challenge,
                        "query": query,
                        "label": labels_by_id.get(str(query["id"])),
                        "client_observed_ms": round(observed_ms, 3),
                        "shadow_internal_ms": _shadow_internal_ms(_performance(public)),
                        "fixed_hash_parity": (
                            shadow.get("selected_chunk_hashes") == fixed_hashes
                            and int(shadow.get("selected_count", -1)) == len(fixed)
                        ),
                        "baseline_hash_parity": (
                            shadow.get("baseline_chunk_hashes") == baseline_hashes
                        ),
                        "alpha_selection_differs": fixed_hashes != screen_hashes,
                        "shadow_status": shadow.get("status"),
                        "baseline_items": public_items,
                        "alpha008_items": fixed,
                        "alpha015_items": screen,
                    }
                )
                request_id += 1
    finally:
        client.close()

    elapsed_ms = round((time.perf_counter() - started_run) * 1000.0, 3)
    for row in rows:
        row["scores"] = {
            arm: _score_selection(row[f"{arm}_items"], row["label"])
            for arm in ("baseline", "alpha008", "alpha015")
        }
    summary = _summarize(rows)
    holdout_summaries = {
        name: _summarize([row for row in rows if row["challenge"] == name])
        for name in loaded
    }
    comparison = _comparison(rows)
    integrity = {
        "requests": len(rows),
        "fixed_hash_parity": sum(bool(row["fixed_hash_parity"]) for row in rows),
        "baseline_hash_parity": sum(bool(row["baseline_hash_parity"]) for row in rows),
        "shadow_ok": sum(row["shadow_status"] == "ok" for row in rows),
        "timing_receipts": sum(row["shadow_internal_ms"] is not None for row in rows),
        "alpha_selection_differences": sum(
            bool(row["alpha_selection_differs"]) for row in rows
        ),
    }
    decision = _decision(summary, holdout_summaries, comparison, integrity)
    result = {
        "schema_version": 1,
        "protocol": "2026-09-13-source-conditioning-alpha015-development-screen",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "generation_id": EXPECTED_GENERATION_ID,
        "calibration_id": EXPECTED_CALIBRATION_ID,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "model_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "model_artifact_fingerprint": fixed_model.artifact_fingerprint,
        "fixed_alpha": fixed_model.alpha,
        "screen_alpha": screen_model.alpha,
        "contracts": {name: values[2] for name, values in loaded.items()},
        "integrity": integrity,
        "elapsed_ms": elapsed_ms,
        "decision": decision,
        "summary": summary,
        "holdout_summaries": holdout_summaries,
        "comparison": comparison,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": decision,
                "integrity": integrity,
                "summary": summary,
                "holdout_summaries": holdout_summaries,
                "comparison": comparison,
            }
        )
    )


if __name__ == "__main__":
    main()
