"""Validate the frozen source conditioning model on an independent memory holdout."""

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
    _audit,
    _score_selection,
)
from scripts.run_live_source_conditioning_shadow import (  # noqa: E402
    _call_query,
    _performance,
    _shadow_internal_ms,
    _summarize,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


EXPECTED_QUERY_SHA256 = "cf91bb848a518991a49863c79ae3014a4aa9c4ba3f717877c2019266173b5940"
EXPECTED_FACT_SHA256 = "4d5c359a53f87ab5264920e3ba2aa9ef19ad4e1cd6a0b1bb5864d4c3af4b1fee"
BURIED_QUERY_SHA256 = "9eefe96c8bb07cb71b2bde71f359c222f6966e488c85097249e5536b6b074794"
BURIED_FACT_SHA256 = "9e824f03caef370490f9be53b16451bc8fa29ee8d616aff7ff05351f6d0b86c5"
EXPECTED_ARTIFACT_SHA256 = "fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f"
EXPECTED_ARTIFACT_FINGERPRINT = (
    "9c3e4a2d1a56d4c3bf4d77800dbdf9b02261549199af8ddbeb4ac11aafdc66c7"
)
EXPECTED_GENERATION_ID = "gen_18d5edd2e5e847c0af1ee37e40d27893"
EXPECTED_CALIBRATION_ID = "cal_23d8708ac550444fa4274ac617df0870"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "c054e26b28e62760c0fefa64815da8033433e94c2474f079ed8a2aec1b93fb1a"
EXPECTED_REQUESTS = 36
MIN_TARGET_QUERIES = 4
MIN_SELECTION_DIFFERENCES = 2
PRECISION_TOLERANCE = 0.05


def _challenge_config(name: str) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {
        "independent": {
            "protocol": "2026-09-13-source-conditioning-independent-holdout",
            "query_set": (
                "docs/preregistrations/"
                "2026-09-13-memory-source-conditioning-holdout-queries.json"
            ),
            "fact_labels": (
                "docs/preregistrations/"
                "2026-09-13-memory-source-conditioning-holdout-facts.json"
            ),
            "query_sha256": EXPECTED_QUERY_SHA256,
            "fact_sha256": EXPECTED_FACT_SHA256,
            "prior_query_sets": [
                "docs/preregistrations/2026-09-13-memory-queries-source-gold.json"
            ],
        },
        "buried": {
            "protocol": "2026-09-13-source-conditioning-buried-fact-challenge",
            "query_set": (
                "docs/preregistrations/"
                "2026-09-13-memory-buried-fact-challenge-queries.json"
            ),
            "fact_labels": (
                "docs/preregistrations/"
                "2026-09-13-memory-buried-fact-challenge-facts.json"
            ),
            "query_sha256": BURIED_QUERY_SHA256,
            "fact_sha256": BURIED_FACT_SHA256,
            "prior_query_sets": [
                "docs/preregistrations/2026-09-13-memory-queries-source-gold.json",
                (
                    "docs/preregistrations/"
                    "2026-09-13-memory-source-conditioning-holdout-queries.json"
                ),
            ],
        },
    }
    return configs[name]


def _identity(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    observed = (
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
    if observed != expected:
        raise RuntimeError(f"serving lineage mismatch: expected {expected}, got {observed}")
    return observed


def _validate_holdout_contract(
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    prior_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    query_ids = [str(value["id"]) for value in queries]
    answerable_ids = {
        str(value["id"]) for value in queries if bool(value.get("answerable"))
    }
    label_ids = {str(value["query_id"]) for value in labels}
    if len(queries) != EXPECTED_REQUESTS or len(set(query_ids)) != EXPECTED_REQUESTS:
        raise ValueError("holdout must contain 36 unique query identifiers")
    if len(answerable_ids) != 18 or len(labels) != 18 or label_ids != answerable_ids:
        raise ValueError("holdout labels must cover exactly 18 answerable queries")
    holdout_sources = {str(value["source"]) for value in labels}
    prior_sources = {
        str(source)
        for value in prior_queries
        for source in value.get("relevant_files", [])
    }
    overlap = sorted(holdout_sources & prior_sources)
    if overlap:
        raise ValueError(f"holdout source overlap with prior gold: {overlap}")
    return {
        "queries": len(queries),
        "answerable_queries": len(answerable_ids),
        "unanswerable_queries": len(queries) - len(answerable_ids),
        "gold_sources": len(holdout_sources),
        "prior_source_overlap": overlap,
    }


def _target_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    complete_gains: list[str] = []
    complete_losses: list[str] = []
    fact_gains: list[str] = []
    fact_losses: list[str] = []
    for row in rows:
        if row["label"] is None:
            continue
        query_id = str(row["query"]["id"])
        baseline = row["scores"]["baseline"]
        candidate = row["scores"]["candidate"]
        baseline_fact_count = len(baseline["covered_facts"])
        candidate_fact_count = len(candidate["covered_facts"])
        if bool(candidate["complete"]) and not bool(baseline["complete"]):
            complete_gains.append(query_id)
        if bool(baseline["complete"]) and not bool(candidate["complete"]):
            complete_losses.append(query_id)
        if candidate_fact_count > baseline_fact_count:
            fact_gains.append(query_id)
        if candidate_fact_count < baseline_fact_count:
            fact_losses.append(query_id)
        if bool(baseline["source_hit"]) and not bool(baseline["complete"]):
            targets.append(
                {
                    "query_id": query_id,
                    "candidate_complete": bool(candidate["complete"]),
                    "candidate_source_hit": bool(candidate["source_hit"]),
                    "candidate_fact_gain": candidate_fact_count > baseline_fact_count,
                }
            )
    return {
        "target_queries": len(targets),
        "target_query_ids": [value["query_id"] for value in targets],
        "target_fact_rescues": sum(bool(value["candidate_fact_gain"]) for value in targets),
        "target_source_losses": sum(not bool(value["candidate_source_hit"]) for value in targets),
        "complete_gain_ids": complete_gains,
        "complete_loss_ids": complete_losses,
        "fact_gain_ids": fact_gains,
        "fact_loss_ids": fact_losses,
    }


def _decision(
    summary: dict[str, Any],
    target: dict[str, Any],
    integrity: dict[str, int],
) -> str:
    if (
        integrity["requests"] != EXPECTED_REQUESTS
        or integrity["candidate_hash_parity"] != EXPECTED_REQUESTS
        or integrity["baseline_hash_parity"] != EXPECTED_REQUESTS
        or integrity["shadow_ok"] != EXPECTED_REQUESTS
        or integrity["timing_receipts"] != EXPECTED_REQUESTS
        or integrity["selection_differences"] < MIN_SELECTION_DIFFERENCES
    ):
        return "REPAIR"
    if int(target["target_queries"]) < MIN_TARGET_QUERIES:
        return "INSUFFICIENT"
    baseline = summary["arms"]["baseline"]
    candidate = summary["arms"]["candidate"]
    baseline_precision = float(baseline["context_precision"] or 0.0)
    candidate_precision = float(candidate["context_precision"] or 0.0)
    if (
        int(target["target_fact_rescues"]) >= 1
        and not target["complete_loss_ids"]
        and not target["fact_loss_ids"]
        and int(candidate["complete_queries"]) >= int(baseline["complete_queries"])
        and int(candidate["covered_facts"]) >= int(baseline["covered_facts"])
        and int(candidate["unanswerable_answers"])
        <= int(baseline["unanswerable_answers"])
        and candidate_precision >= baseline_precision - PRECISION_TOLERANCE
    ):
        return "ADVANCE"
    return "CLOSE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", choices=("independent", "buried"), default="independent")
    parser.add_argument("--query-set")
    parser.add_argument("--fact-labels")
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
        raise ValueError("generation differs from the registered holdout generation")

    config = _challenge_config(args.challenge)
    query_path = Path(args.query_set or str(config["query_set"]))
    label_path = Path(args.fact_labels or str(config["fact_labels"]))
    artifact_path = Path(args.artifact)
    query_bytes = query_path.read_bytes()
    label_bytes = label_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != str(config["query_sha256"]):
        raise RuntimeError("query set differs from the registered digest")
    if hashlib.sha256(label_bytes).hexdigest() != str(config["fact_sha256"]):
        raise RuntimeError("fact labels differ from the registered digest")
    if hashlib.sha256(artifact_bytes).hexdigest() != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("source conditioning artifact differs from the registered digest")

    queries = json.loads(query_bytes.decode("utf-8"))
    labels = json.loads(label_bytes.decode("utf-8"))
    prior_queries = [
        query
        for path in config["prior_query_sets"]
        for query in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
    contract = _validate_holdout_contract(queries, labels, prior_queries)
    labels_by_id = {str(value["query_id"]): value for value in labels}
    model = load_source_conditioning_artifact(artifact_path)
    if model.artifact_fingerprint != EXPECTED_ARTIFACT_FINGERPRINT:
        raise RuntimeError("source conditioning artifact fingerprint mismatch")
    model.assert_compatible(
        pipeline_fingerprint=EXPECTED_PIPELINE,
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
    )
    client = TTYMCP(command, args.timeout)
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"holdout {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            public = _call_query(client, request_id, str(query["query"]))
            observed_ms = (time.perf_counter() - started) * 1000.0
            _identity(public)
            shadow = _audit(public, "source_conditioning_shadow")
            leg_audit = _audit(public, "retrieval_leg_benchmark_audit")
            pool_audit = _audit(public, "source_admission_benchmark_audit")
            selected = select_source_conditioned(
                model,
                pool_audit["items"],
                leg_audit["dense"],
                leg_audit["sparse"],
                threshold=float(pool_audit["threshold"]),
            )
            public_items = list(public.get("trusted_evidence", {}).get("items", []))
            expected_candidate_hashes = [
                chunk_identifier_hash(value["chunk_id"]) for value in selected
            ]
            expected_baseline_hashes = [
                chunk_identifier_hash(value["chunk_id"]) for value in public_items
            ]
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": labels_by_id.get(str(query["id"])),
                    "client_observed_ms": round(observed_ms, 3),
                    "shadow_internal_ms": _shadow_internal_ms(_performance(public)),
                    "candidate_hash_parity": (
                        shadow.get("selected_chunk_hashes") == expected_candidate_hashes
                        and int(shadow.get("selected_count", -1)) == len(selected)
                    ),
                    "baseline_hash_parity": (
                        shadow.get("baseline_chunk_hashes") == expected_baseline_hashes
                    ),
                    "selection_differs": expected_candidate_hashes != expected_baseline_hashes,
                    "shadow_status": shadow.get("status"),
                    "baseline_items": public_items,
                    "candidate_items": selected,
                }
            )
            request_id += 1
    finally:
        client.close()

    elapsed_ms = round((time.perf_counter() - started_run) * 1000.0, 3)
    for row in rows:
        row["scores"] = {
            "baseline": _score_selection(row["baseline_items"], row["label"]),
            "candidate": _score_selection(row["candidate_items"], row["label"]),
        }
    summary = _summarize(rows)
    target = _target_analysis(rows)
    integrity = {
        "requests": len(rows),
        "candidate_hash_parity": sum(bool(row["candidate_hash_parity"]) for row in rows),
        "baseline_hash_parity": sum(bool(row["baseline_hash_parity"]) for row in rows),
        "shadow_ok": sum(row["shadow_status"] == "ok" for row in rows),
        "timing_receipts": sum(row["shadow_internal_ms"] is not None for row in rows),
        "selection_differences": sum(bool(row["selection_differs"]) for row in rows),
    }
    decision = _decision(summary, target, integrity)
    result = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "challenge": args.challenge,
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
        "contract": contract,
        "integrity": integrity,
        "elapsed_ms": elapsed_ms,
        "decision": decision,
        "summary": summary,
        "target_analysis": target,
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
                "target_analysis": target,
            }
        )
    )


if __name__ == "__main__":
    main()
