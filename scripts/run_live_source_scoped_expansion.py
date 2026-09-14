"""Measure source scoped expansion on the pinned Context 4 memory gold set."""

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
from scripts.run_live_source_conditioned_admission import (  # noqa: E402
    ITEM_BUDGET,
    RAW_COSINE_FLOOR,
    _audit,
    _leave_one_query_out_support,
    _score_selection,
    _source_rows,
    _source_select,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


SOURCE_BUDGETS = (1, 2, 3)
SCOPED_ALPHAS = (0.00, 0.08, 0.15)
GLOBAL_ALPHAS = (0.08, 0.15)
CHUNKS_PER_SOURCE = 8
EXPECTED_QUERY_SHA256 = "06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f"
EXPECTED_FACT_SHA256 = "45c38731e138f6aed425635b78ee79b692e142f5ef040e0efffeb042ad47186b"
EXPECTED_CALIBRATION_ID = "cal_23d8708ac550444fa4274ac617df0870"


def _rank_sources(row: dict[str, Any], support: dict[str, float]) -> list[str]:
    best_pool_rank: dict[str, int] = {}
    for item in row["source_admission_audit"]["items"]:
        source = str(item["source"])
        rank = int(item["pool_rank"])
        best_pool_rank[source] = min(rank, best_pool_rank.get(source, rank))
    if set(best_pool_rank) != set(support):
        raise RuntimeError("source support and global candidate sources differ")
    return sorted(
        support,
        key=lambda source: (-support[source], best_pool_rank[source], source),
    )


def _source_scoped_candidates(
    source_audits: dict[str, dict[str, Any]],
    ranked_sources: list[str],
    support: dict[str, float],
    source_budget: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_rank, source in enumerate(ranked_sources[:source_budget], start=1):
        audit = source_audits[source]
        source_items = sorted(
            audit["items"],
            key=lambda item: (
                -float(item["cosine"]),
                int(item["pool_rank"]),
                str(item["chunk_id"]),
            ),
        )[:CHUNKS_PER_SOURCE]
        for item in source_items:
            if str(item["source"]) != source:
                raise RuntimeError("source scoped result escaped its source filter")
            chunk_id = str(item["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidates.append(
                {
                    **item,
                    "source_support": float(support[source]),
                    "source_rank": source_rank,
                    "source_scoped_pool_rank": int(item["pool_rank"]),
                }
            )
    return candidates


def _scoped_select(
    candidates: list[dict[str, Any]], threshold: float, alpha: float
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("verdict") not in {"ok", "low_confidence"}:
            continue
        cosine = float(item["cosine"])
        adjusted = cosine + alpha * (float(item["source_support"]) - 0.5)
        if cosine < RAW_COSINE_FLOOR or adjusted < threshold:
            continue
        eligible.append({**item, "adjusted_score": adjusted})
    eligible.sort(
        key=lambda item: (
            -float(item["adjusted_score"]),
            -float(item["source_support"]),
            int(item["source_rank"]),
            int(item["source_scoped_pool_rank"]),
            str(item["chunk_id"]),
        )
    )
    return eligible[:ITEM_BUDGET]


def _arm_name(source_budget: int, alpha: float) -> str:
    return f"scoped_sources_{source_budget}_alpha_{alpha:.2f}"


def _summarize(rows: list[dict[str, Any]], arm_names: list[str]) -> dict[str, Any]:
    answerable = [row for row in rows if row["label"] is not None]
    unanswerable = [row for row in rows if row["label"] is None]
    total_facts = sum(len(row["label"]["facts"]) for row in answerable)
    arms: dict[str, Any] = {}
    for name in arm_names:
        answer_scores = [row["scores"][name] for row in answerable]
        negative_scores = [row["scores"][name] for row in unanswerable]
        total_items = sum(int(score["items"]) for score in answer_scores)
        gold_items = sum(int(score["gold_items"]) for score in answer_scores)
        arms[name] = {
            "complete_queries": sum(bool(score["complete"]) for score in answer_scores),
            "answerable_queries": len(answerable),
            "covered_facts": sum(len(score["covered_facts"]) for score in answer_scores),
            "total_facts": total_facts,
            "source_hit_queries": sum(bool(score["source_hit"]) for score in answer_scores),
            "gold_context_items": gold_items,
            "total_context_items": total_items,
            "context_precision": round(gold_items / total_items, 4) if total_items else None,
            "unanswerable_answers": sum(
                bool(score["answered_unanswerable"]) for score in negative_scores
            ),
            "unanswerable_abstentions": sum(
                not bool(score["answered_unanswerable"]) for score in negative_scores
            ),
        }

    baseline = arms["baseline"]
    for name, values in arms.items():
        values["delta_vs_baseline"] = {
            key: values[key] - baseline[key]
            for key in ("complete_queries", "covered_facts", "unanswerable_answers")
        }
        if not name.startswith("scoped_sources_"):
            continue
        alpha = float(name.rsplit("_", 1)[1])
        control_name = f"global_alpha_{alpha:.2f}"
        if control_name not in arms:
            continue
        control = arms[control_name]
        values["delta_vs_global_control"] = {
            key: values[key] - control[key]
            for key in ("complete_queries", "covered_facts", "unanswerable_answers")
        }
    return {
        "queries": len(rows),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "essential_facts": total_facts,
        "arms": arms,
    }


def _meets_build(values: dict[str, Any], control: dict[str, Any]) -> bool:
    quality_gain = (
        int(values["complete_queries"]) >= int(control["complete_queries"]) + 1
        or int(values["covered_facts"]) >= int(control["covered_facts"]) + 2
    )
    return bool(
        quality_gain
        and int(values["complete_queries"]) >= int(control["complete_queries"])
        and int(values["covered_facts"]) >= int(control["covered_facts"])
        and int(values["unanswerable_answers"]) <= int(control["unanswerable_answers"])
        and int(values["unanswerable_answers"]) <= 2
        and float(values["context_precision"] or 0.0) >= 0.48
    )


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    for source_budget in SOURCE_BUDGETS:
        name = _arm_name(source_budget, 0.08)
        if _meets_build(arms[name], arms["global_alpha_0.08"]):
            return {"verdict": "BUILD SOURCE SCOPED EXPANSION", "selected_arm": name}
    for source_budget in SOURCE_BUDGETS:
        name = _arm_name(source_budget, 0.15)
        if _meets_build(arms[name], arms["global_alpha_0.15"]):
            return {"verdict": "VALIDATE ALPHA 0.15", "selected_arm": name}

    for source_budget in SOURCE_BUDGETS:
        for alpha in (0.08, 0.15):
            name = _arm_name(source_budget, alpha)
            values = arms[name]
            control = arms[f"global_alpha_{alpha:.2f}"]
            complete_delta = int(values["complete_queries"]) - int(control["complete_queries"])
            fact_delta = int(values["covered_facts"]) - int(control["covered_facts"])
            safe = int(values["unanswerable_answers"]) <= int(
                control["unanswerable_answers"]
            )
            if safe and (
                (fact_delta == 1 and complete_delta >= 0)
                or (complete_delta == 1 and fact_delta == -1)
            ):
                return {"verdict": "GATE", "selected_arm": name}
    return {"verdict": "CLOSE", "selected_arm": None}


def _support_and_rank(rows: list[dict[str, Any]]) -> None:
    source_rows = [source_row for row in rows for source_row in _source_rows(row)]
    all_support = _leave_one_query_out_support(source_rows)
    for row in rows:
        query_index = int(row["query_index"])
        support = {
            source: value
            for (index, source), value in all_support.items()
            if index == query_index
        }
        ranked = _rank_sources(row, support)
        if len(ranked) < max(SOURCE_BUDGETS):
            raise RuntimeError("every query must expose at least three candidate sources")
        gold = set(str(value) for value in row["query"].get("relevant_files", []))
        row["source_support"] = [
            {
                "source": source,
                "support": support[source],
                "rank": rank,
                "gold": bool(row["query"].get("answerable") and source in gold),
            }
            for rank, source in enumerate(ranked, start=1)
        ]
        row["ranked_sources"] = ranked


def _score_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arm_names = ["baseline", "global_alpha_0.08", "global_alpha_0.15"] + [
        _arm_name(source_budget, alpha)
        for source_budget in SOURCE_BUDGETS
        for alpha in SCOPED_ALPHAS
    ]
    for row in rows:
        support = {
            str(item["source"]): float(item["support"]) for item in row["source_support"]
        }
        threshold = float(row["source_admission_audit"]["threshold"])
        global_pool = list(row["source_admission_audit"]["items"])
        selections = {
            "baseline": list(row["baseline_items"]),
            "global_alpha_0.08": _source_select(global_pool, support, threshold, 0.08),
            "global_alpha_0.15": _source_select(global_pool, support, threshold, 0.15),
        }
        for source_budget in SOURCE_BUDGETS:
            candidates = _source_scoped_candidates(
                row["source_scoped_audits"],
                row["ranked_sources"],
                support,
                source_budget,
            )
            for alpha in SCOPED_ALPHAS:
                selections[_arm_name(source_budget, alpha)] = _scoped_select(
                    candidates, threshold, alpha
                )
        row["selections"] = selections
        row["scores"] = {
            name: _score_selection(items, row["label"])
            for name, items in selections.items()
        }
    summary = _summarize(rows, arm_names)
    return {"summary": summary, "decision": _decision(summary)}


def _call_query(
    client: TTYMCP,
    request_id: int,
    query: str,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "query": query,
        "k": ITEM_BUDGET,
        "mode": "evidence_assembly",
        "max_steps": 12,
        "max_graph_nodes": 1,
        "max_evidence_tokens": 2048,
        "graph_expansion": "off",
    }
    if source is not None:
        arguments["source"] = source
    response = client.call(
        request_id,
        "tools/call",
        {"name": "recall_reasoning_query", "arguments": arguments},
    )
    payload = json.loads(_extract_payload(response))
    if not isinstance(payload, dict):
        raise ValueError("reasoning query payload must be an object")
    return payload


def _identity(payload: dict[str, Any], generation_id: str) -> tuple[str, str, str]:
    if payload.get("generation_id") != generation_id:
        raise RuntimeError(
            f"pinned generation mismatch: expected {generation_id}, "
            f"got {payload.get('generation_id')}"
        )
    identity = (
        str(payload.get("calibration_id")),
        str(payload.get("pipeline_fingerprint")),
        str(payload.get("corpus_fingerprint")),
    )
    if identity[0] != EXPECTED_CALIBRATION_ID:
        raise RuntimeError(
            f"calibration mismatch: expected {EXPECTED_CALIBRATION_ID}, got {identity[0]}"
        )
    return identity


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    query_path = Path(args.query_set)
    label_path = Path(args.fact_labels)
    query_bytes = query_path.read_bytes()
    label_bytes = label_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != EXPECTED_QUERY_SHA256:
        raise RuntimeError("query set differs from the registered digest")
    if hashlib.sha256(label_bytes).hexdigest() != EXPECTED_FACT_SHA256:
        raise RuntimeError("fact labels differ from the registered digest")
    queries = json.loads(query_bytes.decode("utf-8"))
    labels = json.loads(label_bytes.decode("utf-8"))
    labels_by_id = {str(label["query_id"]): label for label in labels}
    answerable_ids = {str(query["id"]) for query in queries if bool(query["answerable"])}
    if set(labels_by_id) != answerable_ids:
        raise ValueError("essential fact labels must cover exactly the answerable queries")

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
            benchmark_source_admission_audit=True,
        ),
        args.timeout,
    )
    rows: list[dict[str, Any]] = []
    expected_identity: tuple[str, str, str] | None = None
    request_count = 0
    started_run = time.perf_counter()
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"global {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            payload = _call_query(client, request_id, str(query["query"]))
            request_count += 1
            identity = _identity(payload, args.generation_id)
            if expected_identity is None:
                expected_identity = identity
            elif identity != expected_identity:
                raise RuntimeError(
                    f"lineage changed during collection: {expected_identity} then {identity}"
                )
            source_audit = _audit(payload, "source_admission_benchmark_audit")
            if int(source_audit["candidate_k"]) != 20:
                raise RuntimeError("registered source admission candidate_k is 20")
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": labels_by_id.get(str(query["id"])),
                    "global_client_observed_ms": round(
                        (time.perf_counter() - started) * 1000.0, 3
                    ),
                    "baseline_items": list(payload.get("trusted_evidence", {}).get("items", [])),
                    "leg_audit": _audit(payload, "retrieval_leg_benchmark_audit"),
                    "source_admission_audit": source_audit,
                }
            )
            request_id += 1

        _support_and_rank(rows)

        for row_index, row in enumerate(rows):
            row["source_scoped_audits"] = {}
            row["source_scoped_client_observed_ms"] = {}
            for source_index, source in enumerate(row["ranked_sources"][:3], start=1):
                print(
                    f"scoped {row_index + 1}/{len(rows)} source {source_index}/3",
                    flush=True,
                )
                started = time.perf_counter()
                payload = _call_query(
                    client,
                    request_id,
                    str(row["query"]["query"]),
                    source=source,
                )
                request_count += 1
                identity = _identity(payload, args.generation_id)
                if identity != expected_identity:
                    raise RuntimeError(
                        f"lineage changed during collection: {expected_identity} then {identity}"
                    )
                audit = _audit(payload, "source_admission_benchmark_audit")
                row["source_scoped_audits"][source] = audit
                row["source_scoped_client_observed_ms"][source] = round(
                    (time.perf_counter() - started) * 1000.0, 3
                )
                request_id += 1
    finally:
        client.close()

    elapsed_ms = round((time.perf_counter() - started_run) * 1000.0, 3)
    if request_count != 200:
        raise RuntimeError(f"registered audit requires 200 retrieval requests, got {request_count}")
    analysis = _score_experiment(rows)
    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-source-scoped-expansion",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels": str(label_path),
        "fact_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "query_count": len(queries),
        "retrieval_request_count": request_count,
        "client_observed_total_ms": elapsed_ms,
        "generation_id": args.generation_id,
        "calibration_id": None if expected_identity is None else expected_identity[0],
        "pipeline_fingerprint": None if expected_identity is None else expected_identity[1],
        "corpus_fingerprint": None if expected_identity is None else expected_identity[2],
        "source_budgets": list(SOURCE_BUDGETS),
        "chunks_per_source": CHUNKS_PER_SOURCE,
        "scoped_alphas": list(SCOPED_ALPHAS),
        "raw_cosine_floor": RAW_COSINE_FLOOR,
        **analysis,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": artifact["decision"],
                "summary": artifact["summary"],
            }
        )
    )


if __name__ == "__main__":
    main()
