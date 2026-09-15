"""Measure source conditioned per chunk admission on the pinned memory gold set."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


RRF_K = 60
ITEM_BUDGET = 5
RAW_COSINE_FLOOR = 0.30
L2_COEFFICIENT = 1.0
MAX_IRLS_ITERATIONS = 100
IRLS_TOLERANCE = 1e-8
ALPHAS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
FEATURE_NAMES = (
    "max_cosine",
    "best_dense_rr",
    "best_sparse_rr",
    "rrf_mass",
    "cross_leg_fraction",
    "log_chunk_count",
    "source_margin",
)
EXPECTED_BASELINE = {
    "complete_queries": 14,
    "covered_facts": 16,
    "source_hit_queries": 17,
    "context_precision": 0.4677,
    "unanswerable_answers": 4,
}
FloatArray: TypeAlias = NDArray[np.float64]


def _audit(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("values", {})
        .get(name)
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} is missing")
    return value


def _rank_maps(audit: dict[str, Any], candidate_k: int) -> tuple[dict[str, int], dict[str, int]]:
    def ranks(name: str) -> dict[str, int]:
        return {
            str(item["chunk_id"]): int(item["rank"])
            for item in audit[name]
            if int(item["rank"]) <= candidate_k
        }

    return ranks("dense"), ranks("sparse")


def _source_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    pool_audit = row["source_admission_audit"]
    candidate_k = int(pool_audit["candidate_k"])
    dense_ranks, sparse_ranks = _rank_maps(row["leg_audit"], candidate_k)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool_audit["items"]:
        by_source[str(item["source"])].append(item)

    mass: dict[str, float] = {}
    for source, items in by_source.items():
        value = 0.0
        for item in items:
            chunk_id = str(item["chunk_id"])
            if chunk_id in dense_ranks:
                value += 1.0 / (RRF_K + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                value += 1.0 / (RRF_K + sparse_ranks[chunk_id])
        mass[source] = value

    gold = {str(source) for source in row["query"].get("relevant_files", [])}
    source_rows: list[dict[str, Any]] = []
    for source, items in by_source.items():
        chunk_ids = {str(item["chunk_id"]) for item in items}
        dense = [dense_ranks[chunk_id] for chunk_id in chunk_ids if chunk_id in dense_ranks]
        sparse = [sparse_ranks[chunk_id] for chunk_id in chunk_ids if chunk_id in sparse_ranks]
        both = sum(chunk_id in dense_ranks and chunk_id in sparse_ranks for chunk_id in chunk_ids)
        strongest_other = max((value for key, value in mass.items() if key != source), default=0.0)
        features = (
            max(float(item["cosine"]) for item in items),
            0.0 if not dense else 1.0 / (RRF_K + min(dense)),
            0.0 if not sparse else 1.0 / (RRF_K + min(sparse)),
            mass[source],
            both / len(chunk_ids),
            math.log1p(len(chunk_ids)),
            mass[source] - strongest_other,
        )
        source_rows.append(
            {
                "query_index": int(row["query_index"]),
                "source": source,
                "features": features,
                "label": int(bool(row["query"].get("answerable")) and source in gold),
            }
        )
    return source_rows


def _query_and_class_weights(rows: list[dict[str, Any]]) -> FloatArray:
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        counts[int(row["query_index"])] += 1
    weights: FloatArray = np.asarray(
        [1.0 / counts[int(row["query_index"])] for row in rows], dtype=float
    )
    labels: NDArray[np.int64] = np.asarray(
        [int(row["label"]) for row in rows], dtype=np.int64
    )
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0.0 or negative <= 0.0:
        raise ValueError("source model training fold requires both label classes")
    weights[labels == 1] *= 0.5 / positive
    weights[labels == 0] *= 0.5 / negative
    return weights


def _sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _fit_source_model(rows: list[dict[str, Any]]) -> tuple[FloatArray, FloatArray, FloatArray]:
    raw: FloatArray = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels: FloatArray = np.asarray(
        [int(row["label"]) for row in rows], dtype=np.float64
    )
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (raw - means) / scales
    design = np.column_stack((np.ones(len(rows)), standardized))
    sample_weights = _query_and_class_weights(rows)
    coefficients = np.zeros(design.shape[1], dtype=float)
    ridge = np.diag([0.0, *([L2_COEFFICIENT] * len(FEATURE_NAMES))])

    for _ in range(MAX_IRLS_ITERATIONS):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (sample_weights * (probabilities - labels))
        gradient += ridge @ coefficients
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients -= step
        if float(np.max(np.abs(step))) < IRLS_TOLERANCE:
            break
    return means, scales, coefficients


def _leave_one_query_out_support(rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    query_indices = sorted({int(row["query_index"]) for row in rows})
    support: dict[tuple[int, str], float] = {}
    for held_out in query_indices:
        train = [row for row in rows if int(row["query_index"]) != held_out]
        test = [row for row in rows if int(row["query_index"]) == held_out]
        means, scales, coefficients = _fit_source_model(train)
        raw = np.asarray([row["features"] for row in test], dtype=float)
        design = np.column_stack((np.ones(len(test)), (raw - means) / scales))
        probabilities = _sigmoid(design @ coefficients)
        for index, row in enumerate(test):
            support[(held_out, str(row["source"]))] = float(probabilities[index])
    return support


def _trust_backfill(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [item for item in pool if item.get("verdict") == "ok"]
    return sorted(eligible, key=lambda item: int(item["pool_rank"]))[:ITEM_BUDGET]


def _source_select(
    pool: list[dict[str, Any]],
    support: dict[str, float],
    threshold: float,
    alpha: float,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for item in pool:
        if item.get("verdict") not in {"ok", "low_confidence"}:
            continue
        cosine = float(item["cosine"])
        source_support = float(support[str(item["source"])])
        adjusted = cosine + alpha * (source_support - 0.5)
        if cosine < RAW_COSINE_FLOOR or adjusted < threshold:
            continue
        eligible.append(
            {
                **item,
                "source_support": source_support,
                "adjusted_score": adjusted,
            }
        )
    eligible.sort(key=lambda item: (-float(item["adjusted_score"]), int(item["pool_rank"])))
    return eligible[:ITEM_BUDGET]


def _fact_covered(fact: dict[str, Any], items: list[dict[str, Any]], source: str) -> bool:
    terms = [str(term).casefold() for term in fact["terms"]]
    minimum = int(fact["min_matches"])
    return any(
        str(item.get("source")) == source
        and sum(term in str(item.get("text", "")).casefold() for term in terms) >= minimum
        for item in items
    )


def _score_selection(
    items: list[dict[str, Any]], label: dict[str, Any] | None
) -> dict[str, Any]:
    if label is None:
        return {"items": len(items), "answered_unanswerable": bool(items)}
    source = str(label["source"])
    facts = list(label["facts"])
    covered = [str(fact["name"]) for fact in facts if _fact_covered(fact, items, source)]
    gold_items = sum(str(item.get("source")) == source for item in items)
    return {
        "items": len(items),
        "gold_items": gold_items,
        "source_hit": gold_items > 0,
        "covered_facts": covered,
        "fact_count": len(facts),
        "complete": len(covered) == len(facts),
    }


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
    backfill = arms["trust_backfill"]
    for name, values in arms.items():
        values["delta_vs_baseline"] = {
            key: values[key] - baseline[key]
            for key in ("complete_queries", "covered_facts", "unanswerable_answers")
        }
        values["delta_vs_trust_backfill"] = {
            key: values[key] - backfill[key]
            for key in ("complete_queries", "covered_facts", "unanswerable_answers")
        }
    return {
        "queries": len(rows),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "essential_facts": total_facts,
        "arms": arms,
    }


def _score_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [source_row for row in rows for source_row in _source_rows(row)]
    support = _leave_one_query_out_support(source_rows)
    arm_names = ["baseline", "trust_backfill", "cosine_backfill"] + [
        f"source_alpha_{alpha:.2f}" for alpha in ALPHAS if alpha > 0.0
    ]
    for row in rows:
        query_index = int(row["query_index"])
        per_source = {
            source: value
            for (index, source), value in support.items()
            if index == query_index
        }
        pool = list(row["source_admission_audit"]["items"])
        threshold = float(row["source_admission_audit"]["threshold"])
        selections = {
            "baseline": list(row["baseline_items"]),
            "trust_backfill": _trust_backfill(pool),
            "cosine_backfill": _source_select(pool, per_source, threshold, 0.0),
        }
        selections.update(
            {
                f"source_alpha_{alpha:.2f}": _source_select(
                    pool, per_source, threshold, alpha
                )
                for alpha in ALPHAS
                if alpha > 0.0
            }
        )
        row["source_support"] = [
            {
                "source": source,
                "support": value,
                "gold": bool(
                    row["query"].get("answerable")
                    and source in row["query"].get("relevant_files", [])
                ),
            }
            for source, value in sorted(per_source.items(), key=lambda item: item[1], reverse=True)
        ]
        row["selections"] = selections
        row["scores"] = {
            name: _score_selection(items, row["label"])
            for name, items in selections.items()
        }
    summary = _summarize(rows, arm_names)
    observed = {
        key: summary["arms"]["baseline"][key] for key in EXPECTED_BASELINE
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError(
            "served baseline drifted from the registered substrate: "
            f"expected {EXPECTED_BASELINE}, observed {observed}"
        )
    return {
        "feature_names": list(FEATURE_NAMES),
        "alphas": list(ALPHAS),
        "raw_cosine_floor": RAW_COSINE_FLOOR,
        "l2_coefficient": L2_COEFFICIENT,
        "summary": summary,
    }


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
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    query_path = Path(args.query_set)
    label_path = Path(args.fact_labels)
    query_bytes = query_path.read_bytes()
    label_bytes = label_path.read_bytes()
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
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"recorded {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            payload = _call_query(client, request_id, str(query["query"]))
            client_ms = (time.perf_counter() - started) * 1000.0
            if payload.get("generation_id") != args.generation_id:
                raise RuntimeError(
                    f"pinned generation mismatch: expected {args.generation_id}, "
                    f"got {payload.get('generation_id')}"
                )
            identity = (
                str(payload.get("calibration_id")),
                str(payload.get("pipeline_fingerprint")),
                str(payload.get("corpus_fingerprint")),
            )
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
                    "client_observed_ms": round(client_ms, 3),
                    "baseline_items": list(payload.get("trusted_evidence", {}).get("items", [])),
                    "leg_audit": _audit(payload, "retrieval_leg_benchmark_audit"),
                    "source_admission_audit": source_audit,
                }
            )
            request_id += 1
    finally:
        client.close()

    analysis = _score_experiment(rows)
    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-source-conditioned-chunk-admission",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels": str(label_path),
        "fact_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "query_count": len(queries),
        "generation_id": args.generation_id,
        "calibration_id": None if expected_identity is None else expected_identity[0],
        "pipeline_fingerprint": None if expected_identity is None else expected_identity[1],
        "corpus_fingerprint": None if expected_identity is None else expected_identity[2],
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
    print(json.dumps({"output": str(output), "summary": artifact["summary"]}))


if __name__ == "__main__":
    main()
