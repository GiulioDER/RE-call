"""Measure source-scoped document expansion against immutable essential fact labels."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


ARMS = ("baseline", "document_retrieval", "document_bundle", "structural_bundle")


def _document_audit(payload: dict[str, Any]) -> dict[str, Any]:
    value = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("values", {})
        .get("document_expansion_benchmark_audit")
    )
    if not isinstance(value, dict):
        raise ValueError("document expansion benchmark audit is missing")
    if set(value.get("arms", {})) != set(ARMS[1:]):
        raise ValueError("document expansion benchmark arms are incomplete")
    return value


def _fact_covered(fact: dict[str, Any], items: list[dict[str, Any]], source: str) -> bool:
    terms = [str(term).casefold() for term in fact["terms"]]
    min_matches = int(fact["min_matches"])
    for item in items:
        if str(item.get("source")) != source:
            continue
        text = str(item.get("text", "")).casefold()
        if sum(term in text for term in terms) >= min_matches:
            return True
    return False


def _score_arm(
    arm: dict[str, Any],
    label: dict[str, Any] | None,
) -> dict[str, Any]:
    items = list(arm.get("items", []))
    if label is None:
        return {
            "decision": arm.get("decision"),
            "reason_code": arm.get("reason_code"),
            "items": len(items),
            "answered_unanswerable": arm.get("decision") == "answer" or bool(items),
        }
    source = str(label["source"])
    facts = list(label["facts"])
    covered = [str(fact["name"]) for fact in facts if _fact_covered(fact, items, source)]
    gold_items = sum(str(item.get("source")) == source for item in items)
    return {
        "decision": arm.get("decision"),
        "reason_code": arm.get("reason_code"),
        "items": len(items),
        "gold_items": gold_items,
        "source_hit": gold_items > 0,
        "covered_facts": covered,
        "fact_count": len(facts),
        "complete": len(covered) == len(facts),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return round(ordered[index], 3)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["label"] is not None]
    unanswerable = [row for row in rows if row["label"] is None]
    total_facts = sum(len(row["label"]["facts"]) for row in answerable)
    summaries: dict[str, Any] = {}
    for arm_name in ARMS:
        scores = [row["scores"][arm_name] for row in answerable]
        unanswerable_scores = [row["scores"][arm_name] for row in unanswerable]
        retrieval_ms = [
            float(row["arms"][arm_name]["retrieval_ms"])
            for row in rows
            if row["arms"][arm_name].get("retrieval_ms") is not None
        ]
        total_items = sum(int(score["items"]) for score in scores)
        gold_items = sum(int(score["gold_items"]) for score in scores)
        covered_facts = sum(len(score["covered_facts"]) for score in scores)
        false_answers = sum(bool(score["answered_unanswerable"]) for score in unanswerable_scores)
        summaries[arm_name] = {
            "complete_queries": sum(bool(score["complete"]) for score in scores),
            "answerable_queries": len(answerable),
            "covered_facts": covered_facts,
            "total_facts": total_facts,
            "source_hit_queries": sum(bool(score["source_hit"]) for score in scores),
            "gold_context_items": gold_items,
            "total_context_items": total_items,
            "context_precision": round(gold_items / total_items, 4) if total_items else None,
            "unanswerable_answers": false_answers,
            "unanswerable_abstentions": len(unanswerable) - false_answers,
            "mean_retrieval_ms": (
                round(statistics.fmean(retrieval_ms), 3) if retrieval_ms else None
            ),
            "p95_retrieval_ms": _percentile(retrieval_ms, 0.95),
        }
    pool_summaries: dict[str, Any] = {}
    for pool_name in ("document", "structural"):
        scores = [row["pool_scores"][pool_name] for row in answerable]
        pool_summaries[pool_name] = {
            "complete_queries": sum(bool(score["complete"]) for score in scores),
            "covered_facts": sum(len(score["covered_facts"]) for score in scores),
            "source_hit_queries": sum(bool(score["source_hit"]) for score in scores),
        }
    return {
        "queries": len(rows),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "essential_facts": total_facts,
        "arms": summaries,
        "diagnostic_trusted_pools": pool_summaries,
    }


def _baseline_arm(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = payload.get("trusted_evidence", {})
    retrieval_ms = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("spans_ms", {})
        .get("baseline_retrieval_ms")
    )
    return {
        "decision": bundle.get("decision"),
        "reason_code": bundle.get("reason_code"),
        "trust_state": bundle.get("trust_state"),
        "items": list(bundle.get("items", [])),
        "retrieval_ms": retrieval_ms,
    }


def _call_query(
    client: TTYMCP,
    request_id: int,
    query: str,
    *,
    max_steps: int,
    max_evidence_tokens: int,
) -> dict[str, Any]:
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
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-evidence-tokens", type=int, default=2048)
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
            benchmark_document_expansion_audit=True,
        ),
        args.timeout,
    )
    rows: list[dict[str, Any]] = []
    try:
        request_id = _initialize(client)
        for query_index, query in enumerate(queries):
            print(f"recorded {query_index + 1}/{len(queries)}", flush=True)
            started = time.perf_counter()
            payload = _call_query(
                client,
                request_id,
                str(query["query"]),
                max_steps=args.max_steps,
                max_evidence_tokens=args.max_evidence_tokens,
            )
            client_ms = (time.perf_counter() - started) * 1000.0
            if payload.get("generation_id") != args.generation_id:
                raise RuntimeError(
                    f"pinned generation mismatch: expected {args.generation_id}, "
                    f"got {payload.get('generation_id')}"
                )
            audit = _document_audit(payload)
            arms = {"baseline": _baseline_arm(payload), **audit["arms"]}
            label = labels_by_id.get(str(query["id"]))
            pools = {
                name: {
                    "decision": "answer" if items else "abstain",
                    "reason_code": None,
                    "items": items,
                }
                for name, items in audit.get("diagnostic_pools", {}).items()
            }
            rows.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": label,
                    "client_observed_ms": round(client_ms, 3),
                    "generation_id": payload.get("generation_id"),
                    "pipeline_fingerprint": payload.get("pipeline_fingerprint"),
                    "corpus_fingerprint": payload.get("corpus_fingerprint"),
                    "calibration_id": payload.get("calibration_id"),
                    "audit_policy": {
                        key: value
                        for key, value in audit.items()
                        if key not in {"arms", "diagnostic_pools"}
                    },
                    "arms": arms,
                    "scores": {
                        arm_name: _score_arm(arm, label)
                        for arm_name, arm in arms.items()
                    },
                    "diagnostic_pools": pools,
                    "pool_scores": {
                        pool_name: _score_arm(pool, label)
                        for pool_name, pool in pools.items()
                    },
                }
            )
            request_id += 1
    finally:
        client.close()

    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-document-expansion-essential-facts",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "fact_labels": str(label_path),
        "fact_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "generation_id": args.generation_id,
        "summary": _summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": artifact["summary"], "output": str(output)}))


if __name__ == "__main__":
    main()
