"""Generate paired answers from the frozen live graph retrieval artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import ANSWER_PROMPT_DIGEST  # noqa: E402
from recall.evidence import parse_answer_envelope, render_evidence_prompt, validate_answer  # noqa: E402
from recall.reasoning import reasoning_response_from_dict  # noqa: E402
from scripts.run_openrouter_answer_batch import _answer_one  # noqa: E402


ARMS = ("off", "one_hop")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("input must be a live graph performance artifact")
    rows = [row for row in payload["rows"] if isinstance(row, dict) and row.get("pass_index") == 1]
    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    if any(len(by_arm[arm]) != 50 for arm in ARMS):
        raise ValueError(f"expected 50 first pass rows per arm, got { {arm: len(rows) for arm, rows in by_arm.items()} }")
    ids = [str(row["query_index"]) for row in by_arm["off"]]
    for arm in ARMS:
        if [str(row["query_index"]) for row in by_arm[arm]] != ids:
            raise ValueError(f"arm {arm} is not paired by query index")
    return payload


def _normalize_query(query: dict[str, Any]) -> dict[str, Any]:
    result = dict(query)
    result["relevant_ids"] = [
        value if str(value).startswith("recall/") else f"recall/{value}"
        for value in query.get("relevant_ids", [])
    ]
    return result


def _row(
    source: dict[str, Any],
    *,
    model: str,
    api_key: str,
    endpoint: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    query = _normalize_query(source["query"])
    response = reasoning_response_from_dict(json.loads(source["payload"]))
    bundle = response.trusted_evidence
    system, user = render_evidence_prompt(bundle)
    evidence_keys = [
        f"{item.source}:{item.ordinal}"
        for item in bundle.items
        if item.ordinal is not None
    ]
    record: dict[str, Any] = {
        "query_index": source["query_index"],
        "query": query["query"],
        "answerable": query["answerable"],
        "arm": source["arm"],
        "evidence_ids": [item.chunk_id for item in bundle.items],
        "evidence_keys": evidence_keys,
        "retrieval_refusal_reason": source.get("refusal_reason"),
        "graph_candidates": response.diagnostics.graph_candidates_discovered,
        "graph_new_trusted_evidence": response.diagnostics.graph_relation_new_trusted_evidence,
        "answer": None,
        "citations": [],
        "answer_valid": False,
        "answer_errors": [],
        "insufficient_evidence": None,
        "provider": None,
    }
    if not bundle.items:
        record.update({"answer_valid": True, "insufficient_evidence": True})
        return record
    try:
        raw, provider = _answer_one(
            system=system,
            user=user,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            reasoning_effort="none",
            max_tokens=512,
            timeout=timeout,
            retries=retries,
        )
        envelope = parse_answer_envelope(raw)
        validation = validate_answer(envelope, bundle)
        record.update(
            {
                "answer": envelope.answer,
                "citations": list(envelope.citations),
                "answer_valid": validation.valid,
                "answer_errors": list(validation.errors),
                "insufficient_evidence": envelope.insufficient_evidence,
                "provider": provider,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["answer_errors"] = [f"{type(exc).__name__}: {exc}"]
    return record


def _citation_metrics(row: dict[str, Any]) -> dict[str, float | int | None]:
    gold = set(row.get("gold_ids") or [])
    key_by_chunk = dict(zip(row.get("evidence_ids") or [], row.get("evidence_keys") or []))
    cited_keys = [key_by_chunk[citation] for citation in row.get("citations") or [] if citation in key_by_chunk]
    matched = sum(key in gold for key in cited_keys)
    return {
        "cited_count": len(cited_keys),
        "gold_cited_count": matched,
        "gold_citation_precision": matched / len(cited_keys) if cited_keys else None,
        "gold_citation_recall": matched / len(gold) if gold else None,
        "any_gold_citation": int(matched > 0),
        "complete_gold_citation_coverage": int(bool(gold) and gold.issubset(set(cited_keys))),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["citation_metrics"] = _citation_metrics(item)
        enriched.append(item)
    n = len(enriched)
    provider_failures = sum(bool(row["answer_errors"]) for row in enriched)
    valid = sum(bool(row["answer_valid"]) for row in enriched)
    nonempty = sum(isinstance(row.get("answer"), str) and bool(row["answer"].strip()) for row in enriched)
    answerable = [row for row in enriched if row["answerable"]]
    unanswerable = [row for row in enriched if not row["answerable"]]
    correct_abstention = sum(bool(row["insufficient_evidence"]) for row in unanswerable)
    metrics = [row["citation_metrics"] for row in enriched]
    precisions = [m["gold_citation_precision"] for m in metrics if m["gold_citation_precision"] is not None]
    recalls = [m["gold_citation_recall"] for m in metrics if m["gold_citation_recall"] is not None]
    return {
        "rows": n,
        "provider_failures": provider_failures,
        "provider_failure_rate": provider_failures / n if n else None,
        "valid_answer_rate": valid / n if n else None,
        "nonempty_answer_rate": nonempty / n if n else None,
        "correct_abstention_rate_unanswerable": correct_abstention / len(unanswerable) if unanswerable else None,
        "any_gold_citation_rate": sum(int(m["any_gold_citation"]) for m in metrics) / n if n else None,
        "complete_gold_citation_coverage_rate": sum(int(m["complete_gold_citation_coverage"]) for m in metrics) / n if n else None,
        "mean_gold_citation_precision": sum(precisions) / len(precisions) if precisions else None,
        "mean_gold_citation_recall": sum(recalls) / len(recalls) if recalls else None,
        "provider_latency_ms_p50": _percentile([int(row["provider"]["latency_ms"]) for row in enriched if row.get("provider")], 0.50),
        "provider_latency_ms_p95": _percentile([int(row["provider"]["latency_ms"]) for row in enriched if row.get("provider")], 0.95),
        "prompt_tokens": sum(int(row["provider"].get("prompt_tokens") or 0) for row in enriched if row.get("provider")),
        "completion_tokens": sum(int(row["provider"].get("completion_tokens") or 0) for row in enriched if row.get("provider")),
        "answerable_rows": len(answerable),
        "unanswerable_rows": len(unanswerable),
        "rows_detail": enriched,
    }


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--query-index",
        type=int,
        action="append",
        dest="query_indices",
        help="restrict the correction pass to these paired query indices",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    payload = _load(args.input)
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    source_rows = [
        row
        for row in payload["rows"]
        if isinstance(row, dict) and row.get("pass_index") == 1 and row.get("arm") in ARMS
    ]
    if args.query_indices is not None:
        wanted = set(args.query_indices)
        source_rows = [row for row in source_rows if int(row["query_index"]) in wanted]
        found = {int(row["query_index"]) for row in source_rows}
        if found != wanted:
            raise ValueError(f"requested query indices missing from both arms: {sorted(wanted - found)}")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_row, row, model=args.model, api_key=api_key, endpoint=endpoint, timeout=args.timeout, retries=args.retries): row
            for row in source_rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"completed {index}/{len(futures)} {result['arm']} query={result['query_index']}", flush=True)
    for row in results:
        row["gold_ids"] = _normalize_query(next(source["query"] for source in source_rows if source["query_index"] == row["query_index"] and source["arm"] == row["arm"])).get("relevant_ids", [])
    by_arm = {arm: _summary([row for row in results if row["arm"] == arm]) for arm in ARMS}
    baseline = {row["query_index"]: row for row in results if row["arm"] == "off"}
    treatment = {row["query_index"]: row for row in results if row["arm"] == "one_hop"}
    paired = []
    for query_index in sorted(baseline):
        old = baseline[query_index]
        new = treatment[query_index]
        old_metric = old["citation_metrics"] = _citation_metrics(old)
        new_metric = new["citation_metrics"] = _citation_metrics(new)
        paired.append(
            {
                "query_index": query_index,
                "valid_delta": int(new["answer_valid"]) - int(old["answer_valid"]),
                "any_gold_citation_delta": new_metric["any_gold_citation"] - old_metric["any_gold_citation"],
                "gold_citation_recall_delta": (new_metric["gold_citation_recall"] or 0) - (old_metric["gold_citation_recall"] or 0),
            }
        )
    artifact = {
        "artifact": "RE-call live graph answer quality replay",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "query_set_sha256": "63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192",
        "source_commit": args.source_commit,
        "generation_id": payload.get("generation_id"),
        "model": args.model,
        "provider": "OpenRouter chat completions",
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 512,
        "answer_prompt_digest": ANSWER_PROMPT_DIGEST,
        "arms": by_arm,
        "paired": paired,
        "judge": "not run; factual correctness requires a separate fixed judge",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(results), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
