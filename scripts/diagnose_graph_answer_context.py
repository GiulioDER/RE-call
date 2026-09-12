"""Measure answer output reliability while bounding graph evidence context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import ANSWER_PROMPT_DIGEST  # noqa: E402
from recall.evidence import parse_answer_envelope, render_evidence_prompt, validate_answer  # noqa: E402
from recall.reasoning import reasoning_response_from_dict  # noqa: E402


CAPS: tuple[tuple[str, int | None], ...] = (("full", None), ("cap5", 5), ("cap3", 3))


def _post_json(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/recall",
            "X-Title": "RE-call graph answer context diagnostic",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provider returned a non-object response")
    return value


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
    return ""


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, dict)
        and row.get("pass_index") == 1
        and row.get("arm") == "one_hop"
        and bool(row.get("query", {}).get("answerable"))
    ]
    if len(rows) != 22:
        raise ValueError(f"expected 22 answerable one hop rows, got {len(rows)}")
    return rows


def _row(
    source: dict[str, Any],
    *,
    cap_name: str,
    cap: int | None,
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    response = reasoning_response_from_dict(json.loads(source["payload"]))
    bundle = response.trusted_evidence
    original_count = len(bundle.items)
    if cap is not None:
        bundle = replace(bundle, items=bundle.items[:cap])
    system, user = render_evidence_prompt(bundle)
    record: dict[str, Any] = {
        "query_index": source["query_index"],
        "query": source["query"]["query"],
        "cap": cap_name,
        "original_evidence_items": original_count,
        "evidence_items": len(bundle.items),
        "prompt_chars": len(system) + len(user),
        "answer": None,
        "citations": [],
        "answer_valid": False,
        "answer_errors": [],
        "provider": None,
        "raw_provider_response": None,
    }
    if not bundle.items:
        record.update({"answer_valid": True, "insufficient_evidence": True})
        return record
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "none"},
    }
    last_response: dict[str, Any] | None = None
    last_error = "provider request failed"
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            last_response = _post_json(
                "https://openrouter.ai/api/v1/chat/completions",
                api_key,
                payload,
                timeout,
            )
            raw = _message_text(last_response)
            if not raw:
                raise ValueError("provider returned empty content")
            envelope = parse_answer_envelope(raw)
            validation = validate_answer(envelope, bundle)
            usage = last_response.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            record.update(
                {
                    "answer": envelope.answer,
                    "citations": list(envelope.citations),
                    "answer_valid": validation.valid,
                    "answer_errors": list(validation.errors),
                    "insufficient_evidence": envelope.insufficient_evidence,
                    "provider": {
                        "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                        "model_revision": str(last_response.get("model") or "unreported"),
                    },
                }
            )
            return record
        except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt + 1 < retries:
                time.sleep(min(30.0, 2.0**attempt))
    record["answer_errors"] = [last_error]
    record["raw_provider_response"] = last_response
    return record


def _citation_metrics(row: dict[str, Any]) -> dict[str, float | int | None]:
    source = next(
        item
        for item in _SOURCE_ROWS
        if item["query_index"] == row["query_index"]
    )
    gold = {
        value if str(value).startswith("recall/") else f"recall/{value}"
        for value in source["query"].get("relevant_ids", [])
    }
    response = reasoning_response_from_dict(json.loads(source["payload"]))
    items = response.trusted_evidence.items
    cap = next(cap for name, cap in CAPS if name == row["cap"])
    if cap is not None:
        items = items[:cap]
    by_chunk = {item.chunk_id: f"{item.source}:{item.ordinal}" for item in items if item.ordinal is not None}
    cited_keys = [by_chunk[citation] for citation in row.get("citations", []) if citation in by_chunk]
    matched = sum(key in gold for key in cited_keys)
    return {
        "gold_cited_count": matched,
        "gold_citation_precision": matched / len(cited_keys) if cited_keys else None,
        "gold_citation_recall": matched / len(gold) if gold else None,
        "any_gold_citation": int(matched > 0),
    }


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        row["citation_metrics"] = _citation_metrics(row)
    n = len(rows)
    metrics = [row["citation_metrics"] for row in rows]
    precision = [m["gold_citation_precision"] for m in metrics if m["gold_citation_precision"] is not None]
    recall = [m["gold_citation_recall"] for m in metrics if m["gold_citation_recall"] is not None]
    failures = [row for row in rows if row["answer_errors"]]
    provider_rows = [row for row in rows if row.get("provider")]
    return {
        "rows": n,
        "provider_failures": len(failures),
        "provider_failure_rate": len(failures) / n if n else None,
        "failure_types": dict(Counter(error.split(":", 1)[0] for row in failures for error in row["answer_errors"])),
        "valid_answer_rate": sum(bool(row["answer_valid"]) for row in rows) / n if n else None,
        "nonempty_answer_rate": sum(isinstance(row.get("answer"), str) and bool(row["answer"].strip()) for row in rows) / n if n else None,
        "any_gold_citation_rate": sum(int(m["any_gold_citation"]) for m in metrics) / n if n else None,
        "mean_gold_citation_precision": sum(precision) / len(precision) if precision else None,
        "mean_gold_citation_recall": sum(recall) / len(recall) if recall else None,
        "prompt_chars_p50": _percentile([int(row["prompt_chars"]) for row in rows], 0.50),
        "prompt_chars_p95": _percentile([int(row["prompt_chars"]) for row in rows], 0.95),
        "evidence_items_p50": _percentile([int(row["evidence_items"]) for row in rows], 0.50),
        "evidence_items_p95": _percentile([int(row["evidence_items"]) for row in rows], 0.95),
        "provider_latency_ms_p50": _percentile([int(row["provider"]["latency_ms"]) for row in provider_rows], 0.50),
        "provider_latency_ms_p95": _percentile([int(row["provider"]["latency_ms"]) for row in provider_rows], 0.95),
        "rows_detail": rows,
    }


_SOURCE_ROWS: list[dict[str, Any]] = []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    global _SOURCE_ROWS
    _SOURCE_ROWS = _load(args.input)
    tasks = [(source, name, cap) for source in _SOURCE_ROWS for name, cap in CAPS]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _row,
                source,
                cap_name=name,
                cap=cap,
                model=args.model,
                api_key=api_key,
                timeout=args.timeout,
                retries=args.retries,
            ): (source, name)
            for source, name, cap in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"completed {index}/{len(tasks)} {result['cap']} query={result['query_index']}", flush=True)
    artifact = {
        "artifact": "RE-call graph answer context reliability diagnostic",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_commit": args.source_commit,
        "generation_id": "gen_6559b6dbacfc4bc2847b65005f1ba1d4",
        "model": args.model,
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 512,
        "answer_prompt_digest": ANSWER_PROMPT_DIGEST,
        "caps": {name: _summary([row for row in results if row["cap"] == name]) for name, _cap in CAPS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(results), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
