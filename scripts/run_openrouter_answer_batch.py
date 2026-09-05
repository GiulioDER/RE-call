"""Replay frozen reasoning responses through an OpenRouter answer model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import ANSWER_PROMPT_DIGEST  # noqa: E402
from recall.evidence import (  # noqa: E402
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)
from recall.reasoning import reasoning_response_from_dict  # noqa: E402


def _read_payloads(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = json.loads(base64.b64decode(raw).decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input must contain a JSON list")
    return payload


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(
            item["text"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ).strip()
    return ""


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, object],
    timeout: float,
) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openai/recall",
        "X-Title": "RE-call graph answer model replay",
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        response_body = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_body, dict):
            raise ValueError("provider returned a non-object response")
        return response_body, int(response.status)


def _answer_one(
    *,
    system: str,
    user: str,
    model: str,
    endpoint: str,
    api_key: str,
    reasoning_effort: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    last_error = "provider request failed"
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            response, _status = _post_json(
                url=endpoint,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("provider returned no choices")
            message = choices[0].get("message")
            content = _message_text(message.get("content") if isinstance(message, dict) else None)
            if not content:
                raise ValueError("provider returned empty content")
            usage = response.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            provider = {
                "provider_id": "openrouter",
                "model_id": model,
                "model_revision": str(response.get("model") or "unreported"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "monetary_cost_usd": None,
                "prompt_digest": ANSWER_PROMPT_DIGEST,
            }
            return content, provider
        except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, error.HTTPError):
                detail = exc.read(512).decode("utf-8", errors="replace").replace("\n", " ")
                last_error = f"HTTP {exc.code}: {detail[:240]}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"[:280]
            if attempt + 1 < retries:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"answer generation failed after {retries} attempts: {last_error}")


def _row(
    item: dict[str, Any],
    *,
    model: str,
    endpoint: str,
    api_key: str,
    reasoning_effort: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    query = item["query"]
    response = reasoning_response_from_dict(json.loads(item["payload"]))
    bundle = response.trusted_evidence
    gold = set(query.get("relevant_ids", []))
    evidence_keys = [
        f"{evidence_item.source}:{evidence_item.ordinal}"
        for evidence_item in bundle.items
        if evidence_item.ordinal is not None
    ]
    matched = sum(key in gold for key in evidence_keys)
    record: dict[str, object] = {
        "query": query["query"],
        "answerable": query["answerable"],
        "gold_ids": query.get("relevant_ids", []),
        "arm": item["arm"],
        "evidence_ids": [evidence_item.chunk_id for evidence_item in bundle.items],
        "evidence_keys": evidence_keys,
        "evidence_recall": matched / len(gold) if gold else None,
        "evidence_precision": matched / len(evidence_keys) if evidence_keys else None,
        "trusted_items": len(bundle.items),
        "retrieval_latency_ms": response.diagnostics.latency_ms,
        "graph_latency_ms": response.diagnostics.graph_expansion_latency_ms,
        "graph_entities": response.diagnostics.graph_entities_inspected,
        "graph_relations": response.diagnostics.graph_relations_inspected,
        "graph_candidates": response.diagnostics.graph_candidates_discovered,
        "graph_rejected": response.diagnostics.graph_candidates_rejected,
        "graph_diagnostics": response.diagnostics.graph_diagnostics_encountered,
        "response_refusal_reason": response.refusal_reason,
        "bundle_decision": bundle.decision,
        "answer_provider_invoked": False,
        "model": model,
        "human_review": None,
    }
    if bundle.decision != "answer" or not bundle.items:
        record.update(
            {
                "answer": None,
                "citations": [],
                "insufficient_evidence": True,
                "answer_valid": True,
                "answer_errors": [],
                "provider": None,
            }
        )
        return record

    record["answer_provider_invoked"] = True
    system, user = render_evidence_prompt(bundle)
    try:
        raw, provider = _answer_one(
            system=system,
            user=user,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )
        envelope = parse_answer_envelope(raw)
        validation = validate_answer(envelope, bundle)
        record.update(
            {
                "answer": envelope.answer,
                "citations": list(envelope.citations),
                "insufficient_evidence": envelope.insufficient_evidence,
                "answer_valid": validation.valid,
                "answer_errors": list(validation.errors),
                "provider": provider,
            }
        )
    except Exception as exc:
        record.update(
            {
                "answer": None,
                "citations": [],
                "insufficient_evidence": None,
                "answer_valid": False,
                "answer_errors": [f"{type(exc).__name__}: {exc}"],
                "provider": None,
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="medium")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.retries < 1 or args.timeout <= 0:
        raise SystemExit("max tokens, retries, and timeout must be positive")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    items = _read_payloads(args.input)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        print(f"answer {index}/{len(items)} {item['arm']} {item['query']['query']}", flush=True)
        rows.append(
            _row(
                item,
                model=args.model,
                endpoint=endpoint,
                api_key=api_key,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
        )
    artifact = {
        "artifact": "RE-call frozen graph answer model replay",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "answer_provider": "OpenRouter chat completions",
        "judge": "human review required, no model judge used",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
