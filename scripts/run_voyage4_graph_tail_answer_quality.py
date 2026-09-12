"""Generate paired answers from the immutable Voyage 4 graph tail contexts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import ANSWER_PROMPT_DIGEST  # noqa: E402
from recall.evidence import (  # noqa: E402
    EvidenceBundle,
    EvidenceItem,
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)
from scripts.run_openrouter_answer_batch import _answer_one  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("arms"), dict):
        raise ValueError("retrieval artifact must contain arms")
    for arm in ("baseline", "structural_edges"):
        if not isinstance(payload["arms"].get(arm, {}).get("rows"), list):
            raise ValueError(f"retrieval artifact is missing {arm} rows")
    return payload


def _bundle(row: dict[str, Any]) -> tuple[EvidenceBundle, dict[str, str]]:
    items: list[EvidenceItem] = []
    benchmark_by_chunk: dict[str, str] = {}
    for context in row.get("context", []):
        chunk_id = str(context["chunk_id"])
        benchmark_id = str(context["id"])
        benchmark_by_chunk[chunk_id] = benchmark_id
        ordinal = None
        if ":" in benchmark_id:
            try:
                ordinal = int(benchmark_id.rsplit(":", 1)[1])
            except ValueError:
                pass
        items.append(
            EvidenceItem(
                chunk_id=chunk_id,
                text=str(context["text"]),
                source=benchmark_id,
                ordinal=ordinal,
                indexed_at=None,
                valid_from=None,
                valid_until=None,
                cosine=0.0,
                confidence=1.0,
            )
        )
    return (
        EvidenceBundle(
            query=str(row["question"]),
            decision="answer",
            reason_code=None,
            decision_state="supported",
            calibrated=True,
            stale=False,
            embedding_profile="voyage:voyage-4",
            retrieval_profile="voyage4_graph_tail_quality",
            index_generation="20260912-voyage4-graph-tail",
            items=tuple(items),
        ),
        benchmark_by_chunk,
    )


def _one(
    row: dict[str, Any],
    *,
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    bundle, benchmark_by_chunk = _bundle(row)
    system, user = render_evidence_prompt(bundle)
    result: dict[str, Any] = {
        "id": str(row["id"]),
        "arm": str(row["arm"]),
        "category": row.get("category"),
        "question": str(row["question"]),
        "gold": list(row.get("gold") or []),
        "context_ids": list(row.get("context_ids") or []),
        "answer": None,
        "citations": [],
        "answer_valid": False,
        "answer_errors": [],
        "insufficient_evidence": False,
        "provider": None,
    }
    try:
        raw, provider = _answer_one(
            system=system,
            user=user,
            model=model,
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            reasoning_effort="none",
            max_tokens=512,
            timeout=timeout,
            retries=retries,
        )
        envelope = parse_answer_envelope(raw)
        validation = validate_answer(envelope, bundle)
        result.update(
            {
                "answer": envelope.answer,
                "citations": list(envelope.citations),
                "answer_valid": validation.valid,
                "answer_errors": list(validation.errors),
                "insufficient_evidence": envelope.insufficient_evidence,
                "provider": provider,
            }
        )
        result["gold_citations"] = [
            benchmark_by_chunk[citation]
            for citation in envelope.citations
            if citation in benchmark_by_chunk and benchmark_by_chunk[citation] in result["gold"]
        ]
    except Exception as exc:  # noqa: BLE001 - provider failures are recorded per row
        result["answer_errors"] = [f"{type(exc).__name__}: {exc}"]
        result["gold_citations"] = []
    return result


def _metrics(row: dict[str, Any]) -> dict[str, float | int | None]:
    gold = set(str(value) for value in row.get("gold") or [])
    cited = list(row.get("gold_citations") or [])
    return {
        "any_gold_citation": int(bool(set(cited) & gold)),
        "complete_gold_citation": int(bool(gold) and gold.issubset(set(cited))),
        "gold_citation_precision": len(set(cited) & gold) / len(set(row.get("citations") or []))
        if row.get("citations")
        else None,
        "gold_citation_recall": len(set(cited) & gold) / len(gold) if gold else None,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_metrics(row) for row in rows]
    precisions = [float(value["gold_citation_precision"]) for value in values if value["gold_citation_precision"] is not None]
    recalls = [float(value["gold_citation_recall"]) for value in values if value["gold_citation_recall"] is not None]
    providers = [row["provider"] for row in rows if isinstance(row.get("provider"), dict)]
    return {
        "rows": len(rows),
        "provider_failures": sum(bool(row["answer_errors"]) for row in rows),
        "valid_answer_rate": sum(bool(row["answer_valid"]) for row in rows) / len(rows),
        "nonempty_answer_rate": sum(bool(str(row.get("answer") or "").strip()) for row in rows) / len(rows),
        "any_gold_citation_rate": statistics.fmean(int(value["any_gold_citation"]) for value in values),
        "complete_gold_citation_coverage_rate": statistics.fmean(int(value["complete_gold_citation"]) for value in values),
        "mean_gold_citation_precision": statistics.fmean(precisions) if precisions else None,
        "mean_gold_citation_recall": statistics.fmean(recalls) if recalls else None,
        "prompt_tokens": sum(int(provider.get("prompt_tokens") or 0) for provider in providers),
        "completion_tokens": sum(int(provider.get("completion_tokens") or 0) for provider in providers),
        "provider_latency_ms_p50": sorted(int(provider["latency_ms"]) for provider in providers)[len(providers) // 2] if providers else None,
        "provider_latency_ms_p95": sorted(int(provider["latency_ms"]) for provider in providers)[min(len(providers) - 1, int((len(providers) - 1) * 0.95))] if providers else None,
    }


def _bootstrap(values: list[int]) -> tuple[float, float, float]:
    rng = random.Random(20260912)
    samples = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(10000)]
    samples.sort()
    return statistics.fmean(values), samples[25], samples[9974]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    payload = _load(args.input)
    source_rows: list[dict[str, Any]] = []
    for arm_name, artifact_arm in (("baseline", "baseline"), ("structural_edges", "structural_edges")):
        for row in payload["arms"][artifact_arm]["rows"]:
            source_rows.append({**row, "arm": arm_name})
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_one, row, model=args.model, api_key=api_key, timeout=args.timeout, retries=args.retries): row
            for row in source_rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"completed {index}/{len(futures)} {result['arm']} {result['id']}", flush=True)
    by_arm = {arm: [row for row in results if row["arm"] == arm] for arm in ("baseline", "structural_edges")}
    for arm in by_arm:
        by_arm[arm].sort(key=lambda row: row["id"])
    baseline = {row["id"]: row for row in by_arm["baseline"]}
    treatment = {row["id"]: row for row in by_arm["structural_edges"]}
    ids = sorted(set(baseline) & set(treatment))
    if len(ids) != len(by_arm["baseline"]) or len(ids) != len(by_arm["structural_edges"]):
        raise ValueError("answer arms are not paired")
    complete_deltas = [int(_metrics(treatment[key])["complete_gold_citation"]) - int(_metrics(baseline[key])["complete_gold_citation"]) for key in ids]
    any_deltas = [int(_metrics(treatment[key])["any_gold_citation"]) - int(_metrics(baseline[key])["any_gold_citation"]) for key in ids]
    payload_out = {
        "protocol": "2026-09-12-voyage4-graph-tail-answer-quality",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": args.model,
        "provider": "OpenRouter chat completions",
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 512,
        "answer_prompt_digest": ANSWER_PROMPT_DIGEST,
        "arms": {arm: {"summary": _summary(rows), "rows": rows} for arm, rows in by_arm.items()},
        "paired": {
            "questions": len(ids),
            "complete_gold_citation_delta": {"mean": _bootstrap(complete_deltas)[0], "lo95": _bootstrap(complete_deltas)[1], "hi95": _bootstrap(complete_deltas)[2]},
            "any_gold_citation_delta": {"mean": _bootstrap(any_deltas)[0], "lo95": _bootstrap(any_deltas)[1], "hi95": _bootstrap(any_deltas)[2]},
            "complete_rescues": sum(value > 0 for value in complete_deltas),
            "complete_regressions": sum(value < 0 for value in complete_deltas),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "questions": len(ids), "paired": payload_out["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
