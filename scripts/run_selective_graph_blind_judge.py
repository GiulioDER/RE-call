"""Blindly judge paired baseline and selective graph answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_openrouter_answer_batch import _answer_one  # noqa: E402

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
SEED = 20260912
UNCHANGED_SAMPLE = 138
JUDGE_PROMPT = """You are a blind evaluator of two answers to a memory question.
Use only the question and reference evidence below. Do not infer which answer came from which
retrieval system. Evaluate each answer independently.

Return one JSON object and no other text with exactly these integer fields:
correctness_a, support_a, completeness_a, unsupported_claims_a,
correctness_b, support_b, completeness_b, unsupported_claims_b.

For correctness, support, and completeness use 0 for absent or wrong, 1 for partial, and 2 for
fully correct, supported, or complete. Count unsupported factual claims as a nonnegative integer.
An answer that appropriately says the evidence is insufficient can receive full correctness and
support when the evidence is genuinely insufficient.

=== QUESTION ===
{question}

=== REFERENCE EVIDENCE ===
{evidence}

=== ANSWER A ===
{answer_a}

=== ANSWER B ===
{answer_b}
"""


def _complete(row: dict[str, Any]) -> bool:
    gold = set(str(value) for value in row.get("gold") or [])
    cited = set(str(value) for value in row.get("gold_citations") or [])
    return bool(gold) and gold.issubset(cited)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "arms" not in payload:
        raise ValueError("answer artifact must contain arms")
    for arm in ("baseline", "selective_tail"):
        if not isinstance(payload["arms"].get(arm, {}).get("rows"), list):
            raise ValueError(f"answer artifact is missing {arm} rows")
    return payload


def _pairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = {str(row["id"]): row for row in payload["arms"]["baseline"]["rows"]}
    selective = {str(row["id"]): row for row in payload["arms"]["selective_tail"]["rows"]}
    ids = sorted(set(baseline) & set(selective))
    rescues = [item for item in ids if _complete(selective[item]) and not _complete(baseline[item])]
    regressions = [item for item in ids if _complete(baseline[item]) and not _complete(selective[item])]
    unchanged = [item for item in ids if item not in set(rescues) | set(regressions)]
    rng = random.Random(SEED)
    rng.shuffle(unchanged)
    sampled = rescues + regressions + unchanged[:UNCHANGED_SAMPLE]
    return [
        {
            "id": item,
            "bucket": "rescue" if item in rescues else "regression" if item in regressions else "unchanged",
            "baseline": baseline[item],
            "selective": selective[item],
        }
        for item in sampled
    ]


def _evidence(pair: dict[str, Any], retrieval: dict[str, Any]) -> str:
    rows = {arm: {str(row["id"]): row for row in retrieval["arms"][arm]["rows"]} for arm in ("baseline", "selective_tail")}
    seen: set[str] = set()
    parts: list[str] = []
    for arm in ("baseline", "selective_tail"):
        for context in rows[arm][pair["id"]].get("context", []):
            evidence_id = str(context["id"])
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            parts.append(f"[{evidence_id}] {context['text']}")
    return "\n\n".join(parts)


def _parse(raw: str) -> dict[str, int]:
    candidate = raw.strip()
    if "```" in candidate:
        candidate = candidate.replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge response is not JSON")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response is not an object")
    fields = (
        "correctness_a", "support_a", "completeness_a", "unsupported_claims_a",
        "correctness_b", "support_b", "completeness_b", "unsupported_claims_b",
    )
    result: dict[str, int] = {}
    for field in fields:
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"{field} must be an integer")
        if field.startswith("unsupported"):
            if number < 0:
                raise ValueError(f"{field} must be nonnegative")
        elif number not in (0, 1, 2):
            raise ValueError(f"{field} must be 0, 1, or 2")
        result[field] = number
    return result


def _one(pair: dict[str, Any], retrieval: dict[str, Any], *, key: str, timeout: float, retries: int) -> dict[str, Any]:
    baseline_answer = str(pair["baseline"].get("answer") or "")
    selective_answer = str(pair["selective"].get("answer") or "")
    labels = [("baseline", baseline_answer), ("selective_tail", selective_answer)]
    random.Random(f"{SEED}:{pair['id']}").shuffle(labels)
    prompt = JUDGE_PROMPT.format(
        question=pair["baseline"]["question"],
        evidence=_evidence(pair, retrieval),
        answer_a=labels[0][1],
        answer_b=labels[1][1],
    )
    output: dict[str, Any] = {
        "id": pair["id"],
        "bucket": pair["bucket"],
        "category": pair["baseline"].get("category"),
        "baseline_label": "a" if labels[0][0] == "baseline" else "b",
        "selective_label": "a" if labels[0][0] == "selective_tail" else "b",
        "scores": None,
        "judge_error": None,
        "provider": None,
    }
    try:
        raw, provider = _answer_one(
            system="Return only the requested JSON object.",
            user=prompt,
            model=JUDGE_MODEL,
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key=key,
            reasoning_effort="none",
            max_tokens=256,
            timeout=timeout,
            retries=retries,
        )
        output["scores"] = _parse(raw)
        output["provider"] = provider
    except Exception as exc:  # noqa: BLE001 - failures remain explicit in the artifact
        output["judge_error"] = f"{type(exc).__name__}: {exc}"
    return output


def _totals(row: dict[str, Any]) -> tuple[int, int, int, int]:
    scores = row["scores"]
    assert isinstance(scores, dict)
    a = sum(scores[f"{field}_a"] for field in ("correctness", "support", "completeness"))
    b = sum(scores[f"{field}_b"] for field in ("correctness", "support", "completeness"))
    unsupported_a = scores["unsupported_claims_a"]
    unsupported_b = scores["unsupported_claims_b"]
    if row["baseline_label"] == "a":
        return a, b, unsupported_a, unsupported_b
    return b, a, unsupported_b, unsupported_a


def _bootstrap(values: list[int]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    samples = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(10000)]
    samples.sort()
    return statistics.fmean(values), samples[25], samples[9974]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    import os

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    answers = _load(args.answers)
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    pairs = _pairs(answers)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_one, pair, retrieval, key=key, timeout=args.timeout, retries=args.retries): pair
            for pair in pairs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"completed {index}/{len(futures)} {results[-1]['id']}", flush=True)
    results.sort(key=lambda row: row["id"])
    valid = [row for row in results if isinstance(row.get("scores"), dict)]
    deltas = [_totals(row)[1] - _totals(row)[0] for row in valid]
    unsupported_deltas = [_totals(row)[3] - _totals(row)[2] for row in valid]
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        by_bucket.setdefault(str(row["bucket"]), []).append(row)
    payload = {
        "protocol": "2026-09-12-selective-graph-blind-judge",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "answers_artifact": str(args.answers),
        "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest(),
        "retrieval_artifact": str(args.retrieval),
        "retrieval_sha256": hashlib.sha256(args.retrieval.read_bytes()).hexdigest(),
        "judge_model": JUDGE_MODEL,
        "provider": "OpenRouter chat completions",
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 256,
        "sample_seed": SEED,
        "unchanged_sample": UNCHANGED_SAMPLE,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
        "sample": {
            "pairs": len(pairs),
            "rescue": sum(pair["bucket"] == "rescue" for pair in pairs),
            "regression": sum(pair["bucket"] == "regression" for pair in pairs),
            "unchanged": sum(pair["bucket"] == "unchanged" for pair in pairs),
        },
        "summary": {
            "pairs": len(pairs),
            "valid_judgments": len(valid),
            "judge_failures": len(results) - len(valid),
            "valid_judgment_rate": len(valid) / len(results) if results else 0.0,
            "mean_total_score_baseline": statistics.fmean(_totals(row)[0] for row in valid) if valid else None,
            "mean_total_score_selective": statistics.fmean(_totals(row)[1] for row in valid) if valid else None,
            "mean_total_score_delta": _bootstrap(deltas)[0] if deltas else None,
            "total_score_delta_lo95": _bootstrap(deltas)[1] if deltas else None,
            "total_score_delta_hi95": _bootstrap(deltas)[2] if deltas else None,
            "mean_unsupported_claim_delta": statistics.fmean(unsupported_deltas) if valid else None,
        },
        "by_bucket": {
            bucket: {
                "pairs": len(rows),
                "mean_total_score_baseline": statistics.fmean(_totals(row)[0] for row in rows),
                "mean_total_score_selective": statistics.fmean(_totals(row)[1] for row in rows),
                "mean_total_score_delta": statistics.fmean(_totals(row)[1] - _totals(row)[0] for row in rows),
            }
            for bucket, rows in sorted(by_bucket.items())
        },
        "rows": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs), "valid": len(valid)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
