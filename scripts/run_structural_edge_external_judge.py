"""Judge structural edge answer artifacts with one fixed external OpenRouter judge.

This is deliberately a scoring replay. It does not retrieve evidence or regenerate answers.
It joins stored answers to the checked out LoCoMo gold answers, skips rows that failed the locked
answer envelope or abstained, and uses the same judge prompt and denominator for every arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.llm import OpenRouterLLM  # noqa: E402
from benchmarks.pipeline import JUDGE_SYSTEM_PROMPT  # noqa: E402
from benchmarks.run import validate_openrouter_key  # noqa: E402
from recall._env import load_dotenv  # noqa: E402


ARMS = (
    "baseline",
    "semantic_order_more_direct",
    "semantic_order_most_direct",
    "category_selective",
)
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gold_by_id(dataset_path: Path) -> dict[str, dict[str, Any]]:
    dataset = _load_json(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError("LoCoMo dataset must be a JSON array")
    gold: dict[str, dict[str, Any]] = {}
    for conversation_index, sample in enumerate(dataset):
        if not isinstance(sample, dict) or not isinstance(sample.get("qa"), list):
            raise ValueError(f"conversation_{conversation_index} has no qa list")
        for qa_index, qa in enumerate(sample["qa"]):
            if not isinstance(qa, dict) or "question" not in qa:
                raise ValueError(f"conversation_{conversation_index}:qa_{qa_index} is malformed")
            # LoCoMo category 5 is an adversarial refusal control and intentionally has no gold
            # answer. The structural answer artifact being scored contains categories 1 through 4.
            if qa.get("category") == 5 or "answer" not in qa:
                continue
            question_id = f"conversation_{conversation_index}:qa_{qa_index}"
            gold[question_id] = {
                "question": str(qa["question"]),
                "gold": str(qa["answer"]),
                "category": qa.get("category"),
                "adversarial": bool(qa.get("adversarial", False)),
            }
    return gold


def _load_answer_artifact(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("answer artifact must contain a rows array")
    if tuple(payload.get("arms") or ()) != ARMS:
        raise ValueError("answer artifact arms do not match the locked structural edge arms")
    return cast(dict[str, Any], payload)


def _judge_prompt(question: str, gold: str, answer: str) -> str:
    return f"Question: {question}\nGold answer: {gold}\nPredicted answer: {answer}\nCorrect?"


def _prompt_key(question: str, gold: str, answer: str) -> str:
    material = JUDGE_SYSTEM_PROMPT + "\0" + _judge_prompt(question, gold, answer)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _strict_verdict(raw: str) -> bool:
    verdict = raw.strip().casefold()
    if verdict.startswith("yes"):
        return True
    if verdict.startswith("no"):
        return False
    raise ValueError(f"judge returned neither YES nor NO: {raw[:160]!r}")


def _eligible(row: dict[str, Any]) -> tuple[bool, str | None]:
    if not bool(row.get("answer_valid")):
        return False, "invalid_answer_envelope"
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "empty_answer"
    if bool(row.get("insufficient_evidence")):
        return False, "insufficient_evidence"
    return True, None


def _validate_pairing(rows_by_arm: dict[str, list[dict[str, Any]]]) -> None:
    baseline_ids = [str(row.get("id")) for row in rows_by_arm["baseline"]]
    for arm, rows in rows_by_arm.items():
        ids = [str(row.get("id")) for row in rows]
        if ids != baseline_ids[: len(ids)]:
            raise ValueError(f"arm {arm} is not paired with baseline question order")
    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("baseline contains duplicate question ids")


def _judge_one(
    item: tuple[str, dict[str, Any], str, str, str],
    judge: OpenRouterLLM,
) -> dict[str, Any]:
    key, row, question, gold, answer = item
    started = time.perf_counter()
    try:
        raw = judge.complete(JUDGE_SYSTEM_PROMPT, _judge_prompt(question, gold, answer))
        verdict = _strict_verdict(raw)
        return {
            "_prompt_key": key,
            "judge_invoked": True,
            "judge_verdict": verdict,
            "judge_raw": raw,
            "judge_error": None,
            "judge_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "_prompt_key": key,
            "judge_invoked": True,
            "judge_verdict": None,
            "judge_raw": None,
            "judge_error": f"{type(exc).__name__}: {exc}",
            "judge_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }


def _row_with_score(
    row: dict[str, Any],
    gold: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    cache_hit: bool,
) -> dict[str, Any]:
    question = str(row["question"])
    if question != gold["question"]:
        raise ValueError(f"question text mismatch for {row['id']}")
    scored = dict(row)
    eligible, skip_reason = _eligible(row)
    scored.update(
        {
            "gold_answer": gold["gold"],
            "judge_eligible": eligible,
            "judge_skipped_reason": skip_reason,
            "judge_cache_hit": False,
            "judge_invoked": False,
            "judge_verdict": None,
            "judge_raw": None,
            "judge_error": None,
            "judge_latency_ms": None,
            "correct": False,
        }
    )
    if not eligible:
        return scored
    if result is None:
        raise ValueError(f"missing judge result for eligible row {row['id']}")
    scored.update({key: value for key, value in result.items() if key != "_prompt_key"})
    scored["judge_cache_hit"] = cache_hit
    scored["correct"] = result.get("judge_verdict") is True
    return scored


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    eligible = sum(bool(row["judge_eligible"]) for row in rows)
    invoked = sum(bool(row["judge_invoked"]) for row in rows)
    successful = sum(row["judge_verdict"] is not None for row in rows)
    correct = sum(row["correct"] is True for row in rows)
    errors = sum(row["judge_error"] is not None for row in rows)
    latencies = sorted(
        int(row["judge_latency_ms"])
        for row in rows
        if row.get("judge_latency_ms") is not None
    )
    skips = Counter(
        str(row["judge_skipped_reason"])
        for row in rows
        if row.get("judge_skipped_reason") is not None
    )

    def percentile(values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        index = min(len(values) - 1, int((len(values) - 1) * fraction))
        return values[index]

    return {
        "rows": total,
        "valid_answer_rate": sum(bool(row.get("answer_valid")) for row in rows) / total
        if total
        else None,
        "judge_eligible": eligible,
        "judge_invoked_rows": invoked,
        "judge_successful_rows": successful,
        "judge_failures": errors,
        "judge_cache_hits": sum(bool(row["judge_cache_hit"]) for row in rows),
        "correct": correct,
        "answer_score_all_rows": correct / total if total else None,
        "judge_accuracy_successful_only": correct / successful if successful else None,
        "judge_coverage_eligible": successful / eligible if eligible else None,
        "skip_reasons": dict(skips),
        "judge_latency_ms_median": percentile(latencies, 0.50),
        "judge_latency_ms_p95": percentile(latencies, 0.95),
    }


def _paired(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    baseline = {str(row["id"]): row for row in rows_by_arm["baseline"]}
    report: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        if arm == "baseline":
            continue
        paired = [(baseline[str(row["id"])], row) for row in rows]
        differences = [
            int(row["correct"] is True) - int(old["correct"] is True) for old, row in paired
        ]
        mean_difference = sum(differences) / len(differences) if differences else 0.0
        variance = (
            sum((difference - mean_difference) ** 2 for difference in differences)
            / (len(differences) - 1)
            if len(differences) > 1
            else 0.0
        )
        standard_error = (variance / len(differences)) ** 0.5 if differences else 0.0
        report[arm] = {
            "rows": len(paired),
            "treatment_minus_baseline_score_points": round(
                100 * mean_difference,
                4,
            )
            if paired
            else None,
            "treatment_minus_baseline_score_points_ci95_normal": [
                round(100 * (mean_difference - 1.96 * standard_error), 4),
                round(100 * (mean_difference + 1.96 * standard_error), 4),
            ]
            if paired
            else None,
            "rescued_baseline_misses": sum(
                old["correct"] is not True and row["correct"] is True for old, row in paired
            ),
            "regressed_baseline_correct": sum(
                old["correct"] is True and row["correct"] is not True for old, row in paired
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "locomo10.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=None, help="same first N questions per arm")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    if args.output.resolve() == args.input.resolve():
        parser.error("output must differ from input")
    if args.output.exists() and not args.resume:
        parser.error(f"{args.output} already exists, use a new output or --resume")
    if not args.input.exists():
        parser.error(f"{args.input} not found")
    if not args.dataset.exists():
        parser.error(f"{args.dataset} not found")

    load_dotenv()
    try:
        key = validate_openrouter_key(os.environ.get("OPENROUTER_API_KEY"))
    except ValueError as exc:
        parser.error(str(exc))

    artifact = _load_answer_artifact(args.input)
    gold_by_id = _gold_by_id(args.dataset)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        arm_rows = artifact["rows"]
        selected = [row for row in arm_rows if str(row.get("arm")) == arm]
        selected = selected[: args.limit] if args.limit is not None else selected
        if not selected:
            raise ValueError(f"arm {arm} has no rows")
        rows_by_arm[arm] = [cast(dict[str, Any], row) for row in selected]
    _validate_pairing(rows_by_arm)

    groups: dict[str, tuple[str, dict[str, Any], dict[str, Any], str]] = {}
    group_counts: Counter[str] = Counter()
    for arm, rows in rows_by_arm.items():
        for row in rows:
            question_id = str(row["id"])
            if question_id not in gold_by_id:
                raise ValueError(f"no LoCoMo gold row for {question_id}")
            gold = gold_by_id[question_id]
            question = str(row["question"])
            if question != gold["question"]:
                raise ValueError(f"question text mismatch for {question_id}")
            eligible, _ = _eligible(row)
            if not eligible:
                continue
            answer = str(row["answer"])
            key_for_prompt = _prompt_key(question, gold["gold"], answer)
            group_counts[key_for_prompt] += 1
            groups.setdefault(key_for_prompt, (arm, row, gold, answer))

    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            if isinstance(saved, dict) and isinstance(saved.get("_prompt_key"), str):
                completed[saved["_prompt_key"]] = saved

    pending = [
        (key_for_prompt, row, gold["question"], gold["gold"], answer)
        for key_for_prompt, (_, row, gold, answer) in groups.items()
        if key_for_prompt not in completed
    ]
    print(
        f"judge groups {len(groups)}, completed checkpoint {len(completed)}, "
        f"pending {len(pending)}, workers {args.workers}, model {args.model}",
        flush=True,
    )
    judge = OpenRouterLLM(
        model=args.model,
        api_key=key,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=8,
        max_attempts=3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_judge_one, item, judge): item[0] for item in pending}
        with checkpoint.open("a", encoding="utf-8") as handle:
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                completed[futures[future]] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"completed {index}/{len(pending)}", flush=True)

    result_rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            question_id = str(row["id"])
            gold = gold_by_id[question_id]
            eligible, _ = _eligible(row)
            result = None
            key_for_prompt = None
            if eligible:
                key_for_prompt = _prompt_key(str(row["question"]), gold["gold"], str(row["answer"]))
                result = completed[key_for_prompt]
            scored = _row_with_score(
                row,
                gold,
                result,
                cache_hit=bool(key_for_prompt and group_counts[key_for_prompt] > 1),
            )
            result_rows_by_arm[arm].append(scored)

    metadata = judge.provider_metadata().to_dict()
    usage = judge.usage()
    flat_rows = [row for rows in result_rows_by_arm.values() for row in rows]
    summaries = {arm: _summary(rows) for arm, rows in result_rows_by_arm.items()}
    report = {
        "artifact": "RE-call structural edge external judge answer evaluation",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source_commit": artifact.get("source_commit"),
        "preregistration": "docs/preregistrations/2026-09-11-structural-edge-performance.md",
        "arms": list(ARMS),
        "questions_per_arm": len(rows_by_arm["baseline"]),
        "judge_model_requested": args.model,
        "judge_model_returned": metadata.get("model_id"),
        "judge_provider": "OpenRouter",
        "judge_base_url": args.base_url,
        "judge_temperature": 0.0,
        "judge_max_tokens": 8,
        "judge_prompt": "benchmarks.pipeline.JUDGE_SYSTEM_PROMPT",
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "judge_groups": len(groups),
        # `provider_metadata()` intentionally omits the call counter. On a fresh run the usage
        # counter is authoritative; on a resumed run the checkpoint has already captured the
        # completed unique prompt groups, so the group count remains the durable logical count.
        "judge_provider_calls": usage.get("calls") or len(groups),
        "judge_provider_usage": usage,
        "judge_provider_summary": metadata,
        "summaries": summaries,
        "paired": _paired(result_rows_by_arm),
        "rows": flat_rows,
        "notes": [
            "The denominator for answer_score_all_rows is every paired row. Invalid envelopes, empty answers, and insufficient evidence are false for answerable LoCoMo rows.",
            "judge_accuracy_successful_only is conditional on a successful external YES or NO response and is not the primary all row score.",
            "Exact duplicate question, gold, and answer prompts reuse one external verdict across arms.",
            "The closed hypothesis search was unavailable because its local database connection was refused. The locked project preregistration and trusted memory rule were used instead.",
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(flat_rows), "output": str(args.output), "summaries": summaries}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
