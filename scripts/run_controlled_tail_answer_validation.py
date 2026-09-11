"""Replay controlled-tail contexts through the configured answer provider and citation validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recall.answer_provider import resolve_answer_provider  # noqa: E402
from scripts.run_structural_edge_answer_validation import (  # noqa: E402
    _answer_row,
    _request_key,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("arms"), dict):
        raise ValueError("input must be a retrieval artifact with arms")
    arms = payload["arms"]
    if not arms:
        raise ValueError("input contains no arms")
    for name, value in arms.items():
        if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
            raise ValueError(f"arm {name} has no rows")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="same first N questions per arm")
    parser.add_argument("--workers", type=int, default=8, help="bounded concurrent provider calls")
    parser.add_argument("--resume", action="store_true", help="reuse completed checkpoint rows")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    payload = _load(args.input)
    provider = resolve_answer_provider(os.environ)
    if provider is None:
        raise SystemExit("answer provider is disabled; set RECALL_REASONING_ANSWER_ENABLED=1")
    arms = list(payload["arms"])
    baseline_arm = next((arm for arm in arms if arm.startswith("baseline")), arms[0])
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        rows = payload["arms"][arm]["rows"]
        selected = rows[: args.limit] if args.limit is not None else rows
        rows_by_arm[arm] = selected
    baseline_ids = [str(row["id"]) for row in rows_by_arm[baseline_arm]]
    for arm, rows in rows_by_arm.items():
        ids = [str(row["id"]) for row in rows]
        if ids != baseline_ids[: len(ids)]:
            raise ValueError(f"arm {arm} is not paired with {baseline_arm}")

    groups: dict[str, tuple[str, dict[str, Any]]] = {}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            groups.setdefault(_request_key(row), (arm, row))
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed: dict[str, dict[str, object]] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            if saved.get("_type") == "row" and isinstance(saved.get("_prompt_key"), str):
                completed[saved["_prompt_key"]] = saved
    pending = [(key, arm, row) for key, (arm, row) in groups.items() if key not in completed]
    print(
        f"provider groups {len(groups)}, completed checkpoint {len(completed)}, "
        f"pending {len(pending)}, workers {args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _answer_row,
                row,
                provider,
                arm=arm,
                source_commit=payload.get("git_revision"),
            ): (key, arm, row)
            for key, arm, row in pending
        }
        with checkpoint.open("a", encoding="utf-8") as handle:
            for index, future in enumerate(as_completed(futures), start=1):
                key, arm, row = futures[future]
                result = future.result()
                result["_type"] = "row"
                result["_prompt_key"] = key
                completed[key] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"completed {index}/{len(pending)} {arm} {row['id']}", flush=True)

    result_rows: list[dict[str, object]] = []
    for arm, rows in rows_by_arm.items():
        for row in rows:
            key = _request_key(row)
            source = dict(completed[key])
            source.pop("_type", None)
            source.pop("_prompt_key", None)
            source.update(
                {
                    "id": str(row["id"]),
                    "question": str(row["question"]),
                    "category": row.get("category"),
                    "arm": arm,
                    "source_commit": payload.get("git_revision"),
                    "added_items": len(row.get("additions") or []),
                    "provider_cache_hit": source.get("id") != str(row["id"])
                    or source.get("arm") != arm,
                }
            )
            result_rows.append(source)
    metadata = provider.provider_metadata().to_dict()
    artifact = {
        "artifact": "RE-call controlled-tail answer and citation validation",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_commit": payload.get("git_revision"),
        "arms": arms,
        "questions_per_arm": len(rows_by_arm[baseline_arm]),
        "model": metadata.get("model_id"),
        "provider": "configured answer provider",
        "answer_prompt_digest": metadata.get("prompt_digest"),
        "rows": result_rows,
        "provider_requests": len(groups),
        "provider_cache_hits": sum(1 for row in result_rows if row["provider_cache_hit"]),
        "provider_summary": metadata,
        "judge": "not run; factual correctness requires a separate fixed judge",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(result_rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
