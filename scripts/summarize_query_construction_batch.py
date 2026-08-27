"""Score a frozen query construction artifact without using a model judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _hit_keys(row: dict[str, Any]) -> set[str]:
    payload = row.get("final", {})
    if not isinstance(payload, dict):
        return set()
    retrieval = payload.get("retrieval")
    if isinstance(retrieval, dict):
        hits = retrieval.get("hits", [])
        return {
            str(item.get("source") or item.get("chunk_id"))
            for item in hits
            if isinstance(item, dict) and item.get("verdict", "ok") == "ok"
        }
    evidence = payload.get("trusted_evidence")
    if isinstance(evidence, dict):
        items = evidence.get("items", [])
        return {
            str(item.get("source") or item.get("chunk_id"))
            for item in items
            if isinstance(item, dict)
        }
    return set()


def _gold_keys(row: dict[str, Any]) -> set[str]:
    gold = row.get("gold")
    if not isinstance(gold, dict):
        gold = row
    values: list[object] = []
    for key in (
        "gold_sources",
        "gold_ids",
        "relevant_ids",
        "gold_memos",
        "gold_memo",
        "declared_memo",
    ):
        value = gold.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, str):
            values.append(value)
    return {str(value) for value in values if str(value).strip()}


def _matches(hits: set[str], gold: set[str]) -> bool:
    if not hits or not gold:
        return False
    return any(
        hit == target
        or hit.rsplit("/", 1)[-1] == target
        or hit.rsplit("/", 1)[-1].removesuffix(".md") == target.removesuffix(".md")
        for hit in hits
        for target in gold
    )


def summarize(payload: dict[str, Any]) -> dict[str, object]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("artifact rows must be a list")
    by_arm: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        arm = str(row.get("arm", "unknown"))
        hits = _hit_keys(row)
        gold = _gold_keys(row)
        by_arm.setdefault(arm, []).append(
            {
                "task_id": row.get("task_id"),
                "recovered": _matches(hits, gold),
                "hit_count": len(hits),
                "gold_count": len(gold),
                "status": row.get("final", {}).get("status") if isinstance(row.get("final"), dict) else None,
            }
        )
    summary: dict[str, object] = {"artifact": payload.get("artifact"), "arms": {}}
    arms = summary["arms"]
    assert isinstance(arms, dict)
    for arm, arm_rows in sorted(by_arm.items()):
        recovered = sum(bool(row["recovered"]) for row in arm_rows)
        arms[arm] = {
            "rows": len(arm_rows),
            "recovered": recovered,
            "recovery_rate": recovered / len(arm_rows) if arm_rows else None,
            "rows_detail": arm_rows,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("artifact must contain a JSON object")
    result = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
