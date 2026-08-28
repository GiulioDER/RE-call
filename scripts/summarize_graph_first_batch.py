"""Score graph-first retrieval artifacts without a model judge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.summarize_query_construction_batch import _gold_keys, _matches


def _hits(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        return set()
    hits = retrieval.get("hits")
    if not isinstance(hits, list):
        return set()
    return {
        str(item.get("source") or item.get("chunk_id"))
        for item in hits
        if isinstance(item, dict) and item.get("verdict", "ok") == "ok"
    }


def summarize(payload: dict[str, Any]) -> dict[str, object]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("artifact rows must be a list")
    scored: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        final = row.get("final") if isinstance(row.get("final"), dict) else {}
        baseline = final.get("baseline_retrieval") if isinstance(final, dict) else {}
        graph = final.get("diagnostics", {}).get("graph", {}) if isinstance(final, dict) else {}
        gold = _gold_keys(row)
        baseline_hits = _hits(baseline)
        final_hits = _hits(final)
        scored.append(
            {
                "task_id": row.get("task_id"),
                "gold_class": row.get("gold", {}).get("gold_class"),
                "baseline_recovered": _matches(baseline_hits, gold),
                "recovered": _matches(final_hits, gold),
                "candidate_count": len(final.get("candidate_queries", [])) if isinstance(final, dict) else 0,
                "new_trusted_items": len(final.get("new_trusted_chunk_ids", [])) if isinstance(final, dict) else 0,
                "retrieval_calls": final.get("diagnostics", {}).get("retrieval_calls", 0) if isinstance(final, dict) else 0,
                "graph_readiness": graph.get("readiness") if isinstance(graph, dict) else None,
                "graph_reason": graph.get("reason") if isinstance(graph, dict) else None,
            }
        )
    summary: dict[str, object] = {
        "artifact": payload.get("artifact"),
        "mode": payload.get("mode"),
        "rows": len(scored),
        "rows_detail": scored,
    }
    by_class: dict[str, object] = {}
    for label in ("miss", "control"):
        subset = [row for row in scored if row["gold_class"] == label]
        by_class[label] = {
            "rows": len(subset),
            "baseline_recovered": sum(bool(row["baseline_recovered"]) for row in subset),
            "recovered": sum(bool(row["recovered"]) for row in subset),
            "candidate_activation": sum(int(row["candidate_count"]) > 0 for row in subset),
            "new_trusted_items": sum(int(row["new_trusted_items"]) for row in subset),
        }
    summary["by_class"] = by_class
    summary["miss_rescues"] = sum(
        row["gold_class"] == "miss"
        and not bool(row["baseline_recovered"])
        and bool(row["recovered"])
        for row in scored
    )
    summary["control_retention"] = sum(
        row["gold_class"] == "control" and bool(row["recovered"]) for row in scored
    )
    summary["diagnostics"] = {
        "status": dict(Counter(
            str(row.get("final", {}).get("status"))
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("final"), dict)
        )),
        "graph_readiness": dict(Counter(str(row["graph_readiness"]) for row in scored)),
        "candidate_queries": sum(int(row["candidate_count"]) for row in scored),
        "new_trusted_items": sum(int(row["new_trusted_items"]) for row in scored),
        "retrieval_calls": sum(int(row["retrieval_calls"]) for row in scored),
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
