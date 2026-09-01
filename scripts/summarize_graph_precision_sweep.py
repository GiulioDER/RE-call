"""Summarize the preregistered hub threshold and cosine margin sweep."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def _metric(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _payload_metrics(item: dict[str, object]) -> dict[str, object]:
    payload = json.loads(str(item["payload"]))
    gold = set(item["query"].get("relevant_ids", []))
    evidence_keys = [
        f"{entry['source']}:{entry['ordinal']}"
        for entry in payload.get("trusted_evidence", {}).get("items", [])
        if entry.get("ordinal") is not None
    ]
    matched = sum(key in gold for key in evidence_keys)
    diagnostics = payload.get("diagnostics", {})
    return {
        "query": item["query"]["query"],
        "arm": item["arm"],
        "hub_threshold": item["hub_threshold"],
        "cosine_margin": item["cosine_margin"],
        "policy_fingerprint": diagnostics.get("graph_policy_fingerprint"),
        "evidence_precision": matched / len(evidence_keys) if evidence_keys else None,
        "evidence_recall": matched / len(gold) if gold else None,
        "trusted_items": len(evidence_keys),
        "graph_latency_ms": diagnostics.get("graph_expansion_latency_ms", 0.0),
        "graph_candidates": diagnostics.get("graph_candidates_discovered", 0),
        "graph_rejected": diagnostics.get("graph_candidates_rejected", 0),
        "graph_admission_rejections": diagnostics.get("graph_admission_rejections", {}),
        # Separate from the line above on purpose: a refusal of the whole expansion is not a
        # candidate that lost. Summed into one counter they read as admission criteria that
        # need tuning, when the answer is that expansion never started.
        "graph_expansion_refusals": diagnostics.get("graph_expansion_refusals", {}),
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_graph_precision_sweep.py OUTPUT_JSON RAW_JSON ...")
    observations = []
    for filename in sys.argv[2:]:
        for item in json.loads(Path(filename).read_text(encoding="utf-8")):
            observations.append(_payload_metrics(item))

    settings = {}
    for key in sorted(
        {
            (row["hub_threshold"], row["cosine_margin"])
            for row in observations
            if row["arm"] == "one_hop"
        }
    ):
        setting_rows = [
            row
            for row in observations
            if row["arm"] == "one_hop"
            and (row["hub_threshold"], row["cosine_margin"]) == key
        ]
        rejections: Counter[str] = Counter()
        refusals: Counter[str] = Counter()
        for row in setting_rows:
            rejections.update(row["graph_admission_rejections"])
            refusals.update(row["graph_expansion_refusals"])
        settings[f"{key[0]}:{key[1]:.2f}"] = {
            "hub_threshold": key[0],
            "cosine_margin": key[1],
            "rows": len(setting_rows),
            "policy_fingerprints": sorted(
                {row["policy_fingerprint"] for row in setting_rows}
            ),
            "mean_evidence_precision": _metric(setting_rows, "evidence_precision"),
            "mean_evidence_recall": _metric(setting_rows, "evidence_recall"),
            "mean_trusted_items": _metric(setting_rows, "trusted_items"),
            "mean_graph_latency_ms": _metric(setting_rows, "graph_latency_ms"),
            "mean_graph_candidates": _metric(setting_rows, "graph_candidates"),
            "mean_graph_rejected": _metric(setting_rows, "graph_rejected"),
            "rejections": dict(sorted(rejections.items())),
            "expansion_refusals": dict(sorted(refusals.items())),
        }

    report = {
        "artifact": "RE-call graph precision threshold and margin sweep",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "benchmarks/PREREGISTRATION-evidence-graph-precision-tuning-v1.md",
        "rows": len(observations),
        "settings": settings,
        "raw_batches": sys.argv[2:],
        "observations": observations,
    }
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(observations), "settings": len(settings), "output": sys.argv[1]}))


if __name__ == "__main__":
    main()
