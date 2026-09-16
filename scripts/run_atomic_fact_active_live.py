"""Replay the frozen 96-row pool through control and active MCP processes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_atomic_fact_release_confirmation import (  # noqa: E402
    EXPECTED_POOL_SHA256,
    one_sided_paired_probability,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _public_signature  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402

PROTOCOL = "2026-09-16-atomic-fact-active-serving-live"
EXPECTED_ROWS = 96
CUTOFFS = (1, 3, 5)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if private:
        os.chmod(path, 0o600)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise RuntimeError("latency sample is empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pool(path: Path) -> list[dict[str, Any]]:
    if _sha256(path) != EXPECTED_POOL_SHA256:
        raise RuntimeError("private pool hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("private pool must contain exactly 96 rows")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("private pool row is malformed")
    return cast(list[dict[str, Any]], rows)


def _command_for(
    *,
    mode: str,
    artifact_root: str | None,
    generation: str,
    profile: str,
    source_policy_file: str | None = None,
) -> list[str]:
    return cast(
        list[str],
        _command(
            "memory",
            "voyage-context:voyage-context-4",
            "/home/sentiment/recall-repos/memory",
            profile,
            "combined",
            "none",
            20260825,
            32,
            0.10,
            generation,
            atomic_rescue_mode=mode,
            atomic_rescue_artifact_root=artifact_root,
            benchmark_pin=True,
            source_policy_file=source_policy_file,
        ),
    )


def _call(
    client: TTYMCP,
    request_id: int,
    query: str,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, object] = {
        "query": query,
        "k": 5,
        "mode": "retrieval_only",
        "max_steps": 12,
        "max_graph_nodes": 1,
        "max_evidence_tokens": 2048,
        "graph_expansion": "off",
    }
    if source is not None:
        arguments["source"] = source
    response = client.call(
        request_id,
        "tools/call",
        {"name": "recall_reasoning_query", "arguments": arguments},
    )
    payload = json.loads(_extract_payload(response))
    if not isinstance(payload, dict):
        raise RuntimeError("reasoning response must be an object")
    return payload


def _lineage(payload: Mapping[str, object]) -> tuple[str, str, str, str]:
    return tuple(
        str(payload.get(key))
        for key in (
            "generation_id",
            "calibration_id",
            "pipeline_fingerprint",
            "corpus_fingerprint",
        )
    )  # type: ignore[return-value]


def _items(payload: Mapping[str, object]) -> list[dict[str, object]]:
    evidence = payload.get("trusted_evidence")
    items = evidence.get("items") if isinstance(evidence, dict) else None
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def _hit(items: Sequence[Mapping[str, object]], row: Mapping[str, Any], cutoff: int, label: str) -> bool:
    gold_source = str(row["gold_sources"][0])
    gold_ordinal = int(row["gold_ordinal"])
    if label == "exact":
        return any(
            str(item.get("source")) == gold_source
            and int(item.get("ordinal", -1)) == gold_ordinal
            for item in items[:cutoff]
        )
    return any(str(item.get("source")) == gold_source for item in items[:cutoff])


def _paired(rows: Sequence[Mapping[str, object]], label: str, cutoff: int) -> dict[str, object]:
    gains = losses = ties = 0
    for row in rows:
        pool_row = cast(Mapping[str, Any], row["gold"])
        control = _hit(cast(list[dict[str, object]], row["control"]), pool_row, cutoff, label)
        active = _hit(cast(list[dict[str, object]], row["active"]), pool_row, cutoff, label)
        if active and not control:
            gains += 1
        elif control and not active:
            losses += 1
        else:
            ties += 1
    return {
        "gains": gains,
        "losses": losses,
        "ties": ties,
        "net": gains - losses,
        "one_sided_paired_probability": one_sided_paired_probability(gains, losses),
    }


def _atomic_ms(payload: Mapping[str, object]) -> float:
    diagnostics = payload.get("diagnostics")
    stages = diagnostics.get("retrieval_stage_ms") if isinstance(diagnostics, dict) else None
    value = stages.get("atomic_rescue") if isinstance(stages, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("active response lacks atomic_rescue stage latency")
    return float(value)


def _trust_signature(payload: Mapping[str, object]) -> tuple[object, ...]:
    evidence = payload.get("trusted_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return (
        payload.get("trust_state"),
        payload.get("refusal_reason"),
        payload.get("outcome"),
        evidence.get("decision"),
        evidence.get("reason_code"),
        evidence.get("failure_code"),
    )


def _scoped_parity(
    rows: Sequence[Mapping[str, Any]],
    *,
    generation: str,
    artifact_root: str,
    profile: str,
    timeout: float,
    source_policy_file: str | None,
) -> int:
    control = TTYMCP(
        _command_for(
            mode="off",
            artifact_root=None,
            generation=generation,
            profile=profile,
            source_policy_file=source_policy_file,
        ),
        timeout,
    )
    active = TTYMCP(
        _command_for(
            mode="active",
            artifact_root=artifact_root,
            generation=generation,
            profile=profile,
            source_policy_file=source_policy_file,
        ),
        timeout,
    )
    matches = 0
    try:
        _initialize(control)
        _initialize(active)
        for index, row in enumerate(rows[:12], start=2):
            source = None if source_policy_file is not None else str(row["gold_sources"][0])
            baseline = _call(control, index, str(row["query"]), source=source)
            candidate = _call(active, index, str(row["query"]), source=source)
            matches += _public_signature(baseline) == _public_signature(candidate)
    finally:
        control.close()
        active.close()
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--profile", default="combined")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--source-policy-file")
    args = parser.parse_args()
    rows = _pool(args.pool)
    expected_lineage = (args.generation, args.calibration, args.pipeline, args.corpus)
    control = TTYMCP(
        _command_for(
            mode="off",
            artifact_root=None,
            generation=args.generation,
            profile=args.profile,
        ),
        args.timeout,
    )
    active = TTYMCP(
        _command_for(
            mode="active",
            artifact_root=args.artifact_root,
            generation=args.generation,
            profile=args.profile,
        ),
        args.timeout,
    )
    private_rows: list[dict[str, object]] = []
    atomic_latencies: list[float] = []
    errors = 0
    try:
        _initialize(control)
        _initialize(active)
        _call(active, 2, str(rows[0]["query"]))
        for index, row in enumerate(rows, start=3):
            try:
                baseline = _call(control, index, str(row["query"]))
                candidate = _call(active, index, str(row["query"]))
                if _lineage(baseline) != expected_lineage or _lineage(candidate) != expected_lineage:
                    raise RuntimeError("active-serving lineage changed during replay")
                atomic_latencies.append(_atomic_ms(candidate))
                private_rows.append(
                    {
                        "query_id": str(row["id"]),
                        "gold": {
                            "gold_sources": row["gold_sources"],
                            "gold_ordinal": row["gold_ordinal"],
                        },
                        "control": _items(baseline),
                        "active": _items(candidate),
                        "control_trust": _trust_signature(baseline),
                        "active_trust": _trust_signature(candidate),
                    }
                )
            except Exception:
                errors += 1
                raise
    finally:
        control.close()
        active.close()

    comparison = {
        label: {str(cutoff): _paired(private_rows, label, cutoff) for cutoff in CUTOFFS}
        for label in ("exact", "gold")
    }
    source_scope_matches = _scoped_parity(
        rows,
        generation=args.generation,
        artifact_root=args.artifact_root,
        profile=args.profile,
        timeout=args.timeout,
        source_policy_file=None,
    )
    security_scope_matches = (
        _scoped_parity(
            rows,
            generation=args.generation,
            artifact_root=args.artifact_root,
            profile=args.profile,
            timeout=args.timeout,
            source_policy_file=args.source_policy_file,
        )
        if args.source_policy_file
        else 0
    )
    trust_regressions = sum(
        row["control_trust"][0] == "trusted" and row["active_trust"][0] != "trusted"
        for row in private_rows
    )
    unexplained_trust_changes = sum(
        row["control_trust"] != row["active_trust"] and row["control"] == row["active"]
        for row in private_rows
    )
    p95 = _percentile(atomic_latencies, 95)
    p99 = _percentile(atomic_latencies, 99)
    losses_ok = all(
        comparison[label][str(cutoff)]["losses"] <= 1
        for label in ("exact", "gold")
        for cutoff in CUTOFFS
    )
    checks = {
        "rows": len(private_rows) == EXPECTED_ROWS,
        "errors_zero": errors == 0,
        "gold_at_5_positive_net": comparison["gold"]["5"]["net"] > 0,
        "exact_at_5_nonnegative_net": comparison["exact"]["5"]["net"] >= 0,
        "all_cutoff_losses_lte_1": losses_ok,
        "trusted_response_regressions_zero": trust_regressions == 0,
        "unexplained_trust_changes_zero": unexplained_trust_changes == 0,
        "atomic_p95_lte_20_ms": p95 <= 20.0,
        "atomic_p99_lte_45_ms": p99 <= 45.0,
        "source_scoped_parity_12_of_12": source_scope_matches == 12,
        "security_scoped_parity_12_of_12": security_scope_matches == 12,
    }
    decision = "PASS_ATOMIC_ACTIVE_LIVE" if all(checks.values()) else "STOP_ATOMIC_ACTIVE_SERVING"
    private = {"schema_version": 1, "protocol": PROTOCOL, "rows": private_rows}
    _json(args.private_output, private, private=True)
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "generation_id": args.generation,
        "calibration_id": args.calibration,
        "pipeline_fingerprint": args.pipeline,
        "corpus_fingerprint": args.corpus,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "private_rows_sha256": _sha256(args.private_output),
        "comparison": comparison,
        "atomic_stage_latency_ms": {"p95": round(p95, 6), "p99": round(p99, 6)},
        "source_scoped_parity": source_scope_matches,
        "security_scoped_parity": security_scope_matches,
        "trust_regressions": trust_regressions,
        "unexplained_trust_changes": unexplained_trust_changes,
        "checks": checks,
        "decision": decision,
        "serving_route_changed": False,
    }
    _json(args.public_output, public)
    print(json.dumps({"decision": decision, "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
