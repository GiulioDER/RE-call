"""Run the preregistered atomic rescue production shadow through fresh MCP processes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _public_signature  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


PROTOCOL = "2026-09-16-atomic-fact-production-shadow-remediation"
EXPECTED_POOL_SHA256 = "414b441fd95ccc0de7a6ede329515941c09f8aef8e9fc8e20ccbdfc1d7514f0f"
EXPECTED_PRIVATE_ROWS_SHA256 = "590548c5a7ed038bad4b245a6fc0f4bd8ae8c9f37e98f716e39148e6037b2442"
EXPECTED_GENERATION = "gen_dff506e12f494965af9f109671a99e63"
EXPECTED_CALIBRATION = "cal_6171177aeb614d288baa4e28600caca1"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0"
EXPECTED_ROWS = 96
CONCURRENT_REQUESTS = 256
CONCURRENT_WORKERS = 8


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


def build_expected_payload(
    pool: Mapping[str, object], private_retrieval: Mapping[str, object]
) -> dict[str, object]:
    """Build query-digest receipts for private in-process candidate parity checks."""

    queries = pool.get("queries")
    rows = private_retrieval.get("rows")
    if not isinstance(queries, list) or not isinstance(rows, list):
        raise RuntimeError("private confirmation artifacts lack rows")
    query_by_id = {
        str(row["id"]): str(row["query"])
        for row in queries
        if isinstance(row, dict) and row.get("id") and row.get("query")
    }
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("private retrieval row is malformed")
        query_id = str(row.get("query_id", ""))
        query = query_by_id.get(query_id)
        arm = row.get("dense5_atomic1")
        if query is None or not isinstance(arm, list) or len(arm) != 6:
            raise RuntimeError("private retrieval row lacks its query or six candidates")
        selected = arm[5]
        if not isinstance(selected, dict):
            raise RuntimeError("private retrieval selected candidate is malformed")
        output[hashlib.sha256(query.encode("utf-8")).hexdigest()] = {
            "source": str(selected["source"]),
            "parent_ordinal": int(selected["ordinal"]),
            "score": float(selected["score"]),
        }
    if len(output) != EXPECTED_ROWS:
        raise RuntimeError("private expected candidate receipt must contain 96 rows")
    return {"schema_version": 1, "rows": output}


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


def _latencies(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": round(_percentile(values, 50), 6),
        "p95": round(_percentile(values, 95), 6),
        "p99": round(_percentile(values, 99), 6),
        "max": round(max(values), 6),
    }


def _payload(client: TTYMCP, request_id: int, query: str) -> dict[str, Any]:
    response = client.call(
        request_id,
        "tools/call",
        {
            "name": "recall_reasoning_query",
            "arguments": {
                "query": query,
                "k": 5,
                "mode": "retrieval_only",
                "max_steps": 12,
                "max_graph_nodes": 1,
                "max_evidence_tokens": 2048,
                "graph_expansion": "off",
            },
        },
    )
    decoded = json.loads(_extract_payload(response))
    if not isinstance(decoded, dict):
        raise RuntimeError("reasoning response must be an object")
    return decoded


def _batch_payloads(
    client: TTYMCP, request_id: int, queries: Sequence[str]
) -> tuple[list[dict[str, Any]], int]:
    calls = [
        (
            request_id + index,
            "tools/call",
            {
                "name": "recall_reasoning_query",
                "arguments": {
                    "query": query,
                    "k": 5,
                    "mode": "retrieval_only",
                    "max_steps": 12,
                    "max_graph_nodes": 1,
                    "max_evidence_tokens": 2048,
                    "graph_expansion": "off",
                },
            },
        )
        for index, query in enumerate(queries)
    ]
    payloads: list[dict[str, Any]] = []
    for response in client.batch(calls):
        decoded = json.loads(_extract_payload(response))
        if not isinstance(decoded, dict):
            raise RuntimeError("reasoning response must be an object")
        payloads.append(decoded)
    return payloads, request_id + len(calls)


def _identity(payload: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(payload.get("generation_id")),
        str(payload.get("calibration_id")),
        str(payload.get("pipeline_fingerprint")),
        str(payload.get("corpus_fingerprint")),
    )


def _frozen_identity(payload: Mapping[str, object]) -> None:
    expected = (
        EXPECTED_GENERATION,
        EXPECTED_CALIBRATION,
        EXPECTED_PIPELINE,
        EXPECTED_CORPUS,
    )
    if _identity(payload) != expected:
        raise RuntimeError(f"frozen serving lineage mismatch: {_identity(payload)}")


def _atomic(payload: Mapping[str, object]) -> Mapping[str, object]:
    diagnostics = payload.get("diagnostics")
    performance = diagnostics.get("performance") if isinstance(diagnostics, dict) else None
    values = performance.get("values") if isinstance(performance, dict) else None
    atomic = values.get("atomic_rescue_shadow") if isinstance(values, dict) else None
    if not isinstance(atomic, dict):
        raise RuntimeError("atomic rescue shadow diagnostics are missing")
    return atomic


def _span(payload: Mapping[str, object]) -> float:
    diagnostics = payload.get("diagnostics")
    performance = diagnostics.get("performance") if isinstance(diagnostics, dict) else None
    spans = performance.get("spans_ms") if isinstance(performance, dict) else None
    value = spans.get("atomic_rescue_shadow_ms") if isinstance(spans, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("atomic rescue shadow span is missing")
    return float(value)


def _command_for(
    *,
    generation: str | None,
    artifact: str | None,
    expected: str | None,
    mode: str,
    profile: str,
) -> list[str]:
    return cast(list[str], _command(
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
        atomic_rescue_artifact=artifact,
        atomic_rescue_sample_rate=1.0 if mode == "shadow" else 0.0,
        atomic_rescue_expected=expected,
        benchmark_pin=True,
    ))


def _sequential(
    queries: Sequence[Mapping[str, object]],
    *,
    artifact: str,
    expected: str,
    profile: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    control = TTYMCP(
        _command_for(
            generation=EXPECTED_GENERATION,
            artifact=None,
            expected=None,
            mode="off",
            profile=profile,
        ),
        timeout,
    )
    candidate = TTYMCP(
        _command_for(
            generation=EXPECTED_GENERATION,
            artifact=artifact,
            expected=expected,
            mode="shadow",
            profile=profile,
        ),
        timeout,
    )
    private_rows: list[dict[str, Any]] = []
    selector_ms: list[float] = []
    stage_ms: list[float] = []
    request_id_control = 2
    request_id_candidate = 3
    warm: dict[str, object]
    try:
        _initialize(control)
        _initialize(candidate)
        warm_payload = _payload(candidate, 2, str(queries[0]["query"]))
        _frozen_identity(warm_payload)
        warm_atomic = _atomic(warm_payload)
        warm = {
            "status": warm_atomic.get("status"),
            "artifact_load_ms": warm_atomic.get("artifact_load_ms"),
            "resident_memory_delta_bytes": warm_atomic.get("resident_memory_delta_bytes"),
            "stage_ms": _span(warm_payload),
            "public_result_unchanged": (
                warm_atomic.get("benchmark_public_result_unchanged") is True
            ),
            "reference_identity_parity": (
                warm_atomic.get("benchmark_reference_identity_parity") is True
            ),
            "reference_score_parity": (
                warm_atomic.get("benchmark_reference_score_parity") is True
            ),
        }
        for index, row in enumerate(queries):
            query = str(row["query"])
            print(f"sequential {index + 1}/{len(queries)}", flush=True)
            baseline = _payload(control, request_id_control, query)
            shadow = _payload(candidate, request_id_candidate, query)
            request_id_control += 1
            request_id_candidate += 1
            _frozen_identity(baseline)
            _frozen_identity(shadow)
            atomic = _atomic(shadow)
            selector = atomic.get("selector_ms")
            if isinstance(selector, bool) or not isinstance(selector, (int, float)):
                raise RuntimeError("selector latency is missing")
            selector_ms.append(float(selector))
            stage_ms.append(_span(shadow))
            private_rows.append(
                {
                    "query_id": str(row["id"]),
                    "query": query,
                    "public_parity": _public_signature(baseline) == _public_signature(shadow),
                    "public_result_unchanged": (
                        atomic.get("benchmark_public_result_unchanged") is True
                    ),
                    "identity_parity": atomic.get("benchmark_identity_parity") is True,
                    "score_parity": atomic.get("benchmark_score_parity") is True,
                    "reference_identity_parity": (
                        atomic.get("benchmark_reference_identity_parity") is True
                    ),
                    "reference_score_parity": (
                        atomic.get("benchmark_reference_score_parity") is True
                    ),
                    "status": atomic.get("status"),
                    "selector_ms": float(selector),
                    "stage_ms": stage_ms[-1],
                    "baseline_signature": _public_signature(baseline),
                }
            )
    finally:
        control.close()
        candidate.close()
    summary = {
        "rows": len(private_rows),
        "public_parity": sum(bool(row["public_parity"]) for row in private_rows),
        "public_result_unchanged": sum(
            bool(row["public_result_unchanged"]) for row in private_rows
        ),
        "identity_parity": sum(bool(row["identity_parity"]) for row in private_rows),
        "score_parity": sum(bool(row["score_parity"]) for row in private_rows),
        "reference_identity_parity": sum(
            bool(row["reference_identity_parity"]) for row in private_rows
        ),
        "reference_score_parity": sum(
            bool(row["reference_score_parity"]) for row in private_rows
        ),
        "status_ok": sum(row["status"] == "ok" for row in private_rows),
        "selector_latency_ms": _latencies(selector_ms),
        "shadow_stage_latency_ms": _latencies(stage_ms),
        "warm": warm,
    }
    return private_rows, summary


def _concurrent(
    queries: Sequence[Mapping[str, object]],
    baseline_by_id: Mapping[str, object],
    *,
    artifact: str,
    expected: str,
    profile: str,
    timeout: float,
) -> dict[str, Any]:
    client = TTYMCP(
        _command_for(
            generation=EXPECTED_GENERATION,
            artifact=artifact,
            expected=expected,
            mode="shadow",
            profile=profile,
        ),
        timeout,
    )
    request_id = 2
    errors = parity_failures = identity_failures = score_failures = 0
    immutability_failures = reference_identity_failures = reference_score_failures = 0
    selector_ms: list[float] = []
    started = time.perf_counter()
    try:
        _initialize(client)
        for offset in range(0, CONCURRENT_REQUESTS, CONCURRENT_WORKERS):
            batch_rows = [queries[index % len(queries)] for index in range(offset, offset + 8)]
            try:
                payloads, request_id = _batch_payloads(
                    client,
                    request_id,
                    [str(row["query"]) for row in batch_rows],
                )
            except Exception:
                errors += len(batch_rows)
                continue
            for row, payload in zip(batch_rows, payloads, strict=True):
                try:
                    _frozen_identity(payload)
                    atomic = _atomic(payload)
                    selector = atomic.get("selector_ms")
                    if isinstance(selector, bool) or not isinstance(selector, (int, float)):
                        raise RuntimeError("selector latency is missing")
                    selector_ms.append(float(selector))
                    query_id = str(row["id"])
                    if _public_signature(payload) != baseline_by_id[query_id]:
                        parity_failures += 1
                    if atomic.get("benchmark_public_result_unchanged") is not True:
                        immutability_failures += 1
                    if atomic.get("benchmark_identity_parity") is not True:
                        identity_failures += 1
                    if atomic.get("benchmark_score_parity") is not True:
                        score_failures += 1
                    if atomic.get("benchmark_reference_identity_parity") is not True:
                        reference_identity_failures += 1
                    if atomic.get("benchmark_reference_score_parity") is not True:
                        reference_score_failures += 1
                except Exception:
                    errors += 1
    finally:
        client.close()
    return {
        "requests": CONCURRENT_REQUESTS,
        "workers": CONCURRENT_WORKERS,
        "errors": errors,
        "public_parity_failures": parity_failures,
        "public_immutability_failures": immutability_failures,
        "identity_parity_failures": identity_failures,
        "score_parity_failures": score_failures,
        "reference_identity_failures": reference_identity_failures,
        "reference_score_failures": reference_score_failures,
        "selector_latency_ms": _latencies(selector_ms) if selector_ms else None,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _failure_probe(
    query: str,
    baseline_signature: object,
    *,
    artifact: str,
    expected_error: str,
    profile: str,
    timeout: float,
) -> dict[str, object]:
    client = TTYMCP(
        _command_for(
            generation=EXPECTED_GENERATION,
            artifact=artifact,
            expected=None,
            mode="shadow",
            profile=profile,
        ),
        timeout,
    )
    try:
        _initialize(client)
        first = _payload(client, 2, query)
        second = _payload(client, 3, query)
    finally:
        client.close()
    first_atomic = _atomic(first)
    second_atomic = _atomic(second)
    return {
        "expected_error": expected_error,
        "first_error": first_atomic.get("error_code"),
        "second_error": second_atomic.get("error_code"),
        "first_public_parity": _public_signature(first) == baseline_signature,
        "second_public_parity": _public_signature(second) == baseline_signature,
        "first_public_result_unchanged": (
            first_atomic.get("benchmark_public_result_unchanged") is True
        ),
        "second_public_result_unchanged": (
            second_atomic.get("benchmark_public_result_unchanged") is True
        ),
        "process_healthy": True,
        "passed": (
            first_atomic.get("error_code") == expected_error
            and second_atomic.get("error_code") == expected_error
            and first_atomic.get("benchmark_public_result_unchanged") is True
            and second_atomic.get("benchmark_public_result_unchanged") is True
        ),
    }


def _rollover_probe(
    query: str,
    *,
    artifact: str,
    profile: str,
    timeout: float,
) -> dict[str, object]:
    control = TTYMCP(
        _command_for(generation=None, artifact=None, expected=None, mode="off", profile=profile),
        timeout,
    )
    candidate = TTYMCP(
        _command_for(
            generation=None,
            artifact=artifact,
            expected=None,
            mode="shadow",
            profile=profile,
        ),
        timeout,
    )
    try:
        _initialize(control)
        _initialize(candidate)
        baseline = _payload(control, 2, query)
        first = _payload(candidate, 2, query)
        second = _payload(candidate, 3, query)
    finally:
        control.close()
        candidate.close()
    first_atomic = _atomic(first)
    second_atomic = _atomic(second)
    active_identity = _identity(baseline)
    return {
        "active_identity": active_identity,
        "first_identity": _identity(first),
        "second_identity": _identity(second),
        "first_error": first_atomic.get("error_code"),
        "second_error": second_atomic.get("error_code"),
        "first_public_parity": _public_signature(first) == _public_signature(baseline),
        "second_public_parity": _public_signature(second) == _public_signature(baseline),
        "first_public_result_unchanged": (
            first_atomic.get("benchmark_public_result_unchanged") is True
        ),
        "second_public_result_unchanged": (
            second_atomic.get("benchmark_public_result_unchanged") is True
        ),
        "process_healthy": True,
        "passed": (
            active_identity != (
                EXPECTED_GENERATION,
                EXPECTED_CALIBRATION,
                EXPECTED_PIPELINE,
                EXPECTED_CORPUS,
            )
            and _identity(first) == active_identity
            and _identity(second) == active_identity
            and first_atomic.get("error_code") == "lineage_error"
            and second_atomic.get("error_code") == "lineage_error"
            and first_atomic.get("benchmark_public_result_unchanged") is True
            and second_atomic.get("benchmark_public_result_unchanged") is True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--private-retrieval", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--remote-expected", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-build-result", type=Path, required=True)
    parser.add_argument("--missing-artifact", required=True)
    parser.add_argument("--wrong-generation-artifact", required=True)
    parser.add_argument("--wrong-digest-artifact", required=True)
    parser.add_argument("--truncated-artifact", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    if _sha256(args.pool) != EXPECTED_POOL_SHA256:
        raise RuntimeError("private pool hash changed")
    if _sha256(args.private_retrieval) != EXPECTED_PRIVATE_ROWS_SHA256:
        raise RuntimeError("private retrieval rows hash changed")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    private_retrieval = json.loads(args.private_retrieval.read_text(encoding="utf-8"))
    if not isinstance(pool, dict) or not isinstance(private_retrieval, dict):
        raise RuntimeError("private confirmation artifacts must be objects")
    expected_payload = build_expected_payload(pool, private_retrieval)
    _json(args.expected_output, expected_payload, private=True)
    queries = pool.get("queries")
    if not isinstance(queries, list) or len(queries) != EXPECTED_ROWS:
        raise RuntimeError("private pool must contain 96 queries")

    started = time.perf_counter()
    sequential_rows, sequential = _sequential(
        queries,
        artifact=args.artifact,
        expected=args.remote_expected,
        profile=args.profile,
        timeout=args.timeout,
    )
    baseline_by_id = {
        str(row["query_id"]): row["baseline_signature"] for row in sequential_rows
    }
    concurrent = _concurrent(
        queries,
        baseline_by_id,
        artifact=args.artifact,
        expected=args.remote_expected,
        profile=args.profile,
        timeout=args.timeout,
    )
    probe_query = str(queries[0]["query"])
    probe_baseline = baseline_by_id[str(queries[0]["id"])]
    failure_probes = {
        "missing": _failure_probe(
            probe_query,
            probe_baseline,
            artifact=args.missing_artifact,
            expected_error="artifact_error",
            profile=args.profile,
            timeout=args.timeout,
        ),
        "wrong_generation": _failure_probe(
            probe_query,
            probe_baseline,
            artifact=args.wrong_generation_artifact,
            expected_error="lineage_error",
            profile=args.profile,
            timeout=args.timeout,
        ),
        "wrong_digest": _failure_probe(
            probe_query,
            probe_baseline,
            artifact=args.wrong_digest_artifact,
            expected_error="artifact_error",
            profile=args.profile,
            timeout=args.timeout,
        ),
        "truncated": _failure_probe(
            probe_query,
            probe_baseline,
            artifact=args.truncated_artifact,
            expected_error="artifact_error",
            profile=args.profile,
            timeout=args.timeout,
        ),
    }
    rollover = _rollover_probe(
        probe_query,
        artifact=args.artifact,
        profile=args.profile,
        timeout=args.timeout,
    )
    build_result = json.loads(args.artifact_build_result.read_text(encoding="utf-8"))
    selector = sequential["selector_latency_ms"]
    stage = sequential["shadow_stage_latency_ms"]
    build_checks = build_result.get("checks", {}) if isinstance(build_result, dict) else {}
    checks = {
        "sequential_public_immutability": (
            sequential["public_result_unchanged"] == EXPECTED_ROWS
        ),
        "sequential_identity_parity": sequential["identity_parity"] == EXPECTED_ROWS,
        "sequential_reference_identity_parity": (
            sequential["reference_identity_parity"] == EXPECTED_ROWS
        ),
        "sequential_reference_score_parity": (
            sequential["reference_score_parity"] == EXPECTED_ROWS
        ),
        "sequential_status_ok": sequential["status_ok"] == EXPECTED_ROWS,
        "selector_p95_lte_10_ms": float(selector["p95"]) <= 10.0,
        "selector_p99_lte_25_ms": float(selector["p99"]) <= 25.0,
        "shadow_stage_p95_lte_15_ms": float(stage["p95"]) <= 15.0,
        "shadow_stage_p99_lte_30_ms": float(stage["p99"]) <= 30.0,
        "artifact_load": build_checks.get("load_lte_2000_ms") is True,
        "artifact_resident_memory": build_checks.get("resident_delta_lte_128_mib") is True,
        "artifact_disk": build_checks.get("artifact_lte_64_mib") is True,
        "concurrent_zero_errors": concurrent["errors"] == 0,
        "concurrent_public_immutability": (
            concurrent["public_immutability_failures"] == 0
        ),
        "concurrent_reference_identity_parity": (
            concurrent["reference_identity_failures"] == 0
        ),
        "concurrent_reference_score_parity": (
            concurrent["reference_score_failures"] == 0
        ),
        "concurrent_selector_p99_lte_40_ms": (
            concurrent["selector_latency_ms"] is not None
            and float(concurrent["selector_latency_ms"]["p99"]) <= 40.0
        ),
        "failure_probes": all(bool(probe["passed"]) for probe in failure_probes.values()),
        "rollover_probe": bool(rollover["passed"]),
    }
    decision = (
        "PASS_ATOMIC_PRODUCTION_SHADOW_REMEDIATION"
        if all(checks.values())
        else "STOP_ATOMIC_PRODUCTION_SHADOW_REMEDIATION"
    )
    private = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "rows": sequential_rows,
        "concurrent": concurrent,
        "failure_probes": failure_probes,
        "rollover": rollover,
    }
    _json(args.private_output, private, private=True)
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "generation_id": EXPECTED_GENERATION,
        "calibration_id": EXPECTED_CALIBRATION,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "private_retrieval_sha256": EXPECTED_PRIVATE_ROWS_SHA256,
        "expected_receipt_sha256": _sha256(args.expected_output),
        "private_result_sha256": _sha256(args.private_output),
        "artifact": {
            key: build_result.get(key)
            for key in (
                "artifact_bytes",
                "load_ms",
                "resident_memory_delta_bytes",
                "matrix_sha256",
                "metadata_sha256",
                "view_count",
                "parent_count",
            )
        },
        "sequential": sequential,
        "concurrent": concurrent,
        "failure_probes": {
            name: {key: value for key, value in probe.items() if key != "active_identity"}
            for name, probe in failure_probes.items()
        },
        "rollover": rollover,
        "checks": checks,
        "decision": decision,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "serving_route_changed": False,
    }
    _json(args.public_output, public)
    print(json.dumps({"decision": decision, "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
