#!/usr/bin/env python3
"""Benchmark cold write-hook calls against a persistent local relay.

This is a measurement harness for the preregistration beside it. It deliberately does not modify
the installer or the shipped hook configuration. The relay uses the hook's own SQL helper, so a
latency win cannot come from a second retrieval implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OPTIONS = {
    "k": 5,
    "min_chars": 12,
    "connect_timeout": 2.0,
    "cooldown_seconds": 300.0,
}

# Frozen locally before the first live call. These are representative write and command payloads
# from the registered hook probe, including the known hazard and an innocuous command.
PAYLOADS = (
    "version_file.write_text(content, encoding='utf-8')",
    "Path.write_text on Windows injects CRLF against a tree configured eol=lf",
    "ls -la scripts/ | head -20",
    "python -m pytest tests/test_write_time_hook.py -q",
    "with open(path, 'w', encoding='utf-8', newline='\\n') as handle:",
    "subprocess.run([sys.executable, '-m', 'pytest'], check=True)",
    "content = path.read_text(encoding='utf-8')",
    "git diff --check",
    "connection.execute(query, params).fetchall()",
    "json.dumps({'hookSpecificOutput': output})",
)


def _redact_dsn(dsn: str) -> str:
    if "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    if "@" not in rest:
        return f"{scheme}://{rest}"
    userinfo, _, host = rest.rpartition("@")
    user, sep, _password = userinfo.partition(":")
    return f"{scheme}://{user}{':***' if sep else ''}@{host}"


def _dsn_host(dsn: str) -> str:
    try:
        return urlsplit(dsn).hostname or "unknown"
    except ValueError:
        return "invalid"


def _one_request(
    mode: str,
    dsn: str,
    tenant: str,
    query: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    request = {"dsn": dsn, "tenant": tenant, "query": query}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), f"--{mode}-child"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=30,
    )
    elapsed = (time.perf_counter() - started) * 1000
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {"status": "error", "error": result.stderr[-500:] or "child returned no output", "elapsed_ms": elapsed}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        payload = {"status": "error", "error": lines[-1][:500]}
    payload["elapsed_ms"] = elapsed
    return payload


def _relay_requests(dsn: str, tenant: str, queries: list[str]) -> list[dict[str, Any]]:
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--relay-child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUTF8": "1"},
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    rows: list[dict[str, Any]] = []
    try:
        for query in queries:
            started = time.perf_counter()
            process.stdin.write(json.dumps({"dsn": dsn, "tenant": tenant, "query": query}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            elapsed = (time.perf_counter() - started) * 1000
            if not line:
                rows.append({"status": "error", "error": "relay exited", "elapsed_ms": elapsed})
                break
            payload = json.loads(line)
            payload["elapsed_ms"] = elapsed
            rows.append(payload)
    finally:
        if process.stdin:
            process.stdin.close()
        process.terminate()
        process.wait(timeout=10)
    return rows


def _hit_key(payload: dict[str, Any]) -> list[tuple[str, float]]:
    return [(str(row[0]), round(float(row[2]), 9)) for row in payload.get("hits", [])]


def _run(args: argparse.Namespace) -> int:
    dsn = args.dsn
    queries = [query for query in PAYLOADS for _ in range(3)]
    cold = [_one_request("cold", dsn, args.tenant, query) for query in queries]
    relay = _relay_requests(dsn, args.tenant, queries)
    equality = len(cold) == len(relay) and all(
        left.get("status") == right.get("status") == "ok"
        and _hit_key(left) == _hit_key(right)
        for left, right in zip(cold, relay, strict=True)
    )

    unreachable = "postgresql://recall:recall@127.0.0.1:59999/agent_ab"
    unavailable = _relay_requests(unreachable, args.tenant, [queries[0]] * 6)
    safety = all(float(row.get("elapsed_ms", 999999)) <= 500 for row in unavailable[1:])
    artifact = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "dsn": _redact_dsn(dsn),
        "dsn_host": _dsn_host(dsn),
        "tenant": args.tenant,
        "payload_count": len(PAYLOADS),
        "repetitions": 3,
        "cold": cold,
        "relay": relay,
        "result_equality": equality,
        "unreachable_relay": unavailable,
        "unreachable_followup_under_500ms": safety,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"write_time_connection_reuse_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
    cold_times = [float(row["elapsed_ms"]) for row in cold]
    relay_times = [float(row["elapsed_ms"]) for row in relay]
    print(json.dumps({
        "artifact": str(path),
        "result_equality": equality,
        "cold_median_ms": sorted(cold_times)[len(cold_times) // 2] if cold_times else None,
        "relay_median_ms": sorted(relay_times)[len(relay_times) // 2] if relay_times else None,
        "unreachable_followup_under_500ms": safety,
    }, indent=2))
    return 0 if equality and safety else 1


def _child(mode: str) -> int:
    request = json.loads(sys.stdin.readline())
    from recall_hooks.write_time import _search_connection, search

    config = {"dsn": request["dsn"], "tenant": request["tenant"]}
    if mode == "cold":
        started = time.perf_counter()
        try:
            hits = search(request["query"], config, OPTIONS)
            print(json.dumps({"status": "ok", "hits": hits, "query_ms": (time.perf_counter() - started) * 1000}), flush=True)
        except Exception as exc:  # benchmark records failure instead of hiding it
            print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0

    import psycopg

    try:
        connection = psycopg.connect(request["dsn"], connect_timeout=2.0, options="-c statement_timeout=5s")
    except Exception as exc:
        for line in sys.stdin:
            if line.strip():
                print(json.dumps({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0
    with connection:
        for line in sys.stdin:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                hits = _search_connection(connection, row["query"], config, OPTIONS)
                print(json.dumps({"status": "ok", "hits": hits}), flush=True)
            except Exception as exc:
                print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--cold-child", action="store_true")
    parser.add_argument("--relay-child", action="store_true")
    args = parser.parse_args()
    if args.cold_child:
        return _child("cold")
    if args.relay_child:
        return _child("relay")
    if not args.dsn:
        parser.error("--dsn is required")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
