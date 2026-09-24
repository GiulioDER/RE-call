"""Concurrent Adds for the SAME user against a live AML hosted service, retried as AML retries.

Pre-registration: ``docs/preregistrations/2026-09-24-aml-c9-same-user-add-concurrency.md``.

Every Add holds its tenant's advisory lock for the whole request, and a waiter whose lock wait
passes the pool's 25 s ``statement_timeout`` is answered 503. The 2026-09-23 ramp sent one user per
worker with sequential Adds, so it never exercised that wait. This fires ``--concurrency`` Adds at
once in each scenario, spread over ``users`` users, and retries a 5xx or a client timeout with the
same ``request_id`` up to ``--max-attempts`` times, which is AML's documented contract. Adds are
real LoCoMo sessions, so the Add-time compiler and atomic views run at realistic sizes.

    python scripts/aml_same_user_concurrency.py --base-url http://127.0.0.1:18120 \
        --data locomo10.json --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import httpx

# (label, users): each scenario sends `concurrency` Adds at once, split evenly over its users.
SCENARIOS = (("one_per_user", None), ("four_per_user", 4), ("all_one_user", 1))


def sessions(data: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    """The first ``count`` non-empty LoCoMo sessions, as AML message lists, in dataset order."""
    found: list[list[dict[str, Any]]] = []
    for sample in data:
        conversation = sample["conversation"]
        keys = sorted(
            (k for k in conversation if k.startswith("session_") and k.split("_")[1].isdigit()
             and len(k.split("_")) == 2),
            key=lambda k: int(k.split("_")[1]),
        )
        for key in keys:
            turns = conversation[key]
            if not turns:
                continue
            stamp = datetime.strptime(
                " ".join(conversation[f"{key}_date_time"].split()), "%I:%M %p on %d %B, %Y"
            ).replace(tzinfo=timezone.utc)
            millis = int(stamp.timestamp() * 1_000)
            found.append(
                [{"role": "user", "content": f"{t['speaker']}: {t['text']}", "timestamp": millis}
                 for t in turns]
            )
            if len(found) == count:
                return found
    raise SystemExit(f"dataset holds fewer than {count} non-empty sessions")


async def add_with_retries(
    client: httpx.AsyncClient, body: dict[str, Any], max_attempts: int, backoff: float
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            response = await client.post("/v1/add", json=body)
            status: int | str = response.status_code
        except httpx.TimeoutException:
            status = "client_timeout"
        attempts.append({"status": status, "seconds": round(time.perf_counter() - t0, 2)})
        if status == 200:
            break
        if isinstance(status, int) and status < 500:
            break  # a 4xx is not retried by AML and would be a harness bug
        await asyncio.sleep(backoff)
    return {
        "request_id": body["request_id"],
        "user_id": body["user_id"],
        "attempts": attempts,
        "final_status": attempts[-1]["status"],
        "seconds_to_final": round(time.perf_counter() - started, 2),
    }


async def run_scenario(
    client: httpx.AsyncClient,
    label: str,
    users: int,
    payloads: list[list[dict[str, Any]]],
    run_id: str,
    max_attempts: int,
    backoff: float,
) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    for index, messages in enumerate(payloads):
        user = f"c9-same-user-{run_id}-{label}-u{index % users}"
        bodies.append(
            {
                "request_id": hashlib.sha256(f"{run_id}:{label}:{index}".encode()).hexdigest()[:32],
                "user_id": user,
                "session_id": f"{label}-s{index}",
                "messages": messages,
            }
        )
    started = time.perf_counter()
    results = await asyncio.gather(
        *(add_with_retries(client, body, max_attempts, backoff) for body in bodies)
    )
    wall = time.perf_counter() - started
    # Replay every request: an idempotent receipt must come back 200 with a nonzero raw count.
    replays = []
    for body in bodies:
        response = await client.post("/v1/add", json=body)
        payload = response.json() if response.status_code == 200 else {}
        replays.append(response.status_code == 200 and int(payload.get("raw_count", 0)) > 0)
    for user in sorted({body["user_id"] for body in bodies}):
        await client.post("/v1/delete", json={"user_id": user})
    first = Counter(str(r["attempts"][0]["status"]) for r in results)
    return {
        "scenario": label,
        "users": users,
        "adds": len(bodies),
        "wall_seconds": round(wall, 1),
        "first_attempt_status": dict(first),
        "first_attempt_5xx_or_timeout": sum(
            1 for r in results if r["attempts"][0]["status"] != 200
        ),
        "eventual_success": sum(1 for r in results if r["final_status"] == 200),
        "max_attempts_used": max(len(r["attempts"]) for r in results),
        "total_retries": sum(len(r["attempts"]) - 1 for r in results),
        "median_seconds_to_success": statistics.median(
            r["seconds_to_final"] for r in results if r["final_status"] == 200
        ) if any(r["final_status"] == 200 for r in results) else None,
        "replay_receipts_ok": sum(replays),
        "requests": results,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.data.read_bytes())
    # Distinct sessions per scenario: a repeated text would hit the server's embedding cache,
    # shorten the Add, and so shorten the very lock hold this measures.
    payloads = sessions(data, args.concurrency * len(SCENARIOS))
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    timeout = httpx.Timeout(args.client_timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=timeout) as client:
        version = (await client.get("/version")).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        scenarios = []
        for position, (label, users) in enumerate(SCENARIOS):
            scenarios.append(
                await run_scenario(
                    client,
                    label,
                    users or args.concurrency,
                    payloads[position * args.concurrency : (position + 1) * args.concurrency],
                    args.run_id,
                    args.max_attempts,
                    args.backoff,
                )
            )
            print(json.dumps({k: v for k, v in scenarios[-1].items() if k != "requests"}),
                  flush=True)
    return {
        "preregistration": "docs/preregistrations/2026-09-24-aml-c9-same-user-add-concurrency.md",
        "run_id": args.run_id,
        "git_commit": version.get("git_commit"),
        "variant": version.get("variant"),
        "concurrency": args.concurrency,
        "max_attempts": args.max_attempts,
        "backoff_seconds": args.backoff,
        "client_timeout_seconds": args.client_timeout,
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--backoff", type=float, default=5.0)
    parser.add_argument("--client-timeout", type=float, default=300.0)
    parser.add_argument("--expected-variant", default="C9_routed_specialists_grounded_graph_atomic")
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
