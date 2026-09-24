"""Add throughput of a live AML hosted service under the platform's concurrency, one arm per run.

Pre-registration: ``docs/preregistrations/2026-09-24-aml-c9-add-speed.md``.

AML sends Adds for many users at once but never two for one user, and a user's Adds arrive in
order. This reproduces that shape: ``--users`` workers, each owning one user and sending its
sessions one after another, retried on a 5xx or a client timeout with the same ``request_id``
(AML's documented contract). The server's internal ``RECALL_AML_ADD_CONCURRENCY`` is set by
whoever starts the server; this client only records what it was told and what it measured.

Sessions are real LoCoMo sessions. ``arm_sessions`` deals them to arms by size, so every arm
carries about the same text and no arm can hit another arm's embedding cache entries.

    python scripts/aml_add_throughput.py --base-url http://127.0.0.1:18121 \
        --data locomo10.json --arm 0 --arm-label k8a --server-add-concurrency 8 --out k8a.json
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

ARMS = 4
ADDS_PER_USER = 3
# Every sixth session in dataset order is kept out of every arm, for the compiler provider
# measurement, so the two measurements never share a session.
PROVIDER_STRIDE = 6


def all_sessions(data: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Every non-empty LoCoMo session, as AML message lists, in dataset order."""
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
    return found


def size(session: list[dict[str, Any]]) -> int:
    return sum(len(message["content"]) for message in session)


def provider_sessions(data: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    """The sessions reserved for the provider measurement, in dataset order."""
    reserved = all_sessions(data)[::PROVIDER_STRIDE]
    if len(reserved) < count:
        raise SystemExit(f"only {len(reserved)} reserved sessions, {count} asked")
    return reserved[:count]


def arm_sessions(data: list[dict[str, Any]], arm: int, per_arm: int) -> list[list[dict[str, Any]]]:
    """``per_arm`` sessions for ``arm``, dealt largest first in snake order across the arms."""
    pool = [s for i, s in enumerate(all_sessions(data)) if i % PROVIDER_STRIDE]
    pool.sort(key=size, reverse=True)
    needed = per_arm * ARMS
    if len(pool) < needed:
        raise SystemExit(f"only {len(pool)} sessions for {needed}")
    dealt: list[list[list[dict[str, Any]]]] = [[] for _ in range(ARMS)]
    for index, session in enumerate(pool[:needed]):
        lap, seat = divmod(index, ARMS)
        dealt[seat if lap % 2 == 0 else ARMS - 1 - seat].append(session)
    return dealt[arm]


async def add_with_retries(
    client: httpx.AsyncClient, body: dict[str, Any], max_attempts: int, backoff: float
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(max_attempts):
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
            break  # AML does not retry a 4xx, and here it would be a harness bug
        await asyncio.sleep(backoff)
    return {
        "request_id": body["request_id"],
        "user_id": body["user_id"],
        "characters": size(body["messages"]),
        "attempts": attempts,
        "final_status": attempts[-1]["status"],
        "seconds_to_final": round(time.perf_counter() - started, 2),
    }


async def user_worker(
    client: httpx.AsyncClient, bodies: list[dict[str, Any]], max_attempts: int, backoff: float
) -> list[dict[str, Any]]:
    """One user's Adds, strictly one after another, as AML sends them."""
    return [await add_with_retries(client, body, max_attempts, backoff) for body in bodies]


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.data.read_bytes())
    per_arm = args.users * ADDS_PER_USER
    payloads = arm_sessions(data, args.arm, per_arm)
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    timeout = httpx.Timeout(args.client_timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=timeout) as client:
        version = (await client.get("/version")).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        per_user: list[list[dict[str, Any]]] = [[] for _ in range(args.users)]
        for index, messages in enumerate(payloads):
            user = index % args.users
            per_user[user].append(
                {
                    "request_id": hashlib.sha256(
                        f"{args.run_id}:{args.arm_label}:{index}".encode()
                    ).hexdigest()[:32],
                    "user_id": f"c9-speed-{args.run_id}-{args.arm_label}-u{user}",
                    "session_id": f"{args.arm_label}-s{index}",
                    "messages": messages,
                }
            )
        started = time.perf_counter()
        results = [
            r
            for batch in await asyncio.gather(
                *(user_worker(client, bodies, args.max_attempts, args.backoff) for bodies in per_user)
            )
            for r in batch
        ]
        wall = time.perf_counter() - started
        for bodies in per_user:
            await client.post("/v1/delete", json={"user_id": bodies[0]["user_id"]})
    ok = [r for r in results if r["final_status"] == 200]
    latencies = sorted(r["seconds_to_final"] for r in ok)
    characters = sum(r["characters"] for r in results)
    return {
        "preregistration": "docs/preregistrations/2026-09-24-aml-c9-add-speed.md",
        "measurement": "add_throughput",
        "run_id": args.run_id,
        "arm": args.arm,
        "arm_label": args.arm_label,
        "server_add_concurrency": args.server_add_concurrency,
        "git_commit": version.get("git_commit"),
        "variant": version.get("variant"),
        "users": args.users,
        "adds": len(results),
        "characters": characters,
        "wall_seconds": round(wall, 1),
        "adds_per_minute": round(len(ok) / wall * 60, 2),
        "characters_per_second": round(characters / wall, 1),
        "eventual_success": len(ok),
        "first_attempt_status": dict(Counter(str(r["attempts"][0]["status"]) for r in results)),
        "total_retries": sum(len(r["attempts"]) - 1 for r in results),
        "max_attempts_used": max(len(r["attempts"]) for r in results),
        "add_seconds_p50": statistics.median(latencies) if latencies else None,
        "add_seconds_p90": latencies[int(len(latencies) * 0.9)] if latencies else None,
        "requests": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--arm", type=int, required=True, choices=range(ARMS))
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--server-add-concurrency", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--backoff", type=float, default=5.0)
    parser.add_argument("--client-timeout", type=float, default=300.0)
    parser.add_argument("--expected-variant", default="C9_routed_specialists_grounded_graph_atomic")
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
