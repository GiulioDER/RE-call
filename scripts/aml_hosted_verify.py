"""External contract and 16 by 16 concurrency verifier for a frozen hosted endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import statistics
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


@dataclass(frozen=True)
class Call:
    status: int
    payload: dict[str, Any]
    latency_ms: float


class Client:
    def __init__(self, base_url: str, key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.key = key

    def call(self, path: str, payload: dict[str, Any] | None = None) -> Call:
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.base_url + path,
            data=body,
            method="GET" if payload is None else "POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                status = response.status
                result = json.loads(response.read().decode())
        except HTTPError as exc:
            status = exc.code
            result = json.loads(exc.read().decode())
        return Call(status, result, (time.perf_counter() - started) * 1_000)


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentage)))
    return ordered[index]


def verify_contract(client: Client) -> dict[str, Any]:
    run = uuid4().hex
    user_a = f"verify-a-{run}"
    user_b = f"verify-b-{run}"
    request = {
        "request_id": f"add-{run}",
        "user_id": user_a,
        "session_id": "sessions/contract/part-1",
        "messages": [
            {
                "role": "user",
                "content": f"Exact marker {run} failed in Symbol.verify_path with E_VERIFY_42",
                "timestamp": 1_726_133_200_000,
            }
        ],
    }
    health = client.call("/health")
    version = client.call("/version")
    added = client.call("/v1/add", request)
    replayed = client.call("/v1/add", request)
    conflicting = client.call(
        "/v1/add", {**request, "messages": [{"role": "user", "content": "different"}]}
    )
    second_chunk = client.call(
        "/v1/add",
        {
            "request_id": f"part-2-{run}",
            "user_id": user_a,
            "session_id": "sessions/contract/part-1",
            "messages": [{"role": "assistant", "content": "pytest validation passed"}],
        },
    )
    other = client.call(
        "/v1/add",
        {
            "request_id": f"other-{run}",
            "user_id": user_b,
            "session_id": "sessions/contract/other",
            "messages": [{"role": "user", "content": f"forbidden-other-{run}"}],
        },
    )
    searches = {
        width: client.call(
            "/v1/search",
            {"query": "E_VERIFY_42 Symbol.verify_path", "user_id": user_a, "top_k": width},
        )
        for width in (1, 5, 100)
    }
    visible = "\n".join(
        item["content"] for result in searches.values() for item in result.payload.get("data", [])
    )
    checks = {
        "health": health.status == 200,
        "version": version.status == 200 and version.payload.get("product") == "RE-call Hosted 1.0",
        "add": added.status == 200
        and {
            "success": True,
            "request_id": request["request_id"],
            "user_id": user_a,
            "session_id": request["session_id"],
        }.items()
        <= added.payload.items(),
        "immediate_search": all(
            result.status == 200 and result.payload.get("data") for result in searches.values()
        ),
        "identical_replay": replayed.status == 200 and replayed.payload == added.payload,
        "conflicting_replay": conflicting.status == 409,
        "cross_chunk": second_chunk.status == 200,
        "peer_add": other.status == 200,
        "tenant_isolation": f"forbidden-other-{run}" not in visible,
        "top_k": all(
            len(result.payload.get("data", [])) <= width for width, result in searches.items()
        ),
    }
    deleted_a = client.call("/v1/delete", {"user_id": user_a})
    after_delete = client.call(
        "/v1/search", {"query": "E_VERIFY_42", "user_id": user_a, "top_k": 100}
    )
    peer_after = client.call(
        "/v1/search", {"query": f"forbidden-other-{run}", "user_id": user_b, "top_k": 5}
    )
    checks["delete"] = (
        deleted_a.status == 200
        and after_delete.status == 200
        and not after_delete.payload.get("data")
        and bool(peer_after.payload.get("data"))
    )
    client.call("/v1/delete", {"user_id": user_b})
    return {"passed": all(checks.values()), "checks": checks}


def verify_concurrency(client: Client) -> dict[str, Any]:
    run = uuid4().hex
    user = f"soak-{run}"

    def add(index: int) -> Call:
        return client.call(
            "/v1/add",
            {
                "request_id": f"{run}-{index}",
                "user_id": user,
                "session_id": f"sessions/soak/{index}",
                "messages": [{"role": "user", "content": f"soak marker {index}"}],
            },
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        adds = list(pool.map(add, range(16)))

    def search(index: int) -> Call:
        return client.call(
            "/v1/search",
            {"query": f"soak marker {index}", "user_id": user, "top_k": 12},
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        searches = list(pool.map(search, range(16)))
    client.call("/v1/delete", {"user_id": user})
    add_ms = [call.latency_ms for call in adds]
    search_ms = [call.latency_ms for call in searches]
    return {
        "passed": (
            all(call.status == 200 for call in adds + searches)
            and percentile(add_ms, 0.95) < 30_000
            and percentile(search_ms, 0.95) < 5_000
        ),
        "add_errors": sum(call.status != 200 for call in adds),
        "search_errors": sum(call.status != 200 for call in searches),
        "add_p95_ms": percentile(add_ms, 0.95),
        "search_p95_ms": percentile(search_ms, 0.95),
        "add_mean_ms": statistics.fmean(add_ms),
        "search_mean_ms": statistics.fmean(search_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=("contract", "concurrency", "all"), default="all")
    args = parser.parse_args()
    key = os.environ.get("RECALL_AML_API_KEY", "")
    if not key:
        raise SystemExit("RECALL_AML_API_KEY is required")
    client = Client(args.base_url, key)
    result: dict[str, Any] = {}
    if args.mode in ("contract", "all"):
        result["contract"] = verify_contract(client)
    if args.mode in ("concurrency", "all"):
        result["concurrency"] = verify_concurrency(client)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(section["passed"] for section in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
