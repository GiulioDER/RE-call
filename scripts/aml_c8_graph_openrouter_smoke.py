"""Verify the isolated C8 graph path through the configured OpenRouter compiler.

This is a service compatibility check.  It creates and deletes one unique private tenant and does
not invoke any official AML endpoint or evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


EXPECTED_VARIANT = "C8_routed_specialists_grounded_graph"
EXPECTED_PROVIDER = "openrouter"
EXPECTED_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True)
class Call:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str]


class Client:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def call(self, path: str, payload: dict[str, Any] | None = None) -> Call:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self._base_url + path,
            data=body,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                return Call(
                    response.status,
                    json.loads(response.read().decode("utf-8")),
                    dict(response.headers.items()),
                )
        except HTTPError as exc:
            return Call(
                exc.code,
                json.loads(exc.read().decode("utf-8")),
                dict(exc.headers.items()),
            )


def verify(client: Client, *, clock=time.time) -> dict[str, Any]:
    """Exercise real Add compilation and graph Search, then always delete the tenant."""
    nonce = uuid4().hex
    user_id = f"c8-graph-openrouter-{nonce}"
    marker = f"C8_GRAPH_OPENROUTER_{nonce[:12]}"
    version = client.call("/version")
    add = Call(599, {}, {})
    status = Call(599, {}, {})
    search = Call(599, {}, {})
    cleanup = Call(599, {}, {})
    try:
        add = client.call(
            "/v1/add",
            {
                "request_id": f"graph-add-{nonce}",
                "user_id": user_id,
                "session_id": f"c8-graph-openrouter/{nonce}",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{marker}: src/graph_target.py raised GraphTargetError. "
                            "Apply the exact repair in src/graph_target.py."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"{marker}: GraphTargetError was fixed in src/graph_target.py. "
                            "pytest tests/test_graph_target.py passed."
                        ),
                    },
                ],
            },
        )
        status = client.call("/v1/corpus/status", {"user_id": user_id})
        search = client.call(
            "/v1/search", {"query": marker, "user_id": user_id, "top_k": 12}
        )
    finally:
        cleanup = client.call("/v1/delete", {"user_id": user_id})

    components = version.payload.get("active_components", {})
    relation_hits = search.headers.get("X-Recall-Graph-Relation-Hits", "0")
    checks = {
        "isolated_c8": (
            version.status == 200
            and version.payload.get("variant") == EXPECTED_VARIANT
            and components.get("graph_sidecar") is True
        ),
        "openrouter_gpt_4o_mini": (
            version.payload.get("generation_provider") == EXPECTED_PROVIDER
            and version.payload.get("generation_model") == EXPECTED_MODEL
        ),
        "live_compiler_created_records": (
            add.status == 200
            and add.payload.get("compiler_fallback") is False
            and int(add.payload.get("compiled_count", 0)) > 0
        ),
        "grounded_relations_persisted": (
            status.status == 200
            and int(status.payload.get("authored_relation_count", 0)) > 0
            and int(status.payload.get("eligible_relation_count", 0)) > 0
        ),
        "graph_search_used_relation": (
            search.status == 200
            and search.headers.get("X-Recall-Graph-Attempted") == "1"
            and search.headers.get("X-Recall-Graph-Fallback") == "0"
            and int(relation_hits) > 0
            and bool(search.payload.get("data"))
        ),
        "cleanup": cleanup.status == 200,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "provider": EXPECTED_PROVIDER,
        "model": EXPECTED_MODEL,
        "variant": EXPECTED_VARIANT,
        "timestamp": int(clock()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    api_key = os.environ.get("RECALL_AML_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RECALL_AML_API_KEY is required")
    result = verify(Client(args.base_url, api_key))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
