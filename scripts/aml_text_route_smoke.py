"""Verify a routed specialist AML service keeps text memories on every specialist route.

This is a pre-run service check for the C8 and C9 routed specialist variants. It adds three
text-only sessions to one private tenant, searches them through each route the deterministic
router can choose for text (``multimodal`` via a visual word, ``code`` and ``context``), and
always deletes the tenant afterwards. It never calls an official AML endpoint or evaluator.

The multimodal cases guard the defect fixed in PR 719: text-only Adds carry no multimodal
manifest, and before that fix a text query containing a visual word ("image", "UI", ...)
routed to ``multimodal`` and returned an empty evidence list. Each case asserts the route it
exercised, so a green result cannot come from a different route than the one under test.
Under the ``data`` route gates (``/version`` ``route_gates``, PR 810) a visual word in plain text
routes to ``code`` instead, so those two cases expect ``code`` and must still return their text.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


#: Statuses AML itself retries for Add; a routed specialist service serialises Adds per tenant on
#: an advisory lock and answers a lock-wait timeout with 503.
ADD_RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524})
ADD_ATTEMPTS = 8


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
            with urlopen(request, timeout=180) as response:  # noqa: S310
                return Call(
                    response.status,
                    json.loads(response.read().decode("utf-8")),
                    {key.casefold(): value for key, value in response.headers.items()},
                )
        except HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": raw[:200]}
            return Call(
                exc.code,
                payload,
                {key.casefold(): value for key, value in exc.headers.items()},
            )


@dataclass(frozen=True)
class SearchCase:
    name: str
    query: str
    route: str
    must_contain: str


def _memories(marker: str) -> list[tuple[str, str]]:
    """Three text-only sessions: (session suffix, content). Each carries a unique marker."""
    return [
        (
            "registry",
            f"{marker}-REGISTRY: encrypt the container image with an RSA key pair before "
            "pushing it to the private registry.",
        ),
        (
            "frontend",
            f"{marker}-FRONTEND: the settings page button overlapped the footer; the fix "
            "moved it into the flex container in settings.css.",
        ),
        (
            "decision",
            f"{marker}-DECISION: in yesterday's planning discussion the team agreed to ship "
            "the migration on Thursday.",
        ),
    ]


def _cases(marker: str) -> list[SearchCase]:
    return [
        SearchCase(
            "visual_word_image",
            f"How do I create an encrypted image for {marker}-REGISTRY?",
            "multimodal",
            f"{marker}-REGISTRY",
        ),
        SearchCase(
            "visual_word_ui",
            f"What fixed the UI overlap in {marker}-FRONTEND?",
            "multimodal",
            f"{marker}-FRONTEND",
        ),
        SearchCase(
            "code_route",
            f"Which css file fixed {marker}-FRONTEND?",
            "code",
            f"{marker}-FRONTEND",
        ),
        SearchCase(
            "context_route",
            f"What did we decide yesterday in {marker}-DECISION?",
            "context",
            f"{marker}-DECISION",
        ),
    ]


def _add(client: Client, body: dict[str, Any], sleep: Callable[[float], None]) -> Call:
    result = Call(599, {}, {})
    for attempt in range(ADD_ATTEMPTS):
        result = client.call("/v1/add", body)
        if result.status not in ADD_RETRYABLE:
            return result
        sleep(min(30.0, 2.0 * 2**attempt))
    return result


def _contains(call: Call, text: str) -> bool:
    return any(text in json.dumps(item.get("content", "")) for item in call.payload.get("data", []))


def verify(
    client: Client,
    *,
    expected_variant: str,
    user_id: str | None = None,
    expect_commit: str | None = None,
    expect_compiler: bool | None = None,
    expect_reranker: bool | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Add three text memories, search every text route, then always delete the tenant."""
    nonce = uuid4().hex
    tenant_user = user_id or f"text-route-smoke-{nonce}"
    marker = f"TRS{nonce[:10].upper()}"
    version = client.call("/version")
    adds: list[Call] = []
    searches: dict[str, Call] = {}
    cleanup = Call(599, {}, {})
    cases = _cases(marker)
    try:
        for index, (suffix, content) in enumerate(_memories(marker)):
            adds.append(
                _add(
                    client,
                    {
                        "request_id": f"text-route-{nonce}-{index}",
                        "user_id": tenant_user,
                        "session_id": f"text-route-smoke/{nonce}/{suffix}",
                        "messages": [{"role": "user", "content": content}],
                    },
                    sleep,
                )
            )
        for case in cases:
            searches[case.name] = client.call(
                "/v1/search", {"query": case.query, "user_id": tenant_user, "top_k": 10}
            )
    finally:
        cleanup = client.call("/v1/delete", {"user_id": tenant_user})

    components = version.payload.get("active_components", {})
    route_gates = version.payload.get("route_gates", "route")
    # The grounded graph reads compiled records only, so it serves exactly when the compile does
    # (an inert sidecar is skipped since the 2026-10-20 CCA fix U-07). The operator states what the
    # deployment must serve; the service's own report is checked against it, never trusted as the
    # expectation, so a lost drop-in fails the smoke instead of changing what it expects.
    reported_compiler = components.get("compiler") is True
    reported_reranker = version.payload.get("reranker_enabled") is True
    graph_on = reported_compiler if expect_compiler is None else expect_compiler
    rerank_on = reported_reranker if expect_reranker is None else expect_reranker
    checks: dict[str, bool] = {
        "variant": version.status == 200 and version.payload.get("variant") == expected_variant,
        "graph_sidecar_matches_compiler": components.get("graph_sidecar") is graph_on,
        "compiler_as_expected": reported_compiler is graph_on,
        "reranker_as_expected": reported_reranker is rerank_on,
        "adds_stored": len(adds) == 3 and all(
            add.status == 200 and add.payload.get("status") == "stored" for add in adds
        ),
    }
    for case in cases:
        call = searches.get(case.name, Call(599, {}, {}))
        expected_route = case.route
        if route_gates == "data" and case.route == "multimodal":
            expected_route = "code"
        checks[f"{case.name}_route"] = (
            call.headers.get("x-recall-specialist-route") == expected_route
        )
        checks[f"{case.name}_returns_text_memory"] = (
            call.status == 200 and _contains(call, case.must_contain)
        )
        checks[f"{case.name}_graph_no_fallback"] = (
            call.headers.get("x-recall-graph-attempted") == ("1" if graph_on else "0")
            and call.headers.get("x-recall-graph-fallback") == "0"
        )
        if rerank_on:
            # A reranker that falls back on every Search serves the fused order and still
            # passes every check above, so its own headers are checked when it is on.
            checks[f"{case.name}_reranked"] = (
                call.headers.get("x-recall-reranker-completed") == "1"
                and call.headers.get("x-recall-reranker-fallback") == "0"
            )
    if expect_commit:
        checks["served_commit"] = str(version.payload.get("git_commit", "")).startswith(
            expect_commit
        )
    checks["cleanup"] = cleanup.status == 200
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "variant": expected_variant,
        "route_gates": route_gates,
        "served_commit": version.payload.get("git_commit"),
        "atomic": {
            name: {
                key: call.headers.get(f"x-recall-atomic-rescue-{key}")
                for key in ("attempted", "active", "fallback")
            }
            for name, call in searches.items()
        },
        "timestamp": int(clock()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--expected-variant", default="C9_routed_specialists_grounded_graph_atomic"
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=(
            "use this user id (for a service bound to one authorized user); default is unique. "
            "The tenant is DELETED at the end, so never pass a user id that holds real data."
        ),
    )
    parser.add_argument(
        "--expect-compiler",
        choices=("0", "1"),
        required=True,
        help="1 if the deployment must compile at Add, 0 if not (the graph follows it)",
    )
    parser.add_argument(
        "--expect-reranker",
        choices=("0", "1"),
        required=True,
        help="1 if every Search must be reranked, 0 if not",
    )
    parser.add_argument(
        "--expect-commit",
        default=None,
        help="fail unless /version git_commit starts with this (the build just deployed)",
    )
    args = parser.parse_args()
    api_key = os.environ.get("RECALL_AML_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RECALL_AML_API_KEY is required")
    result = verify(
        Client(args.base_url, api_key),
        expected_variant=args.expected_variant,
        user_id=args.user_id,
        expect_commit=args.expect_commit,
        expect_compiler=args.expect_compiler == "1",
        expect_reranker=args.expect_reranker == "1",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
