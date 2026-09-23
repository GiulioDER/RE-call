"""Contract proofs for the routed specialist text-route smoke verifier.

Red proof receipt: replacing ``call.status == 200 and _contains(call, case.must_contain)`` in
``verify`` with ``call.status == 200`` makes
``test_an_empty_multimodal_result_fails_the_smoke`` fail on ``result["passed"] is False``,
because the empty evidence list the pre-PR-719 service returned would then pass.
"""

from __future__ import annotations

import json

from recall_aml.specialists import route_query
from scripts.aml_text_route_smoke import Call, _cases, verify


class _Client:
    """A service stub answering every route with the memory the query names."""

    def __init__(self, *, empty_multimodal: bool = False, add_503s: int = 0) -> None:
        self.paths: list[str] = []
        self.memories: list[str] = []
        self.empty_multimodal = empty_multimodal
        self.add_503s = add_503s

    def call(self, path, payload=None):
        self.paths.append(path)
        if path == "/version":
            return Call(
                200,
                {
                    "variant": "C9_routed_specialists_grounded_graph_atomic",
                    "git_commit": "abc123",
                    "active_components": {"graph_sidecar": True, "atomic_rescue": True},
                },
                {},
            )
        if path == "/v1/add":
            if self.add_503s:
                self.add_503s -= 1
                return Call(503, {"error": "service_unavailable"}, {})
            self.memories.append(payload["messages"][0]["content"])
            return Call(200, {"status": "stored"}, {})
        if path == "/v1/search":
            route = route_query(payload["query"])
            headers = {
                "x-recall-specialist-route": route,
                "x-recall-graph-attempted": "1",
                "x-recall-graph-fallback": "0",
            }
            if route == "multimodal" and self.empty_multimodal:
                return Call(200, {"data": []}, headers)
            data = [
                {"id": str(index), "content": memory}
                for index, memory in enumerate(self.memories)
                if any(token in memory for token in payload["query"].split())
            ]
            return Call(200, {"data": data}, headers)
        if path == "/v1/delete":
            return Call(200, {"status": "deleted"}, {})
        raise AssertionError(path)


def test_every_case_query_reaches_the_route_it_claims() -> None:
    """The smoke is only meaningful if each query really takes the route it asserts."""
    for case in _cases("TRSABCDEF0123"):
        assert route_query(case.query) == case.route, case.name
    assert {case.route for case in _cases("M")} == {"multimodal", "code", "context"}


def test_a_healthy_service_passes_and_is_always_cleaned_up() -> None:
    client = _Client(add_503s=2)
    result = verify(
        client,
        expected_variant="C9_routed_specialists_grounded_graph_atomic",
        sleep=lambda _: None,
    )

    assert result["passed"] is True, json.dumps(result["checks"])
    assert client.paths[0] == "/version" and client.paths[-1] == "/v1/delete"
    assert client.paths.count("/v1/add") == 5
    assert client.paths.count("/v1/search") == 4


def test_an_empty_multimodal_result_fails_the_smoke() -> None:
    """The pre-PR-719 behaviour: a visual word routes to multimodal and returns no evidence."""
    result = verify(
        _Client(empty_multimodal=True),
        expected_variant="C9_routed_specialists_grounded_graph_atomic",
        sleep=lambda _: None,
    )

    assert result["passed"] is False
    assert result["checks"]["visual_word_image_returns_text_memory"] is False
    assert result["checks"]["visual_word_ui_returns_text_memory"] is False
    assert result["checks"]["code_route_returns_text_memory"] is True


def test_the_wrong_variant_fails_the_smoke() -> None:
    result = verify(
        _Client(),
        expected_variant="C8_routed_specialists_grounded_graph",
        sleep=lambda _: None,
    )

    assert result["passed"] is False and result["checks"]["variant"] is False
