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

    def __init__(
        self,
        *,
        empty_multimodal: bool = False,
        add_503s: int = 0,
        gates: str | None = None,
        visual_words: bool = True,
        compiler: bool = True,
        reranker: bool = False,
        rerank_fallback: bool = False,
    ) -> None:
        self.paths: list[str] = []
        self.memories: list[str] = []
        self.empty_multimodal = empty_multimodal
        self.add_503s = add_503s
        self.gates = gates
        self.visual_words = visual_words
        self.compiler = compiler
        self.reranker = reranker
        self.rerank_fallback = rerank_fallback

    def call(self, path, payload=None):
        self.paths.append(path)
        if path == "/version":
            version = {
                "variant": "C9_routed_specialists_grounded_graph_atomic",
                "git_commit": "abc123",
                # The graph serves exactly when the compile does (CCA U-07, 2026-10-20).
                "active_components": {
                    "graph_sidecar": self.compiler,
                    "compiler": self.compiler,
                    "atomic_rescue": True,
                },
                "reranker_enabled": self.reranker,
            }
            if self.gates is not None:
                version["route_gates"] = self.gates
            return Call(200, version, {})
        if path == "/v1/add":
            if self.add_503s:
                self.add_503s -= 1
                return Call(503, {"error": "service_unavailable"}, {})
            self.memories.append(payload["messages"][0]["content"])
            return Call(200, {"status": "stored"}, {})
        if path == "/v1/search":
            route = route_query(payload["query"], visual_words=self.visual_words)
            headers = {
                "x-recall-specialist-route": route,
                "x-recall-graph-attempted": "1" if self.compiler else "0",
                "x-recall-graph-fallback": "0",
                "x-recall-reranker-completed": "1" if self.reranker and not self.rerank_fallback else "0",
                "x-recall-reranker-fallback": "1" if self.rerank_fallback else "0",
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


def test_the_smoke_follows_the_data_route_gates() -> None:
    """Under the data route gates a visual word in plain text routes to code (PR 810), and a
    correctly deployed service must pass; one that still routes it to multimodal must not.

    Red proof, audit of #810: the pre-fix verifier expects ``multimodal`` for both visual-word
    cases whatever ``/version`` says, so the healthy data-gated service fails the first assertion.
    The second assertion fails if the verifier ignores the route instead (expected route removed
    from the check).
    """
    data_gated = verify(
        _Client(gates="data", visual_words=False),
        expected_variant="C9_routed_specialists_grounded_graph_atomic",
        sleep=lambda _: None,
    )
    assert data_gated["passed"] is True, json.dumps(data_gated["checks"])
    assert data_gated["route_gates"] == "data"

    misrouted = verify(
        _Client(gates="data", visual_words=True),
        expected_variant="C9_routed_specialists_grounded_graph_atomic",
        sleep=lambda _: None,
    )
    assert misrouted["passed"] is False
    assert misrouted["checks"]["visual_word_image_route"] is False


VARIANT = "C9_routed_specialists_grounded_graph_atomic"


def _verify(client: _Client, **expect: object) -> dict:
    return verify(client, expected_variant=VARIANT, sleep=lambda _: None, **expect)  # type: ignore[arg-type]


def test_the_full2_configuration_passes() -> None:
    """CCA V-09 (2026-10-20). Invariant: the Full 2 deployment (compile off, so the graph is
    skipped; reranker on and completing) passes when the operator states exactly that.

    Red proof: at 43693e37 `verify` takes no expect_compiler/expect_reranker, so this call raises
    TypeError; on the fix, mutating `graph_on` to always True fails `graph_sidecar_matches_compiler`.
    """
    result = _verify(
        _Client(compiler=False, reranker=True), expect_compiler=False, expect_reranker=True,
        expect_commit="abc1",
    )

    assert result["passed"] is True, json.dumps(result["checks"])
    assert result["checks"]["served_commit"] is True


def test_a_lost_compile_off_drop_in_fails_the_smoke() -> None:
    """CCA V-09. Invariant: when the operator expects the compile off and the service reports it
    on (a drop-in lost), the smoke fails instead of changing its expectation.

    Red proof: deriving the expectation from /version (as 43693e37's smoke did) passes this case.
    """
    result = _verify(_Client(compiler=True), expect_compiler=False, expect_reranker=False)

    assert result["passed"] is False
    assert result["checks"]["compiler_as_expected"] is False


def test_a_reranker_that_falls_back_fails_the_smoke() -> None:
    """CCA V-09. Invariant: with the reranker expected on, a Search served in fused order
    (reranker fallback) fails the smoke.

    Red proof: mutating the `_reranked` check away leaves this run passing.
    """
    result = _verify(
        _Client(compiler=False, reranker=True, rerank_fallback=True),
        expect_compiler=False,
        expect_reranker=True,
    )

    assert result["passed"] is False
    assert any(name.endswith("_reranked") and not ok for name, ok in result["checks"].items())


def test_a_lost_reranker_drop_in_fails_the_smoke() -> None:
    """CCA V-09. Invariant: the operator expects the reranker on and the service reports it off."""
    result = _verify(_Client(compiler=False, reranker=False), expect_compiler=False, expect_reranker=True)

    assert result["passed"] is False
    assert result["checks"]["reranker_as_expected"] is False


def test_the_wrong_served_commit_fails_the_smoke() -> None:
    """CCA V-09. Invariant: --expect-commit fails a service serving another build."""
    result = _verify(_Client(), expect_commit="ffff")

    assert result["passed"] is False
    assert result["checks"]["served_commit"] is False
