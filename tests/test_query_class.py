import pytest

from benchmarks.evidence_tokens import prompt_token_cost
from recall.query_class import classify_query, route_query, routing_mode


def test_query_class_precedence_is_deterministic() -> None:
    result = classify_query("When did the rollout change and why?")
    assert result.query_class == "temporal"
    assert result.matched_rules
    assert result.classifier_version == "query-class-v1"


def test_routing_uses_related_quality_for_relational_queries() -> None:
    result = route_query("Why did the rollout change?")
    assert result.profile == "quality"
    assert result.related_expansion is True
    assert result.expansion_mode == "structure"


def test_unknown_query_is_safe_fast_fallback() -> None:
    result = route_query("🌱")
    assert result.query_class == "unknown"
    assert result.profile == "fast"
    assert result.related_expansion is False


def test_routing_mode_defaults_to_shadow_and_validates_active_opt_in() -> None:
    assert routing_mode() == "shadow"
    assert routing_mode("active") == "active"


@pytest.mark.parametrize(
    ("query", "expected_graph_expansion"),
    [
        ("How many rollout changes were there?", "off"),
        ("Who owns the rollout?", "off"),
        ("What is the answer to this single-hop lookup?", "off"),
        ("What is the chain from the project to the service?", "one_hop"),
        ("When did the rollout change?", "one_hop"),
        ("List every project that uses the service.", "one_hop"),
        ("Compare project A versus project B.", "one_hop"),
        ("Is project A better than project B?", "one_hop"),
    ],
)
def test_reasoning_query_uses_category_aware_graph_activation(
    query: str, expected_graph_expansion: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automatic graph activation is conservative and category aware.

    Invariant: numeric and direct single hop questions keep graph expansion off, while multi hop,
    temporal, list completion, and explicit comparison questions select one hop. Red proof for
    node ``tests/test_query_class.py::test_reasoning_query_uses_category_aware_graph_activation``:
    the baseline service default at ``recall_mcp/service.py::reasoning_query`` was ``off``, so the
    active cases failed the intended assertion with ``off`` instead of ``one_hop``. The test
    observes the policy received by the reasoning consumer, not a new classifier symbol.
    """
    from recall_mcp import service

    class Store:
        tenant = "tenant-a"
        generation_id = "legacy"

    captured: list[object] = []
    monkeypatch.setattr(service, "reason", lambda request: captured.append(request) or "ok")

    assert service.reasoning_query(Store(), object(), query) == "ok"
    assert captured[0].policy.graph_expansion == expected_graph_expansion


def test_exact_cost_counts_evidence_and_full_input_separately() -> None:
    class Counter:
        tokenizer_id = "cl100k_base"
        tokenizer_revision = "fixture"

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    cost = prompt_token_cost("system prompt", "evidence payload", Counter())
    assert cost == {"evidence_tokens_exact": 2, "input_tokens_exact": 4}
