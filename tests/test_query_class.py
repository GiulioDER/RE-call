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


def test_routing_assigns_distinct_graph_budgets_by_query_category() -> None:
    """Each category gets the graph shape its retrieval objective needs.

    Invariant: list recall has the widest node and entity budgets, temporal retrieval is tight,
    and multi hop retrieval has the deepest operation budget. The failure mode is one shared
    default silently starving list breadth or allowing temporal over expansion. The intended red
    proof is to mutate ``route_query`` so every branch returns ``DEFAULT_GRAPH_BUDGET``; this node
    then fails on the first category specific assertion.
    """
    list_budget = route_query("List every project mentioned in the notes").graph_budget
    temporal_budget = route_query("When did the rollout change?").graph_budget
    multi_hop_budget = route_query("Why did the rollout change?").graph_budget

    assert list_budget.name == "list_recall"
    assert list_budget.max_graph_nodes > temporal_budget.max_graph_nodes
    assert list_budget.max_graph_entities > temporal_budget.max_graph_entities
    assert temporal_budget.name == "temporal"
    assert temporal_budget.max_graph_nodes < multi_hop_budget.max_graph_nodes
    assert multi_hop_budget.name == "multi_hop"
    assert multi_hop_budget.max_steps > list_budget.max_steps


@pytest.mark.parametrize(
    ("query", "expected_graph_expansion"),
    [
        ("How many rollout changes were there?", "off"),
        ("Who owns the rollout?", "off"),
        ("What is the answer to this single hop lookup?", "off"),
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
    this node mutates the reasoning service default from auto to off, which fails the active cases
    at the consumer policy assertion. The test observes the policy received by the reasoning
    consumer, not only the classifier helper.
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
