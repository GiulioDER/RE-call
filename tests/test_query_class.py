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


def test_exact_cost_counts_evidence_and_full_input_separately() -> None:
    class Counter:
        tokenizer_id = "cl100k_base"
        tokenizer_revision = "fixture"

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    cost = prompt_token_cost("system prompt", "evidence payload", Counter())
    assert cost == {"evidence_tokens_exact": 2, "input_tokens_exact": 4}
