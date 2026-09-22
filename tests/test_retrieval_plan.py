from __future__ import annotations

import json

import pytest

from recall.retrieval_plan import (
    DEFAULT_RETRIEVAL_ROUTE_ID,
    RetrievalPlan,
    RetrievalPlanConfigurationError,
    RetrievalPlanError,
    RetrievalPlanResolver,
)
from recall_mcp.stores import StoreRegistry


def _resolver() -> RetrievalPlanResolver:
    return RetrievalPlanResolver.from_env(
        {
            "RECALL_RETRIEVAL_PLANS_JSON": json.dumps(
                {
                    "version": 1,
                    "default_route": "memory",
                    "routes": [
                        {
                            "id": "memory",
                            "primary_tenant": "memory",
                            "rescue_tenant": "re-call-code-gen",
                            "allowed_tenants": ["memory", "re-call-code-gen", "re-call-docs"],
                            "scopes": ["memory"],
                            "modalities": ["text"],
                            "primary_limit": 5,
                            "rescue_limit": 2,
                            "latency_budget_ms": 900,
                        },
                        {
                            "id": "docs",
                            "primary_tenant": "re-call-docs",
                            "allowed_tenants": ["re-call-docs"],
                            "source_prefixes": ["docs/"],
                            "query_any": ["api", "documentation"],
                        },
                    ],
                }
            )
        }
    )


def test_unconfigured_planning_preserves_single_tenant_compatibility() -> None:
    """Mutation proof: changing the fallback tenant or adding a rescue leg must fail this node.

    The targeted production symbol is ``_compatibility_plan``. The invariant is that an unset
    route plan is observationally one tenant, which protects existing callers from federation.
    """
    plan = RetrievalPlanResolver.from_env({}).resolve(
        current_tenant="default", query="anything"
    )

    assert plan.route_id == DEFAULT_RETRIEVAL_ROUTE_ID
    assert plan.selected_tenants == ("default",)
    assert plan.max_fanout == 1
    assert plan.rescue_tenant is None
    assert plan.selection_reason == "compatibility_default"


def test_from_env_without_mapping_reads_process_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "RECALL_RETRIEVAL_PLANS_JSON",
        json.dumps(
            {
                "version": 1,
                "routes": [
                    {
                        "id": "env-route",
                        "primary_tenant": "memory",
                        "allowed_tenants": ["memory"],
                    }
                ],
            }
        ),
    )
    resolver = RetrievalPlanResolver.from_env()
    assert resolver.configured


def test_explicit_scope_and_modality_select_a_primary_without_score_fusion() -> None:
    """Mutation proof: removing the scope or modality match must select the wrong physical tenant.

    The targeted production symbol is ``RetrievalPlanResolver.resolve``. The assertion observes
    the selected leg and its bound limit, not a score or a retrieval side effect.
    """
    plan = _resolver().resolve(
        current_tenant="logical-user",
        query="show the memory",
        request_scope="memory",
        modality="text",
    )

    assert plan.primary_tenant == "memory"
    assert plan.selected_tenants == ("memory",)
    assert plan.primary_limit == 5
    assert plan.selection_reason == "request_scope+modality"
    assert "score" not in plan.as_dict()


def test_ambiguous_request_names_one_bounded_rescue_leg() -> None:
    """Mutation proof: dropping the rescue leg or raising fanout breaks the bounded ambiguity rule.

    The targeted production symbol is ``RetrievalPlanResolver.resolve``. The exact assertion is
    intended to fail if the implementation fans out to every configured tenant or rescues twice.
    """
    plan = _resolver().resolve(current_tenant="logical-user", query="unclear request")

    assert plan.primary_tenant == "memory"
    assert plan.rescue_tenant == "re-call-code-gen"
    assert plan.selected_tenants == ("memory", "re-call-code-gen")
    assert len(plan.selected_legs) == 2
    assert plan.max_fanout == 2
    assert plan.rescue_limit == 2
    assert "ambiguous_bounded_rescue" in plan.selection_reason


def test_specific_source_route_does_not_rescue() -> None:
    plan = _resolver().resolve(
        current_tenant="logical-user",
        query="documentation api",
        source="docs/API.md",
        modality="text",
    )

    assert plan.route_id == "docs"
    assert plan.selected_tenants == ("re-call-docs",)
    assert plan.rescue_tenant is None


def test_invalid_config_fails_closed_before_request_selection() -> None:
    with pytest.raises(RetrievalPlanConfigurationError, match="version"):
        RetrievalPlanResolver.from_env(
            {"RECALL_RETRIEVAL_PLANS_JSON": json.dumps({"version": 2, "routes": []})}
        )


def test_plan_rejects_unallowed_rescue_and_unbounded_fanout() -> None:
    with pytest.raises(RetrievalPlanError, match="rescue_tenant"):
        RetrievalPlan(
            route_id="bad",
            primary_tenant="memory",
            rescue_tenant="code",
            allowed_tenants=frozenset({"memory"}),
        )
    with pytest.raises(RetrievalPlanError, match="max_fanout"):
        RetrievalPlan(
            route_id="bad",
            primary_tenant="memory",
            allowed_tenants=frozenset({"memory"}),
            max_fanout=2,
        )


def test_registry_rejects_a_selected_tenant_outside_its_allowlist() -> None:
    """Mutation proof: removing the allowlist subtraction must make this assertion fail.

    The targeted production symbol is ``StoreRegistry.validate_retrieval_plan``. This is the
    security boundary that prevents a valid route document from expanding the serving surface.
    """
    registry = object.__new__(StoreRegistry)
    registry._allowed = frozenset({"memory"})
    plan = RetrievalPlan(
        route_id=DEFAULT_RETRIEVAL_ROUTE_ID,
        primary_tenant="re-call-docs",
        allowed_tenants=frozenset({"re-call-docs"}),
    )

    with pytest.raises(PermissionError, match="allowlist"):
        registry.validate_retrieval_plan(plan)
