from __future__ import annotations

import pytest

from recall.runtime_route import RouteConfigurationError, resolve_runtime_route


def test_one_explicit_route_is_shared_by_read_and_write_paths() -> None:
    route = resolve_runtime_route(
        {
            "RECALL_ENV": "development",
            "RECALL_INDEX_MODE": "generation",
            "RECALL_TABLE": "legacy_chunks",
        }
    )

    assert route.uses_generation
    assert route.identity() == {
        "mode": "generation",
        "environment": "development",
        "source": "RECALL_INDEX_MODE",
        "table": "recall_chunks_v1",
        "explicit": True,
        "enterprise": False,
    }


def test_production_cannot_select_legacy() -> None:
    with pytest.raises(RouteConfigurationError, match="production requires"):
        resolve_runtime_route({"RECALL_ENV": "production", "RECALL_INDEX_MODE": "legacy"})


def test_enterprise_cannot_select_legacy() -> None:
    with pytest.raises(RouteConfigurationError, match="enterprise control plane"):
        resolve_runtime_route({"RECALL_ENV": "development"}, enterprise=True)


def test_development_compatibility_default_is_visible() -> None:
    route = resolve_runtime_route({"RECALL_ENV": "development"})

    assert route.mode == "legacy"
    assert not route.explicit
    assert "compatibility default" in route.describe()


def test_invalid_route_configuration_names_the_setting() -> None:
    with pytest.raises(RouteConfigurationError, match="RECALL_INDEX_MODE"):
        resolve_runtime_route({"RECALL_INDEX_MODE": "surprise"})
