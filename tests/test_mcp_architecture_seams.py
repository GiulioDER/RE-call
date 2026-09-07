from __future__ import annotations

from types import SimpleNamespace

import recall_mcp.compat as compat
import recall_mcp.generation_admin as generation_admin
import recall_mcp.retrieval as retrieval
import recall_mcp.service as service


def test_retrieval_search_is_owned_by_retrieval_module() -> None:
    assert retrieval.search_memory.__module__ == "recall_mcp.retrieval"
    assert service.search_memory.__module__ == "recall_mcp.service"


def test_retrieval_evidence_and_card_registry_are_owned_by_retrieval_module() -> None:
    assert retrieval.evidence_memory.__module__ == "recall_mcp.retrieval"
    assert service.register_evidence_cards is retrieval.register_evidence_cards


def test_generation_boundary_forwards_without_requiring_service_at_import_time(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(service, "generation_ingest", lambda *args: sentinel)

    assert generation_admin.generation_ingest("store", "embedder", "stage", "text") is sentinel


def test_legacy_service_serialization_name_is_the_compatibility_implementation() -> None:
    assert service.serving_json is compat.serving_json


def test_retrieval_profile_startup_is_owned_by_retrieval_module() -> None:
    assert service.startup_retrieval_profile is retrieval.startup_retrieval_profile


def test_retrieval_execution_owner_is_not_service() -> None:
    assert service._Retrieval is retrieval._Retrieval
    assert service.MAX_QUERY_CHARS == retrieval.MAX_QUERY_CHARS
    assert service.MAX_SEARCH_K == retrieval.MAX_SEARCH_K
    assert retrieval._retrieve_trusted.__module__ == "recall_mcp.retrieval"
    assert service._cost_surface is retrieval._cost_surface
    assert service._evidence_advice is retrieval._evidence_advice


def test_compatibility_serialization_omits_empty_additive_fields() -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        explanation=None,
        related_items=[],
        related_diagnostics=[],
        model_dump_json=lambda **kwargs: captured.update(kwargs) or "{}",
    )

    assert compat.serving_json(result) == "{}"

    assert captured["exclude"] == {"explanation", "related_items", "related_diagnostics"}
