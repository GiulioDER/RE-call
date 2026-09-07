from __future__ import annotations

from types import SimpleNamespace

import recall_mcp.compat as compat
import recall_mcp.generation_admin as generation_admin
import recall_mcp.retrieval as retrieval
import recall_mcp.service as service


def test_retrieval_boundary_forwards_to_the_legacy_service(monkeypatch) -> None:
    sentinel = object()

    def fake_search(*args, **kwargs):
        assert args == ("store", "embedder", "query", None, 5, None, None, False, False, "source", 3, False)
        assert kwargs == {}
        return sentinel

    monkeypatch.setattr(service, "search_memory", fake_search)

    assert retrieval.search_memory("store", "embedder", "query") is sentinel


def test_generation_boundary_forwards_without_requiring_service_at_import_time(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(service, "generation_ingest", lambda *args: sentinel)

    assert generation_admin.generation_ingest("store", "embedder", "stage", "text") is sentinel


def test_legacy_service_serialization_name_is_the_compatibility_implementation() -> None:
    assert service.serving_json is compat.serving_json


def test_retrieval_profile_startup_is_owned_by_retrieval_module() -> None:
    assert service.startup_retrieval_profile is retrieval.startup_retrieval_profile


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
