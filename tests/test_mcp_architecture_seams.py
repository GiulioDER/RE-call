from __future__ import annotations

from types import SimpleNamespace

import recall_mcp.compat as compat
import recall_mcp.generation_admin as generation_admin
import recall_mcp.graph_projection as graph_projection
import recall_mcp.indexing as indexing
import recall_mcp.lifecycle as lifecycle
import recall_mcp.models as models
import recall_mcp.provenance as provenance
import recall_mcp.reasoning_common as reasoning_common
import recall_mcp.reasoning_admin as reasoning_admin
import recall_mcp.reasoning_api as reasoning_api
import recall_mcp.retrieval as retrieval
import recall_mcp.service as service
import recall_mcp.status as status


def test_retrieval_boundary_forwards_to_the_legacy_service(monkeypatch) -> None:
    sentinel = object()

    def fake_search(*args, **kwargs):
        assert args == ("store", "embedder", "query", None, 5, None, None, False, False, "source", 3, False)
        assert kwargs == {}
        return sentinel

    monkeypatch.setattr(service, "search_memory", fake_search)

    assert retrieval.search_memory("store", "embedder", "query") is sentinel


def test_retrieval_boundary_forwards_the_settings_environment_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(service, "search_memory", fake_search)
    environment = {"RECALL_MCP_TOOLS": "search", "RECALL_RETRIEVAL_PROFILE": "fast"}

    assert retrieval.search_memory("store", "embedder", "query", env=environment) == "sentinel"
    assert captured["kwargs"] == {"env": environment}


def test_generation_boundary_forwards_without_requiring_service_at_import_time(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(service, "generation_ingest", lambda *args: sentinel)

    assert generation_admin.generation_ingest("store", "embedder", "stage", "text") is sentinel


def test_generation_boundary_forwards_the_settings_environment_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_ingest(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(service, "generation_ingest", fake_ingest)
    environment = {"RECALL_MCP_TOOLS": "search", "RECALL_INDEX_MODE": "generation"}

    assert (
        generation_admin.generation_ingest("store", "embedder", "stage", "text", env=environment)
        == "sentinel"
    )
    assert captured["kwargs"] == {"env": environment}


def test_legacy_service_serialization_name_is_the_compatibility_implementation() -> None:
    assert service.serving_json is compat.serving_json


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


def test_service_result_models_are_compatibility_aliases() -> None:
    for name in (
        "ForgetResult",
        "IndexResult",
        "MemoryStatsResult",
        "ReasoningAuditResult",
        "ReasoningProjectionResult",
        "ReasoningProposalItem",
        "ReasoningProposalResult",
        "RewritePlanResult",
    ):
        assert getattr(service, name) is getattr(models, name)


def test_reasoning_response_apis_are_owned_by_reasoning_api() -> None:
    assert reasoning_api.reasoning_query.__module__ == "recall_mcp.reasoning_api"
    assert reasoning_api.reasoning_audit.__module__ == "recall_mcp.reasoning_api"
    assert service.reasoning_query is reasoning_api.reasoning_query
    assert service.reasoning_audit is reasoning_api.reasoning_audit


def test_lifecycle_operations_are_owned_by_lifecycle_module() -> None:
    assert service.current_state_memory is lifecycle.current_state_memory
    assert service.forget_memory is lifecycle.forget_memory
    assert service.memory_inventory is lifecycle.memory_inventory
    assert service.memory_stats is lifecycle.memory_stats
    assert service.MAX_FORGET_SOURCES == lifecycle.MAX_FORGET_SOURCES


def test_indexing_operations_are_owned_by_indexing_module() -> None:
    assert service.index_memory is indexing.index_memory
    assert service.IndexPreflightError is indexing.IndexPreflightError
    assert service.DEFAULT_MAX_INDEX_FILES == indexing.DEFAULT_MAX_INDEX_FILES
    assert service.DEFAULT_MAX_INDEX_BYTES == indexing.DEFAULT_MAX_INDEX_BYTES


def test_provenance_operations_are_owned_by_provenance_module() -> None:
    assert service.apply_fact_memory is provenance.apply_fact_memory
    assert service.current_facts_memory is provenance.current_facts_memory
    assert service.register_evidence_cards is provenance.register_evidence_cards
    assert service.FACT_WRITE_DSN_ENV == provenance.FACT_WRITE_DSN_ENV


def test_graph_projection_operations_are_owned_by_graph_projection_module() -> None:
    assert service.reasoning_projection is graph_projection.reasoning_projection
    assert service._store_graph is graph_projection._store_graph
    assert service._store_graph_with_readiness is graph_projection._store_graph_with_readiness


def test_reasoning_contract_helpers_are_owned_by_reasoning_common() -> None:
    assert service._reasoning_generation is reasoning_common._reasoning_generation
    assert service._query_construction_retrieval is reasoning_common._query_construction_retrieval
    assert service._same_generation is reasoning_common._same_generation


def test_reasoning_administration_is_owned_by_reasoning_admin() -> None:
    assert service.apply_command_for is reasoning_admin.apply_command_for
    assert service.reasoning_proposals is reasoning_admin.reasoning_proposals
    assert service.rewrite_plan is reasoning_admin.rewrite_plan


def test_status_operations_are_owned_by_status_module() -> None:
    assert service.JobLedger is status.JobLedger
    assert service.job_status is status.job_status
    assert service.calibration_status is status.calibration_status
