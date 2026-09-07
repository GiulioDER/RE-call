from __future__ import annotations

from types import SimpleNamespace

import recall_mcp.compat as compat
import recall_mcp.generation_admin as generation_admin
import recall_mcp.graph_first_api as graph_first_api
import recall_mcp.graph_expansion as graph_expansion
import recall_mcp.graph_projection as graph_projection
import recall_mcp.indexing as indexing
import recall_mcp.lifecycle as lifecycle
import recall_mcp.provenance as provenance
import recall_mcp.reasoning_common as reasoning_common
import recall_mcp.retrieval as retrieval
import recall_mcp.service as service
import recall_mcp.status as status


def test_retrieval_search_is_owned_by_retrieval_module() -> None:
    assert retrieval.search_memory.__module__ == "recall_mcp.retrieval"
    assert service.search_memory.__module__ == "recall_mcp.service"


def test_retrieval_evidence_and_card_registry_are_owned_by_retrieval_module() -> None:
    assert retrieval.evidence_memory.__module__ == "recall_mcp.retrieval"
    assert retrieval.related_memory.__module__ == "recall_mcp.retrieval"
    assert service.register_evidence_cards is retrieval.register_evidence_cards
    assert service.related_memory is retrieval.related_memory


def test_generation_ingest_is_owned_by_generation_admin() -> None:
    assert generation_admin.generation_ingest.__module__ == "recall_mcp.generation_admin"
    assert service.generation_ingest is generation_admin.generation_ingest


def test_generation_calibration_operations_are_owned_by_generation_admin() -> None:
    assert generation_admin.run_calibration.__module__ == "recall_mcp.generation_admin"
    assert generation_admin.publish_calibration.__module__ == "recall_mcp.generation_admin"
    assert service.run_calibration.__module__ == "recall_mcp.service"
    assert service.publish_calibration.__module__ == "recall_mcp.service"


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


def test_provenance_operations_are_owned_by_provenance_module() -> None:
    assert provenance.apply_fact_memory.__module__ == "recall_mcp.provenance"
    assert provenance.current_facts_memory.__module__ == "recall_mcp.provenance"
    assert service.apply_fact_memory is provenance.apply_fact_memory
    assert service.current_facts_memory is provenance.current_facts_memory
    assert service._fact_write_dsn is provenance._fact_write_dsn


def test_lifecycle_operations_are_owned_by_lifecycle_module() -> None:
    assert service.current_state_memory is lifecycle.current_state_memory
    assert service.forget_memory is lifecycle.forget_memory
    assert service.memory_stats is lifecycle.memory_stats
    assert service.memory_inventory is lifecycle.memory_inventory
    assert service.MAX_FORGET_SOURCES == lifecycle.MAX_FORGET_SOURCES


def test_status_operations_are_owned_by_status_module() -> None:
    assert service.JobLedger is status.JobLedger
    assert service.job_status is status.job_status
    assert service.calibration_status is status.calibration_status


def test_indexing_implementation_is_owned_by_indexing_module() -> None:
    assert indexing.index_memory.__module__ == "recall_mcp.indexing"
    assert service._scrub_paths is indexing._scrub_paths
    assert service.DEFAULT_MAX_INDEX_FILES == indexing.DEFAULT_MAX_INDEX_FILES
    assert service.DEFAULT_MAX_INDEX_BYTES == indexing.DEFAULT_MAX_INDEX_BYTES


def test_graph_projection_cache_is_owned_by_graph_projection_module() -> None:
    assert service._GRAPH_PROJECTIONS is graph_projection._GRAPH_PROJECTIONS
    assert service._GRAPH_PROJECTION_CACHE_MAX == graph_projection._GRAPH_PROJECTION_CACHE_MAX
    assert service._GRAPH_PROJECTION_LOCK is graph_projection._GRAPH_PROJECTION_LOCK


def test_reasoning_contract_helpers_are_owned_by_reasoning_common() -> None:
    assert service._reasoning_generation is reasoning_common._reasoning_generation
    assert service._query_construction_generation is reasoning_common._query_construction_generation
    assert service._query_construction_retrieval is reasoning_common._query_construction_retrieval
    assert service._query_construction_evidence is reasoning_common._query_construction_evidence
    assert service._query_construction_anchors is reasoning_common._query_construction_anchors
    assert service._same_generation is reasoning_common._same_generation


def test_graph_first_retrieval_is_owned_by_graph_first_api() -> None:
    assert graph_first_api.graph_first_retrieval.__module__ == "recall_mcp.graph_first_api"
    assert service.graph_first_retrieval.__module__ == "recall_mcp.service"


def test_graph_expansion_is_owned_by_graph_expansion_module() -> None:
    assert graph_expansion._retrieval_graph.__module__ == "recall_mcp.graph_expansion"
    assert graph_expansion._expand_semantic_graph.__module__ == "recall_mcp.graph_expansion"
    assert service._retrieval_graph is graph_expansion._retrieval_graph
    assert service._expand_semantic_graph is graph_expansion._expand_semantic_graph
    assert service.MAX_GRAPH_RESCORING_CANDIDATES == graph_expansion.MAX_GRAPH_RESCORING_CANDIDATES


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
