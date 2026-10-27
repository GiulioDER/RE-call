from __future__ import annotations

from types import SimpleNamespace

import recall_mcp.compat as compat
import recall_mcp.generation_admin as generation_admin
import recall_mcp.graph_first_api as graph_first_api
import recall_mcp.graph_projection as graph_projection
import recall_mcp.indexing as indexing
import recall_mcp.lifecycle as lifecycle
import recall_mcp.models as models
import recall_mcp.provenance as provenance
import recall_mcp.query_construction_api as query_construction_api
import recall_mcp.reasoning_admin as reasoning_admin
import recall_mcp.reasoning_api as reasoning_api
import recall_mcp.reasoning_common as reasoning_common
import recall_mcp.retrieval as retrieval
import recall_mcp.service as service
import recall_mcp.status as status
from recall_mcp import desktop_ingest as _desktop_ingest


def test_legacy_service_search_forwards_to_retrieval(monkeypatch) -> None:
    sentinel = object()

    def fake_search(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(retrieval, "search_memory", fake_search)

    assert service.search_memory("store", "embedder", "query") is sentinel


def test_legacy_service_search_forwards_the_settings_environment_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(retrieval, "search_memory", fake_search)
    environment = {"RECALL_MCP_TOOLS": "search", "RECALL_RETRIEVAL_PROFILE": "fast"}

    assert service.search_memory("store", "embedder", "query", env=environment) == "sentinel"
    assert captured["args"][-1] == environment


def test_generation_boundary_forwards_without_requiring_service_at_import_time(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(_desktop_ingest, "generation_ingest", lambda *args: sentinel)

    assert generation_admin.generation_ingest("store", "embedder", "stage", "text") is sentinel


def test_generation_boundary_forwards_the_settings_environment_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_ingest(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(_desktop_ingest, "generation_ingest", fake_ingest)
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


def test_graph_first_retrieval_is_owned_by_graph_first_api() -> None:
    assert graph_first_api.graph_first_retrieval.__module__ == "recall_mcp.graph_first_api"
    assert service.graph_first_retrieval.__module__ == "recall_mcp.service"


def test_query_construction_is_owned_by_query_construction_api() -> None:
    assert query_construction_api.query_construction_challenge.__module__ == (
        "recall_mcp.query_construction_api"
    )
    assert query_construction_api._query_construction_graph.__module__ == (
        "recall_mcp.query_construction_api"
    )
    assert service.query_construction_challenge.__module__ == "recall_mcp.service"


def test_reasoning_contract_helpers_are_owned_by_reasoning_common() -> None:
    assert service._reasoning_generation is reasoning_common._reasoning_generation
    assert service._query_construction_retrieval is reasoning_common._query_construction_retrieval
    assert service._same_generation is reasoning_common._same_generation


def test_reasoning_administration_is_owned_by_reasoning_admin() -> None:
    assert service.apply_command_for is reasoning_admin.apply_command_for
    assert service.reasoning_proposals is reasoning_admin.reasoning_proposals
    assert service.rewrite_plan is reasoning_admin.rewrite_plan


def test_retrieval_profile_startup_is_owned_by_retrieval_module() -> None:
    assert service.startup_retrieval_profile is retrieval.startup_retrieval_profile


def test_retrieval_execution_owner_is_not_service() -> None:
    assert service._Retrieval is retrieval._Retrieval
    assert service.MAX_QUERY_CHARS == retrieval.MAX_QUERY_CHARS
    assert service.MAX_SEARCH_K == retrieval.MAX_SEARCH_K
    assert retrieval._retrieve_trusted.__module__ == "recall_mcp.retrieval"
    assert service._cost_surface is retrieval._cost_surface
    assert service._search_hit_model is retrieval._search_hit_model
    assert service._evidence_item_model is retrieval._evidence_item_model
    assert service._search_advice is retrieval._search_advice
    assert service._evidence_advice is retrieval._evidence_advice
    assert retrieval.search_memory.__module__ == "recall_mcp.retrieval"
    assert retrieval.evidence_memory.__module__ == "recall_mcp.retrieval"


def test_status_operations_are_owned_by_status_module() -> None:
    assert service.JobLedger is status.JobLedger
    assert service.job_status is status.job_status
    assert service.calibration_status is status.calibration_status


def test_the_facade_contract_lists_every_server_module() -> None:
    """Every recall_mcp module but the server is a source of the contract forbidding the facade.

    The contract lists modules one by one, because import-linter reads an import inside a source
    package as internal; a module added to `recall_mcp` and not listed would be unchecked.
    Red proof (2026-10-27): removing "recall_mcp.retrieval" from the contract's `source_modules`
    in pyproject.toml fails the equality below with that module missing.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contracts = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "importlinter"
    ]["contracts"]
    (contract,) = [c for c in contracts if c["id"] == "mcp-modules-do-not-reach-the-service-facade"]
    modules = {
        f"recall_mcp.{path.stem}"
        for path in (root / "recall_mcp").glob("*.py")
        if path.stem not in {"__init__", "server", "service"}
    }

    assert contract["forbidden_modules"] == ["recall_mcp.service"]
    assert set(contract["source_modules"]) == modules, sorted(modules ^ set(contract["source_modules"]))


def test_trusted_retrieval_reads_the_reranker_and_admission_from_factories(monkeypatch) -> None:
    """`factories` is the one patch point of the reranker builder and the admission.

    Red proof (2026-10-27): master 970dbeb7b's `retrieval._retrieve_trusted` resolves both
    from names it imported from factories at load, so the patches below reach nothing and it
    fails on "factories was not consulted".
    """
    import contextlib

    import pytest

    import recall_mcp.factories as factories

    consulted: list[str] = []

    def builder(_profile, env=None):
        consulted.append("reranker")
        return None

    def admission(_profile):
        consulted.append("admission")
        return contextlib.nullcontext()

    class _Stop(Exception):
        pass

    def search(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr(factories, "_build_reranker", builder)
    monkeypatch.setattr(factories, "_admission", admission)
    store = SimpleNamespace(generation_id="g1")
    embedder = SimpleNamespace(dim=2, name="seam-fixture", embed=lambda texts: [[1.0, 0.0] for _ in texts])

    with pytest.raises(_Stop):
        retrieval._retrieve_trusted(store, embedder, "q", None, 3, None, None, env={}, trusted_search_fn=search)

    assert consulted == ["admission", "reranker"], f"factories was not consulted: {consulted}"
