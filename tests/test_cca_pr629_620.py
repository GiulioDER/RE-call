"""Regressions for the CCA audit of merged PRs 629 to 620 (DEEP tier, run 20261030T085223Z).

Each test names the finding it pins, the invariant, and the defect it catches. Red proof: every
test was run against the pre-fix code at `63e89b1d0` (the audited base) and failed in its own
assertion, then passed on the fixed tree; the receipts are in the pull request.
"""

from __future__ import annotations

import asyncio
import importlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import recall.query_entity_resolution as resolution
from recall.answer_provider import OllamaAnswerProvider, resolve_answer_provider
from recall.query_class import (
    DEFAULT_GRAPH_BUDGET,
    LIST_RECALL_GRAPH_BUDGET,
    MULTI_HOP_GRAPH_BUDGET,
    TEMPORAL_GRAPH_BUDGET,
)
from recall.semantic_graph import SemanticEntity, SemanticGraphProjection, SemanticRelation
from recall.trust_policy import TrustPolicy
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport
from recall_mcp import reasoning_engine
from recall_mcp import retrieval as retrieval_module
from recall_mcp import semantic_graph_cache
from recall_mcp import service
from recall_mcp.graph_expansion import GRAPH_FIRST_SEED_K

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 30, tzinfo=UTC)


def _entity(index: int, name: str) -> SemanticEntity:
    return SemanticEntity(
        id=f"entity-{index}",
        tenant_id="tenant-a",
        generation_id="generation-a",
        canonical_name=name,
        normalized_name=name.casefold(),
        kind="project",
    )


def _graph(entities: list[SemanticEntity], relations: tuple = ()) -> SemanticGraphProjection:
    return SemanticGraphProjection(
        schema_version=1,
        graph_id="graph-a",
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        entities=tuple(entities),
        mentions=(),
        relations=relations,
        diagnostics=(),
    )


def test_bug001_a_0000_token_does_not_break_query_entity_resolution() -> None:
    """BUG-001. Invariant: a 4 digit token that is not a valid year is skipped, never raised.

    Defect caught: `_YEAR_RE` matched `0000` and `date(0, 1, 1)` raised ValueError, so one entity
    labelled `v0000` failed every graph first reasoning query on the tenant.
    Production line: the `_YEAR_RE` loop of `recall.query_entity_resolution.extract_query_dates`.
    """
    graph = _graph([_entity(1, "v0000"), _entity(2, "Release")])
    try:
        dates = resolution.extract_query_dates("incident 0000", reference_time=NOW)
        resolved = resolution.resolve_query_entities(graph, "Release plan", reference_time=NOW)
    except ValueError as exc:  # the defect: an uncaught ValueError from date(0, 1, 1)
        pytest.fail(f"a 0000 token raised instead of being skipped: {exc}")
    assert dates == ()
    assert resolved.entity_ids == ("entity-2",)


def test_perf001_label_work_is_not_repeated_per_request(monkeypatch) -> None:
    """PERF-001. Invariant: label normalization is done once per loaded graph, not per request.

    Defect caught: `resolve_query_entities` normalized every label (and re-normalized every
    clause per label) on every request, so its cost grew with the corpus.
    Production line: `_label_index` in `recall.query_entity_resolution`.
    """
    graph = _graph([_entity(index, f"Project {index}") for index in range(200)])
    query = "Where is the rollout plan? And who owns it"
    first = resolution.resolve_query_entities(graph, query, reference_time=NOW)

    calls = 0
    real = resolution.normalize_entity_name

    def counting(value: str) -> str:
        nonlocal calls
        calls += 1
        return real(value)

    monkeypatch.setattr(resolution, "normalize_entity_name", counting)
    second = resolution.resolve_query_entities(graph, query, reference_time=NOW)

    assert second == first
    assert calls <= len(first.clauses), f"{calls} normalizations for 200 labels on a repeat query"


def test_perf002_adjacency_cache_hit_does_not_walk_every_relation() -> None:
    """PERF-002. Invariant: a repeat request for one loaded graph reuses its adjacency in O(1).

    Defect caught: the cache key embedded a tuple of every relation, built and hashed per request.
    Production line: the key in `recall_mcp.semantic_graph_cache._semantic_graph_indexes`.
    """

    class CountingRelations(tuple):
        walks = 0

        def __iter__(self):
            type(self).walks += 1
            return super().__iter__()

    relation = SemanticRelation(
        id="relation-1",
        tenant_id="tenant-a",
        generation_id="generation-a",
        subject_id="entity-1",
        object_id="entity-2",
        relation="supports",
        evidence_chunk_ids=("seed",),
        extraction_method="metadata",
        confidence=1.0,
    )
    semantic = _graph([_entity(1, "Seed"), _entity(2, "Neighbor")], (relation,))
    object.__setattr__(semantic, "relations", CountingRelations(semantic.relations))
    semantic_graph_cache._reset_graph_projection_cache()

    first = semantic_graph_cache._semantic_graph_indexes(semantic)
    CountingRelations.walks = 0
    second = semantic_graph_cache._semantic_graph_indexes(semantic)

    assert second is first
    assert CountingRelations.walks == 0


def test_num001_every_category_budget_leaves_room_beyond_the_graph_first_seeds() -> None:
    """NUM-001. Invariant: each category budget admits at least one graph neighbour.

    Defect caught: the temporal budget was 8 nodes, exactly the 8 trusted graph first seeds that
    count against it, so `candidate_budget = max(0, 8 - 8)` switched expansion off.
    Production line: `TEMPORAL_GRAPH_BUDGET` in `recall.query_class`.
    """
    for budget in (
        LIST_RECALL_GRAPH_BUDGET,
        TEMPORAL_GRAPH_BUDGET,
        MULTI_HOP_GRAPH_BUDGET,
        DEFAULT_GRAPH_BUDGET,
    ):
        assert budget.max_graph_nodes - GRAPH_FIRST_SEED_K >= 1, budget


def test_num002_default_ollama_window_holds_the_default_evidence_budget() -> None:
    """NUM-002. Invariant: the default Ollama num_ctx leaves room for 2048 evidence tokens.

    Defect caught: `num_ctx` defaulted to 1024 with 512 output tokens inside it, so the default
    2048 token evidence budget was silently truncated; a window no larger than the output was
    also accepted. Production lines: the `context_tokens` defaults and the new bound in
    `recall.answer_provider`.
    """
    provider = resolve_answer_provider(
        {"RECALL_REASONING_ANSWER_ENABLED": "1", "RECALL_REASONING_ANSWER_MODEL": "m"}
    )
    assert provider is not None
    assert provider.context_tokens - provider.max_tokens >= 2048
    with pytest.raises(ValueError, match="exceed answer max tokens"):
        OllamaAnswerProvider(object(), model_id="m", max_tokens=1024, context_tokens=1024)


def test_num002_control_hosted_providers_do_not_check_the_ollama_window() -> None:
    """NUM-002 control, nonbehavioural for the original defect: the new window bound must not
    reject a hosted provider, which never sends a context window. It guards the fix itself
    (the L5.5 review found the first version refused `MAX_TOKENS=4096` on openrouter), so red
    proof against the pre-fix base does not apply: the base had no bound to misapply.
    """
    for provider_name in ("openrouter", "openai"):
        for context in ("", "1024"):
            env = {
                "RECALL_REASONING_ANSWER_ENABLED": "1",
                "RECALL_REASONING_ANSWER_PROVIDER": provider_name,
                "RECALL_REASONING_ANSWER_MODEL": "m",
                "RECALL_REASONING_ANSWER_API_KEY": "k",
                "RECALL_REASONING_ANSWER_MAX_TOKENS": "4096",
            }
            if context:
                env["RECALL_REASONING_ANSWER_CONTEXT_TOKENS"] = context
            assert resolve_answer_provider(env) is not None


def test_deploy001_desktop_stack_keeps_the_paid_answer_provider_opt_in() -> None:
    """DEPLOY-001. Invariant: with nothing exported, the desktop stack resolves no answer provider.

    Defect caught: the compose anchor set RECALL_REASONING_ANSWER_ENABLED to "1", so a host
    without OPENROUTER_API_KEY failed every desktop MCP server at startup and a host with one
    spent credit without opting in. Production line: `docker-compose.desktop.yml`.
    """
    text = (ROOT / "docker-compose.desktop.yml").read_text(encoding="utf-8")
    anchor = text.split("services:", 1)[0]
    environment: dict[str, str] = {}
    for line in anchor.splitlines():
        match = re.match(r"\s+(RECALL_\w+):\s*(.+?)\s*$", line)
        if match:
            # Compose substitution with an empty host environment: ${VAR:-default} -> default.
            value = re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", match.group(2)).strip('"')
            environment[match.group(1)] = value
    assert environment["RECALL_REASONING_ANSWER_ENABLED"] == "0"
    assert resolve_answer_provider(environment) is None


def test_bug002_reasoning_tool_reads_a_naive_as_of_as_utc(monkeypatch) -> None:
    """BUG-002. Invariant: the as_of the tool passes on is timezone aware.

    Defect caught: `as_of="2026-01-01"` parsed to a naive datetime that was compared with aware
    validity bounds during graph expansion and raised TypeError.
    Production line: the `as_of` parse in `recall_mcp.server.recall_reasoning_query`.
    """
    import recall_mcp.server as server_module
    from recall_mcp.server import build_server
    from recall_mcp.settings import Settings

    class Captured(Exception):
        pass

    seen: dict[str, object] = {}

    def capture(*_args, **kwargs):
        seen["as_of"] = kwargs.get("as_of")
        raise Captured

    monkeypatch.setattr(server_module, "get_access_token", lambda: None)
    monkeypatch.setattr(server_module, "reasoning_query", capture)
    state = {
        "store": SimpleNamespace(tenant="acme"),
        "embedder": object(),
        "settings": Settings.from_env({}),
    }
    tools = {tool.name: tool for tool in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    async def run() -> str:
        return await tools["recall_reasoning_query"].fn(
            ctx=ctx, query="When did the rollout change?", as_of="2026-01-01"
        )

    with pytest.raises(Captured):
        asyncio.run(run())
    as_of = seen["as_of"]
    assert isinstance(as_of, datetime)
    assert as_of.tzinfo is not None
    assert as_of == datetime(2026, 1, 1, tzinfo=UTC)


def test_stakes001_source_scoped_reasoning_admits_no_cross_source_neighbour(monkeypatch) -> None:
    """STAKES-001. Invariant: a `source=` reasoning query never adds graph neighbours from other
    sources to the evidence it trust evaluates.

    Defect caught: graph first expansion ran over the whole store's semantic graph with no source
    filter, so a cross source neighbour entered the trusted context of a source scoped query.
    Production line: the candidate filter in `graph_first_transform`,
    `recall_mcp.reasoning_engine._execute_reasoning_query`.
    """

    @dataclass(frozen=True)
    class Expansion:
        scored_candidates: tuple[ScoredChunk, ...]

    def scored(chunk_id: str, file: str) -> ScoredChunk:
        return ScoredChunk(
            chunk=Chunk(chunk_id, f"/corpus/{file}", f"text {chunk_id}", {"file": file}),
            score=0.8,
            indexed_at=NOW,
        )

    inside, outside = scored("inside", "notes/a.md"), scored("outside", "notes/b.md")
    assembled: dict[str, tuple[str, ...]] = {}

    class Stop(Exception):
        pass

    def assemble(raw, candidates, **_kwargs):
        assembled["ids"] = tuple(candidate.chunk.id for candidate in candidates)
        raise Stop

    def fake_retrieve(*args, **kwargs):
        transform = kwargs.get("pre_trust_transform")
        assert transform is not None, "graph first must be active for this query"
        raw = RetrievalResult(
            query="q",
            hits=[scored("seed", "notes/a.md")],
            gap_warning=False,
            staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        )
        transform(raw)
        raise AssertionError("the assembly stub stops the request")

    monkeypatch.setattr(reasoning_engine, "_provisional_graph_seed_result", lambda raw, *_: raw)
    monkeypatch.setattr(
        reasoning_engine,
        "_expand_semantic_graph",
        lambda *_args, **_kwargs: Expansion((inside, outside)),
    )
    monkeypatch.setattr(reasoning_engine, "_resolve_graph_calibration", lambda *_: None)
    monkeypatch.setattr(reasoning_engine, "_assemble_graph_first_context", assemble)
    monkeypatch.setattr(retrieval_module, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(
        reasoning_engine, "reason", lambda request: request.providers.retriever(request)
    )

    store = SimpleNamespace(tenant="acme", generation_id="legacy")
    with pytest.raises(Stop):
        service.reasoning_query(
            store,
            object(),
            "What does the plan depend on?",
            source="notes/a.md",
            graph_expansion="one_hop",
            policy=TrustPolicy.development(),
        )
    assert assembled["ids"] == ("inside",)


def _cleanup_module(monkeypatch, **env: str):
    for name in (
        "CLAUDE_PID",
        "RECALL_MCP_CLIENT",
        "RECALL_MCP_SESSION_ID",
        "RECALL_MCP_PATTERN",
        "RECALL_MCP_PS_FILE",
        "RECALL_MCP_KILL_FILE",
        "RECALL_MCP_KILL_RC",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    import recall_hooks.mcp_cleanup as cleanup

    # Reloaded so an import time read of RECALL_MCP_PATTERN sees this test's environment.
    return importlib.reload(cleanup)


def test_env003_an_empty_pattern_override_does_not_widen_the_kill(tmp_path, monkeypatch) -> None:
    """ENV-003. Invariant: an empty RECALL_MCP_PATTERN falls back to the default transport pattern.

    Defect caught: `"" in command` is true for every row, so every descendant of the client, a
    shell or language server included, was selected for termination.
    Production line: `_mcp_pattern` in `recall_hooks.mcp_cleanup`.
    """
    table = tmp_path / "processes.txt"
    table.write_text(
        "800 1 client\n"
        "801 800 python -m recall_mcp.server\n"
        "803 800 bash -l\n",
        encoding="utf-8",
    )
    killed = tmp_path / "killed.txt"
    cleanup = _cleanup_module(
        monkeypatch,
        RECALL_MCP_PATTERN="",
        RECALL_MCP_PS_FILE=str(table),
        RECALL_MCP_KILL_FILE=str(killed),
    )
    try:
        status, _detail = cleanup.close_own_mcp_transports("800", cwd=str(tmp_path))
    finally:
        monkeypatch.delenv("RECALL_MCP_PATTERN")
        importlib.reload(cleanup)
    assert status == "closed"
    assert killed.read_text(encoding="utf-8").splitlines() == ["801"]


def test_code011_a_fake_process_table_never_drives_a_real_kill(tmp_path, monkeypatch) -> None:
    """CODE-011. Invariant: the test seams act only as a pair; one alone closes nothing.

    Defect caught: RECALL_MCP_PS_FILE alone fed pids from a file to the real taskkill or SIGTERM,
    and RECALL_MCP_KILL_FILE alone reported real transports as closed while killing nothing.
    Production line: the seam check at the top of `close_own_mcp_transports`.
    """
    table = tmp_path / "processes.txt"
    table.write_text("800 1 client\n801 800 python -m recall_mcp.server\n", encoding="utf-8")
    cleanup = _cleanup_module(monkeypatch, RECALL_MCP_PS_FILE=str(table))
    kills: list[object] = []
    monkeypatch.setattr(cleanup.os, "kill", lambda *args: kills.append(args))
    monkeypatch.setattr(
        cleanup.subprocess,
        "run",
        lambda *args, **kwargs: kills.append(args) or SimpleNamespace(returncode=0, stdout=""),
    )

    status, _detail = cleanup.close_own_mcp_transports("800", cwd=str(tmp_path))

    assert kills == []
    assert status == "unknown"

    monkeypatch.delenv("RECALL_MCP_PS_FILE")
    monkeypatch.setenv("RECALL_MCP_KILL_FILE", str(tmp_path / "killed.txt"))
    status, _detail = cleanup.close_own_mcp_transports("800", cwd=str(tmp_path))
    assert status == "unknown"
    assert not (tmp_path / "killed.txt").exists()
