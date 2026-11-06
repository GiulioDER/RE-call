"""Regression tests for the 2026-11-05 DEEP CCA hunt of the reasoning stack.

Run 20261105T225154Z audited `recall/reasoning.py`, `recall/reasoning_response.py`,
`recall_mcp/reasoning_engine.py`, `recall_mcp/reasoning_diagnostics.py`,
`recall_mcp/semantic_graph_cache.py` and `recall/semantic_graph/`. Each test names its finding,
the production line it pins and how it was proved red: it was run against the pre-fix tree (the
snapshot `cca_checks tautology verify` restores) and failed in its own assertion there.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from recall.evidence import EvidenceValidationError
from recall.observability import PerformanceTrace, current_performance_trace, performance_trace_scope
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    SemanticGraphExpansionResult,
    reason,
    reasoning_response_from_dict,
)
from recall.reasoning_expansion import ExpansionProposal, ExpansionReport
from recall.reasoning_planner import ReasoningBudget
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.semantic_graph import build_semantic_graph
from recall.semantic_graph._model import SemanticGraphStore, _parse_temporal_value
from recall.supersession import SupersessionGraph
from recall.trust_policy import TrustPolicy
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import graph_expansion, reasoning_engine, semantic_graph_cache, service
from recall_mcp import reasoning_diagnostics
from recall_mcp.graph_expansion import GRAPH_FIRST_CONTEXT_K
from recall_mcp import retrieval as retrieval_module
from recall_mcp.settings import (
    Settings,
    _validate_runtime_options,
    activate_runtime_settings,
    reset_runtime_settings,
)

NOW = datetime(2026, 8, 10, tzinfo=UTC)


class _Stop(BaseException):
    """Ends a request at the point under test; BaseException so no fail-open handler eats it."""


def _hit(chunk: Chunk, *, authority: str = "unknown") -> TrustedHit:
    return TrustedHit(
        chunk=chunk,
        cosine=0.91,
        confidence=0.97,
        verdict="ok",
        provenance=Provenance(chunk.source, chunk.metadata.get("file", chunk.source), 0, NOW),
        validity=Validity(NOW, None, None),
        authority=authority,  # type: ignore[arg-type]
    )


def _result(*hits: TrustedHit, generation: str = "gen_1") -> TrustedResult:
    return TrustedResult(
        query="who owns rollout?",
        hits=list(hits),
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(
            embedding_profile="embed-a",
            retrieval_profile="quality",
            index_generation=generation,
            stage_ms={"dense_retrieval": 1.0},
        ),
        calibration_id="cal-1",
        calibration_status="certified",
        tenant_id="acme",
        generation_id=generation,
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        query_set_digest="query-a",
    )


def _chunk(cid: str, file: str, text: str = "text") -> Chunk:
    return Chunk(cid, f"/corpus/{file}", text, {"file": file, "ord": 0})


def test_sec001_denied_neighbour_leaves_no_byte_or_label_trace_in_the_counters() -> None:
    """SEC-001 (P1): counters about a policy-denied graph neighbour reached the client.

    `recall_mcp.graph_expansion._fetch_candidate_chunks` added `candidate_fetched_count` and
    `candidate_payload_bytes` over the whole fetched batch, then rejected denied chunks as
    `security_policy`, and the store counted their bytes into the request's trace. Red on the
    pre-fix tree: `candidate_fetched_count` was 2 (the denied chunk counted).
    """
    allowed = Chunk("a", "public/a.md", "visible text", {"file": "public/a.md"})
    denied = Chunk("b", "restricted/b.md", "secret text " * 20, {"file": "restricted/b.md"})

    class Store:
        def chunks_by_ids(self, ids: tuple[str, ...]) -> dict[str, Chunk]:
            trace = current_performance_trace()
            if trace is not None:
                trace.add("db_result_bytes", 4096)
            return {chunk.id: chunk for chunk in (allowed, denied) if chunk.id in ids}

    stats = graph_expansion.ExpansionStats()
    trace = PerformanceTrace()
    with performance_trace_scope(trace):
        chunks = graph_expansion._fetch_candidate_chunks(
            store=Store(),  # type: ignore[arg-type]
            request=SimpleNamespace(generation=SimpleNamespace(generation_id="g")),  # type: ignore[arg-type]
            candidate_ids=("a", "b"),
            performance=trace,
            security_policy=SourceSecurityPolicy((SourceRule("public", classification="public"),)),
            access_context=AccessContext("alice", "acme", clearance="public"),
            stats=stats,
            generation_scope=lambda *_args: nullcontext(),
        )
    counters = trace.snapshot()["counters"]
    assert set(chunks) == {"a"}
    assert counters["candidate_fetched_count"] == 1
    assert counters["candidate_payload_bytes"] == len(b"visible text") + len(b"public/a.md")
    assert counters.get("db_result_bytes", 0) == 0
    assert stats.rejections == {"missing_chunk": 1}


def test_sec001_scope_metadata_read_counts_no_denied_bytes() -> None:
    """SEC-001 (P1), second channel: the pre-budget metadata read counted denied bytes.

    `_materialize_candidates` reads every neighbour's metadata (text free) before the budget,
    which the SEC-002 fix runs for every source scoped request, and the store counts the result
    into the request's trace as `db_result_bytes`. Proved red on the tree before this read was
    moved to a throwaway trace (`db_result_bytes` was 4096); the base tree has no `source`
    parameter, so the base itself cannot run it.
    """
    metadata = {
        "c_denied": Chunk("c_denied", "restricted/b.md", "", {"file": "restricted/b.md"}),
        "x_in": Chunk("x_in", "public/a.md", "", {"file": "public/a.md"}),
    }

    class Store:
        def chunk_metadata_by_ids(self, ids: tuple[str, ...]) -> dict[str, Chunk]:
            trace = current_performance_trace()
            if trace is not None:
                trace.add("db_result_bytes", 4096)
            return {cid: metadata[cid] for cid in ids}

        def chunks_by_ids(self, ids: tuple[str, ...]) -> dict[str, Chunk]:
            raise _Stop

    trace = PerformanceTrace()
    with performance_trace_scope(trace), pytest.raises(_Stop):
        graph_expansion._materialize_candidates(
            store=Store(),  # type: ignore[arg-type]
            request=SimpleNamespace(  # type: ignore[arg-type]
                generation=SimpleNamespace(generation_id="g"), known_as_of=None
            ),
            candidate_relations_by_chunk={cid: [] for cid in metadata},
            indexes=SimpleNamespace(validity_by_chunk={}),
            seed_entities=set(),
            trusted_seed_ids=set(),
            as_of=NOW,
            supersession={},
            edge_candidates=None,  # type: ignore[arg-type]
            candidate_budget=8,
            defer_trust_evaluation=True,
            max_graph_rescoring_candidates=512,
            performance=trace,
            security_policy=SourceSecurityPolicy((SourceRule("public", classification="public"),)),
            access_context=AccessContext("alice", "acme", clearance="public"),
            stats=graph_expansion.ExpansionStats(),
            generation_scope=lambda *_args: nullcontext(),
            supersedes_key=lambda *_args: None,
            resolve_successor=lambda *_args: None,
            excluded_chunk_ids=frozenset(),
            graph=SupersessionGraph.from_parts({}),
            source="public/a.md",
        )
    assert trace.snapshot()["counters"].get("db_result_bytes", 0) == 0


def test_sec002_source_scope_filters_neighbours_before_the_budget() -> None:
    """SEC-002 (P2): a source scoped query spent its candidate budget on other files.

    `_materialize_candidates` truncated store-wide candidates (sorted by id) to the budget, and
    `reasoning_engine.graph_first_transform` filtered by source only afterwards. The pre-fix
    function has no `source` parameter, so the red proof is a mutation of the fix: deleting the
    `if in_scope_ids is not None and chunk_id not in in_scope_ids: continue` lines sent the one
    budget slot to `c_out1` (`[('c_out1',)] == [('x_in',)]`), as the pre-fix ordering did.
    """
    metadata = {
        "c_out1": _chunk("c_out1", "notes/b.md"),
        "c_out2": _chunk("c_out2", "notes/b.md"),
        "x_in": _chunk("x_in", "notes/a.md"),
    }
    fetched: list[tuple[str, ...]] = []

    class Store:
        def chunk_metadata_by_ids(self, ids: tuple[str, ...]) -> dict[str, Chunk]:
            if len(ids) == 1:
                fetched.append(ids)
                raise _Stop
            return {cid: metadata[cid] for cid in ids}

        def chunks_by_ids(self, ids: tuple[str, ...]) -> dict[str, Chunk]:
            raise AssertionError("the metadata-only path is the one under test")

    with pytest.raises(_Stop):
        graph_expansion._materialize_candidates(
            store=Store(),  # type: ignore[arg-type]
            request=SimpleNamespace(  # type: ignore[arg-type]
                generation=SimpleNamespace(generation_id="g"), known_as_of=None
            ),
            candidate_relations_by_chunk={cid: [] for cid in metadata},
            indexes=SimpleNamespace(validity_by_chunk={}),
            seed_entities=set(),
            trusted_seed_ids=set(),
            as_of=NOW,
            supersession={},
            edge_candidates=None,  # type: ignore[arg-type]
            candidate_budget=1,
            defer_trust_evaluation=True,
            max_graph_rescoring_candidates=512,
            performance=None,
            security_policy=None,
            access_context=None,
            stats=graph_expansion.ExpansionStats(),
            generation_scope=lambda *_args: nullcontext(),
            supersedes_key=lambda *_args: None,
            resolve_successor=lambda *_args: None,
            excluded_chunk_ids=frozenset(),
            graph=SupersessionGraph.from_parts({}),
            source="notes/a.md",
        )
    assert fetched == [("x_in",)]


def test_bug002_graph_first_failure_reaches_the_provider_boundary(monkeypatch) -> None:
    """BUG-002 (P2): a graph failure inside the pre-trust transform escaped `reason()` raw.

    `reasoning_engine.graph_first_transform` ran `_expand_semantic_graph` with no handler, inside
    retrieval, so `run_graph_expansion` never saw it to turn it into GRAPH_PROVIDER_ERROR. Red on
    the pre-fix tree: the RuntimeError escaped the transform. The kept seeds are cut to the
    graph-first context size, as on every success path.
    """
    escaped: list[str] = []

    def failing_expansion(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("semantic graph load failed")

    def fake_retrieve(*_args: Any, **kwargs: Any) -> Any:
        transform = kwargs["pre_trust_transform"]
        raw = RetrievalResult(
            query="q",
            hits=[
                ScoredChunk(chunk=_chunk(f"seed{index:02d}", "notes/a.md"), score=0.8, indexed_at=NOW)
                for index in range(GRAPH_FIRST_CONTEXT_K + 5)
            ],
            gap_warning=False,
            staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        )
        try:
            context = transform(raw)
        except RuntimeError:
            escaped.append("transform")
        else:
            # The context cut every success path applies, so trust sees the same pool size.
            assert [hit.chunk.id for hit in context.hits] == [
                hit.chunk.id for hit in raw.hits[:GRAPH_FIRST_CONTEXT_K]
            ]
        raise _Stop

    provider_errors: list[str] = []

    def fake_reason(request: ReasoningRequest) -> str:
        with pytest.raises(_Stop):
            request.providers.retriever(request)
        assert request.providers.graph_expansion_provider is not None
        try:
            request.providers.graph_expansion_provider(request, _result())
        except RuntimeError as exc:
            provider_errors.append(str(exc))
        return "done"

    monkeypatch.setattr(reasoning_engine, "_provisional_graph_seed_result", lambda raw, *_: raw)
    monkeypatch.setattr(reasoning_engine, "_expand_semantic_graph", failing_expansion)
    monkeypatch.setattr(retrieval_module, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(reasoning_engine, "reason", fake_reason)
    store = SimpleNamespace(tenant="acme", generation_id="legacy")
    assert (
        service.reasoning_query(
            store,
            object(),
            "What does the plan depend on?",
            graph_expansion="one_hop",
            policy=TrustPolicy.development(),
        )
        == "done"
    )
    assert escaped == []
    assert provider_errors == ["semantic graph load failed"]


def test_code001_reuse_audit_recomputes_with_the_configured_shadow_policy(monkeypatch) -> None:
    """CODE-001 (P2): the reuse audit's duplicate payload used the default shadow policy.

    `reasoning_engine` passed the configured policy to the reused payload and none to the
    duplicate, so `guarded_spare_slot` was compared against `alpha008`. Red on the pre-fix tree:
    the second call carried no policy.
    """
    monkeypatch.setenv("RECALL_SOURCE_CONDITIONING_MODE", "shadow")
    monkeypatch.setenv("RECALL_SOURCE_CONDITIONING_SHADOW_POLICY", "guarded_spare_slot")
    monkeypatch.setenv("RECALL_SOURCE_CONDITIONING_ARTIFACT", "/artifact.json")
    policies: list[str] = []

    def shadow_payload(**kwargs: Any) -> dict[str, object]:
        policies.append(str(kwargs.get("policy", "<default>")))
        if len(policies) == 2:
            raise _Stop
        return {}

    executed = SimpleNamespace(
        result=_result(),
        profile=SimpleNamespace(name="quality"),
        candidate_trace=(None, SimpleNamespace(diagnostics=SimpleNamespace(stage_ms={}))),
        query_vector=[0.1],
    )
    monkeypatch.setattr(retrieval_module, "_retrieve_trusted", lambda *_a, **_k: executed)
    monkeypatch.setattr(reasoning_engine, "_source_conditioning_shadow_sampled", lambda *_a: True)
    monkeypatch.setattr(reasoning_engine, "_source_conditioning_reused_audits", lambda *_a: ({}, {}))
    monkeypatch.setattr(
        reasoning_engine, "_source_conditioning_reuse_benchmark_audit_enabled", lambda: True
    )
    monkeypatch.setattr(reasoning_engine, "_source_conditioning_shadow_payload", shadow_payload)
    monkeypatch.setattr(
        reasoning_diagnostics, "_retrieval_leg_benchmark_audit_payload", lambda *_a: {}
    )
    monkeypatch.setattr(
        reasoning_diagnostics, "_source_admission_benchmark_audit_payload", lambda *_a: {}
    )
    monkeypatch.setattr(
        reasoning_engine, "reason", lambda request: request.providers.retriever(request)
    )
    store = SimpleNamespace(tenant="acme", generation_id="legacy")
    with pytest.raises(_Stop):
        service.reasoning_query(
            store, object(), "q", graph_expansion="off", policy=TrustPolicy.development()
        )
    assert policies == ["guarded_spare_slot", "guarded_spare_slot"]


def test_dat001_evidence_authority_survives_the_round_trip() -> None:
    """DAT-001 (P2): `reasoning_response_from_dict` dropped every item's `authority`.

    The `EvidenceItem` decoder in `recall/reasoning_response.py` never read the field, so it came
    back `unknown`. Red on the pre-fix tree: the decoded authority was `unknown`.
    """
    chunk = _chunk("c1", "rollout.md", "Ada owns rollout.")
    response = reason(
        ReasoningRequest(
            query="who owns rollout?",
            tenant_id="acme",
            generation=GenerationSelection(generation_id="gen_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: _result(_hit(chunk, authority="canonical"))
            ),
            policy=ReasoningPolicy(name="retrieval_only"),
        )
    )
    assert response.trusted_evidence.items[0].authority == "canonical"
    decoded = reasoning_response_from_dict(response.to_dict())
    assert decoded.trusted_evidence.items[0].authority == "canonical"


def test_doc003_decoder_refuses_an_unknown_provider_failure_kind() -> None:
    """DOC-003 (P2): the strict decoder let an unknown `ProviderFailure.kind` through a cast.

    `_provider_failure_from_dict` used `cast(Any, ...)` against `_checked_literal`'s own contract.
    Red on the pre-fix tree: the payload decoded without error.
    """
    response = reason(
        ReasoningRequest(
            query="who owns rollout?",
            tenant_id="acme",
            generation=GenerationSelection(generation_id="gen_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: _result(_hit(_chunk("c1", "rollout.md")))
            ),
            policy=ReasoningPolicy(name="retrieval_only"),
        )
    )
    payload = response.to_dict()
    payload["provider_failures"] = [
        {
            "kind": "bogus",
            "provider_id": "p",
            "model_id": "m",
            "provider_revision": "r",
            "message": "x",
        }
    ]
    with pytest.raises(EvidenceValidationError, match="provider failure kind"):
        reasoning_response_from_dict(payload)


def test_doc001_expansion_model_call_is_counted_without_a_planner() -> None:
    """DOC-001 (P2): `evidence_assembly` with retrieval expansion reported one model call.

    `_budget_used` read only the planner's usage, and this policy runs no planner, so the paid
    expansion call was recorded nowhere. Red on the pre-fix tree: `model_calls` was 1.
    """
    first = _chunk("first", "first.md", "The first document identifies the project.")
    second = _chunk("second", "second.md", "The second document contains the owner: Ada.")

    def expansion_provider(_request: Any) -> ExpansionReport:
        return ExpansionReport(
            proposals=(
                ExpansionProposal(
                    id="rewrite_1",
                    mode="rewrite",
                    query="who owns the project",
                    rationale="the initial evidence does not contain an owner",
                    parent_chunk_ids=("first",),
                ),
            )
        )

    response = reason(
        ReasoningRequest(
            query="who owns the project?",
            tenant_id="acme",
            generation=GenerationSelection(generation_id="gen_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: _result(_hit(first)),
                expansion_provider=expansion_provider,
                expansion_retriever=lambda _request, _proposal, _initial: _result(_hit(second)),
                answer_provider=lambda _system, _user: {
                    "answer": "Ada owns the project.",
                    "citations": ["second"],
                    "insufficient_evidence": False,
                },
            ),
            policy=ReasoningPolicy(allow_retrieval_expansion=True),
            budget=ReasoningBudget(max_model_calls=2),
        )
    )
    assert response.outcome == "answered"
    assert response.diagnostics.budget_used is not None
    assert response.diagnostics.budget_used.model_calls == 2


def test_stakes001_graph_expansion_cannot_relabel_an_unpinned_request() -> None:
    """STAKES-001 (P2): graph expansion from another generation replaced an unpinned seed.

    `run_graph_expansion` (recall/reasoning_graph_service.py) checked the expansion only against
    the request, which skips every field the request leaves unset. Red on the pre-fix tree: the
    response came back labelled `gen_2` with the foreign chunk.
    """
    seed = _result(_hit(_chunk("a", "a.md")))
    foreign = _result(_hit(_chunk("a", "a.md")), _hit(_chunk("z", "z.md")), generation="gen_2")
    response = reason(
        ReasoningRequest(
            query="q",
            tenant_id="acme",
            generation=GenerationSelection(),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: seed,
                graph_expansion_provider=lambda _request, _retrieval: SemanticGraphExpansionResult(
                    retrieval=foreign, readiness="ready"
                ),
            ),
            policy=ReasoningPolicy(name="retrieval_only", graph_expansion="one_hop"),
            budget=ReasoningBudget(max_graph_hops=1),
        )
    )
    assert response.generation_id == "gen_1"
    assert [item.chunk_id for item in response.trusted_evidence.items] == ["a"]
    assert response.diagnostics.graph_readiness == "GRAPH_PROVIDER_ERROR"


def test_num001_unpadded_valid_from_starts_at_midnight() -> None:
    """NUM-001 (P2): an unpadded date moved `valid_from` to the end of its day.

    The `strptime` fallback in `_parse_temporal_value` set `end_of_day = True` for every field.
    Red on the pre-fix tree: `2026-1-5` parsed to 23:59:59.999999.
    """
    assert _parse_temporal_value("2026-1-5", "valid_from") == datetime(2026, 1, 5, tzinfo=UTC)
    assert _parse_temporal_value("2026-1-5", "valid_until", end_of_day=True) == datetime(
        2026, 1, 5, 23, 59, 59, 999999, tzinfo=UTC
    )


def _relations(graph: Any) -> list[str]:
    return sorted(relation.relation for relation in graph.relations)


def test_bug001_authored_file_entity_keeps_the_documents_own_edges() -> None:
    """BUG-001 (P2): an authored `file` entity counted as the chunk's own file identity.

    `build_semantic_graph` added every `file` kind entity to `file_entities_by_source`, so the
    source had two identities and its `supersedes` edge was skipped. Red on the pre-fix tree: no
    `supersedes` relation was built.
    """
    chunks = [
        Chunk(
            "a",
            "/r/a.md",
            "alpha",
            {
                "file": "a.md",
                "supersedes": "b.md",
                "recall_graph": {"entities": [{"name": "config.yaml", "kind": "file"}]},
            },
        ),
        Chunk("b", "/r/b.md", "beta", {"file": "b.md"}),
    ]
    graph = build_semantic_graph(chunks, tenant_id="t", generation_id="g")
    assert _relations(graph).count("supersedes") == 1


def test_doc002_graph_reads_supersedes_as_the_trust_layer_does() -> None:
    """DOC-002 (P2): the graph read `supersedes` with its own parser, not the trust layer's.

    A compiled record declares no file reference to the trust layer (`chunk_supersedes_targets`),
    and a legacy flow sequence string names two targets. Red on the pre-fix tree: the compiled
    record produced a `supersedes` edge.
    """
    compiled = build_semantic_graph(
        [
            Chunk("a", "/r/a.md", "alpha", {"file": "a.md", "record_type": "compiled", "supersedes": ["b"]}),
            Chunk("b", "/r/b.md", "beta", {"file": "b.md"}),
        ],
        tenant_id="t",
        generation_id="g",
    )
    assert "supersedes" not in _relations(compiled)
    flow = build_semantic_graph(
        [
            Chunk("a", "/r/a.md", "alpha", {"file": "a.md", "supersedes": "[b.md, c.md]"}),
            Chunk("b", "/r/b.md", "beta", {"file": "b.md"}),
            Chunk("c", "/r/c.md", "gamma", {"file": "c.md"}),
        ],
        tenant_id="t",
        generation_id="g",
    )
    assert _relations(flow).count("supersedes") == 2


def test_doc004_graph_store_protocol_matches_generation_store() -> None:
    """DOC-004 (P2): the exported Protocol named parameters GenerationStore does not take.

    Red on the pre-fix tree: the Protocol's `graph_readiness` took `(tenant_id, generation_id)`.
    """
    from recall.generation_store import GenerationStore

    for name in ("graph_readiness", "delete_generation_graph", "load_generation_graph"):
        declared = list(inspect.signature(getattr(SemanticGraphStore, name)).parameters)
        implemented = list(inspect.signature(getattr(GenerationStore, name)).parameters)
        assert declared == implemented, name


def test_dat002_a_refused_graph_rereads_its_readiness_marker() -> None:
    """DAT-002 (P2): after a graph rebuild a running server kept the old marker until restart.

    `_cached_semantic_graph` cached a refusal under the stale marker's fingerprint, and the
    store's readiness cache never re-read the marker. Red on the pre-fix tree: no invalidation.
    """
    graph = build_semantic_graph([_chunk("a", "a.md")], tenant_id="acme", generation_id="g")
    invalidations: list[str] = []

    class Store:
        tenant = "acme"

        def load_semantic_graph(self, _generation_id: str) -> Any:
            return graph

        def invalidate_graph_readiness(self) -> None:
            invalidations.append("invalidated")

    stale = SimpleNamespace(ready=True, graph_fingerprint="stale-marker")
    semantic_graph_cache._SEMANTIC_GRAPH_CACHE.clear()
    try:
        for _ in range(2):
            assert (
                semantic_graph_cache._cached_semantic_graph(Store(), "g", stale, None)  # type: ignore[arg-type]
                is None
            )
    finally:
        semantic_graph_cache._SEMANTIC_GRAPH_CACHE.clear()
    assert invalidations == ["invalidated", "invalidated"]


def test_env001_benchmark_audits_read_the_servers_environment(monkeypatch) -> None:
    """ENV-001 (P2): the audit gates read `os.environ`, the engine the server's settings.

    Red on the pre-fix tree: with the settings pin off and the process pin on, the gate opened.
    """
    monkeypatch.setenv("RECALL_BENCHMARK_PIN", "1")
    monkeypatch.setenv("RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT", "1")
    token = activate_runtime_settings(Settings.from_env({"RECALL_BENCHMARK_PIN": "0"}))
    try:
        assert not reasoning_diagnostics._retrieval_leg_benchmark_audit_enabled()
    finally:
        reset_runtime_settings(token)


def test_env002_a_mistyped_graph_control_refuses_startup() -> None:
    """ENV-002 (P2): an unknown `RECALL_GRAPH_RELATION_CONTROL` silently ran the real graph.

    Red on the pre-fix tree: `_validate_runtime_options` accepted `shufled`.
    """
    with pytest.raises(ValueError, match="RECALL_GRAPH_RELATION_CONTROL"):
        _validate_runtime_options({"RECALL_GRAPH_RELATION_CONTROL": "shufled"})
