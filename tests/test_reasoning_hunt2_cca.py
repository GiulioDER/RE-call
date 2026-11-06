"""Regression tests for the second DEEP CCA hunt of the reasoning stack (run 20261106T124749Z).

The run verified the 23 P3 findings the first hunt (PR 1227) left unverified and hunted the same
files again. Each behaviour test names its finding and the production line it pins, and was run
against the pre-fix tree (the snapshot `cca_checks tautology verify` restores) and failed in its
own assertion there, or against a mutation of the fix where the pre-fix tree lacks the symbol.
Tests marked "control" pin a single definition and are not red proofs.
"""

from __future__ import annotations

import gc
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from recall.evidence import EvidenceValidationError
from recall.observability import METRICS
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    reason,
    reasoning_response_from_dict,
)
from recall.reasoning_graph import build_reasoning_graph
from recall.semantic_graph import build_semantic_graph
from recall.semantic_graph._model import (
    ENTITY_KINDS,
    RELATION_KINDS,
    RELATION_STATUSES,
    EntityKind,
    RelationKind,
    RelationStatus,
    _parse_temporal_value,
)
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import graph_expansion, reasoning_engine, semantic_graph_cache, service

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def _chunk(cid: str, file: str, text: str = "text") -> Chunk:
    return Chunk(cid, f"/corpus/{file}", text, {"file": file, "ord": 0})


def _result(*chunks: Chunk) -> TrustedResult:
    return TrustedResult(
        query="who owns rollout?",
        hits=[
            TrustedHit(
                chunk=chunk,
                cosine=0.91,
                confidence=0.97,
                verdict="ok",
                provenance=Provenance(chunk.source, chunk.metadata["file"], 0, NOW),
                validity=Validity(NOW, None, None),
            )
            for chunk in chunks
        ],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(
            embedding_profile="embed-a",
            retrieval_profile="quality",
            index_generation="gen_1",
            stage_ms={"dense_retrieval": 1.0},
        ),
        calibration_id="cal-1",
        calibration_status="certified",
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        corpus_fingerprint="corpus-a",
        query_set_digest="query-a",
    )


def _retrieval_only_payload() -> dict[str, Any]:
    response = reason(
        ReasoningRequest(
            query="who owns rollout?",
            tenant_id="acme",
            generation=GenerationSelection(generation_id="gen_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: _result(_chunk("c1", "rollout.md", "Ada owns it."))
            ),
            policy=ReasoningPolicy(name="retrieval_only"),
        )
    )
    return response.to_dict()


def test_bug004_malformed_trust_state_and_date_raise_the_decoders_error() -> None:
    """BUG-004: the strict decoder raised TypeError and a bare ValueError on malformed payloads.

    `_trust_state` tested membership in a set, so a list raised TypeError, and
    `_optional_datetime` did not wrap `fromisoformat`. Red on the pre-fix tree: TypeError and
    ValueError escaped `pytest.raises(EvidenceValidationError)` (ValueError is its base class, so
    the second case is pinned by `type(...) is`).
    """
    payload = _retrieval_only_payload()
    payload["trust_state"] = ["trusted"]
    with pytest.raises(EvidenceValidationError):
        reasoning_response_from_dict(payload)

    payload = _retrieval_only_payload()
    payload["trusted_evidence"]["items"][0]["indexed_at"] = "notadate"
    with pytest.raises(ValueError) as raised:
        reasoning_response_from_dict(payload)
    assert type(raised.value) is EvidenceValidationError


def test_stakes003_decoder_refuses_a_citation_outside_the_trusted_evidence() -> None:
    """STAKES-003: a stored response citing evidence outside its bundle decoded cleanly.

    The producer (`recall.reasoning._citations`) refuses such a citation; the decoder did not.
    Red on the pre-fix tree: the forged payload decoded without error.
    """
    payload = _retrieval_only_payload()
    assert reasoning_response_from_dict(payload).trusted_evidence.items
    payload["citations"] = [{"chunk_id": "not-in-evidence", "source": "x", "ordinal": None}]
    with pytest.raises(EvidenceValidationError, match="citation"):
        reasoning_response_from_dict(payload)


def test_num002_an_explicit_midnight_valid_until_is_not_widened() -> None:
    """NUM-002: an explicit instant at midnight UTC was moved to the end of its day.

    `_parse_temporal_value` widened on `parsed.time() == time.min`, not on a bare date. Red on
    the pre-fix tree: `2026-03-01T00:00:00Z` became 23:59:59.999999.
    """
    midnight = datetime(2026, 3, 1, tzinfo=UTC)
    assert _parse_temporal_value("2026-03-01T00:00:00Z", "valid_until", end_of_day=True) == midnight
    assert (
        _parse_temporal_value("2026-03-01T01:00:00+01:00", "valid_until", end_of_day=True)
        == midnight
    )
    assert _parse_temporal_value("2026-03-01", "valid_until", end_of_day=True) == datetime(
        2026, 3, 1, 23, 59, 59, 999999, tzinfo=UTC
    )


def test_num003_a_multi_chunk_file_raises_each_diagnostic_once() -> None:
    """NUM-003: a diagnostic whose id names no chunk was raised once per chunk of its file.

    Frontmatter is copied to every chunk, and `_finalize_graph` did not deduplicate. Red on the
    pre-fix tree: three chunks gave four diagnostics for two distinct ids.
    """
    metadata = {"file": "notes/a.md", "person": "Ada", "project": "Ada", "entity_aliases": {"Ada": ["AL"]}}
    one = build_semantic_graph(
        [Chunk("a0", "/r/notes/a.md", "alpha", dict(metadata))], tenant_id="t", generation_id="g"
    )
    three = build_semantic_graph(
        [Chunk(f"a{i}", "/r/notes/a.md", f"alpha {i}", dict(metadata)) for i in range(3)],
        tenant_id="t",
        generation_id="g",
    )
    assert len(three.diagnostics) == len({item.id for item in three.diagnostics})
    assert len(three.diagnostics) == len(one.diagnostics)
    assert three.readiness().diagnostic_count == one.readiness().diagnostic_count


def test_code005_a_strict_refusal_keeps_the_profile_and_reaches_the_latency_metric(
    monkeypatch,
) -> None:
    """CODE-005: a strict refusal reported the `plain` profile and skipped the latency histogram.

    `reasoning_engine._strict_reasoning_refusal` built its own response and only counted the
    outcome. Red on the pre-fix tree: `answer_profile` was `plain` for a `dated` request.
    """

    def refuse(_request: ReasoningRequest) -> Any:
        raise TrustRefusal(
            code=TrustFailureCode.INDEX_NOT_READY,
            calibration_status="missing",
            tenant_id="acme",
            generation_id=None,
        )

    monkeypatch.setattr(reasoning_engine, "reason", refuse)
    METRICS.drain_histogram("recall_reasoning_latency_ms")
    response = service.reasoning_query(
        SimpleNamespace(tenant="acme", generation_id="legacy"),
        object(),
        "q",
        graph_expansion="off",
        policy=TrustPolicy.development(),
    )
    assert response.trust_state == "refused"
    assert response.diagnostics.answer_profile == "dated"
    _samples, observed = METRICS.drain_histogram("recall_reasoning_latency_ms")
    assert observed == 1


def test_perf002_the_checked_graph_cache_keeps_a_graph_it_reuses(monkeypatch) -> None:
    """PERF-002: `_CHECKED_GRAPHS` evicted in insertion order and a hit did not refresh it.

    So per-request graphs pushed out the store graph every query reuses. Red on the pre-fix tree:
    the reused graph's entry was evicted and rebuilt.
    """
    from recall import reasoning

    monkeypatch.setattr(reasoning, "_CHECKED_GRAPHS", {})

    def graph() -> Any:
        return build_reasoning_graph(
            [_chunk("c1", "owner.md")], tenant_id="acme", generation_id="gen-1", include_text=True
        )

    store_graph = graph()
    first = reasoning._checked(store_graph)
    others = [graph() for _ in range(reasoning._CHECKED_GRAPHS_MAX * 2)]
    for other in others:
        reasoning._checked(other)
        assert reasoning._checked(store_graph) is first


def test_perf002_dead_checked_graphs_are_evicted_before_live_ones(monkeypatch) -> None:
    """PERF-002: a slot held by a graph that no longer exists was kept over a live graph.

    Red on the pre-fix tree, and by mutation (removing the dead-entry sweep in
    `recall.reasoning._checked`): the live graph was evicted instead of a dead slot.
    """
    from recall import reasoning

    monkeypatch.setattr(reasoning, "_CHECKED_GRAPHS", {})
    live = build_reasoning_graph(
        [_chunk("c1", "owner.md")], tenant_id="acme", generation_id="gen-1", include_text=True
    )
    first = reasoning._checked(live)
    # Held until every slot is taken, so no two of them share an id, then released.
    others = [
        build_reasoning_graph(
            [_chunk("c2", "other.md")], tenant_id="acme", generation_id="gen-1", include_text=True
        )
        for _ in range(reasoning._CHECKED_GRAPHS_MAX - 1)
    ]
    for other in others:
        reasoning._checked(other)
    assert len(reasoning._CHECKED_GRAPHS) == reasoning._CHECKED_GRAPHS_MAX
    del others, other
    gc.collect()
    reasoning._checked(
        build_reasoning_graph(
            [_chunk("c3", "new.md")], tenant_id="acme", generation_id="gen-1", include_text=True
        )
    )
    assert reasoning._CHECKED_GRAPHS.get(id(live)) is first


def test_perf004_a_relation_control_projection_is_built_once_per_graph() -> None:
    """PERF-004: the shuffled and removed controls rebuilt their projection on every request.

    The pre-fix tree has no `_controlled_semantic_graph`, so the red proof is a mutation: removing
    its cache lookup returned a new projection on the second call.
    """
    graph = build_semantic_graph(
        [
            Chunk("a", "/r/a.md", "see [[b]]", {"file": "a.md"}),
            Chunk("b", "/r/b.md", "see [[a]]", {"file": "b.md"}),
        ],
        tenant_id="t",
        generation_id="g",
    )
    semantic_graph_cache._CONTROLLED_GRAPHS.clear()
    try:
        for control in ("removed", "shuffled"):
            first = semantic_graph_cache._controlled_semantic_graph(graph, control, 7)
            assert first is not graph
            assert semantic_graph_cache._controlled_semantic_graph(graph, control, 7) is first
        assert semantic_graph_cache._controlled_semantic_graph(graph, "none", 7) is graph
    finally:
        semantic_graph_cache._CONTROLLED_GRAPHS.clear()


def test_code009_env003_single_definitions_control() -> None:
    """CODE-009 and ENV-003: each vocabulary and ranking constant has one source.

    The values agreed before the fix, so the vocabulary half is a control. The ranking half is a
    red proof: on the pre-fix tree `semantic_graph_cache` defined its own `GRAPH_RERANK_WEIGHTS`,
    which the identity assertion refused. It also pins that the graph relation policy partitions
    the relation kinds.
    """
    assert ENTITY_KINDS == get_args(EntityKind)
    assert RELATION_KINDS == get_args(RelationKind)
    assert RELATION_STATUSES == get_args(RelationStatus)
    directional = semantic_graph_cache.GRAPH_DIRECTIONAL_RELATIONS
    diagnostic_only = semantic_graph_cache.GRAPH_DIAGNOSTIC_ONLY_RELATIONS
    assert not directional & diagnostic_only
    assert directional | diagnostic_only == set(RELATION_KINDS)
    for name in (
        "GRAPH_RERANK_WEIGHTS",
        "GRAPH_RERANK_CORROBORATION_CAP",
        "GRAPH_FILL_SLOT_COUNT",
        "GRAPH_FIRST_SEED_K",
        "GRAPH_FIRST_CONTEXT_K",
    ):
        assert getattr(semantic_graph_cache, name) is getattr(graph_expansion, name), name


def test_perf002_a_source_scoped_query_leaves_the_proposal_cache_alone(monkeypatch) -> None:
    """PERF-002: a source scoped query put a single use entry in the proposal cache.

    Its graph is built for that request alone (`reasoning_engine.graph_provider`), so the entry
    only evicted the store graph's. Red by mutation: deleting the `if source is not None:` branch
    of `proposal_provider` in `recall_mcp/reasoning_engine.py` left an entry after the scoped
    query. The unscoped query is the positive control: it does fill the cache.
    """
    from recall.calibration import Calibration
    from recall.guards import DEFAULT_GAP_THRESHOLD

    allowed = Chunk("allowed", "allowed.md", "decision: owner. Ada owns it.", {"file": "allowed.md"})

    class Store:
        tenant = "acme"
        generation_id = "legacy"

        def iter_chunks(self):
            return iter([allowed])

        def supersession_all(self):
            return {}, frozenset(), {}

    class Embedder:
        dim = 3
        name = "dict"

        def embed(self, texts):
            return [[0.0, 0.0, 1.0] for _ in texts]

    def fake_retrieve(*_args, **_kwargs):
        return SimpleNamespace(
            result=replace(
                _result(allowed),
                generation_id="legacy",
                pipeline_fingerprint=None,
                corpus_fingerprint=None,
                diagnostics=RetrievalDiagnostics(index_generation="legacy"),
            )
        )

    monkeypatch.setattr("recall_mcp.retrieval._retrieve_trusted", fake_retrieve)

    def ask(source: str | None) -> None:
        service.reasoning_query(
            Store(),
            Embedder(),
            "owner",
            source=source,
            graph_expansion="off",
            policy=TrustPolicy.development(),
            calibration=Calibration(embedder="dict", threshold=DEFAULT_GAP_THRESHOLD),
        )

    semantic_graph_cache._reset_graph_projection_cache()
    try:
        ask("allowed.md")
        assert len(semantic_graph_cache._DETERMINISTIC_PROPOSAL_CACHE) == 0
        ask(None)
        assert len(semantic_graph_cache._DETERMINISTIC_PROPOSAL_CACHE) == 1
    finally:
        semantic_graph_cache._reset_graph_projection_cache()
