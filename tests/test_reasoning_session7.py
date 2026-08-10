from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from recall.calibration import Calibration
from recall.guards import DEFAULT_GAP_THRESHOLD
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
from recall.reasoning_planner import ReasoningBudgetUsage
from recall.reasoning_proposals import ProposalProtocolReport, ProviderFailure
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
from recall_mcp.service import reasoning_audit, reasoning_projection, reasoning_query
from tests.conftest import TEST_DSN, requires_db


class DictEmbedder:
    dim = 3
    name = "dict"

    def __init__(self, mapping, default):
        self._mapping = mapping
        self._default = default

    def embed(self, texts):
        return [self._mapping.get(text, self._default) for text in texts]


def _trusted_result():
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    chunk = Chunk("c1", "/corpus/a.md", "decision: owner. Ada owns it.", {"file": "a.md"})
    return TrustedResult(
        query="who owns it?",
        hits=[
            TrustedHit(
                chunk=chunk,
                cosine=0.95,
                confidence=0.98,
                verdict="ok",
                provenance=Provenance(chunk.source, "a.md", 0, now),
                validity=Validity(now, None, None),
            )
        ],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, now, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen_1"),
        calibration_id="cal_1",
        calibration_status="certified",
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe_1",
        corpus_fingerprint="corpus_1",
    )


def test_provider_failure_returns_review_and_metrics_without_answering() -> None:
    METRICS.reset()
    retrieval = _trusted_result()
    graph = build_reasoning_graph(
        [retrieval.hits[0].chunk],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe_1",
        corpus_fingerprint="corpus_1",
        include_text=True,
    )
    failure = ProviderFailure(
        kind="timeout",
        provider_id="test-provider",
        model_id="test-model",
        provider_revision="rev",
        message="timed out",
    )

    response = reason(
        ReasoningRequest(
            query=retrieval.query,
            tenant_id="acme",
            generation=GenerationSelection("gen_1", "pipe_1", "corpus_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: retrieval,
                graph_provider=lambda _request, _retrieval: graph,
                proposal_provider=lambda _request, _graph, _retrieval: ProposalProtocolReport(
                    schema_version=1,
                    generation_id="gen_1",
                    pipeline_id="pipe_1",
                    proposals=(),
                    rejected_proposals=(),
                    provider_failures=(failure,),
                ),
                answer_provider=lambda _system, _user: {
                    "answer": "unsupported",
                    "citations": ["c1"],
                    "insufficient_evidence": False,
                },
            ),
            policy=ReasoningPolicy(name="proposal_assisted"),
        )
    )

    assert response.outcome == "needs_review"
    assert response.refusal_reason == "provider_failure"
    assert response.provider_failures == (failure,)
    assert response.diagnostics.generator_invoked is False
    counters = METRICS.snapshot()["counters"]
    assert (
        counters[
            "recall_reasoning_provider_failure_total{kind=timeout,model_id=test-model,provider_id=test-provider}"
        ]
        == 1
    )
    assert "recall_reasoning_budget_exhausted_total" not in counters


def test_proposal_provider_exception_returns_provider_failure() -> None:
    retrieval = _trusted_result()
    graph = build_reasoning_graph(
        [retrieval.hits[0].chunk],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe_1",
        corpus_fingerprint="corpus_1",
        include_text=True,
    )

    response = reason(
        ReasoningRequest(
            query=retrieval.query,
            tenant_id="acme",
            generation=GenerationSelection("gen_1", "pipe_1", "corpus_1"),
            providers=ReasoningProviderPorts(
                retriever=lambda _request: retrieval,
                graph_provider=lambda _request, _retrieval: graph,
                proposal_provider=lambda _request, _graph, _retrieval: (_ for _ in ()).throw(
                    TimeoutError("query leaked here")
                ),
                answer_provider=lambda _system, _user: {
                    "answer": "unsupported",
                    "citations": ["c1"],
                    "insufficient_evidence": False,
                },
            ),
            policy=ReasoningPolicy(name="proposal_assisted"),
        )
    )

    assert response.outcome == "needs_review"
    assert response.refusal_reason == "provider_failure"
    assert response.provider_failures[0].kind == "timeout"
    assert response.provider_failures[0].message == "TimeoutError"
    assert "query leaked here" not in json.dumps(response.to_dict())

    from recall.cli import _reasoning_trace_export

    try:
        _reasoning_trace_export(response)
    except SystemExit as exc:
        assert "provider_failure" in str(exc)
    else:
        raise AssertionError("trace export should refuse when no trace is available")


def test_budget_exhaustion_metric_uses_planner_stop_reason() -> None:
    from recall.reasoning import _record_reasoning_metrics

    METRICS.reset()
    response = reason(
        ReasoningRequest(
            query="who owns it?",
            tenant_id="acme",
            generation=GenerationSelection("gen_1", "pipe_1", "corpus_1"),
            providers=ReasoningProviderPorts(retriever=lambda _request: _trusted_result()),
            policy=ReasoningPolicy(name="retrieval_only"),
        )
    )
    exhausted_response = response.__class__(
        **{
            **response.__dict__,
            "refusal_reason": "no_answer_provider",
            "diagnostics": response.diagnostics.__class__(
                **{
                    **response.diagnostics.__dict__,
                    "budget_used": ReasoningBudgetUsage(
                        steps=0,
                        graph_nodes=0,
                        model_calls=0,
                        evidence_tokens=0,
                        wall_time_ms=0,
                    ),
                }
            ),
        }
    )

    METRICS.reset()
    _record_reasoning_metrics(exhausted_response)
    assert "recall_reasoning_budget_exhausted_total" not in METRICS.snapshot()["counters"]


def test_reasoning_query_filters_graph_to_source_scoped_retrieval(monkeypatch) -> None:
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    allowed = Chunk(
        "allowed", "allowed.md", "decision: allowed owner. Ada owns it.", {"file": "allowed.md"}
    )
    other = Chunk("other", "other.md", "decision: other owner. Bob owns it.", {"file": "other.md"})

    class Store:
        tenant = "acme"
        generation_id = "legacy"

        def iter_chunks(self):
            return iter([allowed, other])

        def supersession_all(self):
            return {}, frozenset(), {}

    def fake_retrieve(*_args, **_kwargs):
        result = TrustedResult(
            query="owner",
            hits=[
                TrustedHit(
                    chunk=allowed,
                    cosine=0.99,
                    confidence=0.99,
                    verdict="ok",
                    provenance=Provenance(allowed.source, "allowed.md", 0, now),
                    validity=Validity(now, None, None),
                )
            ],
            abstained=False,
            reason="",
            gap_warning=False,
            staleness=StalenessReport(False, now, timedelta(0), timedelta(days=1)),
            diagnostics=RetrievalDiagnostics(index_generation="legacy"),
            calibration_status="certified",
            tenant_id="acme",
            generation_id="legacy",
        )

        class Wrapper:
            pass

        wrapper = Wrapper()
        wrapper.result = result
        return wrapper

    captured_sources: list[set[str]] = []

    def fake_proposals(graph, *, pipeline_id):
        captured_sources.append({node.source for node in graph.nodes})
        return ()

    monkeypatch.setattr("recall_mcp.service._retrieve_trusted", fake_retrieve)
    monkeypatch.setattr("recall_mcp.service.deterministic_inference_proposals", fake_proposals)

    response = reasoning_query(
        Store(),
        DictEmbedder({}, default=[0.0, 0.0, 1.0]),
        "owner",
        source="allowed.md",
        policy=TrustPolicy.development(),
        calibration=Calibration(embedder="dict", threshold=DEFAULT_GAP_THRESHOLD),
    )

    assert response.tenant_id == "acme"
    assert captured_sources == [{"allowed.md"}]


def test_reasoning_query_returns_structured_strict_refusal(monkeypatch) -> None:
    class Store:
        tenant = "acme"
        generation_id = "legacy"

    def refuse(*_args, **_kwargs):
        raise TrustRefusal(
            code=TrustFailureCode.CALIBRATION_MISSING,
            calibration_status="missing",
            tenant_id="acme",
            generation_id="legacy",
        )

    monkeypatch.setattr("recall_mcp.service._retrieve_trusted", refuse)

    response = reasoning_query(
        Store(),
        DictEmbedder({}, default=[0.0, 0.0, 1.0]),
        "sensitive query text",
        policy=TrustPolicy.strict_policy(),
    )

    payload = response.to_dict()
    assert response.trust_state == "refused"
    assert response.refusal_reason == "CALIBRATION_MISSING"
    assert payload["trusted_evidence"]["items"] == []
    assert payload["trusted_evidence"]["query"] == ""
    assert "sensitive query text" not in json.dumps(payload)
    assert reasoning_response_from_dict(payload) == response


def test_reasoning_audit_handles_structured_strict_refusal(monkeypatch) -> None:
    class Store:
        tenant = "acme"
        generation_id = "legacy"

        def iter_chunks(self):
            return iter(())

        def supersession_all(self):
            return {}, frozenset(), {}

    def refuse(*_args, **_kwargs):
        raise TrustRefusal(
            code=TrustFailureCode.CALIBRATION_MISSING,
            calibration_status="missing",
            tenant_id="acme",
            generation_id="legacy",
        )

    monkeypatch.setattr("recall_mcp.service._retrieve_trusted", refuse)

    result = reasoning_audit(
        Store(),
        DictEmbedder({}, default=[0.0, 0.0, 1.0]),
        policy=TrustPolicy.strict_policy(),
    )

    assert result.trust_state == "refused"
    assert result.refusal_reasons == ["CALIBRATION_MISSING"]
    assert result.checks["tenant_scoped"] is True


@requires_db
def test_reasoning_service_projection_and_query_preserve_operational_metadata(
    make_store,
) -> None:
    store = make_store(3)
    store.upsert(
        [Chunk("a", "notes.md", "decision: rollout owner. Ada owns rollout.", {"file": "a.md"})],
        [[1.0, 0.0, 0.0]],
    )
    embedder = DictEmbedder(
        {"rollout owner": [1.0, 0.0, 0.0]},
        default=[0.0, 0.0, 1.0],
    )

    projection = reasoning_projection(store)
    response = reasoning_query(
        store,
        embedder,
        "rollout owner",
        policy=TrustPolicy.development(),
        calibration=Calibration(embedder="dict", threshold=DEFAULT_GAP_THRESHOLD),
    )

    assert projection.tenant_id == store.tenant
    assert projection.generation_id == "legacy"
    assert response.tenant_id == store.tenant
    assert response.generation_id == "legacy"
    assert response.trust_state in {"trusted", "degraded"}
    assert response.reasoning_trace is not None
    assert response.refusal_reason == "no_answer_provider"


@requires_db
def test_cli_reasoning_trace_exports_structured_json(
    tmp_path: Path, capsys, cli_table, monkeypatch
) -> None:
    from recall.cli import main

    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    note = tmp_path / "note.md"
    note.write_text("decision: rollout owner. Ada owns rollout.", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    main([*base, "index", str(tmp_path)])
    capsys.readouterr()

    trace_path = tmp_path / "trace.json"
    main([*base, "reasoning", "trace", "rollout owner", "--output", str(trace_path)])
    out = capsys.readouterr().out

    assert "trace:" in out
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["initial_retrieval"]["generation_id"] == "legacy"
    assert "reason" not in payload["initial_retrieval"]
    assert "expansion_steps" in payload


@requires_db
def test_cli_reasoning_projection_requires_explicit_development(
    tmp_path: Path, capsys, cli_table, monkeypatch
) -> None:
    from recall.cli import main

    monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
    note = tmp_path / "note.md"
    note.write_text("decision: rollout owner. Ada owns rollout.", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    main([*base, "index", str(tmp_path)])
    capsys.readouterr()

    try:
        main([*base, "reasoning", "projection"])
    except SystemExit as exc:
        assert "strict mode" in str(exc)
    else:
        raise AssertionError("strict projection inspection should refuse legacy artifacts")
    assert capsys.readouterr().out == ""
