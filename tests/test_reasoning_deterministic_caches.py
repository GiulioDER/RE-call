from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

import recall_mcp.service as service
from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_planner import plan_multi_hop_evidence
from recall.reasoning_proposals import deterministic_inference_proposals
from recall.reasoning_proposals import _deterministic as deterministic_rules
from recall.types import Chunk, Provenance, RetrievalDiagnostics, StalenessReport, TrustedHit, TrustedResult, Validity


UTC = timezone.utc


class _Store:
    tenant = "acme"

    def __init__(self, generation_id: str = "gen-1") -> None:
        self.generation_id = generation_id


def _chunk(
    chunk_id: str,
    file: str,
    text: str,
    *,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> Chunk:
    metadata: dict[str, object] = {"file": file, "ord": 0}
    if valid_from is not None:
        metadata["valid_from"] = valid_from
    if valid_until is not None:
        metadata["valid_until"] = valid_until
    return Chunk(chunk_id, f"/corpus/{file}", text, metadata)


def _proposal_graph(generation_id: str = "gen-1"):
    return build_reasoning_graph(
        [
            _chunk("old", "policy_v1.md", "decision: policy. Status: active."),
            _chunk("new", "policy_v2.md", "decision: policy. Status: active."),
        ],
        tenant_id="acme",
        generation_id=generation_id,
        pipeline_fingerprint="pipeline-a",
        include_text=True,
    )


@pytest.fixture(autouse=True)
def _reset_reasoning_caches():
    reset_service = getattr(service, "_reset_graph_projection_cache", None)
    if reset_service is not None:
        reset_service()
    yield
    if reset_service is not None:
        reset_service()


def test_repeated_proposal_queries_reuse_deterministic_output(monkeypatch) -> None:
    """The same immutable generation must invoke deterministic rules once.

    Invariant: two calls through the MCP proposal consumer share one proposal tuple. The failure
    mode is rebuilding every deterministic rule set for every query. Red proof against the current
    implementation is two calls to the patched production symbol
    ``recall_mcp.service.deterministic_inference_proposals``; the cache implementation must reduce
    that to one call.
    """
    graph = _proposal_graph()
    store = _Store()
    calls: list[tuple[object, str]] = []
    original = service.deterministic_inference_proposals

    monkeypatch.setattr(service, "_store_graph", lambda *_args, **_kwargs: graph)

    def counted(graph_arg, *, pipeline_id):
        calls.append((graph_arg, pipeline_id))
        return original(graph_arg, pipeline_id=pipeline_id)

    monkeypatch.setattr(service, "deterministic_inference_proposals", counted)

    first = service.reasoning_proposals(store)
    second = service.reasoning_proposals(store)

    assert len(calls) == 1
    assert first.model_dump() == second.model_dump()


def test_generation_change_invalidates_deterministic_proposals(monkeypatch) -> None:
    """A new generation must never reuse proposals from the prior generation."""
    graphs = {generation: _proposal_graph(generation) for generation in ("gen-1", "gen-2")}
    store = _Store()
    calls: list[str] = []

    def project(_store, **_kwargs):
        return graphs[store.generation_id]

    original = service.deterministic_inference_proposals

    monkeypatch.setattr(service, "_store_graph", project)

    def counted(graph_arg, *, pipeline_id):
        calls.append(graph_arg.generation_id)
        return original(graph_arg, pipeline_id=pipeline_id)

    monkeypatch.setattr(service, "deterministic_inference_proposals", counted)

    first = service.reasoning_proposals(store)
    store.generation_id = "gen-2"
    second = service.reasoning_proposals(store)

    assert calls == ["gen-1", "gen-2"]
    assert first.generation_id == "gen-1"
    assert second.generation_id == "gen-2"


def test_cached_proposal_output_is_byte_stable(monkeypatch) -> None:
    """A provider that changes iteration order cannot change the serialized query result."""
    graph = _proposal_graph()
    store = _Store()
    proposals = deterministic_inference_proposals(graph)
    calls = 0

    monkeypatch.setattr(service, "_store_graph", lambda *_args, **_kwargs: graph)

    def unstable(_graph, *, pipeline_id):
        nonlocal calls
        calls += 1
        return proposals if calls == 1 else tuple(reversed(proposals))

    monkeypatch.setattr(service, "deterministic_inference_proposals", unstable)

    first = service.reasoning_proposals(store)
    second = service.reasoning_proposals(store)
    first_bytes = json.dumps(first.model_dump(), sort_keys=True, separators=(",", ":"))
    second_bytes = json.dumps(second.model_dump(), sort_keys=True, separators=(",", ":"))

    assert calls == 1
    assert first_bytes == second_bytes


def _trusted_result(chunk: Chunk) -> TrustedResult:
    hit = TrustedHit(
        chunk=chunk,
        cosine=0.99,
        confidence=0.99,
        verdict="ok",
        provenance=Provenance(chunk.source, chunk.metadata["file"], 0, None),
        validity=Validity(None, None, None),
    )
    return TrustedResult(
        query="q",
        hits=[hit],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen-1"),
        calibration_status="certified",
        tenant_id="acme",
        generation_id="gen-1",
        pipeline_fingerprint="pipeline-a",
    )


def test_planner_indexes_are_built_once_per_graph_and_policy_scope(monkeypatch) -> None:
    """Planner adjacency construction is reusable but remains partitioned by policy scope.

    Invariant: identical graph, generation, pipeline, proposals, and policy scope build indexes
    once. Red proof against the current implementation is three calls to ``_build_indexes`` for
    two identical scopes and one changed scope. The production boundary is
    ``recall.reasoning_planner.plan_multi_hop_evidence``.
    """
    import recall.reasoning_planner as planner

    reset = getattr(planner, "_reset_planner_index_cache", None)
    if reset is not None:
        reset()
    graph = _proposal_graph()
    proposals = deterministic_inference_proposals(graph)
    original = planner._build_indexes
    calls = 0

    def counted(graph_arg, proposals_arg):
        nonlocal calls
        calls += 1
        return original(graph_arg, proposals_arg)

    monkeypatch.setattr(planner, "_build_indexes", counted)

    seed = _chunk("old", "policy_v1.md", "decision: policy. Status: active.")
    plan_multi_hop_evidence(
        _trusted_result(seed), graph, proposals=proposals, policy_scope="scope-a"
    )
    plan_multi_hop_evidence(
        _trusted_result(seed), graph, proposals=proposals, policy_scope="scope-a"
    )
    plan_multi_hop_evidence(
        _trusted_result(seed), graph, proposals=proposals, policy_scope="scope-b"
    )

    assert calls == 2


def test_disjoint_contradictory_windows_do_not_scan_all_pairs(monkeypatch) -> None:
    """Disjoint validity windows must not pay a pairwise overlap check for every claim pair.

    Invariant: the interval sweep is near linear when no windows overlap. Red proof against the
    current implementation counts ``_windows_overlap`` calls in the deterministic provider, which
    is quadratic for this fixture. The production symbol is
    ``recall.reasoning_proposals._deterministic._contradictory_validity_window_proposals``.
    """
    count = 120
    chunks = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(count):
        left = start + timedelta(days=index * 2)
        right = left + timedelta(days=1)
        status = "active" if index % 2 == 0 else "disabled"
        chunks.append(
            _chunk(
                f"claim-{index}",
                f"policy_{index:03d}.md",
                f"decision: policy. Status: {status}.",
                valid_from=left.isoformat(),
                valid_until=right.isoformat(),
            )
        )
    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen-large",
        pipeline_fingerprint="pipeline-a",
        include_text=True,
    )
    overlap_calls = 0
    original = deterministic_rules._windows_overlap

    def counted(left, right):
        nonlocal overlap_calls
        overlap_calls += 1
        return original(left, right)

    monkeypatch.setattr(deterministic_rules, "_windows_overlap", counted)
    deterministic_inference_proposals(graph)

    assert overlap_calls <= count * 4
