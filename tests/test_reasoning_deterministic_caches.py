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


def _alternating_policy_graph(windows: list[tuple[int, int]]):
    """One subject, alternating active and disabled claims, one validity window (in days) each."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    chunks = [
        _chunk(
            f"claim-{index}",
            f"policy_{index:03d}.md",
            f"decision: policy. Status: {'active' if index % 2 == 0 else 'disabled'}.",
            valid_from=(start + timedelta(days=first)).date().isoformat(),
            valid_until=(start + timedelta(days=last)).date().isoformat(),
        )
        for index, (first, last) in enumerate(windows)
    ]
    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen-large",
        pipeline_fingerprint="pipeline-a",
        include_text=True,
    )
    # Validity is YYYY-MM-DD. A full ISO datetime is rejected as malformed and the window is
    # dropped, which makes every window unbounded: the fixture this replaced did exactly that.
    assert [d for d in graph.diagnostics if d.kind == "malformed_metadata"] == []
    return graph


def _count_pair_checks(monkeypatch) -> list[int]:
    """Count the sweep's per-candidate-pair work: one opposing-text check per pair it visits."""
    calls = [0]
    original = deterministic_rules._opposing_validity_text

    def counted(left, right):
        calls[0] += 1
        return original(left, right)

    monkeypatch.setattr(deterministic_rules, "_opposing_validity_text", counted)
    return calls


def _contradictions(graph) -> list:
    return [
        proposal
        for proposal in deterministic_inference_proposals(graph)
        if proposal.rule_id == "deterministic.contradictory_validity_windows"
    ]


def test_disjoint_contradictory_windows_do_not_scan_all_pairs(monkeypatch) -> None:
    """Disjoint validity windows must not pay a pairwise check for every opposing claim pair.

    Invariant: the interval sweep in
    ``recall.reasoning_proposals._deterministic._contradictory_validity_window_proposals`` visits
    a candidate pair only while both windows are active, so 120 disjoint, alternating windows on
    one subject cost no pair checks at all, where a pairwise scan costs 60 * 60 = 3,600.

    This replaces a version that could not fail, for two independent reasons: it counted
    ``_windows_overlap``, which production had stopped calling, so its count was always 0; and
    its ISO datetime validity was rejected as malformed, so every window was unbounded and all
    3,600 opposing pairs were reported as contradictions without anyone seeing it. The counter
    here is the call the sweep really makes per candidate pair, and the overlapping control below
    proves it observes that call.

    Red proof (2026-09-23, base ``c7f2b9bc``), node
    ``tests/test_reasoning_deterministic_caches.py::test_disjoint_contradictory_windows_do_not_scan_all_pairs``:
    deleting the ``while expiry and expiry[0][0] < claim_start`` expiry loop, so every earlier
    claim stays active, fails ``assert calls[0] == 0`` with 3,600 pair checks.
    """
    count = 120
    calls = _count_pair_checks(monkeypatch)

    disjoint = _alternating_policy_graph([(index * 2, index * 2 + 1) for index in range(count)])
    found = _contradictions(disjoint)
    assert calls[0] == 0
    assert found == []

    # Control: the same two opposing claims with overlapping windows must reach the counter and
    # yield the contradiction, so a zero above is the sweep's doing, not an idle counter.
    overlapping = _alternating_policy_graph([(0, 10), (5, 15)])
    assert len(_contradictions(overlapping)) == 1
    assert calls[0] == 1
