from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recall.calibration import Calibration
from recall.reasoning_graph import build_reasoning_graph, project_store_graph
from recall.store import resolve_supersession_candidates
from recall.trust import evaluate
from recall.types import Chunk, RetrievalDiagnostics, RetrievalResult, ScoredChunk, StalenessReport

MON = datetime(2026, 3, 2, tzinfo=timezone.utc)
TUE = datetime(2026, 3, 3, tzinfo=timezone.utc)


def _chunk(cid: str, file: str, *, supersedes: str | None = None) -> Chunk:
    metadata = {"file": file, "ord": 0}
    if supersedes is not None:
        metadata["supersedes"] = supersedes
    return Chunk(cid, f"/corpus/{file}", f"text for {file}", metadata)


def test_graph_projection_is_deterministic_for_identical_inputs() -> None:
    chunks = [_chunk("b", "v2.md", supersedes="v1.md"), _chunk("a", "v1.md")]

    first = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_1")
    second = build_reasoning_graph(list(reversed(chunks)), tenant_id="acme", generation_id="gen_1")

    assert second.graph_id == first.graph_id
    assert [node.id for node in second.nodes] == [node.id for node in first.nodes]
    assert [edge.id for edge in second.authored_edges] == [
        edge.id for edge in first.authored_edges
    ]


def test_authored_edges_reproduce_current_supersession_resolution() -> None:
    chunks = [_chunk("a", "v1.md"), _chunk("b", "v2.md", supersedes="v1.md")]
    expected, unresolved, candidates = resolve_supersession_candidates(
        [("v1.md", None, None), ("v2.md", "v1.md", MON)]
    )

    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen_1",
        authored_edge_candidates=candidates,
        unresolved_references=unresolved,
    )

    assert graph.authored_supersession_map() == expected
    assert graph.authored_edges[0].asserted_at == MON
    assert graph.inferred_candidate_edges == ()


def test_graph_preserves_exact_as_of_boundary_dates() -> None:
    chunks = [_chunk("a", "v1.md"), _chunk("b", "v2.md", supersedes="v1.md")]
    _expected, unresolved, candidates = resolve_supersession_candidates(
        [("v1.md", None, None), ("v2.md", "v1.md", TUE)]
    )

    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen_1",
        authored_edge_candidates=candidates,
        unresolved_references=unresolved,
    )

    assert graph.authored_edges[0].asserted_at == TUE

    _expected, monday_unresolved, monday_candidates = resolve_supersession_candidates(
        [("v1.md", None, None), ("v2.md", "v1.md", MON)]
    )
    monday_graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen_1",
        authored_edge_candidates=monday_candidates,
        unresolved_references=monday_unresolved,
    )

    assert graph.authored_edges[0].id != monday_graph.authored_edges[0].id
    assert graph.graph_id != monday_graph.graph_id


def test_graph_identity_is_tenant_and_generation_scoped() -> None:
    chunks = [_chunk("a", "v1.md")]

    acme = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_1")
    globex = build_reasoning_graph(chunks, tenant_id="globex", generation_id="gen_1")
    next_generation = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_2")

    assert len({acme.graph_id, globex.graph_id, next_generation.graph_id}) == 3
    assert {node.tenant_id for node in acme.nodes} == {"acme"}
    assert {node.generation_id for node in next_generation.nodes} == {"gen_2"}


def test_graph_projection_metadata_is_deeply_immutable() -> None:
    graph = build_reasoning_graph(
        [
            Chunk(
                "a",
                "/a.md",
                "a",
                {"file": "a.md", "calibration": {"scores": [0.9, 0.8]}},
            )
        ],
        tenant_id="acme",
        generation_id="gen_1",
    )
    chunk_node = next(node for node in graph.nodes if node.kind == "chunk")

    with pytest.raises(TypeError):
        chunk_node.metadata["extra"] = "changed"  # type: ignore[index]
    assert isinstance(chunk_node.calibration["scores"], tuple)


def test_graph_reports_ambiguous_references_and_duplicate_candidates() -> None:
    chunks = [
        _chunk("a", "one/x.md"),
        _chunk("b", "two/x.md"),
        _chunk("c", "new.md", supersedes="x.md"),
    ]

    graph = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_1")

    kinds = {diagnostic.kind for diagnostic in graph.diagnostics}
    assert "ambiguous_reference" in kinds
    assert "duplicate_entity_candidate" in kinds
    assert graph.authored_edges == ()


def test_graph_reports_dangling_targets_from_deleted_sources() -> None:
    chunks = [_chunk("b", "v2.md", supersedes="deleted.md")]

    graph = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_1")

    assert graph.authored_supersession_map() == {"deleted.md": "v2.md"}
    assert {diagnostic.kind for diagnostic in graph.diagnostics} >= {"unresolved_reference"}


def test_graph_reports_cycles_conflicts_orphans_and_malformed_metadata() -> None:
    chunks = [
        Chunk("a", "/a.md", "a", {"file": "a.md", "valid_until": "not-a-date"}),
        _chunk("b", "b.md", supersedes="a.md"),
        _chunk("c", "c.md", supersedes="b.md"),
        _chunk("d", "b-newer.md", supersedes="a.md"),
        _chunk("e", "isolated.md"),
    ]
    _winner, unresolved, candidates = resolve_supersession_candidates(
        [
            ("a.md", None, None),
            ("b.md", "a.md", MON),
            ("c.md", "b.md", MON),
            ("b-newer.md", "a.md", TUE),
            ("isolated.md", None, None),
        ]
    )
    candidates["b.md"].append(("a.md", TUE))

    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen_1",
        authored_edge_candidates=candidates,
        unresolved_references=unresolved,
    )

    kinds = {diagnostic.kind for diagnostic in graph.diagnostics}
    assert {
        "conflicting_authored_claim",
        "cycle",
        "orphaned_node",
        "malformed_metadata",
    } <= kinds


def test_graph_cycle_detection_follows_all_conflicting_outgoing_edges() -> None:
    chunks = [
        _chunk("a", "a.md"),
        _chunk("b", "b.md", supersedes="a.md"),
        _chunk("c", "c.md", supersedes="a.md"),
        _chunk("d", "d.md", supersedes="c.md"),
    ]
    _winner, unresolved, candidates = resolve_supersession_candidates(
        [
            ("a.md", None, None),
            ("b.md", "a.md", MON),
            ("c.md", "a.md", TUE),
            ("d.md", "c.md", MON),
        ]
    )
    candidates["d.md"] = [("a.md", TUE)]

    graph = build_reasoning_graph(
        chunks,
        tenant_id="acme",
        generation_id="gen_1",
        authored_edge_candidates=candidates,
        unresolved_references=unresolved,
    )

    cycle_edges = [
        diagnostic.edge_ids for diagnostic in graph.diagnostics if diagnostic.kind == "cycle"
    ]
    assert any(len(edge_ids) == 3 for edge_ids in cycle_edges)


class _Store:
    tenant = "acme"
    generation_id = "legacy"

    def __init__(self) -> None:
        self.supersession_calls = 0
        self.chunks = [_chunk("a", "v1.md"), _chunk("b", "v2.md", supersedes="v1.md")]

    def iter_chunks(self) -> list[Chunk]:
        return self.chunks

    def supersession_all(self):
        self.supersession_calls += 1
        return {"v1.md": "v2.md"}, frozenset(), {"v1.md": [("v2.md", MON)]}


class _GenerationStore(_Store):
    def snapshot(self):
        class _Snapshot:
            def __enter__(self):
                return "gen_pinned"

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Snapshot()

    def generation_binding(self):
        return {
            "tenant_id": self.tenant,
            "generation_id": "gen_pinned",
            "pipeline_fingerprint": "pipeline-fp",
            "corpus_fingerprint": "corpus-fp",
        }


def test_store_projection_uses_one_pinned_generation_view() -> None:
    store = _GenerationStore()

    graph = project_store_graph(store)

    assert graph.generation_id == "gen_pinned"
    assert graph.pipeline_fingerprint == "pipeline-fp"
    assert graph.corpus_fingerprint == "corpus-fp"
    assert graph.authored_edges[0].asserted_at == MON
    assert store.supersession_calls == 1


def test_graph_projection_cannot_make_a_stale_hit_trusted() -> None:
    stale = ScoredChunk(
        Chunk("old", "v1.md", "old", {"file": "v1.md"}),
        score=0.99,
        indexed_at=MON,
        first_indexed_at=MON,
    )
    result = RetrievalResult(
        query="q",
        hits=[stale],
        gap_warning=False,
        staleness=StalenessReport(False, None, None, max_age=timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen_1"),
    )
    graph = build_reasoning_graph(
        [_chunk("old", "v1.md"), _chunk("new", "v2.md", supersedes="v1.md")],
        tenant_id="acme",
        generation_id="gen_1",
    )

    trusted = evaluate(
        result,
        graph.authored_supersession_map(),
        Calibration(embedder="test", threshold=0.5, scale=0.05),
        TUE,
    )

    assert trusted.hits[0].verdict == "superseded"
    assert not trusted.calibrated


def test_a_long_linear_supersession_chain_projects_without_recursion_error() -> None:
    # BUG-006 regression: the cycle walk used to recurse once per file, so an authored
    # chain a few hundred to a thousand files deep raised RecursionError inside
    # build_reasoning_graph. The iterative walk must project it without error.
    chunks = []
    for index in range(1500):
        supersedes = f"m{index - 1:04d}.md" if index else None
        chunks.append(_chunk(f"c{index:04d}", f"m{index:04d}.md", supersedes=supersedes))

    graph = build_reasoning_graph(chunks, tenant_id="acme", generation_id="gen_1")

    assert len(graph.authored_edges) == 1499
    assert not any(diag.kind == "cycle" for diag in graph.diagnostics)


def test_cycle_diagnostics_and_graph_id_survive_the_iterative_walk_unchanged() -> None:
    # Golden identities captured from the recursive walk before the BUG-006 rewrite
    # (commit 3498a06c plus nothing). The fixture holds two cycles sharing a member and
    # one orphan, so it exercises the canonical cycle dedup and diagnostic ordering.
    def golden_chunk(cid: str, file: str, supersedes: str | None = None) -> Chunk:
        metadata: dict[str, object] = {"file": file}
        if supersedes is not None:
            metadata["supersedes"] = supersedes
        return Chunk(cid, file, "text", metadata)

    chunks = [
        golden_chunk("c1", "a.md", "b.md"),
        golden_chunk("c2", "b.md", "c.md"),
        golden_chunk("c3", "c.md", "a.md"),
        golden_chunk("c4", "d.md", "e.md"),
        golden_chunk("c5", "e.md", "d.md"),
        golden_chunk("c6", "f.md", "a.md"),
        golden_chunk("c7", "g.md"),
    ]

    graph = build_reasoning_graph(chunks, tenant_id="tenant-golden", generation_id="gen-golden")

    assert graph.graph_id == "rg_graph_d9d614ef04ce6fa5b7b520ac"
    assert [(diag.id, diag.kind, diag.reference, diag.message) for diag in graph.diagnostics] == [
        (
            "rg_diag_1db3115bcbd4485202e3b776",
            "conflicting_authored_claim",
            "a.md",
            "a.md has 2 authored supersession claims",
        ),
        (
            "rg_diag_684d7199973182aa48c5be29",
            "orphaned_node",
            "g.md",
            "g.md has no authored graph edges",
        ),
        (
            "rg_diag_9446b31e2aeec6a176946550",
            "cycle",
            "d.md",
            "authored supersession cycle includes d.md, e.md",
        ),
        (
            "rg_diag_c1d666e9748a7c5c11847f43",
            "cycle",
            "a.md",
            "authored supersession cycle includes a.md, c.md, b.md",
        ),
    ]


def test_supersession_rows_fallback_is_undated_by_design() -> None:
    # DAT-009 / CODE-009: the fallback path collects only (file, supersedes) pairs.
    # Chunk metadata carries no asserted_at, so the third element is always None, and
    # the edge ids it produces deliberately differ from the dated store path's.
    from recall.reasoning_graph import _supersession_rows

    chunks = [
        _chunk("b", "v2.md", supersedes="v1.md"),
        _chunk("a", "v1.md"),
        _chunk("c", "v2.md", supersedes="v1.md"),
        Chunk("d", "/corpus/x", "no file metadata", {}),
    ]

    rows = _supersession_rows(chunks)

    assert rows == [
        (None, None, None),
        ("v1.md", None, None),
        ("v2.md", "v1.md", None),
    ]
