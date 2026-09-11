"""Measure temporal and supersession admission before ranking.

This is a deterministic mechanism benchmark, not a retrieval quality claim. It compares the
current expansion path with the former order of operations on the same graph candidates.

Run with ``python benchmarks/temporal_supersession_traversal.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from recall.reasoning import GenerationSelection, ReasoningPolicy, ReasoningProviderPorts, ReasoningRequest
from recall.reasoning_planner import ReasoningBudget
from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk, Provenance, StalenessReport, TrustedHit, TrustedResult, Validity
from recall_mcp.service import _expand_semantic_graph


AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def _chunks(*, include_expired: bool = True, include_superseded: bool = True) -> list[Chunk]:
    projects = ["A"]
    relations: list[dict[str, str]] = []
    chunks = [Chunk("seed", "seed.md", "seed", {"file": "seed.md", "project": projects})]
    for project, chunk_id, source, metadata in (
        ("Expired", "a_expired", "expired.md", {"valid_until": "2025-12-31"}),
        ("Superseded", "b_superseded", "superseded.md", {}),
        ("Live", "z_live", "live.md", {"valid_from": "2026-01-01"}),
    ):
        if project == "Expired" and not include_expired:
            continue
        if project == "Superseded" and not include_superseded:
            continue
        projects.append(project)
        relations.append({"relation": "supports", "subject": "A", "object": project})
        chunks.append(
            Chunk(
                chunk_id,
                source,
                project.lower(),
                {"file": source, "project": project, **metadata},
            )
        )
    chunks[0] = Chunk(
        "seed",
        "seed.md",
        "seed",
        {"file": "seed.md", "project": projects, "relations": relations},
    )
    return chunks


def _legacy_effect(chunks: list[Chunk], max_graph_nodes: int) -> dict[str, Any]:
    potential = sorted(chunk.id for chunk in chunks if chunk.id != "seed")
    charged = potential[: max(0, max_graph_nodes - 1)]
    accepted = [
        chunk_id
        for chunk_id in charged
        if chunk_id == "z_live"
        and not ("a_expired" == chunk_id or "b_superseded" == chunk_id)
    ]
    if "a_expired" in charged:
        accepted.remove("a_expired") if "a_expired" in accepted else None
    if "b_superseded" in charged:
        accepted.remove("b_superseded") if "b_superseded" in accepted else None
    return {"scored": charged, "accepted": accepted}


def _aware_effect(chunks: list[Chunk], max_graph_nodes: int) -> dict[str, Any]:
    graph = build_semantic_graph(
        chunks,
        tenant_id="benchmark",
        generation_id="generation",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    scored: list[str] = []

    class Store:
        tenant = "benchmark"
        generation_id = "generation"

        def load_semantic_graph(self, generation_id: str | None = None) -> Any:
            return graph

        def graph_readiness(self) -> Any:
            return graph.readiness()

        def supersession_all(self) -> tuple[dict[str, str], frozenset[str], dict[str, str]]:
            return {"superseded": "replacement"}, frozenset(), {}

        def iter_chunks(self) -> Iterator[Chunk]:
            return iter(chunks)

        def cosines_for(self, ids: Sequence[str], vector: Sequence[float]) -> dict[str, float]:
            del vector
            scored.extend(ids)
            return {chunk_id: 0.9 for chunk_id in ids}

    seed = TrustedHit(
        chunks[0], 1.0, 1.0, "ok", Provenance("seed.md", "seed.md", 0, None), Validity(None, None, None)
    )
    retrieval = TrustedResult(
        query="which project is live",
        hits=[seed],
        abstained=False,
        reason="",
        gap_warning=True,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        tenant_id="benchmark",
        generation_id="generation",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_status="legacy_unbound",
    )
    request = ReasoningRequest(
        query="which project is live",
        tenant_id="benchmark",
        generation=GenerationSelection("generation", "p" * 64, "c" * 64),
        providers=ReasoningProviderPorts(retriever=lambda _: retrieval),
        policy=ReasoningPolicy(graph_expansion="one_hop"),
        budget=ReasoningBudget(max_graph_nodes=max_graph_nodes, max_graph_hops=1),
        as_of=AS_OF,
    )
    result = _expand_semantic_graph(
        Store(),  # type: ignore[arg-type]
        request,
        retrieval,
        None,
        type("Embedder", (), {"embed_query": lambda self, _: [1.0]})(),
    )
    return {
        "scored": scored,
        "accepted": [hit.chunk.id for hit in result.retrieval.hits if hit.chunk.id != "seed"],
        "rejections": dict(result.admission_rejections),
    }


def main() -> None:
    cases = (
        ("one_slot_stale_first", _chunks(), 2),
        ("two_slots_stale_first", _chunks(), 3),
        ("current_only_control", _chunks(include_expired=False, include_superseded=False), 2),
    )
    rows = []
    for name, chunks, budget in cases:
        legacy = _legacy_effect(chunks, budget)
        aware = _aware_effect(chunks, budget)
        rows.append(
            {
                "case": name,
                "max_graph_nodes": budget,
                "legacy": legacy,
                "temporal_supersession_aware": aware,
                "legacy_live_neighbor_preserved": legacy["accepted"] == ["z_live"],
                "aware_live_neighbor_preserved": aware["accepted"] == ["z_live"],
                "scoring_calls_saved": len(legacy["scored"]) - len(aware["scored"]),
            }
        )
    print(json.dumps({"as_of": AS_OF.isoformat(), "cases": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
