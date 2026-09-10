from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from recall.reasoning_graph import build_reasoning_graph
from recall.semantic_graph import build_semantic_graph
from recall.timing import TimedEmbedder
from recall.trust_policy import TrustPolicy
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import service


class _CountingEmbedder:
    dim = 2
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def _fixture() -> tuple[TrustedResult, object, object, list[Chunk]]:
    seed_chunk = Chunk(
        "seed",
        "seed.md",
        "The seed evidence.",
        {
            "file": "seed.md",
            "project": ["A", "B"],
            "relations": [{"relation": "supports", "subject": "A", "object": "B"}],
        },
    )
    neighbor_chunk = Chunk(
        "neighbor",
        "neighbor.md",
        "The neighboring evidence.",
        {"file": "neighbor.md", "project": "B"},
    )
    chunks = [seed_chunk, neighbor_chunk]
    semantic = build_semantic_graph(
        chunks,
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    projected = build_reasoning_graph(
        chunks,
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        include_text=True,
        semantic_graph=semantic,
    )
    indexed_at = None
    seed_hit = TrustedHit(
        seed_chunk,
        cosine=0.95,
        confidence=0.99,
        verdict="ok",
        provenance=Provenance("seed.md", "seed.md", 0, indexed_at),
        validity=Validity(None, None, None),
    )
    result = TrustedResult(
        query="q",
        hits=[seed_hit],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, None, None, timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="generation-a"),
        calibration_status="certified",
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    return result, semantic, projected, chunks


class _Store:
    tenant = "tenant-a"
    generation_id = "generation-a"

    def __init__(self, semantic, projected, chunks: list[Chunk]) -> None:
        self.semantic = semantic
        self.projected = projected
        self.chunks = chunks
        self.seen_vectors: list[list[float]] = []

    def graph_readiness(self):
        return self.semantic.readiness()

    def load_semantic_graph(self, generation_id=None):
        assert generation_id == self.generation_id
        return self.semantic

    def iter_chunks(self):
        return iter(self.chunks)

    def cosines_for(self, ids, vector):
        self.seen_vectors.append(list(vector))
        return {chunk_id: 0.90 for chunk_id in ids}

    def supersession(self):
        return {}, frozenset()


def _run(monkeypatch, graph_expansion: str):
    baseline, semantic, projected, chunks = _fixture()
    store = _Store(semantic, projected, chunks)
    embedder = _CountingEmbedder()

    def fake_retrieve(_store, local_embedder, query, *_args, **_kwargs):
        timed = TimedEmbedder(local_embedder)
        timed.embed_query(query)
        return SimpleNamespace(result=baseline, query_vector=timed.last_query_vector)

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(service, "_store_graph", lambda *_args, **_kwargs: projected)
    response = service.reasoning_query(
        store,
        embedder,
        "q",
        mode="retrieval_only",
        graph_expansion=graph_expansion,
        policy=TrustPolicy.development(),
    )
    return response, embedder, store


@pytest.mark.parametrize("graph_expansion", ["off", "one_hop"])
def test_reasoning_query_embeds_the_query_once(graph_expansion, monkeypatch) -> None:
    """The baseline and one hop graph path each make one query embedding call.

    Invariant: the provider call count is one for both graph modes. The failure mode is a second
    provider call from ``recall_mcp.service._expand_semantic_graph`` after baseline retrieval.
    Red proof for node ``tests/test_reasoning_embedding_reuse.py::test_reasoning_query_embeds_the_query_once``:
    mutate the production assignment at ``recall_mcp.service._expand_semantic_graph`` so it always
    calls ``embed_query(embedder, request.query)``. The one hop parameter then fails with
    ``assert 2 == 1`` while graph off remains at one, proving the assertion reaches the graph
    expansion consumer. Restore the assignment before the green run.
    """
    _response, embedder, _store = _run(monkeypatch, graph_expansion)

    assert embedder.calls == 1


def test_graph_reuse_preserves_baseline_ranking_and_trust(monkeypatch) -> None:
    """Graph expansion must not rescore or re-verdict the baseline evidence."""
    off, _off_embedder, _off_store = _run(monkeypatch, "off")
    one_hop, _one_hop_embedder, store = _run(monkeypatch, "one_hop")

    off_item = off.trusted_evidence.items[0]
    one_hop_item = one_hop.trusted_evidence.items[0]
    assert (
        off_item.chunk_id,
        off_item.cosine,
        off_item.confidence,
        off_item.verdict,
    ) == (
        one_hop_item.chunk_id,
        one_hop_item.cosine,
        one_hop_item.confidence,
        one_hop_item.verdict,
    )
    assert (
        off.trusted_evidence.calibrated,
        off.trusted_evidence.decision_state,
        off.trusted_evidence.trust_state,
        off.trust_state,
    ) == (
        one_hop.trusted_evidence.calibrated,
        one_hop.trusted_evidence.decision_state,
        one_hop.trusted_evidence.trust_state,
        one_hop.trust_state,
    )
    assert one_hop.trusted_evidence.items[1].chunk_id == "neighbor"
    assert store.seen_vectors == [[1.0, 0.0]]
