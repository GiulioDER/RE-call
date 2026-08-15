"""The Voyage reranker must be reachable from the SERVING path, not only from benchmarks.

`benchmarks/voyage_rerank.py` has held a complete, protocol-satisfying `VoyageReranker` since the
EnterpriseRAG work. It could not be served: `pyproject.toml` builds `packages = ["recall",
"recall_mcp"]`, so `benchmarks/` is absent from the wheel, and no value of `RECALL_RERANK_MODEL`
resolved to it. A reranker that exists and cannot be selected is a reranker nobody runs.

The fallback is the part that needs pinning hardest. Falling back from Voyage to the local
cross-encoder keeps retrieval alive through a Voyage outage, and it also means a run can silently
measure a BLEND of two rerankers. That confound is named in
`docs/preregistrations/2026-08-15-bge-large-voyage-splade-memory-corpus.md`, so the fallback has to
be observable: it counts, it logs, and it reports which reranker actually served.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from recall.types import Chunk, ScoredChunk


def _hits(*texts: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id=f"c{i}", source="s.md", text=t, metadata={}), score=0.5 - i / 100)
        for i, t in enumerate(texts)
    ]


@dataclass
class _Item:
    index: int


class _FakeVoyage:
    """Minimal stand-in for `voyageai.Client`: reverses the candidate order."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, documents, model=None, top_k=None):  # noqa: ANN001, ARG002
        self.calls += 1
        order = list(range(len(documents)))[::-1][: (top_k or len(documents))]

        class _R:
            results = [_Item(i) for i in order]

        return _R()


class _BrokenVoyage:
    def rerank(self, *a, **k):  # noqa: ANN002, ANN003, ARG002
        raise RuntimeError("voyage is down")


def test_it_is_importable_from_the_shipped_package() -> None:
    """`benchmarks/` is not in the wheel, so serving code cannot import from there."""
    from recall.rerank import VoyageReranker  # noqa: F401


def test_it_reorders_and_never_rescores() -> None:
    """Same contract as CrossEncoderReranker: identity and `.score` survive the reordering.

    The Voyage relevance score is in different units from a cosine. Leaking it into `.score` would
    corrupt the trust layer's thresholds and its calibrated confidence, which read `.score` as a
    cosine.
    """
    from recall.rerank import VoyageReranker

    hits = _hits("alpha", "beta", "gamma")
    before = [(h.chunk.id, h.score) for h in hits]
    out = VoyageReranker(client=_FakeVoyage()).rerank("q", hits)

    assert [h.chunk.text for h in out] == ["gamma", "beta", "alpha"], "did not reorder"
    assert [(h.chunk.id, h.score) for h in out] == list(reversed(before)), (
        "a hit's identity or score changed; this reranker must reorder only"
    )


def test_empty_input_is_returned_untouched() -> None:
    from recall.rerank import VoyageReranker

    client = _FakeVoyage()
    assert VoyageReranker(client=client).rerank("q", []) == []
    assert client.calls == 0, "an empty pool must not cost a network call"


def test_a_missing_key_raises_rather_than_degrading_silently() -> None:
    from recall.rerank import VoyageReranker

    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageReranker(api_key=None, client=None)


class TestFallback:
    """`FallbackReranker` keeps retrieval alive, and refuses to hide that it did."""

    def test_it_uses_the_primary_when_the_primary_works(self) -> None:
        from recall.rerank import FallbackReranker, VoyageReranker

        primary = VoyageReranker(client=_FakeVoyage())
        r = FallbackReranker(primary=primary, fallback=_ReverseNoop())
        out = r.rerank("q", _hits("a", "b"))
        assert [h.chunk.text for h in out] == ["b", "a"]
        assert r.served_by == "primary"
        assert r.fallback_count == 0

    def test_it_falls_back_when_the_primary_raises(self) -> None:
        from recall.rerank import FallbackReranker, VoyageReranker

        primary = VoyageReranker(client=_BrokenVoyage())
        r = FallbackReranker(primary=primary, fallback=_ReverseNoop())
        out = r.rerank("q", _hits("a", "b"))
        assert [h.chunk.text for h in out] == ["b", "a"], "fallback did not run"
        assert r.served_by == "fallback"
        assert r.fallback_count == 1

    def test_the_fallback_is_counted_not_swallowed(self) -> None:
        """A run that silently mixes two rerankers measures neither.

        The count is the thing that lets a measurement say which reranker served it. Without it a
        Voyage outage mid-run turns a comparison into a blend, and nothing in the result says so.
        """
        from recall.rerank import FallbackReranker, VoyageReranker

        r = FallbackReranker(primary=VoyageReranker(client=_BrokenVoyage()), fallback=_ReverseNoop())
        for _ in range(3):
            r.rerank("q", _hits("a", "b"))
        assert r.fallback_count == 3

    def test_a_failing_fallback_propagates(self) -> None:
        """Two failures is a real error. Returning the unranked pool would look like a result."""
        from recall.rerank import FallbackReranker, VoyageReranker

        class _Boom:
            def rerank(self, query, hits):  # noqa: ANN001, ARG002
                raise RuntimeError("local reranker unavailable too")

        r = FallbackReranker(primary=VoyageReranker(client=_BrokenVoyage()), fallback=_Boom())
        with pytest.raises(RuntimeError, match="local reranker unavailable too"):
            r.rerank("q", _hits("a", "b"))


class _ReverseNoop:
    def rerank(self, query, hits):  # noqa: ANN001, ARG002
        return list(reversed(hits))


class TestSelection:
    """`RECALL_RERANK_MODEL` must be able to name it, or none of the above is reachable."""

    def test_voyage_prefix_selects_the_voyage_reranker(self) -> None:
        from recall.rerank import VoyageReranker, reranker_from_name

        r = reranker_from_name("voyage:rerank-2.5", api_key="k")
        assert isinstance(r, VoyageReranker)
        assert r.model == "rerank-2.5"

    def test_bare_voyage_uses_the_default_model(self) -> None:
        from recall.rerank import VoyageReranker, reranker_from_name

        r = reranker_from_name("voyage", api_key="k")
        assert isinstance(r, VoyageReranker)
        assert r.model == "rerank-2.5"

    def test_an_unprefixed_name_still_means_a_local_cross_encoder(self) -> None:
        """The existing spelling must keep working; this is additive, not a migration."""
        from recall.rerank import reranker_from_name

        # Constructed lazily on purpose: building a CrossEncoder would download weights.
        assert reranker_from_name("BAAI/bge-reranker-base", build=False) == (
            "cross-encoder",
            "BAAI/bge-reranker-base",
        )
