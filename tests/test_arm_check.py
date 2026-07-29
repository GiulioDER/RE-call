"""The self-ablation preflight: disable the mechanism, re-retrieve, require the output to differ.

This is CCA's red-state proof transposed. There, a test claimed as proof of a fix is re-run against
the PRE-fix code and must fail; a test that passes both ways proves nothing. Here, an arm claiming
to measure a mechanism is re-run with the mechanism OFF and must return something different; an arm
that returns the same thing either way measured nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from recall.eval.arm_check import InertArmError, Verdict, ablation_verdicts, enforce


def test_enforce_passes_when_every_mechanism_differs() -> None:
    enforce([Verdict("reranker", "DIFFERS", 25, 19)], metric_class="set", allow_inert=False)


def test_enforce_blocks_identical_on_any_metric_class() -> None:
    verdicts = [Verdict("reranker", "IDENTICAL", 25, 0)]
    with pytest.raises(InertArmError, match="IDENTICAL"):
        enforce(verdicts, metric_class="set", allow_inert=False)
    with pytest.raises(InertArmError, match="IDENTICAL"):
        enforce(verdicts, metric_class="ranked", allow_inert=False)


def test_enforce_blocks_set_identical_only_for_set_metrics() -> None:
    """Same ids, different order: inert for hit@k, live for a rank-sensitive metric."""
    verdicts = [Verdict("reranker", "SET_IDENTICAL", 25, 0)]
    with pytest.raises(InertArmError, match="SET_IDENTICAL"):
        enforce(verdicts, metric_class="set", allow_inert=False)
    enforce(verdicts, metric_class="ranked", allow_inert=False)


def test_allow_inert_lets_it_through() -> None:
    enforce([Verdict("reranker", "IDENTICAL", 25, 0)], metric_class="set", allow_inert=True)


def test_enforce_rejects_an_unknown_metric_class() -> None:
    """The caller declares its metric class; inference would hand a new harness the permissive
    branch by default."""
    with pytest.raises(ValueError, match="metric_class"):
        enforce([], metric_class="whatever", allow_inert=False)


def test_verdicts_are_json_serialisable() -> None:
    """They are stamped into the artifact's `_provenance`, so they must survive json.dumps."""
    payload = [Verdict("reranker", "DIFFERS", 25, 19).as_dict()]
    assert json.loads(json.dumps(payload)) == [
        {"mechanism": "reranker", "verdict": "DIFFERS", "sampled": 25, "differing": 19}
    ]


@dataclass
class _Chunk:
    id: str
    text: str = ""
    source: str = ""


@dataclass
class _Hit:
    chunk: _Chunk
    score: float = 0.5
    indexed_at: object = None


class _StubStore:
    """Returns a fixed dense list and a fixed sparse list, sliced to the requested k."""

    def __init__(self, dense: list[str], sparse: list[str]) -> None:
        self._dense = dense
        self._sparse = sparse

    def query_dense(self, qvec: object, k: int, source: object = None) -> list[_Hit]:
        return [_Hit(_Chunk(cid), 1.0 - i / 100) for i, cid in enumerate(self._dense[:k])]

    def query_sparse(
        self, query: str, k: int, source: object = None, vec: object = None
    ) -> list[_Hit]:
        return [_Hit(_Chunk(cid), 0.5) for cid in self._sparse[:k]]

    def newest_indexed_at(self) -> object:
        return None


class _StubEmbedder:
    dim = 3
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 1.0] for _ in texts]


class _ReversingReranker:
    """Reverses the pool — changes the SET only when the pool is wider than k."""

    def rerank(self, query: str, hits: list[_Hit]) -> list[_Hit]:
        return list(reversed(hits))


class _NoOpReranker:
    def rerank(self, query: str, hits: list[_Hit]) -> list[_Hit]:
        return list(hits)


def test_reranker_over_a_wide_pool_differs() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20,
        reranker=_ReversingReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["DIFFERS"]


def test_reranker_on_a_pool_equal_to_k_cannot_change_the_set() -> None:
    """candidate_k == k on a dense-only arm: the pool IS the answer, so reranking reorders it but
    cannot change which ids survive. This is the documented inert-reranker case."""
    store = _StubStore(dense=[f"d{i}" for i in range(5)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=5,
        reranker=_ReversingReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["SET_IDENTICAL"]


def test_a_reranker_that_changes_nothing_is_identical() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20,
        reranker=_NoOpReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["IDENTICAL"]


def test_a_sparse_leg_that_contributes_nothing_is_identical() -> None:
    """The sparse leg was silently inert for a whole artifact generation, pre-#81/#84."""
    dense = [f"d{i}" for i in range(20)]
    store = _StubStore(dense=dense, sparse=dense)  # sparse returns exactly what dense returns
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20, reranker=None, use_sparse=True,
    )
    assert [(v.mechanism, v.verdict) for v in verdicts] == [("sparse", "IDENTICAL")]


def test_no_mechanisms_configured_yields_no_verdicts() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    assert ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20, reranker=None, use_sparse=False,
    ) == []
