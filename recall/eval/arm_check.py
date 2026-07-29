"""Refuse a benchmark run whose mechanism under test provably changes nothing.

The rule of thumb "candidate_k == k renders the reranker inert" is exactly true only when the
REALIZED fused pool equals k: `HybridRetriever` reranks the whole fused pool and truncates to k
afterwards, and a hybrid pool can reach 2 * candidate_k. So inertness is measured at runtime rather
than asserted from configuration — which also catches inertness nobody predicted.

Retrieval only: no generator, no judge, so this costs nothing and runs ahead of all LLM spend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from recall.embeddings import Embedder
from recall.rerank import Reranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore

#: Questions sampled by default. Deterministic (the first N of the caller's list, no RNG), so a
#: preflight verdict is reproducible for a given slice.
DEFAULT_SAMPLE = 25

METRIC_CLASSES = ("set", "ranked")


class InertArmError(RuntimeError):
    """The arm under test does not differ from the arm with its mechanism disabled."""


class EmptySampleError(RuntimeError):
    """`_compare` was asked to compare a mechanism over zero questions.

    Kept distinct from `InertArmError` on purpose: with no questions, the comparison never ran,
    so nothing was PROVEN about the mechanism. `_compare([], [])`'s vacuous `zip` would otherwise
    return `("IDENTICAL", 0)`, which reads exactly like a mechanism that WAS exercised and found
    to make no difference — and `enforce`'s `allow_inert` flag exists to let a caller deliberately
    record a mechanism that was tested and found inert, not to launder a preflight that tested
    nothing. `_compare` raises this unconditionally, before `allow_inert` is ever consulted.
    """


@dataclass(frozen=True)
class Verdict:
    """One mechanism's ablation result."""

    mechanism: str
    verdict: Literal["DIFFERS", "SET_IDENTICAL", "IDENTICAL"]
    sampled: int
    differing: int

    def as_dict(self) -> dict[str, Any]:
        """For stamping into an artifact's `_provenance`."""
        return {
            "mechanism": self.mechanism,
            "verdict": self.verdict,
            "sampled": self.sampled,
            "differing": self.differing,
        }


def _ids(retriever: HybridRetriever, query: str, k: int) -> list[str]:
    return [hit.chunk.id for hit in retriever.search(query, k=k).hits]


def _compare(
    baseline: list[list[str]], ablated: list[list[str]]
) -> tuple[Literal["DIFFERS", "SET_IDENTICAL", "IDENTICAL"], int]:
    """Aggregate per-question comparisons into one verdict plus a differing count.

    Raises `EmptySampleError` on a zero-length sample rather than returning a vacuous
    `("IDENTICAL", 0)` — an empty `zip` compares nothing and proves nothing, and a caller (or a
    ratchet reading the artifact later) must not be able to mistake "never compared" for "compared
    and found identical".
    """
    if not baseline:
        raise EmptySampleError(
            "_compare called with zero questions — there is nothing to compare, so no verdict "
            "can be produced. A caller reached the preflight with an empty in-scope/sample "
            "question list; that is a caller bug, not evidence the mechanism is inert."
        )
    set_differs = sum(1 for a, b in zip(baseline, ablated) if set(a) != set(b))
    if set_differs:
        return "DIFFERS", set_differs
    order_differs = sum(1 for a, b in zip(baseline, ablated) if a != b)
    if order_differs:
        return "SET_IDENTICAL", order_differs
    return "IDENTICAL", 0


def ablation_verdicts(
    store: PgVectorStore,
    embedder: Embedder,
    questions: Sequence[str],
    *,
    k: int,
    candidate_k: int,
    reranker: Reranker | None = None,
    use_sparse: bool = True,
) -> list[Verdict]:
    """One verdict per CONFIGURED mechanism. A mechanism that is off yields no verdict."""
    baseline = HybridRetriever(
        store, embedder, reranker=reranker, use_sparse=use_sparse, candidate_k=candidate_k
    )
    base_ids = [_ids(baseline, q, k) for q in questions]

    verdicts: list[Verdict] = []
    if reranker is not None:
        without_rerank = HybridRetriever(
            store, embedder, reranker=None, use_sparse=use_sparse, candidate_k=candidate_k
        )
        verdict, differing = _compare(base_ids, [_ids(without_rerank, q, k) for q in questions])
        verdicts.append(Verdict("reranker", verdict, len(questions), differing))
    if use_sparse:
        without_sparse = HybridRetriever(
            store, embedder, reranker=reranker, use_sparse=False, candidate_k=candidate_k
        )
        verdict, differing = _compare(base_ids, [_ids(without_sparse, q, k) for q in questions])
        verdicts.append(Verdict("sparse", verdict, len(questions), differing))
    return verdicts


def enforce(verdicts: Sequence[Verdict], *, metric_class: str, allow_inert: bool) -> None:
    """Raise `InertArmError` when a configured mechanism is inert for the metric being reported.

    `metric_class` is DECLARED by the caller, not inferred: inference would hand a new harness the
    permissive branch by default, which is the failure mode this guard exists to prevent.
    """
    if metric_class not in METRIC_CLASSES:
        raise ValueError(f"metric_class must be one of {METRIC_CLASSES}, got {metric_class!r}")
    if allow_inert:
        return
    blocking = {"IDENTICAL"} if metric_class == "ranked" else {"IDENTICAL", "SET_IDENTICAL"}
    bad = [v for v in verdicts if v.verdict in blocking]
    if bad:
        detail = "; ".join(f"{v.mechanism}={v.verdict} over {v.sampled} questions" for v in bad)
        raise InertArmError(
            f"arm is inert for a '{metric_class}' metric: {detail}. The run would measure nothing. "
            f"Widen candidate_k, fix the mechanism, or pass --allow-inert-arm to record it "
            f"deliberately."
        )
