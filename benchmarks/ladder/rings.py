"""Excision rings: gold, then a widening ring of BM25 neighbours, then the whole cluster.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

The x-axis is a COUNT of excised documents, not a similarity score. BM25 only decides the ORDER in
which neighbours are removed; the resulting id lists are then frozen into the manifest, so every
lab excises identically no matter what embedder it runs. That is what keeps the axis non-circular:
a system under test never computes its own distances.

BM25 deciding the order is still a choice, and `random_rings` exists to price it — P4 in the
pre-registration. If the curve's shape depends on which neighbour function ordered the removal,
the curve is measuring BM25 and does not ship as an answerability result.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.ladder.manifest import RING_MAX
from recall.eval.bm25 import BM25Index


@dataclass(frozen=True)
class RingSpec:
    """Ring widths, fixed in `benchmarks/PREREGISTRATION-ladder.md` before the builder ran.

    A width is a count of NEIGHBOURS excised on top of gold, so width 0 is "gold only".
    """

    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.widths:
            raise ValueError("a ladder with no rungs is not a ladder")
        if any(w < 0 for w in self.widths):
            raise ValueError("ring widths are counts and cannot be negative")
        if list(self.widths) != sorted(set(self.widths)):
            raise ValueError("widths must be strictly increasing — rings nest")


def _assemble(
    gold: Sequence[str],
    ordered_neighbours: Sequence[str],
    spec: RingSpec,
    cluster: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    gold_set = set(gold)
    rings: dict[int, tuple[str, ...]] = {}
    for width in spec.widths:
        # Saturates rather than erroring: a width wider than the cluster is d=max under another
        # name, and refusing it would make the ladder's top rung depend on conversation length.
        taken = ordered_neighbours[:width]
        rings[width] = tuple(sorted(gold_set | set(taken)))
    rings[RING_MAX] = tuple(sorted(gold_set | set(cluster)))
    return rings


def build_rings(
    index: BM25Index,
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: RingSpec,
) -> dict[int, tuple[str, ...]]:
    """Ring level -> excised doc ids. Neighbours are drawn from the cluster only, BM25 order."""
    gold_set = set(gold_doc_ids)
    cluster_set = set(cluster_doc_ids)
    ordered = [
        doc_id
        for doc_id, _ in index.rank(question)
        if doc_id in cluster_set and doc_id not in gold_set
    ]
    return _assemble(gold_doc_ids, ordered, spec, cluster_doc_ids)


def random_rings(
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: RingSpec,
    *,
    seed: int,
) -> dict[int, tuple[str, ...]]:
    """The P4 robustness arm: same widths, neighbours drawn at random within the cluster.

    `question` is unused and kept in the signature on purpose, so this is a drop-in for
    `build_rings` at the call site rather than a second code path in the builder.
    """
    gold_set = set(gold_doc_ids)
    ordered = sorted(d for d in cluster_doc_ids if d not in gold_set)
    random.Random(seed).shuffle(ordered)
    return _assemble(gold_doc_ids, ordered, spec, cluster_doc_ids)
