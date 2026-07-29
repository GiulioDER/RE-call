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

**v2 — fractional rings.** `RingSpec`/`build_rings` are v1: absolute neighbour COUNTS, calibrated
for clusters an order of magnitude smaller than LOCOMO's conversations (see
`PREREGISTRATION-ladder-v2.md` §0). `FractionSpec`/`build_fractional_rings` are v2: rungs are
FRACTIONS of the question's own cluster, so a rung cannot be mis-scaled by corpus size. Widths and
fractions are unrelated ring systems — `RingSpec(widths=(0, 4))` and `FractionSpec(fractions=(0.0,
0.5))` are never combined in one call, and a manifest is built under exactly one of them.

`Instance.ring` is an `int` (v1's contract, unchanged — see `manifest.py`), and a fraction is not
an int, so v2 rungs are encoded as **basis points**: `fraction_to_ring(f) = round(f * 10000)`, and
`ring_to_fraction` is its inverse. `build_fractional_rings` itself returns a `dict[str, ...]` keyed
by a rung LABEL (`"r0.00"`, `"r0.25"`, …), not by ring int — the caller that constructs `Instance`
rows is what applies `fraction_to_ring` to get the int for the `ring` field; label -> basis points
is a 1:1 mapping (`f"r{ring_to_fraction(fraction_to_ring(f)):.2f}"` recovers the label from the
int) but it is not automatic, so state it here where a reader looking for "where do fractions
become ints" will find it.

Basis points were chosen, over say fixed-point-1000 or a separate float column, because they keep
the existing `int` field (no schema change to `Instance` or the manifest digest, whose `_FIELDS`
tuple is untouched by this module), they sort correctly as plain integers exactly like v1's
widths, and their range (0-10000) cannot collide with v1's sentinels `RING_MAX = -1` /
`RING_ORIGINAL = -2` or with v1's small absolute widths (0/4/16/64) — EXCEPT at the value `0`,
which both `RingSpec(widths=(0, ...))` (v1, "gold only") and `fraction_to_ring(0.0)` (v2, "gold
only") produce. That collision is safe and not accidental: `0` means the same thing in both
systems ("gold excised, nothing else"), and a manifest is built under v1 XOR v2 rungs, never a mix
of both in the same file — `MANIFEST_VERSION` (`manifest.py`) tags which one a given manifest
uses, so a reader is never asked to disambiguate a `ring: 0` between the two meanings within one
file.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.ladder.manifest import RING_MAX
from recall.eval.bm25 import BM25Index

#: v2 rung widths are expressed as basis points of the question's own cluster (see module
#: docstring for why basis points rather than a float field).
_BASIS_POINTS = 10_000


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


def _validate_gold(gold_set: set[str], cluster_set: set[str]) -> None:
    """Shared by every ring function (v1 and v2, BM25-ordered and random): gold must be
    non-empty and a subset of its own cluster, or the instance built from it is broken upstream.
    """
    if not gold_set:
        raise ValueError("a question with no gold evidence has nothing to excise")
    outside = sorted(gold_set - cluster_set)
    if outside:
        raise ValueError(
            f"gold evidence {outside} is not in its own cluster. Excising it would produce an "
            f"instance labelled unanswerable whose answer never lived in the ingested slice — a "
            f"question that is broken upstream, not hard."
        )


def _ordered_cluster_neighbours(
    index: BM25Index,
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
) -> tuple[set[str], set[str], list[str]]:
    """Validate gold/cluster/index coverage and return the BM25-ordered neighbour list.

    Shared by `build_rings` (v1, absolute widths) and `build_fractional_rings` (v2, fractions) —
    the ONLY code that ranks and validates against a live `BM25Index`; `_assemble` and its
    fractional counterpart just slice the resulting list at different widths. Ties break by
    `doc_id` ascending — that is `BM25Index.rank`'s own contract, not reimplemented here.
    """
    gold_set = set(gold_doc_ids)
    cluster_set = set(cluster_doc_ids)
    ordered = [
        doc_id
        for doc_id, _ in index.rank(question)
        if doc_id in cluster_set and doc_id not in gold_set
    ]
    missing = sorted(cluster_set - set(index.doc_ids))
    if missing:
        raise ValueError(
            f"{len(missing)} cluster documents are absent from the BM25 index ({missing[:5]}). "
            f"Ring widths are drawn from the index, so a saturating width would not reach them and "
            f"would silently disagree with d=max. Build the index over the whole corpus."
        )
    _validate_gold(gold_set, cluster_set)
    return gold_set, cluster_set, ordered


def _assemble(
    gold: Sequence[str],
    ordered_neighbours: Sequence[str],
    spec: RingSpec,
    cluster: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    gold_set = set(gold)
    cluster_set = set(cluster)
    _validate_gold(gold_set, cluster_set)
    rings: dict[int, tuple[str, ...]] = {}
    for width in spec.widths:
        # Saturates rather than erroring: a width wider than the cluster is d=max under another
        # name, and refusing it would make the ladder's top rung depend on conversation length.
        taken = ordered_neighbours[:width]
        rings[width] = tuple(sorted(gold_set | set(taken)))
    rings[RING_MAX] = tuple(sorted(gold_set | cluster_set))
    return rings


def build_rings(
    index: BM25Index,
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: RingSpec,
) -> dict[int, tuple[str, ...]]:
    """Ring level -> excised doc ids. Neighbours are drawn from the cluster only, BM25 order."""
    _gold_set, _cluster_set, ordered = _ordered_cluster_neighbours(
        index, question, gold_doc_ids, cluster_doc_ids
    )
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


@dataclass(frozen=True)
class FractionSpec:
    """v2 rungs as FRACTIONS of the question's own cluster.

    v1 used absolute counts and they were calibrated for clusters an order of magnitude smaller
    than LOCOMO's conversations, so its widest non-degenerate rung removed ~10 % of the topic. A
    fraction cannot be mis-scaled by corpus size, which is the whole reason for this type.
    """

    fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("a ladder with no rungs is not a ladder")
        if any(f < 0.0 or f > 1.0 for f in self.fractions):
            raise ValueError("fractions are proportions of a cluster and must be in [0.0, 1.0]")
        if list(self.fractions) != sorted(set(self.fractions)):
            raise ValueError("fractions must be strictly increasing — rings nest")


def fraction_to_ring(f: float) -> int:
    """Map a v2 fraction to the basis-points int stored in `Instance.ring` (see module docstring
    for why basis points rather than a new float field).
    """
    return round(f * _BASIS_POINTS)


def ring_to_fraction(ring: int) -> float:
    """Inverse of `fraction_to_ring`."""
    return ring / _BASIS_POINTS


def _fraction_label(f: float) -> str:
    """`0.25 -> "r0.25"`. The five pre-registered fractions (0.00/0.25/0.50/0.75/1.00) all render
    to two decimal places with no rounding ambiguity; this is not meant to generalise beyond them.
    """
    return f"r{f:.2f}"


def build_fractional_rings(
    index: BM25Index,
    question: str,
    gold_doc_ids: Sequence[str],
    cluster_doc_ids: Sequence[str],
    spec: FractionSpec,
) -> dict[str, tuple[str, ...]]:
    """Rung label -> excised doc ids, for v2's fractional ladder.

    Mirrors `build_rings`: neighbours come only from the question's own cluster, in the same BM25
    order (ties by `doc_id` ascending), and gold is always excised — including at `f = 0.00`. At
    fraction `f`, the excised set is gold plus the first `round(f * n_non_gold)` BM25-ordered
    neighbours, where `n_non_gold = len(cluster) - len(gold)` (gold is a subset of the cluster,
    enforced by `_validate_gold`). At `f = 1.00` this is exactly the whole cluster: `round(1.00 *
    n_non_gold) == n_non_gold`, so every non-gold cluster document is taken.

    Keyed by a STRING rung label (`"r0.00"`, …), not by `Instance.ring` — see the module docstring
    for the `fraction_to_ring`/`ring_to_fraction` mapping the caller applies when constructing
    `Instance` rows from this dict.

    Validation is the same helper `build_rings` uses (`_ordered_cluster_neighbours` /
    `_validate_gold`) — there is exactly one copy of "gold must be a non-empty subset of its
    cluster, and the index must cover the cluster."
    """
    gold_set, cluster_set, ordered = _ordered_cluster_neighbours(
        index, question, gold_doc_ids, cluster_doc_ids
    )
    n_non_gold = len(cluster_set - gold_set)
    rings: dict[str, tuple[str, ...]] = {}
    for f in spec.fractions:
        width = round(f * n_non_gold)
        taken = ordered[:width]
        rings[_fraction_label(f)] = tuple(sorted(gold_set | set(taken)))
    return rings
