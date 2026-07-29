"""Ring construction — the x-axis, and the two ways it could quietly become circular.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Two properties carry the design. Rings must NEST (d=4 removes everything d=0 removed, plus more),
or "distance" is not a distance. And gold must always be excised at every level, or an
"unanswerable" instance is answerable and the label is a lie.
"""
from __future__ import annotations

import pytest

from benchmarks.ladder.manifest import RING_MAX
from benchmarks.ladder.rings import RingSpec, build_rings, random_rings
from recall.eval.bm25 import BM25Index

CLUSTER = [f"d{i}" for i in range(10)]
DOCS = [
    ("d0", "caroline attended the lgbtq support group on may seventh"),
    ("d1", "caroline mentioned the support group again"),
    ("d2", "the support group meets weekly"),
    ("d3", "melanie ran a charity race"),
    ("d4", "melanie trained for the race"),
    ("d5", "the weather was cold"),
    ("d6", "they discussed dinner plans"),
    ("d7", "a new job application"),
    ("d8", "the cat needed a vet"),
    ("d9", "holiday travel arrangements"),
]
SPEC = RingSpec(widths=(0, 2, 4))
QUESTION = "when did caroline go to the support group"


def test_ring_zero_excises_exactly_the_gold_documents():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert rings[0] == ("d0",)


def test_rings_nest_so_distance_is_a_distance():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert set(rings[0]) <= set(rings[2]) <= set(rings[4]) <= set(rings[RING_MAX])


def test_ring_widths_add_that_many_neighbours():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert len(rings[2]) == 3  # 1 gold + 2 neighbours
    assert len(rings[4]) == 5  # 1 gold + 4 neighbours


def test_ring_max_excises_the_whole_cluster():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert set(rings[RING_MAX]) == set(CLUSTER)


def test_gold_is_excised_at_every_level_including_max():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    assert all("d0" in ids for ids in rings.values())


def test_neighbours_come_only_from_the_cluster():
    outside = DOCS + [("other0", "caroline support group elsewhere")]
    rings = build_rings(BM25Index(outside), QUESTION, ["d0"], CLUSTER, SPEC)
    assert all(d in CLUSTER for d in rings[4])


def test_a_width_wider_than_the_cluster_saturates_rather_than_erroring():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, RingSpec(widths=(0, 500)))
    assert set(rings[500]) == set(CLUSTER)


def test_excised_ids_are_sorted_so_two_builds_agree():
    rings = build_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, SPEC)
    for ids in rings.values():
        assert list(ids) == sorted(ids)


def test_random_rings_are_reproducible_from_their_seed():
    a = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    b = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    c = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=8)
    assert a == b
    assert a != c


def test_random_rings_obey_the_same_nesting_and_gold_rules():
    rings = random_rings(QUESTION, ["d0"], CLUSTER, SPEC, seed=7)
    assert set(rings[0]) <= set(rings[2]) <= set(rings[4])
    assert all("d0" in ids for ids in rings.values())


def test_an_index_that_does_not_cover_the_cluster_is_refused():
    """Otherwise a saturating width silently disagrees with d=max."""
    partial = BM25Index([(d, t) for d, t in DOCS if d != "d2"])
    with pytest.raises(ValueError, match="absent from the BM25 index"):
        build_rings(partial, QUESTION, ["d0"], CLUSTER, SPEC)


def test_gold_outside_its_own_cluster_is_refused():
    with pytest.raises(ValueError, match="not in its own cluster"):
        build_rings(BM25Index(DOCS), QUESTION, ["dGHOST"], CLUSTER, SPEC)


def test_a_question_with_no_gold_is_refused():
    with pytest.raises(ValueError, match="nothing to excise"):
        build_rings(BM25Index(DOCS), QUESTION, [], CLUSTER, SPEC)


def test_random_rings_refuse_the_same_malformed_gold():
    """The two ring functions are drop-in for each other, so they must reject the same inputs."""
    with pytest.raises(ValueError, match="not in its own cluster"):
        random_rings(QUESTION, ["dGHOST"], CLUSTER, SPEC, seed=7)
