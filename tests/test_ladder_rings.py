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
from benchmarks.ladder.rings import (
    FractionSpec,
    RingSpec,
    build_fractional_rings,
    build_rings,
    fraction_to_ring,
    random_rings,
    ring_to_fraction,
)
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


# --- v2: fractional rings -------------------------------------------------------------------

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
FRACTION_SPEC = FractionSpec(fractions=FRACTIONS)


def test_fraction_spec_rejects_empty():
    with pytest.raises(ValueError, match="no rungs"):
        FractionSpec(fractions=())


def test_fraction_spec_rejects_out_of_range():
    with pytest.raises(ValueError, match="0.0, 1.0|range"):
        FractionSpec(fractions=(0.0, 1.5))


def test_fraction_spec_rejects_non_increasing():
    with pytest.raises(ValueError, match="increasing"):
        FractionSpec(fractions=(0.5, 0.25))


def test_fraction_spec_rejects_negative():
    with pytest.raises(ValueError, match="0.0, 1.0|range"):
        FractionSpec(fractions=(-0.1, 0.5))


def test_fraction_to_ring_uses_basis_points():
    assert fraction_to_ring(0.0) == 0
    assert fraction_to_ring(0.25) == 2500
    assert fraction_to_ring(0.5) == 5000
    assert fraction_to_ring(0.75) == 7500
    assert fraction_to_ring(1.0) == 10000


@pytest.mark.parametrize("f", FRACTIONS)
def test_fraction_ring_round_trip(f):
    assert ring_to_fraction(fraction_to_ring(f)) == pytest.approx(f)


def test_r0_excises_exactly_gold_including_at_the_zero_rung():
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert rings["r0.00"] == ("d0",)


def test_gold_is_excised_at_every_fraction():
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert all("d0" in ids for ids in rings.values())


def test_fractional_rings_nest():
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    ordered_labels = ["r0.00", "r0.25", "r0.50", "r0.75", "r1.00"]
    sets = [set(rings[label]) for label in ordered_labels]
    for smaller, larger in zip(sets, sets[1:]):
        assert smaller <= larger


def test_r1_excises_exactly_the_whole_cluster():
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert set(rings["r1.00"]) == set(CLUSTER)


def test_fractional_rings_are_keyed_by_string_labels():
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert set(rings.keys()) == {"r0.00", "r0.25", "r0.50", "r0.75", "r1.00"}


def test_fractional_neighbours_come_only_from_the_cluster():
    outside = DOCS + [("other0", "caroline support group elsewhere")]
    rings = build_fractional_rings(BM25Index(outside), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert all(d in CLUSTER for d in rings["r0.75"])


def test_fractional_rings_reuse_the_same_validation_as_build_rings():
    with pytest.raises(ValueError, match="not in its own cluster"):
        build_fractional_rings(BM25Index(DOCS), QUESTION, ["dGHOST"], CLUSTER, FRACTION_SPEC)
    with pytest.raises(ValueError, match="nothing to excise"):
        build_fractional_rings(BM25Index(DOCS), QUESTION, [], CLUSTER, FRACTION_SPEC)
    partial = BM25Index([(d, t) for d, t in DOCS if d != "d2"])
    with pytest.raises(ValueError, match="absent from the BM25 index"):
        build_fractional_rings(partial, QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)


def test_fractional_ring_widths_match_the_rounding_rule():
    # cluster has 10 docs, 1 gold -> n_non_gold = 9. round(f * 9) neighbours on top of gold.
    rings = build_fractional_rings(BM25Index(DOCS), QUESTION, ["d0"], CLUSTER, FRACTION_SPEC)
    assert len(rings["r0.00"]) == 1  # 1 gold + round(0.0*9)=0
    assert len(rings["r0.25"]) == 1 + round(0.25 * 9)
    assert len(rings["r0.50"]) == 1 + round(0.5 * 9)
    assert len(rings["r0.75"]) == 1 + round(0.75 * 9)
    assert len(rings["r1.00"]) == 10
