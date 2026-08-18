"""The stratified sampler must not reproduce the head bias it exists to remove.

The published `dense_floor_strat100` sample is the first ten ids of each category block, drawn by
code that is not in the tree. This project has been bitten twice by sorting a sample and then
truncating it, so the properties under test here are: the draw depends on the seed, it is
reproducible from the seed alone, it is NOT the head, and a stratum it cannot fill is refused
rather than silently shortened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.enterprise_rag_sample import (
    head_of_each_stratum,
    sampling_provenance,
    stratify,
    strata_of,
)


@dataclass(frozen=True)
class Q:
    question_id: str
    question_type: str


def _population(sizes: dict[str, int]) -> list[Q]:
    # Each stratum gets its OWN id space. Numbering sequentially across strata would renumber
    # `rare` whenever `basic` is resized, and the isolation test below would then be comparing
    # different questions rather than a different shuffle.
    out: list[Q] = []
    for kind, count in sizes.items():
        out.extend(Q(question_id=f"{kind}_{i:04d}", question_type=kind) for i in range(1, count + 1))
    return out


SIZES = {"basic": 40, "semantic": 30, "rare": 12}


def test_it_draws_exactly_per_stratum_from_every_stratum() -> None:
    chosen = stratify(_population(SIZES), per_stratum=10, seed=7)

    counts: dict[str, int] = {}
    for q in chosen:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1

    assert counts == {"basic": 10, "semantic": 10, "rare": 10}
    assert len(chosen) == 30


def test_the_same_seed_reproduces_the_same_draw() -> None:
    pop = _population(SIZES)

    assert [q.question_id for q in stratify(pop, per_stratum=10, seed=7)] == [
        q.question_id for q in stratify(pop, per_stratum=10, seed=7)
    ]


def test_a_different_seed_draws_differently() -> None:
    """If the seed did not matter, the sampler would be a fixed selection wearing a seed."""
    pop = _population(SIZES)

    a = [q.question_id for q in stratify(pop, per_stratum=10, seed=7)]
    b = [q.question_id for q in stratify(pop, per_stratum=10, seed=8)]

    assert a != b


def test_the_draw_is_not_the_head_of_each_block() -> None:
    """🔑 The defect being removed. `basic` holds 40 consecutive ids; the head is the first 10."""
    pop = _population(SIZES)

    chosen = [q.question_id for q in stratify(pop, per_stratum=10, seed=7)]
    head = [q.question_id for q in head_of_each_stratum(
        pop, per_stratum=10, type_of=lambda q: q.question_type)]

    assert chosen != head


def test_one_stratum_is_unaffected_by_another_being_resized() -> None:
    """One RNG per stratum, so adding questions to `basic` must not reshuffle `rare`. A single
    shared generator would make every stratum depend on the iteration order of every other."""
    small = stratify(_population(SIZES), per_stratum=10, seed=7)
    bigger = stratify(_population({**SIZES, "basic": 60}), per_stratum=10, seed=7)

    rare_small = [q.question_id for q in small if q.question_type == "rare"]
    rare_bigger = [q.question_id for q in bigger if q.question_type == "rare"]

    assert rare_small == rare_bigger


def test_output_is_sorted_by_question_id_for_a_diffable_artifact() -> None:
    chosen = stratify(_population(SIZES), per_stratum=10, seed=7)

    assert [q.question_id for q in chosen] == sorted(q.question_id for q in chosen)


class TestRefusals:
    def test_a_stratum_smaller_than_the_design_is_refused_not_truncated(self) -> None:
        """A short stratum silently changes what a population-weighted rate is a rate over."""
        with pytest.raises(ValueError, match="rare"):
            stratify(_population({**SIZES, "rare": 3}), per_stratum=10, seed=7)

    def test_the_error_names_the_sizes_so_the_operator_can_fix_the_design(self) -> None:
        with pytest.raises(ValueError, match=r"\('rare', 3\)"):
            stratify(_population({**SIZES, "rare": 3}), per_stratum=10, seed=7)

    def test_per_stratum_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="per_stratum"):
            stratify(_population(SIZES), per_stratum=0, seed=7)


def test_provenance_records_what_a_stranger_needs_to_repeat_the_draw() -> None:
    pop = _population(SIZES)
    chosen = stratify(pop, per_stratum=10, seed=7)

    prov = sampling_provenance(pop, chosen, per_stratum=10, seed=7)

    assert prov["seed"] == 7
    assert prov["per_stratum"] == 10
    assert prov["population_n"] == 82
    assert prov["sample_n"] == 30
    assert prov["question_ids"] == [q.question_id for q in chosen]


def test_strata_of_counts_the_population_the_weights_must_agree_with() -> None:
    assert strata_of(_population(SIZES), type_of=lambda q: q.question_type) == SIZES


def test_two_strata_of_equal_size_do_not_select_the_same_positions() -> None:
    """🔑 Mutation-driven. Seeding every stratum with the bare `seed` still isolates them from one
    another's SIZE, so the isolation test above passes, but two strata of equal length then draw
    the identical set of indices. That is hidden structure in a sample whose whole purpose is to
    have none: whatever position correlates with in one category, it correlates with in the other.
    """
    pop = _population({"alpha": 30, "beta": 30})

    chosen = stratify(pop, per_stratum=10, seed=7)
    positions = {}
    for kind in ("alpha", "beta"):
        ids = [q.question_id for q in chosen if q.question_type == kind]
        positions[kind] = sorted(int(qid.split("_")[1]) for qid in ids)

    assert positions["alpha"] != positions["beta"]
