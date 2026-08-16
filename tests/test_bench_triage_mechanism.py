"""The statistics behind the triage mechanism probe, pinned before they are used on real data.

Registered at `results/enterprise_rag/PREREGISTRATION-triage-mechanism.md`. The probe asks whether
`ratio_8_over_1` is reading the disagreement between the dense leg and the fused ranking, so its
whole answer rests on two primitives: a rank correlation and an inversion count. An error in
either produces a plausible number rather than a crash, which is the failure mode this file
exists to prevent.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.explore_triage_signal import build_features
from benchmarks.probe_triage_mechanism import (
    guarded_ratio,
    inversions,
    inverted,
    published_ratio,
    ratio_disagrees_with_inverted,
    spearman,
)


class TestSpearman:
    def test_a_perfectly_increasing_pair_is_exactly_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_a_perfectly_decreasing_pair_is_exactly_minus_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_it_is_rank_based_so_a_monotone_transform_does_not_change_it(self) -> None:
        """The reason for using Spearman rather than Pearson: dense cosines and RRF positions are
        on unrelated scales, and only the ordering is comparable."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 9.0, 11.0, 30.0, 300.0]

        assert spearman(xs, ys) == pytest.approx(spearman(xs, [math.log(y) for y in ys]))

    def test_ties_are_averaged_rather_than_broken_by_position(self) -> None:
        """An all-tied side has no ordering to correlate with, so the answer is undefined, not 0.0
        arrived at by luck of the input order."""
        assert math.isnan(spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))

    def test_a_nan_is_refused_rather_than_silently_ordered(self) -> None:
        """NaN never equals itself, so it neither ties nor sorts predictably: any correlation
        computed over one is a function of row order. `auc()` refuses these for the same reason."""
        with pytest.raises(ValueError, match="NaN"):
            spearman([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])

    def test_fewer_than_two_points_is_undefined_not_zero(self) -> None:
        assert math.isnan(spearman([1.0], [2.0]))

    def test_matches_scipy_across_random_inputs_including_heavy_ties(self) -> None:
        """Ties are where a hand-rolled rank correlation goes wrong, so they are over-represented
        here: values are drawn from a tiny alphabet so collisions are the common case."""
        stats = pytest.importorskip("scipy.stats")
        import random

        rng = random.Random(20260816)
        for _ in range(400):
            n = rng.randint(3, 30)
            xs = [float(rng.randint(0, 3)) for _ in range(n)]
            ys = [float(rng.randint(0, 3)) for _ in range(n)]
            mine, theirs = spearman(xs, ys), float(stats.spearmanr(xs, ys).statistic)
            if math.isnan(theirs):
                assert math.isnan(mine)
            else:
                assert mine == pytest.approx(theirs, abs=1e-9)


class TestInversions:
    def test_a_descending_sequence_has_no_inversions(self) -> None:
        """The list is meant to be best-first, so descending is the agreeing case."""
        assert inversions([0.9, 0.8, 0.7, 0.6]) == 0

    def test_an_ascending_sequence_inverts_every_pair(self) -> None:
        assert inversions([0.6, 0.7, 0.8, 0.9]) == 6  # 4 choose 2

    def test_one_promoted_item_counts_only_the_pairs_it_breaks(self) -> None:
        """`0.5` sits above three items it should be below, and nothing else is out of order."""
        assert inversions([0.5, 0.9, 0.8, 0.7]) == 3

    def test_ties_are_not_inversions(self) -> None:
        assert inversions([0.7, 0.7, 0.7]) == 0

    def test_a_sequence_shorter_than_two_has_none(self) -> None:
        assert inversions([]) == 0
        assert inversions([0.4]) == 0


class TestRatioDefinitions:
    """`published_ratio` must reproduce the number behind the 0.642, warts included; the guarded
    one must refuse the inputs that make a ratio meaningless rather than inventing a value."""

    def test_published_ratio_is_rank_8_over_rank_1(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.45]

        assert published_ratio(scores) == pytest.approx(0.45 / 0.9)

    def test_published_ratio_returns_zero_on_a_short_list_as_it_always_has(self) -> None:
        """⚠️ Pinning a known WART. `at()` returned 0.0 out of range, so a pool shorter than 8
        scored below every genuine value. No fixture row triggered it, and the published numbers
        assume this behaviour, so it is reproduced here rather than fixed here."""
        assert published_ratio([0.9, 0.8]) == 0.0

    def test_published_ratio_returns_zero_when_rank_1_is_zero(self) -> None:
        assert published_ratio([0.0] * 8) == 0.0

    def test_guarded_ratio_agrees_with_the_published_one_on_ordinary_input(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.45]

        assert guarded_ratio(scores) == pytest.approx(published_ratio(scores))

    def test_guarded_ratio_refuses_a_short_list_instead_of_manufacturing_a_zero(self) -> None:
        assert guarded_ratio([0.9, 0.8]) is None

    def test_guarded_ratio_refuses_a_negative_denominator(self) -> None:
        """5 of the 500 fixture rows contain a negative score. A ratio across a sign change is not
        a flatness measure, it is a sign artefact, and it would sort among the genuine values."""
        assert guarded_ratio([-0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]) is None

    def test_guarded_ratio_refuses_a_denominator_at_the_epsilon(self) -> None:
        assert guarded_ratio([1e-12, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]) is None

    def test_guarded_ratio_keeps_a_negative_numerator_because_that_is_real_signal(self) -> None:
        """A dense cosine that goes negative at rank 8 is exactly the disagreement being measured.
        Only the DENOMINATOR has to be well behaved for the ratio to mean anything."""
        assert guarded_ratio([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, -0.09]) == pytest.approx(-0.1)


class TestPublishedRatioMatchesTheScriptItClaimsToReproduce:
    """🔑 Apparatus check A4. `published_ratio` is a SECOND copy of the expression in
    `explore_triage_signal.build_features`, and the whole value of the probe rests on it being the
    same feature that carries the published 0.642. Two copies drift into a plausible number rather
    than an error, so they are pinned against each other here as well as row by row at runtime."""

    @staticmethod
    def _row(scores: list[float]) -> dict[str, object]:
        return {"ranked": [{"doc_id": f"d{i}", "score": s} for i, s in enumerate(scores)],
                "expected_doc_ids": ["d0"], "gap_warning": False}

    @pytest.mark.parametrize("scores", [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],       # ordinary descending
        [0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.95],      # rank 8 above rank 1
        [-0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],      # negative denominator
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, -0.2],      # negative numerator
        [0.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],       # zero denominator
        [0.9, 0.8],                                      # pool shorter than 8
    ])
    def test_the_two_definitions_agree_to_1e_9(self, scores: list[float]) -> None:
        mine = published_ratio(scores)
        theirs = build_features(self._row(scores), "a question", 8)["ratio_8_over_1"]

        assert mine == pytest.approx(theirs, abs=1e-9)


class TestInvertedIsOnePredicate:
    """M2 counts the inverted regime and M6 excludes it, so they must be the same set. The
    registration glosses "dense(rank 8) > dense(rank 1)" as "ratio > 1"; those part company on a
    negative denominator, and 5 of the 500 fixture rows carry a negative score."""

    def test_a_plain_inversion_is_inverted(self) -> None:
        assert inverted([0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.95]) is True

    def test_a_descending_list_is_not(self) -> None:
        assert inverted([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]) is False

    def test_a_pool_shorter_than_eight_is_not_inverted_rather_than_raising(self) -> None:
        assert inverted([0.9, 0.8]) is False

    def test_the_two_phrasings_agree_while_the_denominator_is_positive(self) -> None:
        assert not ratio_disagrees_with_inverted([0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.95])
        assert not ratio_disagrees_with_inverted([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    def test_the_two_phrasings_diverge_on_a_negative_denominator(self) -> None:
        """rank 8 (-0.05) is greater than rank 1 (-0.10), so the raw comparison says inverted;
        the ratio is 0.5, which is below 1, so the gloss says it is not. This row is the reason
        one predicate is defined and reused rather than written twice."""
        scores = [-0.10, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.05]

        assert inverted(scores) is True
        assert published_ratio(scores) == pytest.approx(0.5)
        assert ratio_disagrees_with_inverted(scores) is True
