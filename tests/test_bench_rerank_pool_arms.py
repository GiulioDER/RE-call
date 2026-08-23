"""Scoring and statistics behind the pool-rerank benchmark and the fine-tune null.

These two modules produced numbers that are now cited (MiniLM captures 63% of Voyage's gain; the
fine-tune is a null). Everything they report is built from four small pure functions, so those are
what get tested. Both modules must stay importable under a bare `.[dev]` install -- CI omits the
heavy extras on purpose -- which is asserted here rather than assumed.
"""
from __future__ import annotations

import pytest

from benchmarks.rerank_finetune import _arm
from benchmarks.rerank_pool_arms import _ceiling, _hit_at, _paired_bootstrap, _summarise

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


class TestHitAt:
    def test_hit_when_gold_inside_cutoff(self) -> None:
        assert _hit_at(["a", "b", "c"], {"b"}, 2) == 1.0

    def test_miss_when_gold_below_cutoff(self) -> None:
        """The boundary that decides every hit@k in both scripts."""
        assert _hit_at(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_gold_exactly_at_cutoff_counts(self) -> None:
        assert _hit_at(["a", "b", "c"], {"b"}, 2) == 1.0

    def test_any_of_several_gold_docs_counts(self) -> None:
        """292 of the 400 ladder instances have 1 gold doc; the rest have up to 19."""
        assert _hit_at(["a", "b"], {"z", "b"}, 2) == 1.0

    def test_no_gold_in_order_is_a_miss(self) -> None:
        assert _hit_at(["a", "b"], {"z"}, 2) == 0.0

    def test_k_larger_than_order_does_not_raise(self) -> None:
        assert _hit_at(["a"], {"a"}, 10) == 1.0


class TestCeiling:
    @pytest.mark.parametrize(
        ("hit5", "expected"),
        [(0.640, 0.820), (0.785, 0.8925), (0.870, 0.935), (1.0, 1.0), (0.0, 0.5)],
    )
    def test_published_ceilings(self, hit5: float, expected: float) -> None:
        """The three published ceiling@5 values must fall out of the formula, not a constant."""
        assert _ceiling(hit5) == pytest.approx(expected)


class TestPairedBootstrap:
    def test_constant_positive_difference_has_ci_excluding_zero(self) -> None:
        a = [0.0] * 200
        b = [1.0] * 200
        mean, lo, hi = _paired_bootstrap(a, b, n_resamples=200)
        assert mean == pytest.approx(1.0)
        assert lo > 0 and hi >= lo

    def test_identical_arms_give_zero_delta_and_degenerate_ci(self) -> None:
        """The fine-tune's hit@10 delta was exactly this shape."""
        vals = [1.0, 0.0] * 100
        mean, lo, hi = _paired_bootstrap(vals, vals, n_resamples=200)
        assert (mean, lo, hi) == (0.0, 0.0, 0.0)

    def test_sign_follows_b_minus_a_not_a_minus_b(self) -> None:
        """Orientation bug would flip every reported delta, including the null's direction."""
        mean, _lo, _hi = _paired_bootstrap([1.0] * 50, [0.0] * 50, n_resamples=100)
        assert mean == pytest.approx(-1.0)

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        a = [1.0, 0.0, 1.0, 1.0, 0.0] * 40
        b = [0.0, 1.0, 1.0, 0.0, 0.0] * 40
        assert _paired_bootstrap(a, b, n_resamples=300, seed=7) == _paired_bootstrap(
            a, b, n_resamples=300, seed=7
        )

    def test_is_paired_not_independent(self) -> None:
        """Per-instance pairing is the whole point: a perfectly offsetting pair nets to zero."""
        mean, _lo, _hi = _paired_bootstrap([1.0, 0.0], [0.0, 1.0], n_resamples=100)
        assert mean == pytest.approx(0.0)


class TestSummarise:
    def test_reports_mean_per_k_and_derives_the_ceiling(self) -> None:
        got = _summarise({1: [1.0, 0.0], 5: [1.0, 1.0], 10: [1.0, 1.0]})
        assert got[1] == pytest.approx(0.5)
        assert got[5] == pytest.approx(1.0)
        assert got["ceiling@5"] == pytest.approx(1.0)


class TestArm:
    def test_scores_orderings_against_gold_and_matches_hit_at(self) -> None:
        ids = ["q1", "q2"]
        order = {"q1": ["g", "x", "y"], "q2": ["x", "y", "g"]}
        gold = {"q1": {"g"}, "q2": {"g"}}
        per, summary = _arm(order, ids, gold)
        assert per[1] == [1.0, 0.0]
        assert summary[1] == pytest.approx(0.5)
        assert summary[5] == pytest.approx(1.0)
        assert summary["ceiling@5"] == pytest.approx(1.0)


def test_modules_import_without_the_heavy_extras() -> None:
    """CI installs `.[dev]`, which has no torch / sentence-transformers / datasets.

    Reaching this test at all proves both modules imported under that install. Asserting it
    explicitly documents the constraint so a future module-level `import torch` fails here with a
    readable reason instead of as an opaque collection error.
    """
    import benchmarks.rerank_finetune as ft
    import benchmarks.rerank_pool_arms as pa

    assert ft.BASE_HIT5 == 0.785, "pins the differential-oracle invariant to the measured baseline"
    assert pa.PUBLISHED["retrieval"][5] == 0.640
    assert pa.PUBLISHED["voyage"][5] == 0.870
    for heavy in ("torch", "sentence_transformers", "datasets"):
        assert not hasattr(ft, heavy), f"{heavy} must not be a module global of rerank_finetune"
        assert not hasattr(pa, heavy), f"{heavy} must not be a module global of rerank_pool_arms"
