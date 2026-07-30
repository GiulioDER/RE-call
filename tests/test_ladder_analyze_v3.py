"""The v3 analyzer, pinned against the arm whose numbers are already published.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

The load-bearing test here is the differential oracle: the analyzer must reproduce the FIGURES
ALREADY PUBLISHED in `results/ladder/H1_VERDICT_v2.md` from the frozen v2 artifacts. An analyzer
validated only against its own fixtures proves it is self-consistent, not correct.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.ladder.analyze_v3 import auc, bootstrap_ci, load_cosines

_V2_MANIFEST = Path("results/ladder/manifest_v2.jsonl")
_V2_RESPONSES = Path("results/ladder/responses_v2.jsonl")
_HAVE_V2 = _V2_MANIFEST.exists() and _V2_RESPONSES.exists()


def test_auc_is_one_when_positives_all_outrank_negatives():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0


def test_auc_is_half_on_identical_distributions_not_zero():
    """Ties count 0.5 — a tie is no evidence either way, and scoring it 0 would report a
    perfectly uninformative signal as actively wrong."""
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_is_symmetric_under_swapping_the_classes():
    assert auc([0.9], [0.1]) == pytest.approx(1.0 - auc([0.1], [0.9]))


def test_bootstrap_refuses_empty_input_rather_than_reporting_a_zero():
    """(0,0,0) is indistinguishable from a tightly-measured null — the same defect score.py
    already guards, reproduced here would let absent data read as a result."""
    with pytest.raises(ValueError, match="absent data"):
        bootstrap_ci([])


def test_bootstrap_ci_brackets_the_mean():
    m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0], iterations=500)
    assert lo <= m <= hi


@pytest.mark.skipif(not _HAVE_V2, reason="frozen v2 artifacts not present")
def test_reproduces_the_published_v2_auc_curve():
    """DIFFERENTIAL ORACLE — these five figures are published in H1_VERDICT_v2.md's addendum."""
    from benchmarks.ladder.manifest import RING_ORIGINAL

    cos = load_cosines(_V2_MANIFEST, _V2_RESPONSES)
    ans = [c[RING_ORIGINAL] for c in cos.values() if RING_ORIGINAL in c]
    expected = {0: 0.567, 2500: 0.784, 5000: 0.841, 7500: 0.921, 10000: 0.968}
    for ring, want in expected.items():
        got = auc(ans, [c[ring] for c in cos.values() if ring in c])
        assert got == pytest.approx(want, abs=0.001), f"ring {ring}: {got:.3f} != {want}"


@pytest.mark.skipif(not _HAVE_V2, reason="frozen v2 artifacts not present")
def test_reproduces_the_published_v2_paired_delta():
    """-0.1100 [-0.1172, -0.1028] is published in H1_VERDICT_v2.md section 2."""
    cos = load_cosines(_V2_MANIFEST, _V2_RESPONSES)
    d = [c[10000] - c[0] for c in cos.values() if 10000 in c and 0 in c]
    m, lo, hi = bootstrap_ci(d)
    assert m == pytest.approx(-0.1100, abs=0.0005)
    assert lo == pytest.approx(-0.1172, abs=0.002)
    assert hi == pytest.approx(-0.1028, abs=0.002)
