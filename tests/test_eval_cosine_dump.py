"""The class split behind the cosine artifact, pinned — because the shortcut is silently wrong.

`queries.json` holds three kinds of entry: answerable, unanswerable, and six validity-sensitive
`trust` entries that carry NO `answerable` key at all. Bucketing with a falsy
`entry.get("answerable")` — the obvious one-liner — sends all six trust entries into the
unanswerable class, giving a far-gap class of ELEVEN against a true five.

That does not raise. It returns a distribution, and a plausible one: the trust entries are real
queries with real cosines, so the contaminated class still looks like a class. It just is not the
class the figure and the separability number claim to describe. This is the same shape as the
double-index defect next door in `test_eval_locomo_no_double_index` — a wrong artifact that
passes every check the code performs on itself.

`recall.eval.calibrate.measure_top_cosines` already owns this rule and skips on `trust`. These
tests pin the new bucketer to that behaviour so the two cannot drift apart.
"""
from __future__ import annotations

import json

import pytest

from recall.eval.cosine_dump import _reference_bucket, _summarise
from recall.eval.harness import EVAL_DIR


def _queries() -> list[dict]:
    return json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))


def test_trust_entries_are_skipped_not_bucketed_as_unanswerable() -> None:
    trust_entries = [q for q in _queries() if q.get("trust")]
    assert trust_entries, "fixture no longer has trust entries; this test would be vacuous"
    for entry in trust_entries:
        assert _reference_bucket(entry) is None


def test_labelled_entries_bucket_by_their_explicit_label() -> None:
    assert _reference_bucket({"answerable": True}) == "answerable"
    assert _reference_bucket({"answerable": False}) == "far_gap"


def test_split_sizes_match_the_corpus_labels() -> None:
    buckets = [_reference_bucket(q) for q in _queries()]
    queries = _queries()
    # Derived from the file's own explicit labels rather than typed in, so growing the fixture
    # does not silently turn this into a chore that gets "fixed" by editing the number. It still
    # catches the defect: `is True` and `is False` both miss a trust entry, which carries no
    # `answerable` key at all, so a bucketer that swept those into far_gap would exceed this.
    assert buckets.count("answerable") == sum(1 for q in queries if q.get("answerable") is True)
    assert buckets.count("far_gap") == sum(1 for q in queries if q.get("answerable") is False)
    assert buckets.count(None) == sum(1 for q in queries if q.get("trust"))


def test_the_falsy_shortcut_would_contaminate_the_far_gap_class() -> None:
    """The specific defect this bucketer exists to avoid, asserted rather than described."""
    queries = _queries()
    shortcut = ["answerable" if q.get("answerable") else "far_gap" for q in queries]
    correct = [b for b in (_reference_bucket(q) for q in queries) if b is not None]

    labelled_far_gap = sum(1 for q in queries if q.get("answerable") is False)
    trust = sum(1 for q in queries if q.get("trust"))

    # The shortcut sweeps every trust entry into far_gap; the correct rule skips them entirely.
    assert shortcut.count("far_gap") == labelled_far_gap + trust
    assert correct.count("far_gap") == labelled_far_gap
    assert shortcut.count("far_gap") > correct.count("far_gap")


def test_bucketer_agrees_with_the_calibration_sampling_rule() -> None:
    """`measure_top_cosines` is the single source of this split; the two must not drift."""
    queries = _queries()
    # the rule as measure_top_cosines applies it: skip `trust`, then read `answerable`
    expected = [
        ("answerable" if q["answerable"] else "far_gap") for q in queries if not q.get("trust")
    ]
    actual = [b for b in (_reference_bucket(q) for q in queries) if b is not None]
    assert actual == expected


def test_summarise_reports_an_interval_and_the_class_sizes() -> None:
    row = _summarise("pair", [0.9, 0.8, 0.85, 0.95], [0.1, 0.2, 0.15])
    assert row["pair"] == "pair"
    assert row["n_answerable"] == 4
    assert row["n_other"] == 3
    assert row["separability"] == 1.0
    lo, hi = row["ci95"]
    assert lo <= 1.0 <= hi


def test_summarise_survives_an_empty_class() -> None:
    """An absent class must not be reported as a perfect separation."""
    row = _summarise("pair", [0.9], [])
    assert row["separability"] is None
    assert row["ci95"] is None


def test_chart_renders_both_formats(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from recall.eval.cosine_dump import save_cosine_overlap_chart

    report = {
        "embedder": "hashing-64",
        "k": 5,
        "locomo": {
            "answerable": [0.70, 0.74, 0.78, 0.72],
            "adversarial": [0.69, 0.73, 0.77, 0.71],
        },
        "reference_14doc": {
            "answerable": [0.80, 0.85, 0.90],
            "far_gap": [0.51, 0.55, 0.60],
            "near_miss": [0.68, 0.75, 0.82],
        },
        "separability": [
            _summarise("locomo_answerable_vs_adversarial", [0.70, 0.74], [0.69, 0.73]),
            _summarise("reference_answerable_vs_far_gap", [0.80, 0.85], [0.51, 0.55]),
            _summarise("reference_answerable_vs_near_miss", [0.80, 0.85], [0.68, 0.75]),
        ],
    }
    path = save_cosine_overlap_chart(report, tmp_path)
    assert path.exists() and path.stat().st_size > 0
    assert path.with_suffix(".svg").exists()
