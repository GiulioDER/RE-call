"""The calibration has to say whether a threshold is VIABLE, not just return one.

`best_threshold` already bisects overlapping distributions — its docstring says so — and then
hands back a number that looks exactly like a working threshold. Measured on LongMemEval's
per-question haystacks that silently produced false-abstain 0.443: the gate refused nearly half
the questions retrieval had just answered correctly, and nothing in the artifact or the API said
the classes had not separated.

These tests pin the diagnosis, and pin that it stays a diagnosis: a non-separable calibration is
reported, not quietly disarmed. Changing runtime behaviour on low separability would replace one
silent failure with another.
"""
from __future__ import annotations

import logging

from recall.calibration import (
    MIN_CALIBRATION_SAMPLES,
    MIN_SEPARABILITY,
    Calibration,
    from_samples,
    load_for,
    save,
    separability,
    separability_interval,
)


def _spread(centre: float, n: int, step: float = 0.001) -> list[float]:
    """n distinct values around `centre` — enough samples to clear the sufficiency check."""
    return [centre + i * step for i in range(n)]


# --- separability itself -------------------------------------------------------------------

def test_perfectly_separated_classes_score_one():
    assert separability([0.8, 0.9], [0.1, 0.2]) == 1.0


def test_perfectly_inverted_classes_score_zero():
    assert separability([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_identical_distributions_score_one_half():
    # Every pair is a tie, and a tie is half credit — the same convention the diagnostic used
    # when it measured the real corpus.
    assert separability([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_a_tie_counts_as_half_a_win():
    # one pair, exactly equal
    assert separability([0.5], [0.5]) == 0.5


def test_separability_is_unknown_without_both_classes():
    assert separability([0.8, 0.9], []) is None
    assert separability([], [0.1]) is None


def test_separability_counts_pairs_the_way_the_definition_says():
    # Hand-checkable: 3x3 = 9 pairs. Each of 3,4,5 beats 1 and 2 (6 wins) and loses to 6 (3
    # losses), so 6/9. Verified against the ad-hoc diagnostic on the real 500-question data,
    # where this function and the independent script agree to 4 decimal places (0.7533).
    assert separability([3, 4, 5], [1, 2, 6]) == 6 / 9


# --- the interval on separability ----------------------------------------------------------

def test_the_interval_reproduces_the_published_longmemeval_number():
    # FINDINGS section 10b quotes AUC 0.753 over 470 answerable / 30 unanswerable. Pinned here
    # because the document states the interval as evidence that no signal reaches the 0.90 bar:
    # if this estimator ever moves, that published claim moves with it.
    lo, hi = separability_interval(0.753, 470, 30)

    assert (round(lo, 3), round(hi, 3)) == (0.680, 0.826)
    assert hi < MIN_SEPARABILITY, "the whole interval must sit below the bar for the claim to hold"


def test_the_interval_is_narrower_than_the_crude_small_class_shortcut():
    # sqrt(A(1-A)/n_min) ignores the larger class and reports ~0.079 on these counts, which puts
    # 0.90 back inside the interval. Using it would understate the evidence and make an unusable
    # threshold look merely unproven — the trap the estimator's docstring warns about.
    crude_half = 1.96 * (0.753 * (1 - 0.753) / 30) ** 0.5
    lo, hi = separability_interval(0.753, 470, 30)

    assert (hi - lo) / 2 < crude_half
    assert 0.753 + crude_half > MIN_SEPARABILITY, "the crude bound would not have excluded 0.90"


def test_the_smaller_class_drives_the_width():
    # Labelling more unanswerable queries is the expensive half and the half that pays. Stated as
    # a test so "collect more samples" in the failure message points somewhere specific.
    balanced = separability_interval(0.80, 100, 100)
    lopsided = separability_interval(0.80, 1000, 10)

    assert (balanced[1] - balanced[0]) < (lopsided[1] - lopsided[0])


def test_perfect_separation_has_no_width():
    # AUC 1.0 has a true standard error of zero; the variance term can go fractionally negative
    # on rounding, and a NaN here would silently make `certified` unreachable.
    assert separability_interval(1.0, 50, 50) == (1.0, 1.0)


def test_the_interval_never_escapes_the_unit_range():
    lo, hi = separability_interval(0.99, 20, 20)

    assert 0.0 <= lo <= hi <= 1.0


# --- certification ------------------------------------------------------------------------

def test_a_separable_calibration_with_enough_samples_is_certified():
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES),
                       _spread(0.20, MIN_CALIBRATION_SAMPLES))

    assert cal.separability == 1.0
    assert cal.certified is True


def test_an_overlapping_calibration_is_refused_even_with_plenty_of_samples():
    # This is the LongMemEval failure: hundreds of samples, no separation.
    overlapping_a = _spread(0.70, 200, step=0.001)
    overlapping_u = _spread(0.69, 60, step=0.001)
    cal = from_samples("e", overlapping_a, overlapping_u)

    assert cal.separability < MIN_SEPARABILITY
    assert cal.certified is False
    assert "separab" in cal.certification_reason.lower()


def test_a_point_estimate_above_the_bar_is_refused_when_its_interval_is_not():
    # THE small-sample fail-open. 20 answerable and 20 unanswerable, of which one unanswerable
    # outscores every answerable: 380 of 400 pairs ordered correctly, AUC 0.95 — comfortably past
    # the 0.90 bar on the point estimate, and certified by the rule that read only that point.
    # Its 95% interval reaches down to 0.879, so these samples have NOT established 0.90, and a
    # calibration set this thin clears the bar by luck about as often as by separation.
    answerable = _spread(0.90, 20)
    unanswerable = _spread(0.20, 19) + [0.99]
    cal = from_samples("e", answerable, unanswerable)

    assert cal.separability == 0.95
    assert cal.separability > MIN_SEPARABILITY
    assert cal.separability_ci is not None and cal.separability_ci[0] < MIN_SEPARABILITY
    assert cal.certified is False


def test_the_thin_sample_refusal_is_not_confused_with_an_overlap_refusal():
    # Same verdict, opposite remedy: an overlapping corpus needs a different signal and will
    # never certify, while this one may certify on more labels. A message that read the same for
    # both would send the user to rebuild a pipeline that was fine.
    thin = from_samples("e", _spread(0.90, 20), _spread(0.20, 19) + [0.99])
    overlapping = from_samples("e", _spread(0.70, 200), _spread(0.69, 60))

    assert "interval" in thin.certification_reason
    assert "more" in thin.certification_reason.lower()
    assert "overlap" in overlapping.certification_reason.lower()
    assert "interval" not in overlapping.certification_reason


def test_the_certified_message_carries_the_interval_not_just_the_point():
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES),
                       _spread(0.20, MIN_CALIBRATION_SAMPLES))

    assert cal.certified is True
    assert "[1.000, 1.000]" in cal.certification_reason


def test_too_few_answerable_samples_is_refused():
    # FINDINGS section 6: the q05 floor cannot exclude anything below ~20 samples, so it
    # collapses onto the minimum and one bad retrieval moves the boundary.
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES - 1),
                       _spread(0.20, MIN_CALIBRATION_SAMPLES))

    assert cal.certified is False
    assert "sample" in cal.certification_reason.lower()


def test_too_few_unanswerable_samples_is_refused():
    # The q95 ceiling has the same problem as the q05 floor, from the other side.
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES),
                       _spread(0.20, MIN_CALIBRATION_SAMPLES - 1))

    assert cal.certified is False
    assert "sample" in cal.certification_reason.lower()


def test_a_one_class_calibration_is_unknown_not_certified():
    # There is nothing to separate, so the honest answer is "cannot judge" — which must not be
    # confused with "judged and passed".
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES), [])

    assert cal.separability is None
    assert cal.certified is None


def test_refusing_to_certify_warns_so_a_library_user_sees_it_without_the_cli(caplog):
    # Most callers never run `recall calibrate`. A diagnosis only the CLI prints is a diagnosis
    # a server deployment never receives.
    with caplog.at_level(logging.WARNING):
        from_samples("e", _spread(0.70, 200), _spread(0.69, 60))

    assert any("separab" in r.getMessage().lower() for r in caplog.records), \
        "not-certified calibration produced no log record"


def test_a_certified_calibration_does_not_warn(caplog):
    # The warning has to mean something. If it fires on healthy data too, it is noise and will
    # be filtered out by the first operator who sees it.
    with caplog.at_level(logging.WARNING):
        from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES),
                     _spread(0.20, MIN_CALIBRATION_SAMPLES))

    assert not [r for r in caplog.records if "separab" in r.getMessage().lower()]


# --- the diagnosis has to survive the artifact --------------------------------------------

def test_the_diagnosis_round_trips_through_the_saved_file(tmp_path):
    cal = from_samples("e", _spread(0.90, MIN_CALIBRATION_SAMPLES),
                       _spread(0.20, MIN_CALIBRATION_SAMPLES))
    p = save(cal, tmp_path / "c.json")

    back = load_for("e", p)
    assert back.separability == cal.separability
    assert back.certified is True
    # The interval is recomputed from the persisted (auc, n, n) rather than read back, so a file
    # written by an older version is judged by the same rule as a fresh one.
    assert back.separability_ci == cal.separability_ci


def test_the_saved_file_records_the_interval_the_verdict_was_taken_from(tmp_path):
    # A file showing `separability: 0.95` beside `certified: false` reads as a bug unless the
    # interval that produced the verdict is written down next to it.
    import json

    cal = from_samples("e", _spread(0.90, 20), _spread(0.20, 19) + [0.99])
    p = save(cal, tmp_path / "c.json")
    payload = json.loads(p.read_text(encoding="utf-8"))

    assert payload["separability"] == 0.95
    assert payload["certified"] is False
    assert payload["separability_ci"][0] < MIN_SEPARABILITY


def test_a_calibration_file_written_before_this_check_is_unknown_not_certified(tmp_path):
    # An older artifact carries no diagnosis. Treating "no diagnosis" as "certified" would let
    # exactly the silent failure this check exists to expose survive an upgrade.
    p = tmp_path / "old.json"
    p.write_text('{"embedder": "e", "threshold": 0.7, "scale": 0.05}', encoding="utf-8")

    back = load_for("e", p)
    assert back is not None
    assert back.certified is None


def test_certification_does_not_change_the_threshold_or_the_confidence_mapping():
    # The gate is a diagnosis. If it also moved the boundary, a deployment would silently get
    # different retrieval behaviour on upgrade, which is the failure mode being fixed.
    a, u = _spread(0.70, 200), _spread(0.69, 60)
    cal = from_samples("e", a, u)
    plain = Calibration(embedder="e", threshold=cal.threshold, scale=cal.scale)

    assert cal.certified is False
    assert cal.threshold == plain.threshold
    assert cal.confidence(0.8) == plain.confidence(0.8)
