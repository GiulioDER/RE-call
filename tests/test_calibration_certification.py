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
