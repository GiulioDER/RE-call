"""DAT-001: calibration artifacts cannot escape loading through sample-count overflow.

Proof: run these unchanged nodes on a214dab75 before bounding load_for._count.
The three malformed cases must fail the explicit no-overflow assertion;
ordinary and maximum supported counts are unchanged-diagnosis controls.
"""
import json
import sys

import pytest

from recall.calibration import load_for


@pytest.mark.parametrize('counts', [(10**400, 20), (20, 10**400), (10**200, 10**200)])
def test_oversized_sample_counts_degrade_to_unknown(tmp_path, counts):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(dict(embedder='e', threshold=0.5, scale=0.05,
                                   separability=0.95, n_answerable=counts[0],
                                   n_unanswerable=counts[1])), encoding='utf-8')
    error = None
    try:
        calibration = load_for('e', path)
    except OverflowError as exc:
        error = exc
    assert error is None, f'malformed sample count escaped load_for: {error}'
    assert calibration is not None
    for field, count in zip(('n_answerable', 'n_unanswerable'), counts):
        assert getattr(calibration, field) == (None if count > sys.maxsize else count)
    assert calibration.certified is None
    assert calibration.separability_ci is None
    assert 'not judged' in calibration.certification_reason
    assert calibration.threshold == 0.5 and calibration.scale == 0.05
    assert load_for('e', path) == calibration


@pytest.mark.parametrize('count', [20, sys.maxsize])
def test_valid_sample_counts_keep_their_diagnosis(tmp_path, count):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(dict(embedder='e', threshold=0.5, scale=0.05,
                                   separability=1.0, n_answerable=count,
                                   n_unanswerable=count)), encoding='utf-8')
    calibration = load_for('e', path)
    assert calibration is not None
    assert calibration.n_answerable == calibration.n_unanswerable == count
    assert calibration.certified is True
