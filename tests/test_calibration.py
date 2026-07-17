from __future__ import annotations

from recall.calibration import (
    DEFAULT_SCALE,
    Calibration,
    best_threshold,
    from_samples,
    load_for,
    save,
)


def test_confidence_is_half_at_threshold_and_monotone():
    cal = Calibration(embedder="e", threshold=0.6, scale=0.05)
    assert abs(cal.confidence(0.6) - 0.5) < 1e-9
    assert cal.confidence(0.4) < cal.confidence(0.5) < cal.confidence(0.7) < cal.confidence(0.9)
    assert cal.confidence(-1.0) < 0.01
    assert cal.confidence(1.0) > 0.99


def test_best_threshold_separates_clean_distributions():
    thr = best_threshold(answerable=[0.70, 0.75, 0.90], unanswerable=[0.50, 0.55, 0.64])
    assert 0.64 < thr <= 0.70


def test_from_samples_builds_separating_calibration():
    cal = from_samples("bge", answerable=[0.70, 0.75, 0.90], unanswerable=[0.50, 0.55, 0.64])
    assert cal.embedder == "bge"
    assert 0.64 < cal.threshold <= 0.70
    assert cal.scale >= 0.01
    # answerable cosines map above 0.5 confidence, unanswerable below
    assert cal.confidence(0.75) > 0.5 > cal.confidence(0.55)


def test_from_samples_small_samples_fall_back_to_default_scale():
    cal = from_samples("e", answerable=[0.9], unanswerable=[0.1])
    assert cal.scale == DEFAULT_SCALE


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    cal = Calibration(embedder="bge", threshold=0.7, scale=0.03)
    save(cal, path)
    assert load_for("bge", path) == cal


def test_load_for_wrong_embedder_returns_none(tmp_path):
    path = tmp_path / "calibration.json"
    save(Calibration(embedder="bge", threshold=0.7), path)
    assert load_for("voyage-3", path) is None  # never apply another embedder's threshold


def test_load_for_missing_file_returns_none(tmp_path):
    assert load_for("bge", tmp_path / "nope.json") is None


def test_load_for_env_var_path(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    save(Calibration(embedder="bge", threshold=0.7), path)
    monkeypatch.setenv("RECALL_CALIBRATION", str(path))
    assert load_for("bge") is not None


def test_eval_calibrate_reexports_best_threshold():
    from recall.eval.calibrate import best_threshold as reexported

    assert reexported is best_threshold


def test_load_for_corrupt_file_returns_none(tmp_path, capsys):
    path = tmp_path / "calibration.json"
    path.write_text('{"embedder": "bge", "thresh', encoding="utf-8")  # truncated write
    assert load_for("bge", path) is None


def test_load_for_missing_key_returns_none(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"embedder": "bge", "threshold": 0.7}', encoding="utf-8")  # no scale
    assert load_for("bge", path) is None


def test_load_for_non_dict_json_returns_none(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_for("bge", path) is None


def test_load_for_nan_threshold_returns_none(tmp_path):
    # NaN threshold would make every `score < threshold` False -> abstention silently dead
    path = tmp_path / "calibration.json"
    path.write_text('{"embedder": "bge", "threshold": NaN, "scale": 0.05}', encoding="utf-8")
    assert load_for("bge", path) is None


def test_load_for_nonpositive_scale_returns_none(tmp_path):
    path = tmp_path / "calibration.json"
    for bad in ("0", "-0.05"):
        path.write_text(
            '{"embedder": "bge", "threshold": 0.7, "scale": %s}' % bad, encoding="utf-8"
        )
        assert load_for("bge", path) is None


def test_confidence_tiny_scale_does_not_overflow():
    cal = Calibration(embedder="e", threshold=0.7, scale=0.002)
    assert cal.confidence(-1.0) == 0.0 or cal.confidence(-1.0) < 1e-9  # saturates, no OverflowError
    assert cal.confidence(1.0) > 0.999


def test_best_threshold_never_rounds_above_its_candidate():
    # nearest-rounding could lift the threshold past the chosen boundary cosine, flipping a
    # boundary answerable sample to low_confidence
    thr = best_threshold(answerable=[0.6125001, 0.9], unanswerable=[0.1, 0.2])
    assert thr <= 0.6125001
