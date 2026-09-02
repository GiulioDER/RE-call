"""Regression tests for the audited calibration fixes F-02, F-03, F-05 and F-11.

Each test here fails against the module as it stood before its fix:

- F-02: the q95 unanswerable ceiling used a lower-tail index, so at n <= 20 it landed on the
  sample MAXIMUM and one outlier defined the boundary.
- F-03: the negatives-only branch placed the threshold exactly ON the ceiling; serving confirms
  at score >= threshold (recall.trust), so every sample sitting at the ceiling was confirmed.
- F-05: `save` mapped a directory to ``dir/calibration.json`` and normalized the path while the
  load side used the raw string, so a RECALL_CALIBRATION directory (or ``~`` path) wrote one
  file and read another.
- F-11: `save_for_profile` wrote the file twice (the second write adding the fingerprint), and
  `save` truncated in place, so a crash mid-save left a refused or truncated file.
"""
from __future__ import annotations

import json

from recall import calibration as calibration_module
from recall.calibration import (
    ANSWERABLE_FLOOR_Q,
    PROFILE_FINGERPRINT_KEY,
    UNANSWERABLE_CEILING_Q,
    Calibration,
    _quantile,
    best_threshold,
    load_for,
    save,
    save_for_profile,
)
from recall.calibration_v2 import threshold_error_rates
from recall.embeddings import EmbeddingProfile

PROFILE = EmbeddingProfile(
    profile_id="bge-small-symmetric-v1",
    model_name="BAAI/bge-small-en-v1.5",
    artifact_digest="a" * 64,
    dimension=384,
    query_mode="embed",
    passage_mode="embed",
)


def _calibration(embedder: str = PROFILE.profile_id) -> Calibration:
    return Calibration(
        embedder=embedder, threshold=0.42, scale=0.05, separability=0.99,
        n_answerable=30, n_unanswerable=30,
    )


# ---------------------------------------------------------------------------- F-02


def test_q95_ceiling_ignores_a_single_unanswerable_outlier_at_min_samples():
    """At exactly MIN_CALIBRATION_SAMPLES the ceiling must not be the sample maximum.

    Nineteen unanswerable samples at 0.30 and one outlier at 0.90: the ceiling is 0.30, so the
    threshold lands near the midpoint of (q05 answerable, 0.30). Before the fix the lower-tail
    index put the ceiling ON the 0.90 outlier and the threshold at 0.805.
    """
    answerable = [0.70 + 0.01 * i for i in range(20)]
    unanswerable = [0.30] * 19 + [0.90]
    thr = best_threshold(answerable, unanswerable)
    assert abs(thr - (0.71 + 0.30) / 2) <= 0.005, thr


def test_the_threshold_does_not_increase_as_the_unanswerable_outlier_rises():
    answerable = [0.70 + 0.01 * i for i in range(20)]
    base = best_threshold(answerable, [0.30] * 19 + [0.90])
    with_higher_outlier = best_threshold(answerable, [0.30] * 19 + [0.99])
    assert with_higher_outlier <= base


def test_upper_quantile_mirrors_the_floor_exclusion_count_for_n_5_to_100():
    """The ceiling excludes exactly as many TOP samples as the floor excludes bottom samples.

    Imported inside the test on purpose: the helper does not exist before the fix, and the
    import failure is the red state.
    """
    from recall.calibration import _upper_quantile

    for n in range(5, 101):
        values = [float(i) for i in range(n)]
        floor_excluded = values.index(_quantile(values, ANSWERABLE_FLOOR_Q))
        ceiling_excluded = (n - 1) - values.index(
            _upper_quantile(values, UNANSWERABLE_CEILING_Q)
        )
        assert floor_excluded == ceiling_excluded, (n, floor_excluded, ceiling_excluded)
        assert ceiling_excluded >= 1 if n >= 20 else ceiling_excluded == 0


# ---------------------------------------------------------------------------- F-03


def test_negatives_only_threshold_sits_strictly_above_the_ceiling():
    """Serving confirms at score >= threshold, so a threshold equal to the ceiling confirms
    every sample sitting exactly there. Before the fix this returned 0.5 and the false confirm
    rate was 1.0."""
    thr = best_threshold([], [0.5] * 20)
    assert thr > 0.5
    rates = threshold_error_rates([], [0.5] * 20, thr)
    assert rates["false_confirm_rate"] == 0.0
    # The step above the ceiling is clamped to the top of the cosine range.
    assert best_threshold([], [1.0] * 20) <= 1.0


# ---------------------------------------------------------------------------- F-05


def test_env_var_pointing_at_a_directory_round_trips(tmp_path, monkeypatch):
    """Writer and reader must resolve the SAME file. Before the fix, `save` wrote
    ``dir/calibration.json`` while `load_for` tried to read the directory itself, permanently
    serving the uncalibrated fallback."""
    monkeypatch.setenv("RECALL_CALIBRATION", str(tmp_path))
    written = save(Calibration(embedder="bge", threshold=0.7, scale=0.03))
    assert written == (tmp_path / "calibration.json").resolve()
    loaded = load_for("bge")
    assert loaded is not None
    assert loaded.threshold == 0.7


def test_env_var_tilde_path_round_trips(tmp_path, monkeypatch):
    """A ``~`` path expands identically on the write side and the read side."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("RECALL_CALIBRATION", "~/cal.json")
    written = save(Calibration(embedder="bge", threshold=0.6, scale=0.04))
    assert written == (tmp_path / "cal.json").resolve()
    loaded = load_for("bge")
    assert loaded is not None
    assert loaded.threshold == 0.6


# ---------------------------------------------------------------------------- F-11


def test_save_for_profile_is_exactly_one_write_and_it_carries_the_fingerprint(
    tmp_path, monkeypatch
):
    """The complete payload, fingerprint included, leaves in a SINGLE atomic write.

    Before the fix `save_for_profile` wrote the file twice through `Path.write_text` (the
    second write adding the fingerprint), so a crash between the writes left a fingerprint-less
    file `load_for_profile` permanently refuses. The spy below counts payload writes through
    BOTH mechanisms, so against the pre-fix module it records two writes, the first of them
    fingerprint-less, and the single-write assertion goes red behaviorally.
    """
    from pathlib import Path

    written_payloads = []
    real_writer = getattr(calibration_module, "atomic_write_bytes", None)

    def spy_atomic(path, data):
        written_payloads.append(json.loads(data.decode("utf-8")))
        real_writer(path, data)

    if real_writer is not None:
        monkeypatch.setattr(calibration_module, "atomic_write_bytes", spy_atomic)

    real_write_text = Path.write_text

    def spy_write_text(self, content, *args, **kwargs):
        written_payloads.append(json.loads(content))
        return real_write_text(self, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    path = tmp_path / "calibration.json"
    save_for_profile(_calibration(), PROFILE, path)
    assert len(written_payloads) == 1, "no intermediate fingerprint-less state may reach disk"
    assert written_payloads[0][PROFILE_FINGERPRINT_KEY] == PROFILE.fingerprint()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk[PROFILE_FINGERPRINT_KEY] == PROFILE.fingerprint()
    assert on_disk["threshold"] == 0.42


def test_save_goes_through_the_atomic_writer(tmp_path, monkeypatch):
    """`save` alone must not truncate the target in place either."""
    real_writer = calibration_module.atomic_write_bytes
    calls = []

    def spy(path, data):
        calls.append(path)
        real_writer(path, data)

    monkeypatch.setattr(calibration_module, "atomic_write_bytes", spy)
    save(Calibration(embedder="e", threshold=0.5, scale=0.05), tmp_path / "c.json")
    assert len(calls) == 1
