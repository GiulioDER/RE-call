"""Optional reporting cannot turn a successfully stored fit into a failed calibration.

Red proof against e9ce4a5ac on a Linux test host: collector and report-persistence failures
raise 'optional report blocked the saved calibration' after resolve certifies the fit.
The newly fitted certificate and its current resolution are positive controls.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.lite import calibration
from tests.test_lite_calibration import ContentWordEmbedder, _store


@pytest.mark.parametrize("stage", ["collector", "persistence"])
def test_optional_coverage_failure_does_not_block_saved_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         stage: str) -> None:
    store = _store(tmp_path, 48)

    def failed_report(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("offline diagnostic transport failed")

    if stage == "collector":
        monkeypatch.setattr(calibration, "near_miss_coverage", failed_report)
    else:
        monkeypatch.setattr(store, "save_near_miss_coverage", failed_report)
    try:
        result = calibration.auto_calibrate(store, ContentWordEmbedder())
    except RuntimeError as exc:
        assert store.resolve_calibration().status is CalibrationStatus.CERTIFIED
        raise AssertionError("optional report blocked the saved calibration") from exc
    assert result.status is CalibrationStatus.CERTIFIED and result.artifact is not None
    assert "unavailable" in result.reason and "provisional" in result.reason
    assert store.resolve_calibration().status is CalibrationStatus.CERTIFIED
    report = calibration.status_report(store)
    assert report["certified"] is True
    assert "unavailable" in str(report["message"]) and "provisional" in str(report["message"])
