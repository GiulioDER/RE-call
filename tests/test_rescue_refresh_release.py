"""Refresh must rebuild rescue artifacts whose successful identity no longer applies."""

from dataclasses import replace
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.atomic import BuildReport
from recall.lite.refresh import Outcome, refresh
from tests.test_lite_atomic_rescue import _blank_rescue_environment, _corpus, _env
from tests.test_lite_calibration import DIM, ContentWordEmbedder


@pytest.mark.parametrize("change", ["root", "missing", "matrix", "calibration"])
def test_refresh_rebuilds_an_artifact_missing_from_its_current_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _blank_rescue_environment(monkeypatch)
    memos = _corpus(tmp_path)
    root = tmp_path / "atomic"
    with LiteStore(tmp_path / "memory.db", dim=DIM) as store:
        first = refresh(store, [memos], ContentWordEmbedder(), refusal_env=_env(root))
        assert first.atomic.startswith("built for "), first.to_dict()
        if change == "root":
            root = tmp_path / "other-atomic"
        elif change == "missing":
            next(root.rglob("manifest.json")).unlink()
        elif change == "calibration":
            resolution = store.resolve_calibration()
            changed = replace(resolution, artifact=replace(resolution.artifact, calibration_id="cal_" + "b" * 64))
            monkeypatch.setattr(store, "resolve_calibration", lambda: changed)
        else:
            from recall.atomic_rescue import clear_atomic_rescue_artifact_cache

            clear_atomic_rescue_artifact_cache()
            next(root.rglob("matrix.npy")).unlink()
        result = refresh(store, [memos], ContentWordEmbedder(), refusal_env=_env(root))
        assert result.outcome is Outcome.REFRESHED, "a stale rescue identity suppressed the rebuild"
        assert list(root.rglob("manifest.json")), "refresh left its configured artifact missing"


def test_a_discarded_build_does_not_suppress_the_next_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.lite import atomic

    _blank_rescue_environment(monkeypatch)
    memos = _corpus(tmp_path)
    root = tmp_path / "atomic"
    with LiteStore(tmp_path / "memory.db", dim=DIM) as store:
        with monkeypatch.context() as patch:
            patch.setattr(atomic, "ensure_rescue_artifact", lambda *a, **kw: BuildReport(None, "discarded"))
            refresh(store, [memos], ContentWordEmbedder(), refusal_env=_env(root))
        result = refresh(store, [memos], ContentWordEmbedder(), refusal_env=_env(root))
        assert result.outcome is Outcome.REFRESHED, "a discarded build was recorded as rescued"
        assert list(root.rglob("manifest.json")), "the retry built no artifact"


def test_auto_refresh_rebuilds_after_the_artifact_root_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from recall_mcp.lite_refresh import LiteAutoRefresher

    _blank_rescue_environment(monkeypatch)
    memos = _corpus(tmp_path)
    with LiteStore(tmp_path / "memory.db", dim=DIM) as store:
        refresh(store, [memos], ContentWordEmbedder(), refusal_env=_env(tmp_path / "first"))
        root = tmp_path / "second"
        auto = LiteAutoRefresher(store, ContentWordEmbedder(), [memos], wait_s=30, env=_env(root))
        try:
            auto.before_retrieval()
        finally:
            auto.stop()
        assert list(root.rglob("manifest.json")), "automatic refresh retained the old rescue root"
