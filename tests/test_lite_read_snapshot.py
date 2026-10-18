"""Strict reads keep one calibration and corpus view across another handle's writes.

Baseline: first PR896 repair tree 2ffe5d658216691bbc1cb1f7c067403944138298.
Both test_strict_search_keeps_one_view nodes target the existing trusted_search
snapshot integration and fail "strict search mixed calibration revisions".
test_iteration_uses_the_search_snapshot targets iter_chunks within that consumer
and fails "iteration changed inside strict search". Scheduling hooks perform
real writes or calibrations and preserve the actual binding and resolution.
These nodes must fail in those assertions before snapshot support is added,
then pass unchanged. No missing snapshot symbol is used as a red proof.
"""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationResolution, CalibrationStatus
from recall.lite import LiteStore
from recall.lite import calibration as cal
from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall.types import Chunk
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _store, _strict


@pytest.mark.parametrize("stage", ["after_binding", "after_resolution"])
def test_strict_search_keeps_one_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    with closing(_store(tmp_path, 48)) as store, closing(LiteStore(store.path, dim=DIM)) as writer:
        outcome = cal.auto_calibrate(store, ContentWordEmbedder(), now=datetime(2026, 10, 7, tzinfo=UTC))
        assert outcome.artifact is not None
        selected = outcome.artifact.calibration_id
        baseline = _strict(store)
        changed = False

        def change() -> None:
            nonlocal changed
            if changed:
                return
            if stage == "after_resolution":
                chunks = list(writer.iter_chunks())
                writer.upsert(chunks, [[0.0] * DIM for _ in chunks])
            cal.auto_calibrate(writer, ContentWordEmbedder(name="another-model"), now=datetime(2026, 10, 8, tzinfo=UTC))
            changed = True

        if stage == "after_binding":
            original_binding = store.generation_binding

            def binding() -> dict[str, str]:
                value = original_binding()
                change()
                return value

            monkeypatch.setattr(store, "generation_binding", binding)
        else:
            original_resolution = store.resolve_calibration

            def resolution() -> CalibrationResolution:
                value = original_resolution()
                change()
                return value

            monkeypatch.setattr(store, "resolve_calibration", resolution)
        result = _strict(store)
        assert changed
        assert result.trust_state == "trusted", "strict search mixed calibration revisions"
        assert result.calibration_id == selected, "strict search mixed calibration revisions"
        assert [(hit.chunk.id, hit.verdict) for hit in result.hits] == [(hit.chunk.id, hit.verdict) for hit in baseline.hits], (
            "strict search mixed calibration revisions"
        )
        assert store.generation_binding()["embedder_model"] == "another-model"
        assert store.latest_calibration_json() != cal.artifact_to_json(outcome.artifact)


def test_iteration_uses_the_search_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with closing(_store(tmp_path, 48)) as store, closing(LiteStore(store.path, dim=DIM)) as writer:
        assert cal.auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
        original_binding = store.generation_binding
        original_resolution = store.resolve_calibration
        observed: list[str] = []
        changed = False

        def binding() -> dict[str, str]:
            nonlocal changed
            value = original_binding()
            if not changed:
                writer.upsert([Chunk(id="new", source="/m/new.md", text="new rotor maintenance")], [[0.0] * DIM])
                changed = True
            return value

        def resolution() -> CalibrationResolution:
            value = original_resolution()
            observed.extend(chunk.id for chunk in store.iter_chunks())
            return value

        monkeypatch.setattr(store, "generation_binding", binding)
        monkeypatch.setattr(store, "resolve_calibration", resolution)
        try:
            _strict(store)
        except TrustRefusal as exc:
            assert exc.code is TrustFailureCode.CALIBRATION_STALE
        assert observed and changed
        assert "new" not in observed, "iteration changed inside strict search"
        assert "new" in {chunk.id for chunk in store.iter_chunks()}
