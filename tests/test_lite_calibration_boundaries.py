"""Calibration identity and publication survive public writes.

Baseline: PR896 original head fc9c6617e11c66718d5b33ccef3404e5ae11f799.
The four test_replaced_vectors_require_a_refit nodes target _load's fingerprint
and fail "changed vectors kept the measured identity" before the repair.
The three test_a_concurrent_write_never_relabels_a_fit nodes target auto_calibrate's
binding. They were written against PR 896's refusal ("corpus changed during calibration");
since the restack onto PR 911 a fit measures one frozen `calibration_snapshot()` and binds to
it, because refusing starved a background refit under steady writes. The invariant is the
same: a fit is never labelled with a corpus it did not measure. Red proof (restack, the testbench,
2026-10-13): `fingerprint = store.corpus_fingerprint()` in place of `snapshot.fingerprint` in
`auto_calibrate` fails the after_scores and during_embedding nodes with "the fit claims the corpus
written during it"; the before_save node writes after that line runs, so this mutation cannot
reach it. Scheduling hooks call the real scorer, writer and embedder.
The two test_backdated_fit_keeps_selected_model nodes target save_calibration's
metadata selection and fail "selected threshold accepted another model" through
the strict consumer. Each exact node must then pass with the source repaired.
"""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.lite import LiteStore
from recall.lite import calibration as cal
from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall.types import Chunk
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _store, _strict


@pytest.mark.parametrize("method", ["upsert", "replace_sources"])
@pytest.mark.parametrize("other_writer", [False, True])
def test_replaced_vectors_require_a_refit(tmp_path: Path, method: str, other_writer: bool) -> None:
    with closing(_store(tmp_path, 48)) as store:
        assert cal.auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
        before = store.corpus_fingerprint()
        chunks = list(store.iter_chunks())
        vectors = [[0.0] * DIM for _ in chunks]
        writer = LiteStore(store.path, dim=DIM) if other_writer else store
        try:
            if method == "upsert":
                writer.upsert(chunks, vectors)
            else:
                writer.replace_sources(sorted({c.source for c in chunks}), chunks, vectors)
        finally:
            if other_writer:
                writer.close()
        assert store.corpus_fingerprint() != before, "changed vectors kept the measured identity"
        assert store.resolve_calibration().status is CalibrationStatus.STALE
        with pytest.raises(TrustRefusal) as refused:
            _strict(store)
        assert refused.value.code is TrustFailureCode.CALIBRATION_STALE
        assert cal.ensure_calibrated(store, ContentWordEmbedder()).status is CalibrationStatus.UNCERTIFIED


@pytest.mark.parametrize("stage", ["after_scores", "before_save", "during_embedding"])
def test_a_concurrent_write_never_relabels_a_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    with closing(_store(tmp_path, 48)) as store, closing(LiteStore(store.path, dim=DIM)) as writer:
        embedder = ContentWordEmbedder()
        measured = store.corpus_fingerprint()
        written = False

        def change() -> None:
            nonlocal written
            if not written:
                writer.upsert([Chunk(id="new", source="/m/new.md", text="new rotor maintenance")], [[0.0] * DIM])
                written = True

        if stage == "after_scores":
            original_measure = cal.measure_top_cosines

            def measure(*args: Any, **kwargs: Any) -> tuple[list[float], list[float]]:
                samples = original_measure(*args, **kwargs)
                change()
                return samples

            monkeypatch.setattr(cal, "measure_top_cosines", measure)
        elif stage == "before_save":
            original_save = store.save_calibration

            def save(*args: Any, **kwargs: Any) -> None:
                change()
                original_save(*args, **kwargs)

            monkeypatch.setattr(store, "save_calibration", save)
        else:
            original_embed = embedder.embed

            def embed(texts: list[str]) -> list[list[float]]:
                vectors = original_embed(texts)
                change()
                return vectors

            monkeypatch.setattr(embedder, "embed", embed)
        cal.auto_calibrate(store, embedder)
        assert written
        stored = store.latest_calibration_json()
        assert stored is not None, "a fit measured on one corpus was discarded"
        fitted = cal.artifact_from_json(stored)
        assert fitted.corpus_fingerprint == measured, "the fit claims the corpus written during it"
        assert store.corpus_fingerprint() != measured
        # The write lands after the fit's corpus: it resolves as a changed corpus (carried or stale),
        # never as the corpus the fit measured.
        assert store.resolve_calibration().status is not CalibrationStatus.MISSING


@pytest.mark.parametrize("other_writer", [False, True])
def test_backdated_fit_keeps_selected_model(tmp_path: Path, other_writer: bool) -> None:
    with closing(_store(tmp_path, 48)) as store:
        original = ContentWordEmbedder()
        cal.auto_calibrate(store, original, now=datetime(2026, 10, 7, tzinfo=UTC))
        selected = store.latest_calibration_json()
        writer = LiteStore(store.path, dim=DIM) if other_writer else store
        other = ContentWordEmbedder(name="another-model")
        try:
            cal.auto_calibrate(writer, other, now=datetime(2026, 10, 6, tzinfo=UTC))
        finally:
            if other_writer:
                writer.close()
        assert store.latest_calibration_json() == selected
        try:
            _strict(store, other)
        except TrustRefusal as exc:
            assert exc.code is TrustFailureCode.LINEAGE_MISMATCH
        else:
            pytest.fail("selected threshold accepted another model")
        assert store.generation_binding()["embedder_model"] == original.name
        assert _strict(store, original).trust_state == "trusted"
