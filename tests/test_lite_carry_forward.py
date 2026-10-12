"""Calibration carry-forward on a `LiteStore`: a memory write re-verifies the threshold, not refits it.

Same deterministic content-word embedder as `tests/test_lite_calibration.py`, and the same memos:
made-up subjects, so an added memo shares no words with the fit's gap questions unless a test puts
them there on purpose (`_shift_gap_questions`).

Invariants and the failure each one catches:
- F1 a write within the refit line is carried: strict search stays `trusted` on the fit's threshold,
  bound to the new corpus with its lineage, and `ensure_calibrated` stores no new fit and embeds
  nothing.
- F2 a write that lifts gap questions over the fixed threshold while the classes stay ordered (the
  shifted class separability cannot see) is NOT carried: `stale`, strict search refuses, and
  `ensure_calibrated` refits.
- F3 past `DEFAULT_MAX_CORPUS_DELTA` the carried calibration still serves (the bridge) but a refit
  is due, and the default `ensure_calibrated` performs it inline.
- F4 past `MAX_BRIDGE_CORPUS_DELTA` nothing is carried, however well it would re-verify.
- F5 with `background=True` a due refit runs in a background thread: the write returns the carried
  calibration at once, and the refit lands afterwards.
- F6 with `background=True` a carry that FAILED is never served while the refit runs: `stale` until
  the refit lands.
- F7 a fit whose corpus changed while it was measured binds to the snapshot it measured, never to
  the moved corpus, and the store then carries it forward to the corpus as it is now.
- F8 a carry is refused for an embedder other than the fit's: the threshold binds to one model.
- F9 each carry is recorded in the audit ledger, and `recall_calibration_status` reports it.
- F10 a background refit lands while writes keep arriving during every fit. Found by the
  pre-registered latency run (recall-lab `bc7df98`): a fit that DISCARDED itself when the corpus
  moved never landed under a write every 0.3 s against a 6 s fit, the bridge ran out at memo 49,
  and 36 of 60 writes left strict search refusing.

Red proof, 2026-10-12: each mutation alone against the carry-forward branch, the named test run on
the test host, failing in its intended assertion (JUnit XML), then the file restored byte for byte
(sha256 compared) and the same test green:
- M1 (F1) `resolve` returning `stale` on any corpus change (the rule before carry-forward): "a write
  the threshold survives turned strict search off".
- M2 (F2, C3, D4) `judge_inherited_threshold` certifying on separability alone (`and within_error`
  dropped): "a threshold that confirms the shifted gap questions was carried"; C3 "a changed corpus
  kept its old certificate"; D4 "a stale calibration was reported as certified: carried forward ...
  separability 1.000", which is the shifted class separability cannot see, in the wild.
- M3 (F3) `refit_due=False` in `carry_forward`: "a carry past the refit line was not marked due".
- M4 (F4) the `MAX_BRIDGE_CORPUS_DELTA` refusal disabled: "a threshold was carried across half the
  corpus".
- M5 (F5) the background branch of `ensure_calibrated` disabled (refit inline): "the write waited
  for the refit instead of returning the carried calibration".
- M6 (F6) the background branch returning the fit as certified when the carry failed: "a threshold
  that failed re-verification was offered while the refit ran".
- M7 (F7) the fingerprint comparison after measuring disabled in `auto_calibrate`: the outcome was
  `certified on 40 answerable and 40 unanswerable ...`, a fit over two corpora, stored. (Against
  the first form of F7, which expected the fit discarded; F7 and M7 were replaced by the snapshot
  fix, see M7b.)
- M8 (F8) the embedder identity check in `carry_forward` disabled: "a threshold fitted with one
  model was carried for another".
- M9 (F9) the `_record_carry` call removed: "a carry left no lineage in the audit ledger".
- Baseline (F7, F10): the pre-fix `recall/lite/calibration.py` and `recall/lite/store.py` from
  `c0e7e07d7`, which discarded a fit over a moving corpus: F7 "the memory changed while it was being
  calibrated; the fit was discarded"; F10 "the background refit never landed while writes kept
  arriving".
- M7b (F7) the fit bound to the live store's fingerprint instead of its snapshot's: "a fit was bound
  to a corpus it did not measure".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite import calibration as cal
from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall.types import Chunk
from tests.test_lite_calibration import ContentWordEmbedder, _shift_gap_questions, _store, _strict, _write_memos


class CountingEmbedder(ContentWordEmbedder):
    """Counts the texts it embeds, so a test can prove a step embedded nothing."""

    def __init__(self, name: str = "test-content-words") -> None:
        super().__init__(name)
        self.embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return super().embed(texts)


def _fits(store: LiteStore) -> int:
    with store._lock:  # noqa: SLF001  # the test reads the table the module writes
        return int(store._conn.execute("SELECT count(*) FROM calibrations").fetchone()[0])  # noqa: SLF001


def _fitted(tmp_path: Path, memos: int = 48) -> LiteStore:
    store = _store(tmp_path, memos)
    assert cal.auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    return store


def _add(store: LiteStore, tmp_path: Path, count: int, start: int = 48) -> None:
    _write_memos(tmp_path / "m", count, start=start)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")


def _fit_id(store: LiteStore) -> str:
    raw = store.latest_calibration_json()
    assert raw is not None
    return cal.artifact_from_json(raw).calibration_id


def test_a_small_write_is_carried_without_a_refit(tmp_path: Path) -> None:
    """F1."""
    store = _fitted(tmp_path)
    fit = cal.artifact_from_json(str(store.latest_calibration_json()))
    _add(store, tmp_path, 2)
    resolved = store.resolve_calibration()
    assert resolved.status is CalibrationStatus.CERTIFIED, "a write the threshold survives turned strict search off"
    carried = resolved.artifact
    assert carried is not None and carried.carry_forward is not None
    assert carried.threshold == fit.threshold and carried.scale == fit.scale
    assert carried.corpus_fingerprint == store.corpus_fingerprint() != fit.corpus_fingerprint
    assert carried.generation_id == store.lite_generation_id()
    assert carried.carry_forward["parent_calibration_id"] == fit.calibration_id
    assert carried.carry_forward["sources_added"] == 2
    assert _strict(store).trust_state == "trusted"
    embedder = CountingEmbedder()
    outcome = cal.ensure_calibrated(store, embedder)
    assert outcome.status is CalibrationStatus.CERTIFIED and "carried forward" in outcome.reason
    assert _fits(store) == 1, "a carried write was refitted"
    assert embedder.embedded == 0, "re-verifying the threshold embedded text"


def test_a_shifted_gap_class_is_not_carried(tmp_path: Path) -> None:
    """F2: the classes stay ordered, the fixed cut stops separating them."""
    store = _fitted(tmp_path)
    _shift_gap_questions(store, tmp_path / "m", 6)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    attempt = cal.carry_forward(store)
    assert attempt.artifact is None, "a threshold that confirms the shifted gap questions was carried"
    assert "only the boundary has moved" in attempt.reason, attempt.reason  # certified, but out of bounds
    assert attempt.provenance["false_confirm_rate"] > 0.10
    assert store.resolve_calibration().status is CalibrationStatus.STALE
    with pytest.raises(TrustRefusal) as refused:
        _strict(store)
    assert refused.value.code is TrustFailureCode.CALIBRATION_STALE
    assert cal.ensure_calibrated(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    assert _fits(store) == 2, "a failed carry was not refitted"


def test_past_the_refit_line_the_default_refits_inline(tmp_path: Path) -> None:
    """F3: 20 of 68 sources added is a delta of 0.29, past 0.25 and inside the 0.50 bridge."""
    store = _fitted(tmp_path)
    _add(store, tmp_path, 20)
    attempt = cal.carry_forward(store)
    assert attempt.artifact is not None, attempt.reason
    assert attempt.refit_due, "a carry past the refit line was not marked due"
    assert store.resolve_calibration().status is CalibrationStatus.CERTIFIED, "the bridge did not serve"
    outcome = cal.ensure_calibrated(store, ContentWordEmbedder())
    assert outcome.status is CalibrationStatus.CERTIFIED
    assert _fits(store) == 2, "a due refit was not performed"
    fresh = store.resolve_calibration().artifact
    assert fresh is not None and fresh.carry_forward is None, "after the refit the store still served a carry"


def test_past_the_bridge_nothing_is_carried(tmp_path: Path) -> None:
    """F4: 50 of 98 sources added is a delta of 0.51."""
    store = _fitted(tmp_path)
    _add(store, tmp_path, 50)
    attempt = cal.carry_forward(store)
    assert attempt.artifact is None, "a threshold was carried across half the corpus"
    assert attempt.provenance["corpus_delta"] > cal.MAX_BRIDGE_CORPUS_DELTA
    assert store.resolve_calibration().status is CalibrationStatus.STALE


def test_a_background_refit_does_not_block_the_write(tmp_path: Path) -> None:
    """F5."""
    store = _fitted(tmp_path)
    first = _fit_id(store)
    _add(store, tmp_path, 20)
    outcome = cal.ensure_calibrated(store, ContentWordEmbedder(), background=True)
    assert outcome.status is CalibrationStatus.CERTIFIED
    assert outcome.artifact is not None and outcome.artifact.carry_forward is not None, (
        "the write waited for the refit instead of returning the carried calibration"
    )
    assert "background" in outcome.reason
    landed = cal.wait_for_refit(store, timeout=120)
    assert landed is not None and landed.status is CalibrationStatus.CERTIFIED
    assert _fit_id(store) != first, "the background refit stored nothing"
    resolved = store.resolve_calibration()
    assert resolved.artifact is not None and resolved.artifact.carry_forward is None


def test_a_failed_carry_is_not_served_while_the_refit_runs(tmp_path: Path) -> None:
    """F6."""
    store = _fitted(tmp_path)
    _shift_gap_questions(store, tmp_path / "m", 6)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    outcome = cal.ensure_calibrated(store, ContentWordEmbedder(), background=True)
    assert outcome.status is CalibrationStatus.STALE and outcome.artifact is None, (
        "a threshold that failed re-verification was offered while the refit ran"
    )
    landed = cal.wait_for_refit(store, timeout=120)
    assert landed is not None
    assert store.resolve_calibration().status is landed.status


class _WritesDuringTheFit(ContentWordEmbedder):
    """Indexes one more memo the first time the fit embeds a question: a write racing a refit."""

    def __init__(self, store: LiteStore) -> None:
        super().__init__()
        self._store: LiteStore | None = store

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._store is not None:
            store, self._store = self._store, None
            late = "The late kalomiru flange is serviced by the vozapi team."
            store.upsert([Chunk(id="late-1", source="late.md", text=late, metadata={})], super().embed([late]))
        return super().embed(texts)


def test_a_fit_over_a_moving_corpus_binds_to_what_it_measured(tmp_path: Path) -> None:
    """F7."""
    store = _store(tmp_path, 48)
    measured = store.corpus_fingerprint()
    outcome = cal.auto_calibrate(store, _WritesDuringTheFit(store))
    assert outcome.status is CalibrationStatus.CERTIFIED, outcome.reason
    assert outcome.artifact is not None
    assert outcome.artifact.corpus_fingerprint == measured, "a fit was bound to a corpus it did not measure"
    assert store.corpus_fingerprint() != measured
    resolved = store.resolve_calibration()
    assert resolved.status is CalibrationStatus.CERTIFIED and resolved.artifact is not None
    assert resolved.artifact.carry_forward is not None, "the write during the fit was not carried"


class _KeepsWriting(ContentWordEmbedder):
    """Indexes one more memo every 50 texts it embeds: a server that keeps writing during refits."""

    def __init__(self, store: LiteStore) -> None:
        super().__init__()
        self._store = store
        self._calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._calls % 50 == 1:
            text = f"The kalomiru{self._calls} gasket is serviced by the vozapi{self._calls} team."
            chunk = Chunk(id=f"late-{self._calls}", source=f"late-{self._calls}.md", text=text, metadata={})
            self._store.upsert([chunk], super().embed([text]))
        return super().embed(texts)


def test_a_background_refit_lands_while_writes_keep_arriving(tmp_path: Path) -> None:
    """F10."""
    store = _fitted(tmp_path)
    first = _fit_id(store)
    _add(store, tmp_path, 20)
    cal.ensure_calibrated(store, _KeepsWriting(store), background=True)
    cal.wait_for_refit(store, timeout=120)
    assert _fit_id(store) != first, "the background refit never landed while writes kept arriving"
    assert store.resolve_calibration().status is CalibrationStatus.CERTIFIED


def test_a_carry_is_refused_for_another_embedder(tmp_path: Path) -> None:
    """F8: the content-word embedder under another name gives the same vectors, so only identity differs."""
    store = _fitted(tmp_path)
    _add(store, tmp_path, 2)
    attempt = cal.carry_forward(store, embedder=ContentWordEmbedder(name="another-model"))
    assert attempt.artifact is None, "a threshold fitted with one model was carried for another"
    cal.ensure_calibrated(store, ContentWordEmbedder(name="another-model"))
    raw = store.latest_calibration_json()
    assert raw is not None and cal.artifact_from_json(raw).embedder_identity["model"] == "another-model"


def test_a_carry_is_recorded_and_reported(tmp_path: Path) -> None:
    """F9."""
    store = _fitted(tmp_path)
    fit = _fit_id(store)
    _add(store, tmp_path, 2)
    outcome = cal.ensure_calibrated(store, ContentWordEmbedder())
    assert outcome.artifact is not None
    with store._lock:  # noqa: SLF001
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT payload FROM audit_events WHERE event_type = 'calibration_carried_forward'"
        ).fetchall()
    assert len(rows) == 1, "a carry left no lineage in the audit ledger"
    event = json.loads(rows[0][0])
    assert event["parent_calibration_id"] == fit and event["calibration_id"] == outcome.artifact.calibration_id
    report = cal.status_report(store)
    assert report["status"] == "certified"
    carried = report["carried_forward"]
    assert isinstance(carried, dict) and carried["sources_added"] == 2
    assert "carried forward" in str(report["message"])
