"""Automatic calibration on a `LiteStore`, and strict trust through `trusted_search`.

No database server and no model: the embedder below encodes a text's CONTENT words (stop words
dropped), so a question generated from a memo shares words with it and an off-topic question
shares none. That makes certification deterministic here; the real local model (bge-small) was
checked separately on another host on 2026-10-07: 15 real memos (96 chunks) certified at separability
0.989 and 40 (243 chunks) at 0.993, while 5 memos (15 chunks) stayed under the floor.

Invariants and the failure each one catches:
- C1 under the floor (fewer chunks than a query set needs) nothing is stored, the store resolves
  `missing`, and strict search refuses with CALIBRATION_MISSING rather than answering.
- C2 with enough text the store certifies by itself, and strict search answers `trusted`.
- C3 a changed corpus whose fitted threshold no longer re-verifies resolves `stale`, and strict
  search refuses until `ensure_calibrated` refits. (Before carry-forward, ANY change did this; a
  change the threshold survives is now carried, see `tests/test_lite_carry_forward.py`.)
- C4 searching with a different embedder than the one calibrated is LINEAGE_MISMATCH.
- C5 a stored calibration whose bytes were altered fails closed (DEPENDENCY_UNAVAILABLE): the
  threshold is never read from an artifact whose checksum does not verify.
- C6 an unchanged corpus is not refitted.
- C7 an edit to a stored calibration is noticed even after its resolution was cached.
- C8 storing a calibration goes through the store's one write path, so an error after which SQLite
  already rolled back reaches the caller instead of "cannot rollback".

Red proof, 2026-10-07, each mutation alone against `recall/lite/calibration.py` or
`recall/lite/store.py`, failing in the named assertion (JUnit XML), then restored byte for byte:
- K1 (C1) `resolve` reading an empty store as certified: `CERTIFIED is MISSING` failed.
- K2 (C2) `auto_calibrate` never certifying: the outcome was uncertified.
- K3 (C3) the corpus fingerprint unchecked: "a changed corpus kept its old certificate".
  Re-proved 2026-10-12 for the carry-forward form of C3, with the mutation that lets `resolve` serve
  `carry_forward`'s artifact on separability alone (M2 in `tests/test_lite_carry_forward.py`):
  "a changed corpus kept its old certificate".
- K4 (C4) `generation_binding` without the embedder: `DID NOT RAISE TrustRefusal`.
- K5 (C5) `artifact_from_json` without `verify_checksum`: `DID NOT RAISE TrustRefusal`.
- K6 (C6) `ensure_calibrated` refitting a certified corpus: "an unchanged corpus was calibrated twice".
- P6 (C7) the resolution cache keyed without the stored bytes: `DID NOT RAISE TrustRefusal`.
- K8 (C8) baseline, not a mutation: `LiteStore.save_calibration` at `b3eac0a5`, with its own
  try/ROLLBACK, failed `test_storing_a_calibration_surfaces_the_real_error` in "the rollback masked
  the real error".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import auto_calibrate, ensure_calibrated
from recall.lite.store import ENGLISH_STOP_WORDS
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustRefusal

DIM = 2048
_WORD = re.compile(r"[a-z0-9]+")


class ContentWordEmbedder:
    """Content words hashed into a wide space: shared subject words give a high cosine."""

    dim = DIM

    def __init__(self, name: str = "test-content-words") -> None:
        self.name = name

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * DIM
            for word in _WORD.findall(text.lower()):
                if word not in ENGLISH_STOP_WORDS and len(word) > 2:
                    vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec] if any(vec) else [1.0] + [0.0] * (DIM - 1))
        return out


_SYLLABLES = ("ka", "lo", "mi", "ru", "te", "vo", "za", "ne", "pi", "su")
_NOUNS = ("flange", "valve", "rotor", "gasket", "piston", "bearing", "spindle", "nozzle")


def _word(i: int, salt: int) -> str:
    """A made-up word unique to memo `i`: syllables chosen by its index and a salt."""
    n = i * 7 + salt * 131
    return "".join(_SYLLABLES[(n // 10**k) % 10] for k in range(3)) + _SYLLABLES[(i + salt) % 10]


def _write_memos(root: Path, count: int, *, start: int = 0) -> None:
    root.mkdir(exist_ok=True)
    for i in range(start, start + count):
        subject, part, crew = _word(i, 1), _NOUNS[i % len(_NOUNS)], _word(i, 2)
        (root / f"memo-{i:03d}.md").write_text(
            f"# The {subject} {part}\n\nThe {subject} {part} is serviced by the {crew} team every spring.\n",
            encoding="utf-8",
        )


def _shift_gap_questions(store: LiteStore, root: Path, count: int) -> None:
    """Write `count` memos that lift the fit's first `count` gap questions just over its threshold.

    Each memo holds one gap question's words plus made-up filler, about 2.5 filler words per content
    word, so its cosine with that question lands near 0.53: above the fitted threshold (about 0.45
    with this embedder) and below every answerable score (0.62 and up). The classes stay perfectly
    ORDERED, so separability still certifies, while the fixed threshold now confirms those gap
    questions: the shifted class that only the per-class error check can see.
    """
    from recall.lite.calibration import artifact_from_json

    raw = store.latest_calibration_json()
    assert raw is not None
    evidence = store.calibration_evidence(artifact_from_json(raw).calibration_id)
    assert evidence is not None
    gap = [str(q["query"]) for q in json.loads(evidence[0]) if not q["answerable"]][:count]
    for n, question in enumerate(gap):
        words = {w for w in _WORD.findall(question.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2}
        filler = " ".join(f"fill{n}x{k}" for k in range(math.ceil(2.5 * len(words))))
        (root / f"gap-{n:03d}.md").write_text(f"{question} {filler}\n", encoding="utf-8")


def _store(tmp_path: Path, memos: int) -> LiteStore:
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    _write_memos(tmp_path / "m", memos)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    return store


def _strict(store: LiteStore, embedder: ContentWordEmbedder | None = None):  # noqa: ANN202
    return trusted_search(store, embedder or ContentWordEmbedder(), f"who services the {_word(7, 1)} {_NOUNS[7]}", k=5, env={})


def test_under_the_floor_nothing_is_stored_and_strict_search_refuses(tmp_path: Path) -> None:
    store = _store(tmp_path, 6)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.status is CalibrationStatus.MISSING
    assert "Add a few more memos" in outcome.reason
    assert store.latest_calibration_json() is None
    assert store.resolve_calibration().status is CalibrationStatus.MISSING
    with pytest.raises(TrustRefusal) as refused:
        _strict(store)
    assert refused.value.code is TrustFailureCode.CALIBRATION_MISSING, "strict search answered without a calibration"


def test_enough_text_certifies_and_strict_search_answers(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.status is CalibrationStatus.CERTIFIED, outcome.reason
    resolved = store.resolve_calibration()
    assert resolved.status is CalibrationStatus.CERTIFIED and resolved.artifact is not None
    result = _strict(store)
    assert result.trust_state == "trusted", "a certified lite store did not serve trusted answers"
    assert result.calibrated is True
    assert result.hits[0].provenance.file == "memo-007.md"


def test_a_changed_corpus_is_stale_until_refitted(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    auto_calibrate(store, ContentWordEmbedder())
    _shift_gap_questions(store, tmp_path / "m", 6)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    assert store.resolve_calibration().status is CalibrationStatus.STALE, "a changed corpus kept its old certificate"
    with pytest.raises(TrustRefusal) as refused:
        _strict(store)
    assert refused.value.code is TrustFailureCode.CALIBRATION_STALE
    assert ensure_calibrated(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    assert _strict(store).trust_state == "trusted"


def test_another_embedder_is_a_lineage_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    auto_calibrate(store, ContentWordEmbedder())
    with pytest.raises(TrustRefusal) as refused:
        _strict(store, ContentWordEmbedder(name="another-model"))
    assert refused.value.code is TrustFailureCode.LINEAGE_MISMATCH, "a threshold fitted to one model was applied to another"


def test_an_altered_calibration_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    auto_calibrate(store, ContentWordEmbedder())
    with store._lock:  # noqa: SLF001  # the test plays the part of someone editing the file
        raw = store._conn.execute("SELECT payload FROM calibrations").fetchone()[0]  # noqa: SLF001
        tampered = re.sub(r'"threshold":-?[0-9.eE+-]+', '"threshold":-0.99', raw, count=1)
        assert tampered != raw
        store._conn.execute("UPDATE calibrations SET payload = ?", (tampered,))  # noqa: SLF001
    with pytest.raises(TrustRefusal) as refused:
        _strict(store)
    assert refused.value.code is TrustFailureCode.DEPENDENCY_UNAVAILABLE, "an altered threshold was trusted"


def test_an_unchanged_corpus_is_not_refitted(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    ensure_calibrated(store, ContentWordEmbedder())
    ensure_calibrated(store, ContentWordEmbedder())
    with store._lock:  # noqa: SLF001
        stored = store._conn.execute("SELECT count(*) FROM calibrations").fetchone()[0]  # noqa: SLF001
    assert stored == 1, "an unchanged corpus was calibrated twice"


def test_an_edit_after_a_cached_resolution_is_still_noticed(tmp_path: Path) -> None:
    """C7: the resolution is cached against the stored bytes, so an in-place edit is re-verified."""
    store = _store(tmp_path, 48)
    auto_calibrate(store, ContentWordEmbedder())
    assert _strict(store).trust_state == "trusted"  # the resolution is now cached
    with store._lock:  # noqa: SLF001
        raw = store._conn.execute("SELECT payload FROM calibrations").fetchone()[0]  # noqa: SLF001
        tampered = re.sub(r'"threshold":-?[0-9.eE+-]+', '"threshold":-0.99', raw, count=1)
        store._conn.execute("UPDATE calibrations SET payload = ?", (tampered,))  # noqa: SLF001
    with pytest.raises(TrustRefusal) as refused:
        _strict(store)
    assert refused.value.code is TrustFailureCode.DEPENDENCY_UNAVAILABLE, "a cached threshold hid an altered calibration"


def test_storing_a_calibration_surfaces_the_real_error(tmp_path: Path) -> None:
    """C8: `save_calibration` uses `LiteStore._write()`, which rolls back only an open transaction."""
    store = _store(tmp_path, 1)
    conn = sqlite3.connect(store.path)
    conn.executescript(
        "CREATE TRIGGER refuse_calibration BEFORE INSERT ON calibrations "
        "BEGIN SELECT RAISE(ROLLBACK, 'calibration refused'); END;"
    )
    conn.close()
    with pytest.raises(sqlite3.Error) as caught:
        store.save_calibration("{}", created_at="2026-10-07T00:00:00.000000+00:00", model="m", dimension=8)
    assert "calibration refused" in str(caught.value), "the rollback masked the real error"
    assert not store._conn.in_transaction  # noqa: SLF001


def test_a_store_rebound_to_another_model_does_not_lend_it_the_old_threshold(tmp_path: Path) -> None:
    """CCA STAKES-001 (run 20261106T152931Z). Invariant: once the store is bound to another model
    than its newest fit was fitted for (a switch whose refit saved nothing), that fit decides
    nothing: strict search with the new model is refused as stale, and a refit is wanted.
    Failure caught: the trust gate compares the runtime embedder with the binding only, so the old
    model's certified threshold answered the new model's searches as trusted.

    Red proof, 2026-11-06 on the remote test host: with the `_fitted_for_another_model` check
    removed from `recall.lite.calibration.resolve`, this failed at the STALE assertion
    (`CERTIFIED`); with it removed from `_refit_wanted` instead, it failed at the last assertion;
    restored, it passed."""
    from recall.lite.calibration import _refit_wanted

    store = _store(tmp_path, 48)
    assert auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    store.record_embedder_spec("another", model="another-model", dimension=DIM)
    assert store.resolve_calibration().status is CalibrationStatus.STALE, "the old model's fit still certified"
    with pytest.raises(TrustRefusal) as refused:
        _strict(store, ContentWordEmbedder(name="another-model"))
    assert refused.value.code is TrustFailureCode.CALIBRATION_STALE
    assert _refit_wanted(store), "a fit for another model was taken to cover the corpus"


def test_a_model_is_recorded_only_with_its_dimension(tmp_path: Path) -> None:
    """CCA DAT-002 (deferred pass of run 20261106T152931Z). Invariant: `record_embedder_spec`
    refuses a model without its dimension (or the reverse) and writes nothing. Failure caught: a
    model alone stored the text "None" as `embedder_dimension`, which every later binding read
    fails to parse.

    Red proof, 2026-11-07 on the remote test host: with the `if (model is None) != (dimension is
    None):` check of `recall.lite.store.LiteStore.record_embedder_spec` made `if False:`, this
    failed at `pytest.raises(ValueError)` (DID NOT RAISE); restored, it passed."""
    store = _store(tmp_path, 48)
    assert auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    with pytest.raises(ValueError):
        store.record_embedder_spec("another", model="another-model")
    assert store.bound_embedder() == ("test-content-words", DIM)


def test_the_doctor_report_agrees_with_search_on_a_fit_for_another_model(tmp_path: Path) -> None:
    """Follow-up to CCA STAKES-001 (PR 1246). Invariant: `inspect_lite_file`, which `recall doctor`
    reports from, reads a fit made for another model than the store is bound to as STALE, the
    status `resolve` gives search. Failure caught: the report called the old model's certified fit
    current while strict search refused it.

    Red proof, 2026-11-07 on the remote test host: with the `fitted_for_another_model(bound,
    artifact)` branch of `recall.lite.inspect.inspect_lite_file` made `if False:`, this failed at
    the report assertion (`CERTIFIED`); restored, it passed."""
    from recall.lite.inspect import inspect_lite_file

    store = _store(tmp_path, 48)
    assert auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    assert inspect_lite_file(store.path).calibration is CalibrationStatus.CERTIFIED
    store.record_embedder_spec("another", model="another-model", dimension=DIM)
    assert store.resolve_calibration().status is CalibrationStatus.STALE
    assert inspect_lite_file(store.path).calibration is CalibrationStatus.STALE, "the report called the old fit current"


def _fit_with_identity(identity: dict[str, object]):  # noqa: ANN202
    from types import SimpleNamespace

    return SimpleNamespace(embedder_identity=identity)


def test_a_fit_that_records_no_width_does_not_match_an_unknown_bound_width() -> None:
    """CCA CODE-002 (deferred from run 20261107T115351Z). Invariant: a fit with no recorded width
    is never taken for the bound model's, not even when the bound width did not parse (-1).
    Failure caught: both sides defaulted to -1, so the fit matched and was not read as stale.

    Red proof, 2026-11-07 on the remote test host: with `recall.lite.calibration.fitted_for_another_model`
    restored to the pre-fix body (`dimension` defaulting to -1, compared through `_identity`), this
    failed at the first assertion (False); restored, it passed."""
    from recall.lite.calibration import fitted_for_another_model

    no_width = _fit_with_identity({"model": "m"})
    assert fitted_for_another_model(("m", -1), no_width) is True, "a fit with no width matched an unknown bound width"
    assert fitted_for_another_model(("m", DIM), no_width) is True
    assert fitted_for_another_model(("m", DIM), _fit_with_identity({"model": "m", "dimension": DIM})) is False


def test_a_fit_whose_width_is_not_a_number_reads_as_another_models() -> None:
    """CCA BUG-003 (deferred from run 20261107T115351Z). Invariant: a fit whose recorded width is
    not a positive integer counts as another model's instead of raising, so neither `recall doctor`
    nor search stops on it. Failure caught: `_identity` called int() on it and raised ValueError.

    Red proof, 2026-11-07 on the remote test host: with `recall.lite.calibration.fitted_for_another_model`
    restored to the pre-fix body, this failed at the `pytest.fail` below ("raised ValueError");
    restored, it passed."""
    from recall.lite.calibration import fitted_for_another_model

    for width in ("not-a-width", None, 0, -5):
        try:
            result = fitted_for_another_model(("m", DIM), _fit_with_identity({"model": "m", "dimension": width}))
        except (TypeError, ValueError) as exc:
            pytest.fail(f"raised {type(exc).__name__} for width {width!r}")
        assert result is True, f"a fit with width {width!r} was taken for the bound model's"
