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
- C3 a changed corpus resolves `stale` and strict search refuses until `ensure_calibrated` refits.
- C4 searching with a different embedder than the one calibrated is LINEAGE_MISMATCH.
- C5 a stored calibration whose bytes were altered fails closed (DEPENDENCY_UNAVAILABLE): the
  threshold is never read from an artifact whose checksum does not verify.
- C6 an unchanged corpus is not refitted.
- C8 storing a calibration goes through the store's one write path, so an error after which SQLite
  already rolled back reaches the caller instead of "cannot rollback".

Red proof, 2026-10-07, each mutation alone against `recall/lite/calibration.py` or
`recall/lite/store.py`, failing in the named assertion (JUnit XML), then restored byte for byte:
- K1 (C1) `resolve` reading an empty store as certified: `CERTIFIED is MISSING` failed.
- K2 (C2) `auto_calibrate` never certifying: the outcome was uncertified.
- K3 (C3) the corpus fingerprint unchecked: "a changed corpus kept its old certificate".
- K4 (C4) `generation_binding` without the embedder: `DID NOT RAISE TrustRefusal`.
- K5 (C5) `artifact_from_json` without `verify_checksum`: `DID NOT RAISE TrustRefusal`.
- K6 (C6) `ensure_calibrated` refitting a certified corpus: "an unchanged corpus was calibrated twice".
- K8 (C8) baseline, not a mutation: `LiteStore.save_calibration` at `b3eac0a5`, with its own
  try/ROLLBACK, failed `test_storing_a_calibration_surfaces_the_real_error` in "the rollback masked
  the real error".
"""

from __future__ import annotations

import hashlib
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
    _write_memos(tmp_path / "m", 2, start=48)
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
