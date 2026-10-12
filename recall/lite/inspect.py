"""What a lite store file holds, read without writing anything.

`LiteStore` creates its schema when it opens a file, which is right for indexing and searching and
wrong for a diagnostic: `recall doctor` promises to write nothing, and a doctor that creates the
file it was asked about has changed the answer. This opens the file read-only (`mode=ro`), reads
its identity, size and newest calibration, and judges that calibration against the corpus the
same way `recall.lite.calibration.resolve` does, including the carry-forward a changed corpus gets
(`recall.lite.calibration.carry_forward`, run here over `_ReadOnlyCorpus`), so the doctor and the
dashboard report the calibration search actually serves.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, cast

from recall.calibration_v2 import CalibrationStatus

__all__ = ["LiteFileReport", "inspect_lite_file"]


@dataclass(frozen=True)
class LiteFileReport:
    path: Path
    exists: bool
    tenant: str | None = None
    dim: int | None = None
    chunks: int = 0
    sources: int = 0
    calibration: CalibrationStatus = CalibrationStatus.MISSING
    threshold: float | None = None
    separability: float | None = None
    error: str | None = None
    fingerprint: str = ""  # `corpus_digest` of the rows, as `LiteStore.corpus_fingerprint` computes it
    embedder_model: str | None = None
    calibrated_at: str | None = None
    #: The corpus delta from the fit when the calibration in force was carried forward to this
    #: corpus rather than fitted on it; None for a fitted one.
    carried_delta: float | None = None

    @property
    def generation_id(self) -> str:
        """The id `LiteStore.lite_generation_id` gives this corpus."""
        return "lite-" + self.fingerprint[:16]


class _ReadOnlyCorpus:
    """The few `LiteStore` reads `carry_forward` makes, answered from a read-only connection."""

    def __init__(self, conn: sqlite3.Connection, tenant: str, dim: int, rows: list[Any], fingerprint: str, latest: str) -> None:
        self._conn, self.tenant, self._dim = conn, tenant, dim
        self._rows, self._fingerprint, self._latest = rows, fingerprint, latest

    def latest_calibration_json(self) -> str:
        return self._latest

    def corpus_fingerprint(self) -> str:
        return self._fingerprint

    def lite_generation_id(self) -> str:
        return "lite-" + self._fingerprint[:16]

    def source_manifest(self) -> dict[str, str]:
        from recall.lite.calibration import source_digests

        return source_digests([(str(r[2]), str(r[0]), str(r[1])) for r in self._rows])

    def calibration_evidence(self, calibration_id: str) -> tuple[str, bytes, str] | None:
        try:
            row = self._conn.execute(
                "SELECT queries, vectors, sources FROM calibration_evidence WHERE calibration_id = ?", (calibration_id,)
            ).fetchone()
        except sqlite3.OperationalError:  # a file written before the table existed
            return None
        return (str(row[0]), bytes(row[1]), str(row[2])) if row else None

    @cached_property
    def _vectors(self) -> Any:
        import numpy as np

        blobs = self._conn.execute("SELECT embedding FROM chunks").fetchall()
        vectors: Any = np.zeros((len(blobs), self._dim), dtype=np.float32)
        for i, (blob,) in enumerate(blobs):
            vectors[i] = np.frombuffer(blob, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def top_cosine(self, vector: list[float]) -> float:
        import numpy as np

        if not len(self._vectors):
            return 0.0
        q: Any = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        return float((self._vectors @ (q / norm if norm else q)).max())


def inspect_lite_file(path: str | Path) -> LiteFileReport:
    """Read `path` as a lite store, read-only; an unreadable file is reported, never raised."""
    from recall.lite.calibration import artifact_from_json, carry_forward, corpus_digest

    target = Path(path)
    if not target.exists():
        return LiteFileReport(target, exists=False)
    try:
        conn = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return LiteFileReport(target, exists=True, error=f"{type(exc).__name__}: {exc}")
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        rows = conn.execute("SELECT id, text, source FROM chunks").fetchall()
        latest = conn.execute("SELECT payload FROM calibrations ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        conn.close()
        return LiteFileReport(target, exists=True, error=f"not a lite store ({type(exc).__name__}: {exc})")
    status, threshold, separability, calibrated_at, carried_delta = CalibrationStatus.MISSING, None, None, None, None
    fingerprint = corpus_digest([(str(r[0]), str(r[1])) for r in rows])
    try:
        if latest is not None:
            try:
                artifact = artifact_from_json(str(latest[0]))
            except Exception as exc:  # BROAD-CATCH: fail-open  # a tampered artifact is reported, not raised
                return LiteFileReport(target, exists=True, error=f"the stored calibration does not verify: {exc}")
            threshold, separability, calibrated_at = artifact.threshold, artifact.separability, artifact.created_at
            if artifact.corpus_fingerprint != fingerprint:
                status = CalibrationStatus.STALE
                corpus = _ReadOnlyCorpus(
                    conn, str(meta.get("tenant", "")), int(meta.get("dim", 0)), rows, fingerprint, str(latest[0])
                )
                carried = carry_forward(cast("Any", corpus), artifact)
                if carried.artifact is not None:
                    status, separability = CalibrationStatus.CERTIFIED, carried.artifact.separability
                    carried_delta = float(carried.provenance["corpus_delta"])
            elif not artifact.certified:
                status = CalibrationStatus.UNCERTIFIED
            else:
                status = CalibrationStatus.CERTIFIED
    except sqlite3.Error as exc:
        return LiteFileReport(target, exists=True, error=f"not a lite store ({type(exc).__name__}: {exc})")
    finally:
        conn.close()
    return LiteFileReport(
        target,
        exists=True,
        tenant=meta.get("tenant"),
        dim=int(meta["dim"]) if "dim" in meta else None,
        chunks=len(rows),
        sources=len({r[2] for r in rows}),
        calibration=status,
        threshold=threshold,
        separability=separability,
        fingerprint=fingerprint,
        embedder_model=meta.get("embedder_model"),
        calibrated_at=calibrated_at,
        carried_delta=carried_delta,
    )
