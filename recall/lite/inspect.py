"""What a lite store file holds, read without writing anything.

`LiteStore` creates its schema when it opens a file, which is right for indexing and searching and
wrong for a diagnostic: `recall doctor` promises to write nothing, and a doctor that creates the
file it was asked about has changed the answer. This opens the file read-only (`mode=ro`), reads
its identity, size and newest calibration, and judges that calibration against the corpus the
same way `recall.lite.calibration.resolve` does.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def generation_id(self) -> str:
        """The id `LiteStore.lite_generation_id` gives this corpus."""
        return "lite-" + self.fingerprint[:16]


def lite_file_meta(path: str | Path) -> dict[str, str] | None:
    """A lite file's `meta` table, read over a read-only connection; None when it has none.

    For a command that needs only the width or the tenant: `inspect_lite_file` also reads and
    hashes every chunk and verifies the latest calibration, and opening a `LiteStore` would write
    a lite schema into a file that is not one.
    """
    try:
        conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return {str(key): str(value) for key, value in conn.execute("SELECT key, value FROM meta").fetchall()}
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def inspect_lite_file(path: str | Path) -> LiteFileReport:
    """Read `path` as a lite store, read-only; an unreadable file is reported, never raised."""
    from recall.lite.calibration import artifact_from_json, corpus_digest

    target = Path(path)
    if not target.exists():
        return LiteFileReport(target, exists=False)
    try:
        conn = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return LiteFileReport(target, exists=True, error=f"{type(exc).__name__}: {exc}")
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        rows = conn.execute("SELECT id, text, source, embedding FROM chunks").fetchall()
        latest = conn.execute("SELECT payload FROM calibrations ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        return LiteFileReport(target, exists=True, error=f"not a lite store ({type(exc).__name__}: {exc})")
    finally:
        conn.close()
    try:
        dim = int(meta["dim"])
        if dim < 1:
            raise ValueError("embedding width must be positive")
    except (KeyError, TypeError, ValueError) as exc:
        return LiteFileReport(target, exists=True, error=f"invalid embedding width metadata: {exc}")
    status, threshold, separability, calibrated_at = CalibrationStatus.MISSING, None, None, None
    # The store's identity since PR 896's audit: ids, texts AND stored vectors (restack fix; a
    # pair digest here would never equal `LiteStore.corpus_fingerprint` and read as stale).
    fingerprint = corpus_digest([(str(r[0]), str(r[1]), bytes(r[3])) for r in rows])
    if latest is not None:
        try:
            artifact = artifact_from_json(str(latest[0]))
        except Exception as exc:  # BROAD-CATCH: fail-open  # a tampered artifact is reported, not raised
            return LiteFileReport(target, exists=True, error=f"the stored calibration does not verify: {exc}")
        threshold, separability, calibrated_at = artifact.threshold, artifact.separability, artifact.created_at
        if artifact.corpus_fingerprint != fingerprint:
            status = CalibrationStatus.STALE
        elif not artifact.certified:
            status = CalibrationStatus.UNCERTIFIED
        else:
            status = CalibrationStatus.CERTIFIED
    return LiteFileReport(
        target,
        exists=True,
        tenant=meta.get("tenant"),
        dim=dim,
        chunks=len(rows),
        sources=len({r[2] for r in rows}),
        calibration=status,
        threshold=threshold,
        separability=separability,
        fingerprint=fingerprint,
        embedder_model=meta.get("embedder_model"),
        calibrated_at=calibrated_at,
    )
