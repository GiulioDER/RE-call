"""A content-addressed store of micro-view vectors, decoupled from generations.

The memory tenant refreshes into a new generation about seven times a day, and each refresh
changes only 2 to 14 of roughly 1,650 memos (measured 2026-09-23 over eight generations). A
generation-bound atomic artifact rebuilt from scratch every time would re-embed about 76,000 views
per refresh. This store makes the artifact an assembly instead: the views of one chunk text are
embedded once, with that chunk as their only context, and every later generation containing the
same text reuses the same vectors.

Keys are content, never identity. Chunk IDs are not content-addressed (about 230 change per
generation while about 23 texts do), so a row is keyed by the embedder's profile fingerprint and a
digest of the chunk text together with the view parameters. Changing the model, the window size or
the stride therefore can never return a stale vector; it simply misses.

Reads validate shape and finiteness and treat any mismatch as a miss, so a damaged row costs one
re-embed rather than a wrong vector. The builder is the only writer and runs under the host's
embedding lock; readers never write.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Any

from recall.atomizer import MICRO_VIEW_SIZE, MICRO_VIEW_STRIDE


VIEW_STORE_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS micro_views (
    profile TEXT NOT NULL,
    digest TEXT NOT NULL,
    view_count INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    vectors BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (profile, digest)
);
CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def chunk_view_digest(
    text: str,
    *,
    size: int = MICRO_VIEW_SIZE,
    stride: int = MICRO_VIEW_STRIDE,
    min_content_words: int = 4,
) -> str:
    """The content key of one chunk's micro views, including the view parameters."""

    header = f"micro-views-v1:{size}:{stride}:{min_content_words}\x00"
    return hashlib.sha256((header + text).encode("utf-8")).hexdigest()


class AtomicViewStore:
    """One SQLite file of per-chunk view matrices, keyed by embedder profile and content."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO store_meta (key, value) VALUES ('schema_version', ?)",
                (str(VIEW_STORE_SCHEMA_VERSION),),
            )
            self._conn.commit()
        elif row[0] != str(VIEW_STORE_SCHEMA_VERSION):
            raise ValueError(f"atomic view store schema {row[0]} is unsupported")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AtomicViewStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, profile: str, digest: str, *, view_count: int, dimension: int) -> Any | None:
        """Return a ``(view_count, dimension)`` float32 matrix, or None on any mismatch."""

        import numpy as np

        row = self._conn.execute(
            "SELECT view_count, dimension, vectors FROM micro_views WHERE profile = ? AND digest = ?",
            (profile, digest),
        ).fetchone()
        if row is None:
            return None
        stored_count, stored_dimension, blob = row
        if stored_count != view_count or stored_dimension != dimension:
            return None
        if len(blob) != view_count * dimension * 4:
            return None
        matrix: Any = np.frombuffer(blob, dtype=np.float32).reshape(view_count, dimension)
        if not np.all(np.isfinite(matrix)):
            return None
        return matrix

    def put(self, profile: str, digest: str, vectors: Any) -> None:
        import numpy as np

        matrix: Any = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
            raise ValueError("atomic view store rows must be a finite nonempty 2D matrix")
        self._conn.execute(
            "INSERT OR REPLACE INTO micro_views "
            "(profile, digest, view_count, dimension, vectors, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (profile, digest, int(matrix.shape[0]), int(matrix.shape[1]), matrix.tobytes(), time.time()),
        )

    def commit(self) -> None:
        self._conn.commit()

    def count(self, profile: str | None = None) -> int:
        if profile is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM micro_views").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM micro_views WHERE profile = ?", (profile,)
            ).fetchone()[0]
        )


__all__ = ["AtomicViewStore", "VIEW_STORE_SCHEMA_VERSION", "chunk_view_digest"]
