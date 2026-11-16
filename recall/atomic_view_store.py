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
re-embed rather than a wrong vector. `assemble` is the only writer. The PostgreSQL builder runs it
under the host's embedding lock. A lite store (`recall.lite.atomic`) may run two at once, one
under its write lane and one under its refit lane; that is safe because a row is keyed by its
content, and each batch is written in one short transaction after its embeddings return, under
SQLite's busy timeout. Readers never write. `erase_digests` is the only delete,
which a lite forget runs (`recall.lite.atomic.erase_forgotten`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Sequence

from recall.atomizer import MICRO_VIEW_SIZE, MICRO_VIEW_STRIDE, chunk_micro_views


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

    def erase_digests(self, digests: set[str]) -> int:
        """Delete the views of these chunk texts under every profile, and return how many rows went.

        `secure_delete` zeroes the freed cells and a TRUNCATE checkpoint empties the write-ahead
        log, as `LiteStore.erasing` does for the store, since a plain DELETE leaves the vectors'
        bytes in a free page. Raises on failure: the caller is a forget, which must report it.
        """
        if not digests:
            return 0
        prior = int(self._conn.execute("PRAGMA secure_delete").fetchone()[0])
        self._conn.execute("PRAGMA secure_delete = ON")
        try:
            removed = self._conn.executemany(
                "DELETE FROM micro_views WHERE digest = ?", [(d,) for d in sorted(digests)]
            ).rowcount
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._conn.execute(f"PRAGMA secure_delete = {prior}")
        busy = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        if busy:
            raise sqlite3.OperationalError(
                f"a reader of {self.path} blocked the checkpoint, so erased views may stay in its log"
            )
        return removed

    def count(self, profile: str | None = None) -> int:
        if profile is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM micro_views").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM micro_views WHERE profile = ?", (profile,)
            ).fetchone()[0]
        )


EMBED_BATCH_CHUNKS = 64


def assemble(
    parents: Sequence[tuple[str, str, int, str]],
    store: AtomicViewStore,
    profile: str,
    dimension: int,
    embed_groups: Callable[[list[list[str]]], list[list[list[float]]]],
) -> tuple[Any, list[dict[str, object]], dict[str, int]]:
    """Views and vectors for every parent ``(chunk_id, source, ordinal, text)``, reusing the store.

    Returns the normalized matrix, the artifact metadata rows in parent order, and counts of
    reused and newly embedded chunks and views. Chunks whose text yields no view are skipped.
    """

    import numpy as np

    planned: list[tuple[tuple[str, str, int, str], list[str], str]] = []
    missing: list[int] = []
    for parent in parents:
        views = chunk_micro_views(parent[3])
        if not views:
            continue
        digest = chunk_view_digest(parent[3])
        if store.get(profile, digest, view_count=len(views), dimension=dimension) is None:
            missing.append(len(planned))
        planned.append((parent, views, digest))

    embedded_views = 0
    seen_digests: set[str] = set()
    for offset in range(0, len(missing), EMBED_BATCH_CHUNKS):
        batch = []
        batch_digests: set[str] = set()
        for index in missing[offset : offset + EMBED_BATCH_CHUNKS]:
            item = planned[index]
            digest = item[2]
            if digest in seen_digests or digest in batch_digests:
                continue
            batch.append(item)
            batch_digests.add(digest)
        if not batch:
            continue
        vectors = embed_groups([views for _, views, _ in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("embedder returned the wrong number of view groups")
        for (_, views, digest), group in zip(batch, vectors, strict=True):
            if len(group) != len(views):
                raise RuntimeError("embedder returned the wrong number of views for a chunk")
            store.put(profile, digest, group)
            seen_digests.add(digest)
            embedded_views += len(views)
        store.commit()

    rows: list[Any] = []
    metadata: list[dict[str, object]] = []
    view_ordinal: dict[str, int] = {}
    for (chunk_id, source, ordinal, _), views, digest in planned:
        matrix = store.get(profile, digest, view_count=len(views), dimension=dimension)
        if matrix is None:
            raise RuntimeError(f"view store lacks vectors for chunk {chunk_id} after embedding")
        rows.append(matrix)
        for _ in views:
            index = view_ordinal.get(source, 0)
            view_ordinal[source] = index + 1
            metadata.append(
                {"chunk_id": chunk_id, "source": source, "parent_ordinal": ordinal, "view_ordinal": index}
            )
    if not rows:
        raise RuntimeError("no parent chunk yields a micro view")
    # The rows are float32 already (`get`), so this is the one copy; dropping them as soon as it
    # exists keeps the peak at about two matrices rather than three (CCA PERF-004).
    full: Any = np.concatenate(rows, axis=0, dtype=np.float32)
    rows.clear()
    norms = np.linalg.norm(full, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise RuntimeError("a view vector has zero norm")
    full /= norms
    counts = {
        "parents": len(planned),
        "views": len(metadata),
        "reused_chunks": len(planned) - len(missing),
        "embedded_chunks": len(seen_digests),
        "embedded_views": embedded_views,
    }
    return full, metadata, counts


__all__ = ["AtomicViewStore", "EMBED_BATCH_CHUNKS", "VIEW_STORE_SCHEMA_VERSION", "assemble", "chunk_view_digest"]
