from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from recall.embeddings import Embedder


def cache_key(name: str, dim: int, text: str) -> str:
    """Content-address a chunk's embedding by (embedder name, dim, text).

    Including ``name`` and ``dim`` means switching embedder or model can never return a vector
    computed by a different backend — the key simply misses and the text is re-embedded.
    """
    h = hashlib.sha256()
    for part in (name, str(dim), text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class EmbeddingCache:
    """Content-addressed embedding cache backed by SQLite.

    Opt-in: nothing uses it unless a cache is explicitly passed (``embed_with_cache`` treats a
    ``None`` cache as a plain embed), so existing behaviour and tests are unchanged. Vectors are
    stored as JSON keyed by :func:`cache_key`; identical content is embedded once and reused.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vec TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vec FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, vec: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vec) VALUES (?, ?)",
            (key, json.dumps([float(x) for x in vec])),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def embed_with_cache(
    embedder: Embedder, texts: list[str], cache: EmbeddingCache | None
) -> list[list[float]]:
    """Return one vector per text, serving cached hits and embedding only the misses.

    Misses are embedded in a SINGLE batched call (order preserved) and written back, so a
    re-index of a corpus where most chunks are unchanged only pays to embed what actually
    changed. With ``cache=None`` this is exactly ``embedder.embed(texts)``.
    """
    if cache is None:
        return embedder.embed(texts)
    keys = [cache_key(embedder.name, embedder.dim, t) for t in texts]
    results: list[list[float] | None] = [cache.get(k) for k in keys]
    miss_idx = [i for i, r in enumerate(results) if r is None]
    if miss_idx:
        fresh = embedder.embed([texts[i] for i in miss_idx])
        for i, vec in zip(miss_idx, fresh):
            results[i] = vec
            cache.put(keys[i], vec)
    return [r for r in results if r is not None]
