from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from recall.embeddings import (
    Embedder,
    EmbeddingProfile,
    EmbeddingPurpose,
    embed_passages,
    embed_query,
    embedding_profile,
)


def cache_key(
    profile: EmbeddingProfile, dim: int, text: str, purpose: EmbeddingPurpose = "legacy"
) -> str:
    """Content-address an embedding by (complete profile identity, purpose, dim, text).

    Takes the whole `EmbeddingProfile`, not its ID. The ID alone is not an identity: a
    re-provisioned artifact or a context-mode change moves the vectors while the ID stays fixed,
    and a key that misses those serves a vector computed by different weights, or from different
    text, and nothing downstream can tell, and a cache hit is a plausible vector of the right width.
    `EmbeddingProfile.fingerprint` covers every field; see it for what is in the identity and why.

    Deliberately typed to REFUSE a bare string. The previous signature accepted the profile ID,
    so the unsafe call is the one that used to be correct, and a `str | EmbeddingProfile` union
    would have let every existing caller keep the weaker key without noticing.

    ``purpose`` separates the query, passage and legacy encoders: with an asymmetric model the
    same text embeds to different vectors under each, so one key space per purpose is the
    difference between a cache and a correctness bug.
    """
    h = hashlib.sha256()
    for part in (profile.fingerprint(), purpose, str(dim), text):
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
    embedder: Embedder,
    texts: list[str],
    cache: EmbeddingCache | None,
    *,
    purpose: EmbeddingPurpose = "legacy",
) -> list[list[float]]:
    """Return one vector per text, serving cached hits and embedding only the misses.

    Misses are embedded in a SINGLE batched call (order preserved) and written back, so a
    re-index of a corpus where most chunks are unchanged only pays to embed what actually
    changed. With ``cache=None`` this is exactly ``embedder.embed(texts)``.
    """
    def _embed(values: list[str]) -> list[list[float]]:
        if purpose == "query":
            return [embed_query(embedder, value) for value in values]
        if purpose == "passage":
            return embed_passages(embedder, values)
        return embedder.embed(values)

    if cache is None:
        return _embed(texts)
    profile = embedding_profile(embedder)
    keys = [cache_key(profile, embedder.dim, t, purpose) for t in texts]
    results: list[list[float] | None] = [cache.get(k) for k in keys]
    miss_idx = [i for i, r in enumerate(results) if r is None]
    if miss_idx:
        fresh = _embed([texts[i] for i in miss_idx])
        if len(fresh) != len(miss_idx):
            # A hosted embedder dropping one item used to be absorbed here: the unstrict zip
            # left the unfilled slot as None and the filter below removed it, so the caller
            # got N-1 vectors for N texts and the misalignment surfaced two layers later as a
            # cryptic length mismatch — after the embedding spend. Name the fault at its
            # source instead.
            raise RuntimeError(
                f"embedder {embedder.name!r} returned {len(fresh)} vectors for "
                f"{len(miss_idx)} texts; the Embedder contract is one vector per input"
            )
        for i, vec in zip(miss_idx, fresh):
            results[i] = vec
            cache.put(keys[i], vec)
    filled = [r for r in results if r is not None]
    if len(filled) != len(texts):
        raise RuntimeError(
            f"embedding cache produced {len(filled)} vectors for {len(texts)} texts; "
            f"a cache hit vanished mid-call"
        )
    return filled


def embed_query_with_cache(
    embedder: Embedder, text: str, cache: EmbeddingCache | None
) -> list[float]:
    """Query-specific cache path whose entries cannot alias passage vectors."""
    return embed_with_cache(embedder, [text], cache, purpose="query")[0]
