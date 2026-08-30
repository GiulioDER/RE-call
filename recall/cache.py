from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from recall.embeddings import (
    Embedder,
    EmbeddingProfile,
    EmbeddingPurpose,
    embed_passages,
    embed_query,
    embedding_profile,
)
from recall.observability import get_logger

_log = get_logger("cache")

#: Path of the shared cache, or a disabling word. Read per call rather than at import, so a test
#: (and a long-lived process) can change it without reloading the module.
ENV_CACHE_PATH = "RECALL_EMBED_CACHE"
#: Ceiling on the cache file's vector bytes. A cache that is on by default and grows without a
#: bound is a disk-full incident waiting for the largest corpus somebody indexes.
ENV_CACHE_MAX_MB = "RECALL_EMBED_CACHE_MAX_MB"
DEFAULT_CACHE_MAX_MB = 512
#: Anything here in `RECALL_EMBED_CACHE` means "no cache", so the feature has an off switch that
#: does not require knowing a path. `""` is included: exporting a variable empty is how a shell
#: script disables something, and reading that as a relative path named "" would be a surprise.
_DISABLED_VALUES = frozenset({"", "0", "off", "no", "false", "none"})
#: Rewrite a hit's recency stamp only when it is this stale. Without it a warm build issues one
#: UPDATE per hit, which is write traffic proportional to the work the cache exists to avoid.
_RECENCY_REWRITE_SECONDS = 3600.0


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


def _max_bytes_from_env() -> int:
    """`RECALL_EMBED_CACHE_MAX_MB`, non-negative; anything malformed falls back to the default.

    Malformed values are IGNORED rather than clamped, and logged, for the reason
    `recall.index._batch_chunks_from_env` gives: a host that sets this is bounding a resource, and
    silently substituting a different bound for the one that was configured is how a machine fills
    a disk under a setting somebody believed was in force. `0` means unbounded.
    """
    default = DEFAULT_CACHE_MAX_MB * 1024 * 1024
    raw = os.environ.get(ENV_CACHE_MAX_MB)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _log.warning("ignoring malformed %s=%r", ENV_CACHE_MAX_MB, raw)
        return default
    if value < 0:
        _log.warning("ignoring negative %s=%r", ENV_CACHE_MAX_MB, raw)
        return default
    return value * 1024 * 1024


def default_cache_path() -> Path | None:
    """Where the shared embedding cache lives, or None when it is switched off.

    `RECALL_EMBED_CACHE` names a file, or one of `_DISABLED_VALUES` to disable. Unset, the file
    goes under the platform's cache directory, which is the one place an operating system already
    treats as expendable: nothing here is a source of truth, every entry is recomputable from the
    text its key was derived from, and deleting the file costs one re-embed.
    """
    raw = os.environ.get(ENV_CACHE_PATH)
    if raw is not None:
        if raw.strip().lower() in _DISABLED_VALUES:
            return None
        return Path(raw).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_CACHE_HOME")
        root = Path(base) if base else Path.home() / ".cache"
    return root / "recall" / "embeddings.sqlite"


def open_default_cache() -> "EmbeddingCache | None":
    """Open the shared cache, or return None if it is disabled or cannot be opened.

    Never raises. This is what the indexing entry points call, and a cache is an optimisation: a
    read-only home directory, a corrupt file or a full disk must cost a re-embed, not the run. The
    failure is logged rather than swallowed, because a cache that is silently never warm looks
    exactly like a cache that is working and expensive.
    """
    path = default_cache_path()
    if path is None:
        return None
    try:
        return EmbeddingCache(path)
    except (sqlite3.Error, OSError) as exc:
        _log.warning("embedding cache unavailable at %s (%s); continuing without it", path, exc)
        return None


@contextmanager
def default_cache() -> Iterator["EmbeddingCache | None"]:
    """Hold the shared cache open for one block, closing it however the block ends.

    The indexing entry points use this rather than `open_default_cache` directly, so that the one
    thing a caller can get wrong — leaving a SQLite connection open on a file every other recall
    process wants to write — is not something a caller has to remember. Yields None when the cache
    is disabled or unopenable, which every call site must already handle.
    """
    cache = open_default_cache()
    try:
        yield cache
    finally:
        if cache is not None:
            cache.close()


class EmbeddingCache:
    """Content-addressed embedding cache backed by SQLite.

    Keyed by :func:`cache_key`, so an entry is bound to the complete embedder identity and to the
    encoder purpose. Identical content is embedded once and reused **across tenants, generations,
    corpora and processes**: the key contains nothing scoped to any of them, which is what makes a
    rebuild that changes a pipeline fingerprint without changing any chunk text free, rather than a
    full pass through a metered API.

    Two properties this class must keep, because it is on by default at the indexing entry points:

    - **Bounded.** Vector bytes are capped (`RECALL_EMBED_CACHE_MAX_MB`, default 512 MB) and the
      least recently used entries are evicted past it.
    - **Never fatal.** Any `sqlite3.Error` after construction degrades this object to a permanent
      no-op (every read a miss, every write dropped) with one warning, rather than failing a run
      that would otherwise have succeeded at the cost of embedding.
    """

    def __init__(self, path: str | Path, *, max_bytes: int | None = None) -> None:
        """Open (creating if needed) the cache at ``path``.

        ``max_bytes`` overrides `RECALL_EMBED_CACHE_MAX_MB`; ``0`` means unbounded. Unlike
        `open_default_cache`, this DOES raise if the file cannot be opened: a caller naming a path
        is configuring something, and a silent fallback would leave them measuring a cache that is
        not there.
        """
        self._path = Path(path)
        self._max_bytes = _max_bytes_from_env() if max_bytes is None else max_bytes
        self._degraded = False
        #: Writes are given up on separately from reads, because the two fail independently: a
        #: cache file on a read-only mount (a baked CI image, a shared corpus) still answers every
        #: lookup correctly, and turning the whole object off at the first failed write would throw
        #: away the hits it was built to serve.
        self._writes_disabled = False
        #: Bytes written since the size cap was last checked. The check is a SUM over the table,
        #: so doing it per write would put a full scan on the path of every cached query.
        self._bytes_since_sweep = 0
        #: Vectors served and vectors computed since this object was opened. Exposed so a caller
        #: can report what the cache actually did rather than assert that it did something.
        self.hits = 0
        self.misses = 0
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        # WAL plus a busy timeout, because one cache file is shared: two checkouts indexing
        # different corpora, a build and a query, a hook and an editor. Under the default rollback
        # journal a concurrent reader and writer exclude each other and raise "database is locked"
        # immediately rather than waiting. WAL is refused on some filesystems (network shares,
        # some container mounts); that is a throughput property, not a correctness one, so it is
        # the one statement here allowed to fail.
        with suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        # Durability is deliberately weak: every row is recomputable from the text its key was
        # derived from, so a torn write after a power loss costs one re-embed and nothing else.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vec TEXT NOT NULL)"
        )
        self._migrate()
        self._conn.commit()

    # ---- storage ---------------------------------------------------------------------------
    #
    # Vectors are stored as packed little-endian float32 in the `vec` column, which is declared
    # TEXT. That is neither an accident nor a type error: a SQLite column has affinity rather than
    # a type, and a TEXT-affinity column stores a BLOB as a BLOB (affinity converts numbers to
    # text, never blobs). Reusing the column is what lets a cache file written by an older recall,
    # whose rows hold JSON text, keep serving hits after an upgrade instead of re-embedding its
    # whole corpus. `_decode` dispatches on the storage class of the value it actually finds.
    #
    # float32 rather than float64 because `pgvector`'s `vector` type IS float32: a vector that
    # round-trips through this cache cannot lose precision the corpus would have kept. The one
    # consumer that sees any difference is an in-process float64 comparison (a benchmark scoring
    # in numpy), where it is ~1e-7 relative and cannot reorder a ranking.

    @staticmethod
    def _encode(vec: list[float]) -> bytes:
        return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))

    @staticmethod
    def _decode(raw: object) -> list[float] | None:
        if isinstance(raw, bytes):
            if len(raw) % 4:
                return None  # truncated write: treat as a miss and recompute
            return list(struct.unpack(f"<{len(raw) // 4}f", raw))
        if isinstance(raw, str):
            try:
                return [float(x) for x in json.loads(raw)]
            except (ValueError, TypeError):
                return None
        return None

    def _migrate(self) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(embeddings)")}
        if "used_at" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN used_at REAL")
        self._conn.execute("CREATE INDEX IF NOT EXISTS embeddings_used_at ON embeddings (used_at)")

    def _degrade(self, exc: Exception) -> None:
        if not self._degraded:
            self._degraded = True
            _log.warning(
                "embedding cache at %s failed (%s); continuing without it for this process",
                self._path,
                exc,
            )

    def _stop_writing(self, exc: Exception) -> None:
        if not self._writes_disabled:
            self._writes_disabled = True
            _log.warning(
                "embedding cache at %s is not writable (%s); serving reads only", self._path, exc
            )

    # ---- reads and writes ------------------------------------------------------------------

    def get(self, key: str) -> list[float] | None:
        return self.get_many([key]).get(key)

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        """Look up many keys, counting the hits and recording that they were used.

        One bound statement per key rather than an `IN (...)` list built by interpolation: the
        lookup is a primary-key probe in an in-process database, the caller's batch is a chunk
        batch rather than a corpus, and a query with no interpolated SQL at all cannot grow an
        injection or a host-parameter-limit bug later.
        """
        if self._degraded or not keys:
            return {}
        unique = list(dict.fromkeys(keys))
        found: dict[str, list[float]] = {}
        try:
            for key in unique:
                row = self._conn.execute(
                    "SELECT vec FROM embeddings WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    continue
                vec = self._decode(row[0])
                if vec is not None:
                    found[key] = vec
        except sqlite3.Error as exc:
            self._degrade(exc)
            return {}
        self._touch(list(found))
        self.hits += len(found)
        self.misses += len(unique) - len(found)
        return found

    def _touch(self, keys: list[str]) -> None:
        """Record that these entries were used. Failing here must not cost the caller its hits."""
        if not keys or self._writes_disabled:
            return
        now = time.time()
        stale_before = now - _RECENCY_REWRITE_SECONDS
        try:
            self._conn.executemany(
                "UPDATE embeddings SET used_at = ? WHERE key = ? "
                "AND (used_at IS NULL OR used_at < ?)",
                [(now, key, stale_before) for key in keys],
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._stop_writing(exc)

    def put(self, key: str, vec: list[float]) -> None:
        self.put_many([(key, vec)])

    def put_many(self, items: list[tuple[str, list[float]]]) -> None:
        """Write many vectors in ONE transaction, then enforce the size cap.

        One commit per vector was the previous behaviour, and that is an fsync per vector: on a
        corpus of any size a cache written that way can cost more than the local embedding it
        saves.
        """
        if self._degraded or self._writes_disabled or not items:
            return
        now = time.time()
        rows = [(key, self._encode(vec), now) for key, vec in items]
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vec, used_at) VALUES (?, ?, ?)", rows
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._stop_writing(exc)
            return
        if self._max_bytes <= 0:
            return
        self._bytes_since_sweep += sum(len(row[1]) for row in rows)
        if self._bytes_since_sweep >= self._sweep_interval():
            self._bytes_since_sweep = 0
            try:
                self._evict_over_cap()
            except sqlite3.Error as exc:
                self._stop_writing(exc)

    def _sweep_interval(self) -> int:
        """How many written bytes may accumulate before the cap is re-checked.

        A hundredth of the cap, floored at a megabyte: the check is a SUM over the table, and
        running one per write would make a cached QUERY (one vector, one `put`) pay a full scan.
        The cap can be overshot by this much between sweeps, which is a trade made explicitly
        rather than slack nobody chose.

        Never MORE than the cap itself, which is what keeps the floor from swallowing a small one:
        at a 4 MB cap a megabyte of slack is a quarter of the budget, and at a cap of a few
        kilobytes the floor alone would mean the sweep never ran at all.
        """
        return min(max(1024 * 1024, self._max_bytes // 100), self._max_bytes)

    def _evict_over_cap(self) -> None:
        """Drop least recently used entries until the vector bytes are back under the cap.

        Evicts to 90% of the cap rather than to the cap, so a build sitting exactly at the boundary
        does not pay an eviction scan on every batch it writes.
        """
        if self._max_bytes <= 0:
            return
        row = self._conn.execute("SELECT COALESCE(SUM(length(vec)), 0) FROM embeddings").fetchone()
        used = int(row[0]) if row else 0
        if used <= self._max_bytes:
            return
        target = int(self._max_bytes * 0.9)
        victims: list[str] = []
        # A lazy cursor: only as many rows as it takes to free the overage are ever read, which
        # matters because this runs precisely when the cache is large.
        cursor = self._conn.execute(
            "SELECT key, length(vec) FROM embeddings ORDER BY COALESCE(used_at, 0) ASC, key ASC"
        )
        for key, size in cursor:
            victims.append(str(key))
            used -= int(size)
            if used <= target:
                break
        cursor.close()
        self._conn.executemany(
            "DELETE FROM embeddings WHERE key = ?", [(key,) for key in victims]
        )
        self._conn.commit()
        _log.info("embedding cache evicted %d entries to stay under its size cap", len(victims))

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

    Misses are deduplicated and embedded in a SINGLE batched call (order preserved) and written
    back, so a re-index of a corpus where most chunks are unchanged only pays to embed what
    actually changed. With ``cache=None`` this is exactly ``embedder.embed(texts)``.
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
    cached = cache.get_many(keys)
    # Deduplicated by KEY, not by text: two identical texts share a key by construction, and a
    # corpus repeats itself (a licence header, a template, the same memo filed twice). Embedding
    # one of them twice inside a single call is spend with a known answer.
    missing: dict[str, str] = {}
    for key, text in zip(keys, texts):
        if key not in cached and key not in missing:
            missing[key] = text
    if missing:
        miss_keys = list(missing)
        fresh = _embed([missing[key] for key in miss_keys])
        if len(fresh) != len(miss_keys):
            # A hosted embedder dropping one item used to be absorbed here: the unstrict zip
            # left the unfilled slot as None and the filter below removed it, so the caller
            # got N-1 vectors for N texts and the misalignment surfaced two layers later as a
            # cryptic length mismatch — after the embedding spend. Name the fault at its
            # source instead.
            raise RuntimeError(
                f"embedder {embedder.name!r} returned {len(fresh)} vectors for "
                f"{len(miss_keys)} texts; the Embedder contract is one vector per input"
            )
        cache.put_many(list(zip(miss_keys, fresh)))
        cached.update(zip(miss_keys, fresh))
    results = [cached.get(key) for key in keys]
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
