"""A forget erases the forgotten text's vectors from the shared embedding cache.

Invariant: after `forget_memory` or `withdraw_memory` (the funnel `recall forget`, `recall withdraw`
and the MCP `recall_forget` and `recall_withdraw` tools share, on lite and PostgreSQL alike), the
cache file `RECALL_EMBED_CACHE` names holds no entry embedded from a forgotten chunk's text, under
any embedder identity, and no byte of those vectors survives in the file or its write-ahead log.
The entries of a memo that was not forgotten stay.

Failure mode it catches: the cache is content addressed and on by default, so before this a
forgotten memo's vector stayed in it until LRU eviction, after every store-side erasure had run.

Negative controls: each test first asserts the entries it expects gone are present, so a test that
read the wrong file or the wrong keys could not pass by seeing nothing.

Scope: lite stores, which a test can read byte for byte. The PostgreSQL and generation stores reach
the same `_forget_memory_unlocked` call through their own `texts_for_sources`, which these tests do
not exercise.

Red proof, recorded in the pull request: against `origin/master` `f06b14d58` (no erase path in
`recall/cache.py`) the forget tests fail in their `== set()` / `== []` assertions with every
forgotten key still present; the contextual test also fails with the `EMBEDDING_TEXT_DIGEST_KEY`
line of `recall.index.Indexer` removed; the absent-cache test fails with the `path.exists()` check of
`recall.cache.erase_cached_digests` removed (it passes on master, which never opens the cache);
the byte scan fails with the `secure_delete` of `recall.cache.EmbeddingCache.erase_digests`
removed, alone or with its `wal_checkpoint(TRUNCATE)`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from recall.cache import EmbeddingCache, cache_key
from recall.context import ContextPolicy
from recall.embedding_core import HashingEmbedder
from recall.embeddings import embedding_profile
from recall.index import Indexer
from recall.lite import LiteStore
from recall_mcp.lifecycle import forget_memory, withdraw_memory


def _memos(folder: Path) -> tuple[str, str]:
    folder.mkdir()
    gone = folder / "gone.md"
    gone.write_text("# Locker code\n\nThe locker code is zqx four four seven.\n", encoding="utf-8")
    kept = folder / "kept.md"
    kept.write_text("# Parking\n\nThe car is parked on level three.\n", encoding="utf-8")
    return str(gone.resolve()), str(kept.resolve())


def _index(store: LiteStore, folder: Path, embedder: HashingEmbedder, cache_path: Path,
           policy: ContextPolicy = ContextPolicy()) -> None:
    with EmbeddingCache(cache_path) as cache:
        Indexer(store, embedder, cache=cache, context_policy=policy, env={}).index_path(folder)


def _keys(store: LiteStore, source: str, embedder: HashingEmbedder) -> set[str]:
    profile = embedding_profile(embedder)
    return {
        cache_key(profile, embedder.dim, chunk.text, "passage")
        for chunk in store.chunks_for_source(source)
    }


def _cached(cache_path: Path) -> dict[str, bytes]:
    with closing(sqlite3.connect(cache_path)) as conn:
        return {str(key): bytes(vec) for key, vec in conn.execute("SELECT key, vec FROM embeddings")}


def _file_bytes(cache_path: Path) -> bytes:
    return b"".join(
        p.read_bytes() for p in (cache_path, Path(f"{cache_path}-wal")) if p.exists()
    )


@pytest.mark.parametrize("erase", [forget_memory, withdraw_memory], ids=["forget", "withdraw"])
def test_forget_erases_the_memos_cache_entries_under_every_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, erase
) -> None:
    cache_path = tmp_path / "cache" / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))
    # Debian and Ubuntu build SQLite with SECURE_DELETE on by default, Windows and macOS Pythons
    # do not; without this the byte scan below could not fail on CI however the cache deleted.
    connect = sqlite3.connect

    def connect_without_secure_delete(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.execute("PRAGMA secure_delete = OFF")
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect_without_secure_delete)
    gone, kept = _memos(tmp_path / "mem")
    first, second = HashingEmbedder(64), HashingEmbedder(32)
    with LiteStore(tmp_path / "a.db", dim=64) as store, LiteStore(tmp_path / "b.db", dim=32) as other:
        _index(store, tmp_path / "mem", first, cache_path)
        # The same text in another store under another embedder identity: one cache, two keys.
        _index(other, tmp_path / "mem", second, cache_path)
        gone_keys = _keys(store, gone, first) | _keys(other, gone, second)
        kept_keys = _keys(store, kept, first) | _keys(other, kept, second)
        before = _cached(cache_path)
        assert gone_keys and gone_keys <= set(before), "negative control: the entries were never cached"
        assert kept_keys <= set(before)
        gone_vectors = [before[key] for key in gone_keys]

        result = erase(store, [gone])

        assert result.chunks_removed > 0, result.message
        assert "WARNING" not in result.message, result.message
    after = _cached(cache_path)
    assert gone_keys & set(after) == set(), "the forgotten memo's vectors are still cached"
    assert kept_keys <= set(after), "a memo that was not forgotten lost its cache entries"
    raw = _file_bytes(cache_path)
    assert [vec for vec in gone_vectors if vec in raw] == [], "a forgotten vector's bytes survive"


def test_forget_erases_the_entry_a_contextual_profile_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "cache" / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))
    gone, _kept = _memos(tmp_path / "mem")
    embedder = HashingEmbedder(64)
    with LiteStore(tmp_path / "a.db", dim=64) as store:
        _index(store, tmp_path / "mem", embedder, cache_path, ContextPolicy(mode="document"))
        before = set(_cached(cache_path))
        # The vector was embedded from title-prefixed text, so the chunk text's own key misses:
        # only the digest the indexer recorded can find it.
        assert _keys(store, gone, embedder).isdisjoint(before), "the profile embedded the chunk text"
        assert len(before) == 2, "negative control: one entry per memo"

        result = forget_memory(store, [gone])

        assert result.chunks_removed > 0, result.message
    assert len(_cached(cache_path)) == 1, "the forgotten memo's contextual vector is still cached"


def test_forget_with_no_cache_file_creates_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "cache" / "embeddings.sqlite"
    gone, _kept = _memos(tmp_path / "mem")
    with LiteStore(tmp_path / "a.db", dim=64) as store:
        Indexer(store, HashingEmbedder(64), env={}).index_path(tmp_path / "mem")
        monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))

        result = forget_memory(store, [gone])

        assert result.chunks_removed > 0, result.message
        assert "WARNING" not in result.message, result.message
    assert not cache_path.exists() and not cache_path.parent.exists(), "a forget created a cache"


def test_a_cache_that_cannot_be_erased_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "cache" / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))
    gone, _kept = _memos(tmp_path / "mem")

    def refuse(self, digests):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    with LiteStore(tmp_path / "a.db", dim=64) as store:
        _index(store, tmp_path / "mem", HashingEmbedder(64), cache_path)
        monkeypatch.setattr(EmbeddingCache, "erase_digests", refuse, raising=False)

        result = forget_memory(store, [gone])

        assert result.chunks_removed > 0, result.message
        assert "erasing their vectors from the embedding cache failed" in result.message, result.message
