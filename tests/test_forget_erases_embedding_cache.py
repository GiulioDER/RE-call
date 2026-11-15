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

from recall.cache import EmbeddingCache, cache_key, embed_with_cache, text_digest
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


def test_an_erase_a_reader_kept_from_checkpointing_raises(tmp_path: Path) -> None:
    """A blocked TRUNCATE checkpoint is a failed erase, so the forget can report it.

    Failure mode: `erase_digests` discarded the checkpoint's result row, so with another
    connection holding a read snapshot the deleted vector's old pages stayed in the file and its
    write-ahead log while the forget reported success.

    Red proof: against `origin/master` `98ae7efba` (the checkpoint result unread) this fails at
    `pytest.raises` with `DID NOT RAISE`. The forget turns the raise into a WARNING, which
    `test_a_cache_that_cannot_be_erased_is_reported` pins.
    """
    path = tmp_path / "embeddings.sqlite"
    with EmbeddingCache(path) as cache:
        cache.put_many([("k", [0.25, 0.5])], texts=["the locker code"])
        # Without this the checkpoint waits out the 5 s busy timeout before giving up.
        cache._conn.execute("PRAGMA busy_timeout = 0")
        with closing(sqlite3.connect(path, isolation_level=None)) as reader:
            reader.execute("BEGIN")
            assert reader.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (1,)
            with pytest.raises(sqlite3.OperationalError, match="blocked the checkpoint"):
                cache.erase_digests({text_digest("the locker code")})
            reader.execute("COMMIT")


def _drop_digests(cache_path: Path) -> None:
    """Make every entry look as 0.15.0 wrote it: with no digest of its text."""
    with closing(sqlite3.connect(cache_path)) as conn:
        conn.execute("UPDATE embeddings SET text_sha256 = NULL")
        conn.commit()


def test_forget_warns_while_the_cache_holds_entries_without_a_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forget that cannot tell whether the cache still holds the text says so.

    Failure mode: an entry written before the digest column (0.15.0 and earlier) has a NULL
    digest, which the erase-by-digest never matches, so the forgotten vector stayed cached and
    the forget reported success. The negative control is
    `test_forget_erases_the_memos_cache_entries_under_every_embedder`, which asserts no WARNING
    for a cache whose entries all carry a digest.

    Red proof: against `origin/master` `98ae7efba` this fails at the `record no digest` message
    assertion (`Forgot 1 chunk(s) from 1 source(s).`, no warning).
    """
    cache_path = tmp_path / "cache" / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_path))
    gone, _kept = _memos(tmp_path / "mem")
    with LiteStore(tmp_path / "a.db", dim=64) as store:
        _index(store, tmp_path / "mem", HashingEmbedder(64), cache_path)
        _drop_digests(cache_path)
        assert len(_cached(cache_path)) == 2, "negative control: one entry per memo"

        result = forget_memory(store, [gone])

        assert result.chunks_removed > 0, result.message
        assert "holds 2 embedding(s) that record no digest of their text" in result.message, (
            result.message
        )


def test_a_cache_hit_gives_an_entry_without_a_digest_its_digest(tmp_path: Path) -> None:
    """An entry 0.15.0 wrote becomes erasable the next time its text is embedded through the cache.

    Failure mode: a NULL digest stayed NULL for the entry's lifetime, however often it was
    served, so no forget could ever find it.

    Red proof: against `origin/master` `98ae7efba` this fails at `erase_digests(...) == 1` (0
    erased). Mutation: deleting the `SET text_sha256` statement from
    `recall.cache.EmbeddingCache._touch` fails it the same way.
    """
    path = tmp_path / "embeddings.sqlite"
    embedder = HashingEmbedder(16)
    texts = ["The locker code is zqx four four seven.", "The car is parked on level three."]
    with EmbeddingCache(path) as cache:
        embed_with_cache(embedder, texts, cache, purpose="passage")
    _drop_digests(path)

    with EmbeddingCache(path) as cache:
        embed_with_cache(embedder, texts, cache, purpose="passage")
        assert (cache.hits, cache.misses) == (2, 0), "negative control: both were cache hits"

    with EmbeddingCache(path) as cache:
        assert cache.erase_digests({text_digest(texts[0])}) == 1, "the served entry has no digest"
    assert len(_cached(path)) == 1, "the other text's entry went too"
