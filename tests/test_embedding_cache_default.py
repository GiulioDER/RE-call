"""The shared embedding cache: where it lives, what bounds it, and how it fails.

`tests/test_cache.py` pins what the cache RETURNS and `tests/test_embedding_cache_identity.py` pins
what a key may be served to. This file pins the three properties that only matter once the cache is
on by default at the indexing entry points, none of which the earlier files could have needed:

- it resolves to a disposable location and has an off switch that does not require knowing a path;
- it is BOUNDED, so a default-on cache cannot fill a disk;
- it is NEVER FATAL, so a corrupt or unwritable cache costs a re-embed rather than the run.

The storage change is here too. Vectors are packed float32 in a column declared TEXT, and the one
thing that could silently undo the upgrade path is SQLite deciding to store those bytes as text, so
that is asserted directly rather than inferred from a round trip.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.cache import (
    DEFAULT_CACHE_MAX_MB,
    EmbeddingCache,
    _max_bytes_from_env,
    cache_key,
    default_cache,
    default_cache_path,
    embed_with_cache,
    open_default_cache,
)
from recall.embeddings import legacy_embedding_profile


class CountingEmbedder:
    """Deterministic embedder that records exactly which texts it was asked to embed."""

    dim = 2
    name = "counting"

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]


def _key(text: str, embedder: CountingEmbedder | None = None) -> str:
    embedder = embedder or CountingEmbedder()
    return cache_key(legacy_embedding_profile(embedder), embedder.dim, text)


# ---- where it lives -----------------------------------------------------------------------


@pytest.mark.parametrize("word", ["0", "off", "no", "false", "none", "", "  OFF  "])
def test_a_disabling_word_switches_the_cache_off(monkeypatch, word: str) -> None:
    """Every documented off switch, including an empty value and a shouted, padded one.

    The empty string is the case worth having a test for: exporting a variable empty is how a shell
    disables a feature, and the readings that are NOT wanted here are "a relative path named ''"
    and "unset, so use the default".
    """
    monkeypatch.setenv("RECALL_EMBED_CACHE", word)
    assert default_cache_path() is None
    assert open_default_cache() is None


def test_an_explicit_path_is_honoured_and_expanded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "emb.sqlite"))
    assert default_cache_path() == tmp_path / "emb.sqlite"

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("RECALL_EMBED_CACHE", "~/named.sqlite")
    assert default_cache_path() == Path("~/named.sqlite").expanduser()


def test_unset_resolves_under_the_platform_cache_directory(monkeypatch, tmp_path: Path) -> None:
    """Unset means ON, in a directory the operating system already treats as expendable.

    Asserted through the platform's own variable rather than against a literal, because the point
    is that the file lands somewhere a user can delete without losing anything, not that it lands
    at one particular string.
    """
    monkeypatch.delenv("RECALL_EMBED_CACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    resolved = default_cache_path()

    assert resolved is not None
    assert resolved.parent.name == "recall"
    assert resolved.parent.parent in {tmp_path / "local", tmp_path / "xdg"}


def test_default_cache_closes_the_connection_however_the_block_ends(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "emb.sqlite"))
    with pytest.raises(ZeroDivisionError):
        with default_cache() as cache:
            assert cache is not None
            held = cache
            raise ZeroDivisionError

    # Asserted against the CONNECTION rather than through `get`, which would degrade and answer
    # None: the two states worth telling apart here are "closed" and "still holding the file",
    # and the public read path is deliberately unable to distinguish them.
    with pytest.raises(sqlite3.ProgrammingError):
        held._conn.execute("SELECT 1")


def test_default_cache_yields_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "0")
    with default_cache() as cache:
        assert cache is None


def test_an_unopenable_cache_is_reported_and_survived(monkeypatch, tmp_path: Path) -> None:
    """`open_default_cache` never raises: a cache is an optimisation, not a dependency.

    A directory where the file should be is the cheap portable way to make SQLite refuse; the
    behaviour under test is the refusal being turned into None, not which errno produced it.
    """
    blocked = tmp_path / "emb.sqlite"
    blocked.mkdir()
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(blocked))

    assert open_default_cache() is None


# ---- what it stores -----------------------------------------------------------------------


def test_vectors_are_stored_as_float32_blobs_not_json(tmp_path: Path) -> None:
    """The `vec` column is declared TEXT and must still hold a BLOB.

    SQLite gives a column affinity rather than a type, and TEXT affinity converts numbers to text
    but leaves blobs alone. That subtlety is what lets the new packed format share the column with
    the JSON rows an older recall wrote. If a future SQLite (or a future schema edit) started
    coercing, every write would silently become text and this file would grow five-fold, so the
    storage class is asserted, not assumed.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put(_key("alpha"), [1.5, -2.25])

    row = cache._conn.execute(
        "SELECT typeof(vec), length(vec) FROM embeddings LIMIT 1"
    ).fetchone()

    assert row[0] == "blob"
    assert row[1] == 8  # two float32s, not the 12+ characters JSON would have taken
    assert cache.get(_key("alpha")) == [1.5, -2.25]


def test_a_cache_file_written_by_an_older_recall_still_serves_hits(tmp_path: Path) -> None:
    """The upgrade must not silently re-embed everything already paid for.

    Written through raw SQL in the OLD shape — JSON text, no `used_at` column — because a fixture
    built by the current code could not express the file this is about.
    """
    path = tmp_path / "emb.sqlite"
    legacy = sqlite3.connect(str(path))
    legacy.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, vec TEXT NOT NULL)")
    legacy.execute(
        "INSERT INTO embeddings (key, vec) VALUES (?, ?)",
        (_key("alpha"), json.dumps([5.0, 1.0])),
    )
    legacy.commit()
    legacy.close()

    emb = CountingEmbedder()
    with EmbeddingCache(path) as cache:
        out = embed_with_cache(emb, ["alpha"], cache)

    assert out == [[5.0, 1.0]]
    assert emb.embedded == []  # served from the legacy row, not recomputed


def test_a_truncated_row_is_a_miss_rather_than_a_wrong_vector(tmp_path: Path) -> None:
    """A half-written blob must be recomputed. Returning it would be a plausible short vector."""
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put(_key("alpha"), [5.0, 1.0])
    cache._conn.execute(
        "UPDATE embeddings SET vec = ? WHERE key = ?", (b"\x00\x00\x00", _key("alpha"))
    )
    cache._conn.commit()

    emb = CountingEmbedder()
    assert embed_with_cache(emb, ["alpha"], cache) == [[5.0, 1.0]]
    assert emb.embedded == ["alpha"]  # recomputed rather than served short


# ---- what bounds it ------------------------------------------------------------------------


def test_the_least_recently_used_entries_are_evicted_past_the_cap(tmp_path: Path) -> None:
    """A default-on cache that grows without a bound is a disk-full incident in waiting.

    Recency is stamped explicitly rather than by writing in order and hoping: `used_at` has a
    one-hour rewrite window (see the next test), so within a single test every stamp written by
    `put` is effectively simultaneous and an eviction order inferred from write order would be a
    tie broken by key hash — green today, red on a rename.
    """
    # Filled UNBOUNDED, so the automatic sweep cannot fire while the fixture is being built and
    # evict an arbitrary entry before the recency stamps below are written. The cap is applied
    # afterwards, which is the state this test is about: a table already over its budget.
    cache = EmbeddingCache(tmp_path / "emb.sqlite", max_bytes=0)
    names = ["stale", "next-oldest", "b", "c", "d", "fresh"]
    for name in names:
        cache.put(_key(name), [1.0, 1.0])
    for rank, name in enumerate(names):
        cache._conn.execute(
            "UPDATE embeddings SET used_at = ? WHERE key = ?", (100.0 * (rank + 1), _key(name))
        )
    cache._conn.commit()

    cache._max_bytes = 40  # six 8-byte vectors held, a five-vector budget
    cache._evict_over_cap()  # the ORDER is under test here; the trigger is the next test

    assert cache.get(_key("stale")) is None
    assert cache.get(_key("next-oldest")) is None
    assert cache.get(_key("fresh")) == [1.0, 1.0]
    total = cache._conn.execute("SELECT COALESCE(SUM(length(vec)), 0) FROM embeddings").fetchone()
    assert total[0] <= 40


def test_writing_past_the_cap_triggers_the_sweep_without_being_asked(tmp_path: Path) -> None:
    """Eviction has to happen on its own, or the bound is documentation rather than a bound.

    Asserted on the total only, deliberately: which entries survive is the previous test's
    business, and at this timescale every `used_at` written by `put` is effectively simultaneous,
    so naming survivors here would be asserting a tie-break by key hash.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite", max_bytes=48)

    for index in range(12):  # 96 bytes written into a 48-byte cache
        cache.put(_key(f"k-{index}"), [float(index), 0.0])

    total = cache._conn.execute("SELECT COALESCE(SUM(length(vec)), 0) FROM embeddings").fetchone()
    assert 0 < total[0] <= 48


def test_reading_an_entry_refreshes_its_recency_once_the_stamp_is_stale(tmp_path: Path) -> None:
    """A hit counts as a use, so a long-lived entry that is still being read is not evicted.

    The refresh is deliberately not written on every hit: that would be one UPDATE per hit, which
    is write traffic proportional to the work the cache exists to avoid. The consequence is that
    recency has an hour's granularity, which is the property this pins from both sides.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    cache.put(_key("alpha"), [5.0, 1.0])
    stamp = lambda: cache._conn.execute(  # noqa: E731 - a local alias, not an exported helper
        "SELECT used_at FROM embeddings WHERE key = ?", (_key("alpha"),)
    ).fetchone()[0]

    written_at = stamp()
    assert cache.get(_key("alpha")) == [5.0, 1.0]
    assert stamp() == written_at  # inside the window: not rewritten

    cache._conn.execute("UPDATE embeddings SET used_at = 0 WHERE key = ?", (_key("alpha"),))
    cache._conn.commit()
    assert cache.get(_key("alpha")) == [5.0, 1.0]
    assert stamp() > 0  # outside the window: the hit is recorded


def test_max_bytes_zero_means_unbounded(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "emb.sqlite", max_bytes=0)
    for index in range(20):
        cache.put(_key(f"k-{index}"), [float(index), 0.0])
    rows = cache._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
    assert rows[0] == 20


@pytest.mark.parametrize("raw", ["banana", "-1", "1.5"])
def test_a_malformed_size_cap_falls_back_to_the_default(monkeypatch, raw: str) -> None:
    """Ignored and logged rather than clamped, for the reason `RECALL_INDEX_BATCH_CHUNKS` is:
    substituting a different bound for the one somebody configured is how a resource limit
    disappears while being believed."""
    monkeypatch.setenv("RECALL_EMBED_CACHE_MAX_MB", raw)
    assert _max_bytes_from_env() == DEFAULT_CACHE_MAX_MB * 1024 * 1024


# ---- how it fails --------------------------------------------------------------------------


def test_a_cache_that_cannot_be_written_still_serves_its_reads(tmp_path: Path) -> None:
    """A read-only cache file is a cache, not a broken one.

    A baked CI image or a shared read-only mount answers every lookup correctly and refuses every
    write. Giving up on reads at the first failed write would throw away exactly the hits such a
    file exists to serve, so writes are abandoned on their own.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()
    embed_with_cache(emb, ["alpha"], cache)

    # What a read-only database does to a write, without needing one: reads keep working.
    cache._conn.execute("PRAGMA query_only=ON")
    emb.embedded.clear()

    assert embed_with_cache(emb, ["alpha"], cache) == [[5.0, 1.0]]
    assert emb.embedded == []  # served from cache despite the write failing
    assert cache._writes_disabled and not cache._degraded

    assert embed_with_cache(emb, ["beta"], cache) == [[4.0, 1.0]]
    assert emb.embedded == ["beta"]  # a miss is still computed, just not written back


def test_a_broken_cache_degrades_to_no_cache_instead_of_failing_the_run(tmp_path: Path) -> None:
    """The property that makes default-on defensible: the worst case is a re-embed.

    The table is dropped underneath a live cache, which is what a corrupt file looks like from
    inside a call. Every later read must be a miss and every later write a drop, and the vectors
    the caller receives must still be correct, because they come from the embedder.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()
    embed_with_cache(emb, ["alpha"], cache)
    cache._conn.execute("DROP TABLE embeddings")
    cache._conn.commit()
    emb.embedded.clear()

    out = embed_with_cache(emb, ["alpha", "beta"], cache)

    assert out == [[5.0, 1.0], [4.0, 1.0]]
    assert emb.embedded == ["alpha", "beta"]  # nothing served, nothing raised


# ---- what it saves -------------------------------------------------------------------------


def test_a_text_repeated_in_one_call_is_embedded_once(tmp_path: Path) -> None:
    """A corpus repeats itself: a licence header, a template, the same memo filed twice.

    Before the misses were deduplicated by key, a cold call embedded every occurrence, which is
    spend on a question whose answer is already in flight.
    """
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()

    out = embed_with_cache(emb, ["same", "other", "same"], cache)

    assert emb.embedded == ["same", "other"]
    assert out == [[4.0, 1.0], [5.0, 1.0], [4.0, 1.0]]  # order preserved for the duplicate


def test_recall_index_hands_the_indexer_a_cache(monkeypatch, tmp_path: Path) -> None:
    """The user-facing claim: `recall index` re-embeds only what changed, without being asked to.

    Pinned at the CLI rather than inside `Indexer`, because `Indexer`'s own default is deliberately
    still `None`: a library caller (and every eval harness and benchmark in this repository)
    constructs one directly, and a cache appearing under a run that is MEASURING embedding cost
    would corrupt the measurement silently. The opt-in therefore lives at the entry points a person
    invokes, and this asserts that the busiest one has it.
    """
    from recall.cli_commands import index_search

    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "emb.sqlite"))
    seen: dict[str, object] = {}

    class _Store:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def check_schema(self):
            return None

    class _Indexer:
        def __init__(self, store, embedder, **kwargs):
            seen.update(kwargs)

        def index_path(self, path, glob=None):
            from recall.index import IndexStats

            return IndexStats(files=0, chunks=0)

    monkeypatch.setattr(index_search, "PgVectorStore", lambda *a, **k: _Store())
    monkeypatch.setattr(index_search, "Indexer", _Indexer)
    monkeypatch.setattr(index_search, "_make_embedder", lambda name: CountingEmbedder())
    monkeypatch.setattr(index_search, "head_commit", lambda path: None)

    index_search._cmd_index(
        SimpleNamespace(
            embedder="counting",
            dsn="postgresql://unused",
            table="chunks",
            tenant="t",
            path=str(tmp_path),
            glob="**/*.md",
            project=None,
            no_commit_stamp=True,
            batch_chunks=None,
            allow_prune=False,
        )
    )

    assert isinstance(seen.get("cache"), EmbeddingCache)


def test_hits_and_misses_are_counted_for_the_caller_to_report(tmp_path: Path) -> None:
    """Exposed so a build can say what the cache did. A cache that is silently never warm looks
    exactly like one that is working and expensive."""
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()

    embed_with_cache(emb, ["alpha", "beta"], cache)
    assert (cache.hits, cache.misses) == (0, 2)

    embed_with_cache(emb, ["alpha", "gamma"], cache)
    assert (cache.hits, cache.misses) == (1, 3)
