"""Regression tests for the deferred pass over the 2026-10-31 indexing, embedder and schema hunt.

Each test names the finding it pins (DEF-n, with the hunt's own id), the pre-fix baseline it was
proved red against, and the failure that produced. The baseline is `beb08a605` (master after
PR 1113) unless a mutation is named. The database tests are in tests/test_schema_migrations.py
and tests/test_generations.py; none here needs a database, a network or a model.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

from recall.cache import EmbeddingCache


def _cache(tmp_path: Path) -> EmbeddingCache:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite", max_bytes=1024 * 1024)
    cache.put_many([("k", [0.25, 0.5])])
    return cache


def test_a_failed_touch_ends_its_transaction(tmp_path: Path) -> None:
    """DEF-1 (CCA BUG-002): a write that fails inside the cache leaves no transaction open.

    Red against `beb08a605` (`EmbeddingCache._touch` stops writing without a rollback): node
    tests/test_hunt_index_schema_deferred.py::test_a_failed_touch_ends_its_transaction fails on
    "the failed touch left a transaction open".
    """
    cache = _cache(tmp_path)
    cache._conn.execute("UPDATE embeddings SET used_at = 0")
    cache._conn.commit()
    real = cache._conn

    class _FailingUpdate:
        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

        def executemany(self, sql: str, rows: object) -> object:
            real.execute("UPDATE embeddings SET used_at = used_at WHERE key = 'k'")
            raise sqlite3.OperationalError("database is locked")

    cache._conn = _FailingUpdate()  # type: ignore[assignment]
    try:
        assert cache.get_many(["k"]) == {"k": [0.25, 0.5]}
    finally:
        cache._conn = real
    assert not real.in_transaction, "the failed touch left a transaction open"
    cache.close()


def test_a_fresh_hit_takes_no_write_lock(tmp_path: Path) -> None:
    """DEF-9 (CCA PERF-002): a lookup whose stamp is fresh never asks for the write lock.

    Another connection holds the file's write lock. Red against `beb08a605` (`get_many` touches
    every hit): node tests/test_hunt_index_schema_deferred.py::test_a_fresh_hit_takes_no_write_lock
    fails on "a fresh hit queued behind another writer".
    """
    cache = _cache(tmp_path)
    cache._conn.execute("PRAGMA busy_timeout = 50")
    writer = sqlite3.connect(tmp_path / "embeddings.sqlite", isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        assert cache.get_many(["k"]) == {"k": [0.25, 0.5]}
        assert not cache._writes_disabled, "a fresh hit queued behind another writer"
    finally:
        writer.execute("ROLLBACK")
        writer.close()
        cache.close()


def test_eviction_reads_victims_from_the_index(tmp_path: Path) -> None:
    """DEF-10 (CCA PERF-003): eviction walks the `used_at` index instead of sorting the table.

    Red against `beb08a605` (`ORDER BY COALESCE(used_at, 0)`): node
    tests/test_hunt_index_schema_deferred.py::test_eviction_reads_victims_from_the_index fails on
    "eviction sorts the whole table".
    """
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite", max_bytes=64)
    seen: list[str] = []
    cache._conn.set_trace_callback(seen.append)
    cache.put_many([(f"k{i}", [float(i)] * 8) for i in range(8)])
    cache._evict_over_cap()
    cache._conn.set_trace_callback(None)
    query = next(sql for sql in seen if sql.startswith("SELECT key, length(vec)"))
    plan = " ".join(str(row[-1]) for row in cache._conn.execute(f"EXPLAIN QUERY PLAN {query}"))
    assert "TEMP B-TREE" not in plan, f"eviction sorts the whole table: {plan}"
    cache.close()


def test_an_unresolvable_home_disables_the_cache_instead_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEF-13 (CCA DEPLOY-002): `open_default_cache` never raises, a missing home included.

    Red against `beb08a605` (`default_cache_path()` outside the try): node
    tests/test_hunt_index_schema_deferred.py::test_an_unresolvable_home_disables_the_cache_instead_of_the_run
    fails with RuntimeError "Could not determine home directory".
    """
    import recall.cache as cache_module

    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(cache_module, "default_cache_path", no_home)
    assert cache_module.open_default_cache() is None


def test_retry_refuses_zero_attempts() -> None:
    """DEF-2 (CCA BUG-003): `retry_with_backoff` refuses an attempt count that would never call.

    Red against `beb08a605` (the loop never runs, then `assert last is not None`): node
    tests/test_hunt_index_schema_deferred.py::test_retry_refuses_zero_attempts fails with
    AssertionError instead of ValueError.
    """
    from recall.embedding_core import retry_with_backoff

    calls: list[int] = []
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        retry_with_backoff(lambda: calls.append(1), attempts=0)
    assert calls == []


class _Response:
    def __init__(self, width: int) -> None:
        self.embeddings = [[0.0] * width]


class _RateLimited(Exception):
    status_code = 429


def _fake_voyage(monkeypatch: pytest.MonkeyPatch, *, failures: int) -> list[dict[str, object]]:
    import recall.embedding_providers.voyage as voyage

    built: list[dict[str, object]] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            built.append(kwargs)
            self._failures = failures

        def embed(self, texts: list[str], model: str) -> _Response:
            if self._failures:
                self._failures -= 1
                raise _RateLimited("rate limited")
            return _Response(1024)

    monkeypatch.setattr(voyage, "_voyage_client_class", lambda owner: _Client)
    return built


def test_the_voyage_width_probe_survives_one_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEF-3 (CCA BUG-004): one transient 429 at construction is retried, not fatal.

    Red against `beb08a605` (the probe calls the client directly): node
    tests/test_hunt_index_schema_deferred.py::test_the_voyage_width_probe_survives_one_rate_limit
    fails with _RateLimited "rate limited".
    """
    from recall.embedding_providers.voyage import VoyageEmbedder

    _fake_voyage(monkeypatch, failures=1)
    assert VoyageEmbedder(api_key="test-key").dim == 1024


def test_a_registered_voyage_profile_takes_its_timeout_from_the_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEF-4 (CCA BUG-005): the plain Voyage backend reads the timeout from the env it is given.

    Red against `beb08a605` (`VoyageEmbedder` reads `os.environ`): node
    tests/test_hunt_index_schema_deferred.py::test_a_registered_voyage_profile_takes_its_timeout_from_the_mapping
    fails on "the mapping's timeout was ignored".
    """
    from recall.embedding_registry import registered_profile

    monkeypatch.delenv("RECALL_VOYAGE_TIMEOUT_SECONDS", raising=False)
    built = _fake_voyage(monkeypatch, failures=0)
    registered_profile("voyage-4-v1").build(
        api_key="test-key", env={"RECALL_VOYAGE_TIMEOUT_SECONDS": "5"}
    )
    assert built and built[0]["timeout"] == 5.0, "the mapping's timeout was ignored"


def test_a_bad_thread_cap_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """DEF-15 (CCA ENV-003): a malformed RECALL_EMBED_THREADS says the cap is off.

    Red against `beb08a605` (dropped without a log line): node
    tests/test_hunt_index_schema_deferred.py::test_a_bad_thread_cap_is_reported fails on
    "a bad thread cap was ignored silently".
    """
    from recall.embedding_providers.local import resolve_thread_budget

    with caplog.at_level(logging.WARNING):
        assert resolve_thread_budget({"RECALL_EMBED_THREADS": "four-" + str(time.time_ns())}) is None
    assert any("RECALL_EMBED_THREADS" in record.getMessage() for record in caplog.records), (
        "a bad thread cap was ignored silently"
    )


def test_recall_env_is_read_as_the_serving_route_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEF-18 (CCA ENV-006): GenerationManager strips RECALL_ENV like every other reader.

    Red against `beb08a605` (`.lower()` without `.strip()`): node
    tests/test_hunt_index_schema_deferred.py::test_recall_env_is_read_as_the_serving_route_reads_it
    fails with ValueError "environment must be development, test, or production".
    """
    from recall.generations import GenerationManager

    monkeypatch.setenv("RECALL_ENV", "production ")
    assert GenerationManager("postgresql://unused", "tenant-a").environment == "production"
    # Control: an exported empty value stays refused, as `recall.runtime_route` refuses it, because
    # reading it as development would switch off certification on promote.
    monkeypatch.setenv("RECALL_ENV", "")
    with pytest.raises(ValueError, match="RECALL_ENV"):
        GenerationManager("postgresql://unused", "tenant-a")


def test_a_direct_qwen3_construction_gets_the_registered_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEF-20 (CCA CODE-014): the registry is the only constructor of the Qwen3 profile's identity.

    Built directly with an artifact the registry did not pin, it is refused rather than minting a
    second identity under the registered id. Red against `beb08a605` (an inline copy of the
    profile's fields): node
    tests/test_hunt_index_schema_deferred.py::test_a_direct_qwen3_construction_gets_the_registered_identity
    fails on "DID NOT RAISE ValueError".
    """
    import recall.embedding_providers.local as local

    class _SentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(local, "verify_artifact", lambda path, digest: Path(path))
    with pytest.raises(ValueError):
        local.Qwen3EmbeddingEmbedder(str(tmp_path), "0" * 64, env={})
