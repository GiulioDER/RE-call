"""The compile cache: a paid reply is never paid for twice, and a forgotten source leaves no row.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.cache``):

- ``test_round_trip_and_empty_result_cached``: ``get`` returning None for an empty record list
  fails on ``== []``.
- ``test_key_changes_with_model_prompt_and_body``: the model dropped from the key fails.
- ``test_key_ignores_the_api_key``: the key folded into the cache key fails.
- ``test_damaged_row_is_a_miss``: span offsets accepted without the integer check fails on the
  ``bad_span`` row (a record comes back instead of a miss).
- ``test_unwritable_cache_counts_and_never_raises``: the ``sqlite3.Error`` handler removed fails
  with an exception.
- ``test_erase_sources_removes_every_row``: erasure by key instead of by source fails on the count.
- ``test_erased_rows_leave_no_bytes_in_the_file``: ``secure_delete`` off fails on the file bytes.
- ``test_unreadable_cache_file_degrades_and_forget_removes_it``: the ``unlink`` removed fails on
  ``not path.exists()``.
- ``test_key_covers_every_constant_that_shapes_records``: ``MAX_OUTPUT_TOKENS`` dropped from the
  key fails for that name.
- ``test_key_uses_the_whole_base_url_without_credentials``: the raw ``base_url`` in the key fails
  the credentials half; the host alone fails the path half.
"""

from __future__ import annotations

import sqlite3

import pytest
from pathlib import Path

from recall.compiled_graph.cache import CompileCache, cache_key
from recall.compiled_graph.provider import ProviderConfig
from recall.compiled_graph.records import CompiledRecord

CONFIG = ProviderConfig(api_key="sk-test-a", model="openai/gpt-6-luna-pro")
RECORD = CompiledRecord(
    kind="fact", fields=("Alice owns it.", ""), entities=("Alice",), anchor_ids=("a000_x",),
    spans=((0, 1_600),),
)


def test_round_trip_and_empty_result_cached(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "compile-cache.sqlite")
    key = cache_key(CONFIG, "notes/a.md", "0" * 64)
    assert cache.get(key) is None
    cache.put(key, "notes/a.md", [RECORD])
    assert cache.get(key) == [RECORD]
    empty = cache_key(CONFIG, "notes/b.md", "1" * 64)
    cache.put(empty, "notes/b.md", [])
    assert cache.get(empty) == []


def test_key_changes_with_model_prompt_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    base = cache_key(CONFIG, "notes/a.md", "0" * 64)
    other_model = ProviderConfig(api_key="sk-test-a", model="openai/gpt-4o-mini")
    assert cache_key(other_model, "notes/a.md", "0" * 64) != base
    assert cache_key(CONFIG, "notes/a.md", "1" * 64) != base
    assert cache_key(CONFIG, "notes/b.md", "0" * 64) != base
    other_host = ProviderConfig(api_key="sk-test-a", model=CONFIG.model, base_url="http://localhost:9/v1")
    assert cache_key(other_host, "notes/a.md", "0" * 64) != base
    monkeypatch.setattr("recall.compiled_graph.cache.prompt_digest", lambda: "f" * 64)
    assert cache_key(CONFIG, "notes/a.md", "0" * 64) != base


def test_key_ignores_the_api_key() -> None:
    rotated = ProviderConfig(api_key="sk-test-b", model=CONFIG.model)
    assert cache_key(rotated, "notes/a.md", "0" * 64) == cache_key(CONFIG, "notes/a.md", "0" * 64)


def test_damaged_row_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "compile-cache.sqlite"
    cache = CompileCache(path)
    key = cache_key(CONFIG, "notes/a.md", "0" * 64)
    cache.put(key, "notes/a.md", [RECORD])
    bad_span = '[{"kind": "fact", "fields": [], "entities": [], "anchor_ids": [], "spans": [[0, "x"]]}]'
    for damaged in ("not json", '[{"kind": 1}]', '{"records": []}', bad_span):
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE compiled SET records = ? WHERE key = ?", (damaged, key))
        assert cache.get(key) is None


def test_unwritable_cache_counts_and_never_raises(tmp_path: Path) -> None:
    path = tmp_path / "compile-cache.sqlite"
    cache = CompileCache(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE compiled")
        conn.execute("CREATE TABLE compiled (key TEXT)")
    cache.put(cache_key(CONFIG, "notes/a.md", "0" * 64), "notes/a.md", [RECORD])
    assert cache.write_failures == 1


def test_erase_sources_removes_every_row(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "compile-cache.sqlite")
    for body in ("0", "1"):
        cache.put(cache_key(CONFIG, "notes/a.md", body * 64), "notes/a.md", [RECORD])
    cache.put(cache_key(CONFIG, "notes/b.md", "0" * 64), "notes/b.md", [RECORD])
    assert cache.erase_sources(["notes/a.md"]) == 2
    assert cache.get(cache_key(CONFIG, "notes/a.md", "0" * 64)) is None
    assert cache.get(cache_key(CONFIG, "notes/b.md", "0" * 64)) == [RECORD]


def test_unreadable_cache_file_degrades_and_forget_removes_it(tmp_path: Path) -> None:
    path = tmp_path / "compile-cache.sqlite"
    path.write_bytes(b"this is not a database at all " * 100)
    cache = CompileCache(path)
    key = cache_key(CONFIG, "notes/a.md", "0" * 64)
    assert cache.get(key) is None
    cache.put(key, "notes/a.md", [RECORD])
    assert cache.write_failures == 1
    assert cache.erase_sources(["notes/a.md"]) == 0
    assert not path.exists()


def test_key_covers_every_constant_that_shapes_records(monkeypatch: pytest.MonkeyPatch) -> None:
    base = cache_key(CONFIG, "notes/a.md", "0" * 64)
    for module, name in (
        ("recall.compiled_graph.anchors", "ANCHOR_CHARS"),
        ("recall.compiled_graph.anchors", "ANCHOR_OVERLAP_CHARS"),
        ("recall.compiled_graph.records", "MAX_RECORDS"),
        ("recall.compiled_graph.records", "MAX_ENTITIES"),
        ("recall.compiled_graph.records", "BACKFILL_CHARS"),
        ("recall.compiled_graph.provider", "MAX_OUTPUT_TOKENS"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(f"{module}.{name}", 7)
            assert cache_key(CONFIG, "notes/a.md", "0" * 64) != base, name


def test_key_uses_the_whole_base_url_without_credentials() -> None:
    def key(url: str) -> str:
        return cache_key(ProviderConfig(api_key="k", model="m", base_url=url), "s", "0" * 64)

    assert key("https://gw.example.test/a/v1") != key("https://gw.example.test/b/v1")
    assert key("https://user:pw@gw.example.test/a/v1") == key("https://gw.example.test/a/v1")


def test_erased_rows_leave_no_bytes_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "compile-cache.sqlite"
    cache = CompileCache(path)
    cache.put(cache_key(CONFIG, "notes/a.md", "0" * 64), "notes/a.md", [RECORD])
    cache.erase_sources(["notes/a.md"])
    assert b"Alice owns it." not in path.read_bytes()
