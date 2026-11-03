"""Behavioral regressions for the historical PR109 through PR100 audit.

Red baseline: df339c87af5e9c077a563ef96a816f4ba5424a38. Each test names its
finding and production boundary; the audit receipts preserve the exact red/green
nodes and intended assertion failures. No model weights or paid API are used.
"""
from __future__ import annotations

import asyncio
import os
import stat
import sys
import threading
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest

from recall.atomic_write import atomic_write_bytes
from recall.embeddings import HashingEmbedder
from recall.eval.beir import materialize, safe_filename
from recall.eval.longmemeval_perq import populate_haystack
from recall.index import Indexer
from recall.lite.store import LiteStore
from recall.rerank import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from recall.types import Chunk
from recall_interop.memory_benchmarks import RecallBackend, _UserScope
from recall_mcp.service import forget_memory, withdraw_memory
from tests.conftest import TEST_DSN, requires_db
from tests.test_eval_beir import _write_beir


@requires_db
@pytest.mark.parametrize("storage", ["postgres", "lite"])
@pytest.mark.parametrize("withdraw", [False, True], ids=["forget", "withdraw"])
def test_ambiguous_erasure_alias_refuses_before_any_mutation(make_store, tmp_path, storage, withdraw):
    """STAKES-001: lifecycle must refuse an alias naming unrelated canonical sources.

    Baseline fails the error assertion after deleting both note.md sources.
    Canonical and unique-alias controls preserve the existing useful contract.
    """
    manager = (
        LiteStore(tmp_path / "memory.db", dim=3)
        if storage == "lite" else nullcontext(make_store(3))
    )
    with manager as store:
        store.upsert([
            Chunk("first", "/first/note.md", "first document", {"file": "note.md"}),
            Chunk("second", "/second/note.md", "unrelated document", {"file": "note.md"}),
            Chunk("unique", "/first/unique.md", "unique document", {"file": "unique.md"}),
        ], [[1.0, 0.0, 0.0]] * 3)
        erase = withdraw_memory if withdraw else forget_memory
        error = None
        try:
            erase(store, ["unique.md", "note.md"])
        except ValueError as exc:
            error = exc
        assert error is not None and "ambiguous" in str(error).lower(), (
            "ambiguous erasure proceeded instead of refusing the whole request"
        )
        assert store.count() == 3, "a refusal must leave even the unique batch member intact"
        assert "/first/note.md" in str(error), "the refusal must name an exact retry identifier"
        result = erase(store, ["/first/note.md"])
        assert result.chunks_removed == 1
        assert store.sources_for_identifiers(["note.md"])["note.md"] == ["/second/note.md"]
        result = erase(store, ["unique.md"])
        assert result.chunks_removed == 1 and store.count() == 1


def test_cancelled_add_cannot_overlap_close(tmp_path, monkeypatch):
    """BUG-001: an interrupted threaded add must finish before close touches its store.

    Baseline records close before the blocked index operation finishes. Two
    cancellations also reject a drain that is itself cancelled by a second request.
    """
    backend = RecallBackend("unused", embedder_name="hashing", workspace_root=tmp_path)
    started, release, finished, closed = (threading.Event() for _ in range(4))
    observed = []

    def index(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("test did not release the indexing worker")
        finished.set()

    def close():
        observed.append(finished.is_set())
        closed.set()

    backend._scopes["u"] = _UserScope("u", tmp_path, SimpleNamespace(close=close))
    monkeypatch.setattr(Indexer, "index_path", index)

    async def scenario():
        add = asyncio.create_task(backend.add([{"content": "a memory"}], "u"))
        assert await asyncio.to_thread(started.wait, 2), "indexing worker did not start"
        add.cancel()
        await asyncio.sleep(0)
        add.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(backend.close())
        try:
            await asyncio.to_thread(closed.wait, 0.25)
        finally:
            release.set()
        result = (await asyncio.gather(add, return_exceptions=True))[0]
        await closing
        assert await asyncio.to_thread(finished.wait, 2)
        assert isinstance(result, asyncio.CancelledError)

    asyncio.run(scenario())
    assert observed == [True], "cleanup overlapped the cancelled operation's live worker"


def test_search_queued_behind_delete_observes_the_deleted_user(tmp_path):
    """BUG-002: search decides membership inside serialization, after a queued deletion.

    Baseline returns KeyError from the public search task instead of an empty list.
    """
    backend = RecallBackend("unused", embedder_name="hashing", workspace_root=tmp_path)
    started, release = threading.Event(), threading.Event()

    def chunks():
        started.set()
        if not release.wait(5):
            raise RuntimeError("test did not release deletion")
        return []

    backend._scopes["u"] = _UserScope(
        "u", tmp_path, SimpleNamespace(iter_chunks=chunks, close=lambda: None)
    )

    async def scenario():
        deleting = asyncio.create_task(backend.delete_user("u"))
        assert await asyncio.to_thread(started.wait, 2)
        searching = asyncio.create_task(backend.search("query", "u"))
        try:
            await asyncio.sleep(0)
        finally:
            release.set()
        deleted, result = await asyncio.gather(deleting, searching, return_exceptions=True)
        await backend.close()
        assert deleted is True
        assert result == [], "the queued search dereferenced a user already removed by deletion"

    asyncio.run(scenario())


@requires_db
def test_incremental_backend_add_does_not_reconcile_prior_sources(make_store, tmp_path, monkeypatch):
    """PERF-001: per-turn ingestion must not read or stat every earlier document.

    Baseline records a growing tenant-wide source-hash read and a prune per add.
    This exercises the real adapter, indexer and PostgreSQL writes with hashing.
    """
    store = make_store(64)
    backend = RecallBackend(TEST_DSN, embedder_name="hashing", table=store.table, workspace_root=tmp_path)
    reads, prunes = [], []
    from recall.store import PgVectorStore

    original_hashes = PgVectorStore.source_content_hashes
    original_prune = Indexer._prune_vanished

    def hashes(target):
        result = original_hashes(target)
        reads.append(len(result))
        return result

    def prune(target, *args):
        prunes.append(True)
        return original_prune(target, *args)

    monkeypatch.setattr(PgVectorStore, "source_content_hashes", hashes)
    monkeypatch.setattr(Indexer, "_prune_vanished", prune)

    async def scenario():
        try:
            for i in range(3):
                await backend.add([{"content": f"separate memory number {i}"}], "u")
            assert backend._scopes["u"].store.count() == 3
        finally:
            await backend.close()

    asyncio.run(scenario())
    assert reads == [], "a per-turn append read the entire accumulated source-hash map"
    assert prunes == [], "a per-turn append reconciled earlier workspace files"


@requires_db
def test_unchanged_single_file_reindex_does_not_load_stored_text(make_store, tmp_path, monkeypatch):
    """PERF-001 control: an unchanged file retains its original compact hash-check path.

    Red production mutation: index.py SHA256
    241f1955c55f9f9621588e650c02201b0d448a9da9d6d5c76332cb7f83296fbd,
    the first repair's chunks_for_source branch. It transfers text before skipping.
    The intended red assertion is the recorded payload read, not setup or timing.
    """
    store = make_store(64)
    path = tmp_path / "memo.md"
    path.write_text("an unchanged memory", encoding="utf-8")
    indexer = Indexer(store, HashingEmbedder(dim=64))
    indexer.index_path(path)
    reads = []
    original = store.chunks_for_source

    def chunks(source):
        reads.append(source)
        return original(source)

    monkeypatch.setattr(store, "chunks_for_source", chunks)
    result = indexer.index_path(path)
    assert result.skipped == 1
    assert reads == [], "an unchanged single-file reindex fetched its full stored chunk payload"


def test_materialize_refuses_stale_documents_without_deleting_them(tmp_path):
    """BUG-003: a new manifest must not silently include an older, wider corpus.

    Baseline returns success with three text files after reporting a one-file corpus.
    Refusal preserves the earlier files instead of deleting caller-owned material.
    """
    source, output = tmp_path / "dataset", tmp_path / "output"
    _write_beir(source, {"gold": ("", "answer"), "a": ("", "a"), "b": ("", "b")},
                {"q": "question"}, [("q", "gold", 1)])
    materialize(source, output, max_docs=3)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    error = None
    try:
        materialize(source, output, max_docs=1)
    except ValueError as exc:
        error = exc
    assert error is not None, "a stale materialized corpus was reported as the new selection"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert materialize(source, output, max_docs=3)["documents_written"] == 3
    fresh = tmp_path / "fresh"
    assert materialize(source, fresh, max_docs=1)["documents_written"] == 1
    assert {path.name for path in fresh.glob("*.txt")} == {safe_filename("gold")}


def test_pinned_cross_encoder_refuses_a_local_model_shadow(tmp_path, monkeypatch):
    """SEC-001: the built-in Hub pin must not be overridden by a cwd model directory.

    Baseline invokes the recording loader instead of refusing. The loader double
    avoids importing model runtimes; controls preserve Hub and verified-local loads.
    """
    calls = []

    def loader(model, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=loader))
    monkeypatch.chdir(tmp_path)
    shadow = Path(DEFAULT_RERANKER_MODEL)
    shadow.mkdir(parents=True)
    error = None
    try:
        CrossEncoderReranker()
    except ValueError as exc:
        error = exc
    assert error is not None and "shadow" in str(error), "local weights bypassed the Hub pin"
    assert calls == [], "the refusal must precede the model loader"
    shadow.rmdir()
    CrossEncoderReranker()
    assert calls[-1][0] == DEFAULT_RERANKER_MODEL and calls[-1][1]["revision"]
    verified = tmp_path / "verified"
    verified.mkdir()
    verified_calls = []

    def verify(path, digest):
        verified_calls.append((path, digest))
        return path

    monkeypatch.setattr("recall.rerank.verify_artifact", verify)
    CrossEncoderReranker(str(verified), local_files_only=True, artifact_sha256="digest")
    assert verified_calls == [(verified, "digest")]
    assert calls[-1][1] == {"revision": None, "local_files_only": True}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() != 0, reason="needs POSIX ownership privileges")
def test_atomic_replacement_preserves_owner_group_and_special_mode(tmp_path):
    """DAT-001: replacing a memo retains uid/gid and mode, in that order.

    Baseline leaves the replacement owned by root; ownership-before-mode also
    prevents chown from stripping setuid/setgid bits restored too early.
    """
    path = tmp_path / "memo.md"
    path.write_bytes(b"before")
    os.chown(path, 12345, 12346)
    path.chmod(0o6750)
    atomic_write_bytes(path, b"after")
    info = path.stat()
    assert (info.st_uid, info.st_gid) == (12345, 12346), "atomic replacement changed ownership"
    assert stat.S_IMODE(info.st_mode) == 0o6750
    assert path.read_bytes() == b"after"


@pytest.mark.skipif(os.name != "posix" or os.geteuid() != 0, reason="needs POSIX ownership privileges")
def test_unpreservable_ownership_leaves_the_original_intact(tmp_path, monkeypatch):
    """DAT-001: metadata preservation failure must refuse before replacing the memo.

    Baseline never attempts chown and overwrites the original instead of refusing.
    """
    path = tmp_path / "memo.md"
    path.write_bytes(b"before")
    os.chown(path, 12345, 12346)
    inode = path.stat().st_ino

    def denied(*args, **kwargs):
        raise PermissionError("cannot preserve ownership")

    monkeypatch.setattr(os, "chown", denied)
    error = None
    try:
        atomic_write_bytes(path, b"after")
    except PermissionError as exc:
        error = exc
    assert error is not None, "unpreservable ownership did not refuse replacement"
    assert path.read_bytes() == b"before" and path.stat().st_ino == inode
    assert list(tmp_path.iterdir()) == [path], "a failed write leaked its staged file"


@requires_db
def test_equal_fingerprint_haystack_refill_invalidates_supersession(make_store):
    """DAT-002: a refilled scratch graph describes its current rows, not the previous haystack.

    Baseline keeps old.md superseded after new.md leaves an equal-count/timestamp
    haystack. One upsert transaction supplies an identical timestamp for all rows.
    """
    master = make_store(3)
    master.upsert([
        Chunk("old", "/corpus/old.md", "old", {"file": "old.md"}),
        Chunk("new", "/corpus/new.md", "new", {"file": "new.md", "supersedes": "old.md"}),
        Chunk("other", "/corpus/other.md", "other", {"file": "other.md"}),
    ], [[1.0, 0.0, 0.0]] * 3)
    scratch = "sc_" + uuid.uuid4().hex[:8]
    sub = populate_haystack(TEST_DSN, 3, master.table, scratch, ["old.md", "new.md"])
    try:
        assert "old.md" in sub.supersession()[0], "the first haystack must populate the cache"
        with psycopg.connect(TEST_DSN) as conn:
            first = conn.execute(f"SELECT max(indexed_at), count(*) FROM {scratch}").fetchone()
        populate_haystack(TEST_DSN, 3, master.table, scratch, ["old.md", "other.md"], store=sub)
        with psycopg.connect(TEST_DSN) as conn:
            second = conn.execute(f"SELECT max(indexed_at), count(*) FROM {scratch}").fetchone()
        assert first == second, "the fixture must exercise the cache fingerprint collision"
        assert "old.md" not in sub.supersession()[0], "the removed superseder survived in cache"
    finally:
        sub.drop_table()
        sub.close()
