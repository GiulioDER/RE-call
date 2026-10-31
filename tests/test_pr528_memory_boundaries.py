"""Agent SDK instance boundaries, with offline transport and owned test stores.

Red proof on pre-fix recall_agent/memory.py: SEC-002's root and production
cases fail because indexing returns without refusal; BUG-001's cancellation
case fails because a second store is constructed before the first is released.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import recall_agent.memory as memory_mod
from recall_agent import RecallAgentMemory
from recall_mcp.factories import make_embedder
from recall_mcp.indexing import IndexPreflightError
from tests.conftest import requires_db


@requires_db
@pytest.mark.parametrize("boundary", ["root", "production"])
def test_indexing_uses_the_instance_environment(make_store, tmp_path, monkeypatch, boundary) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("a source outside the configured root", encoding="utf-8")
    monkeypatch.setenv("RECALL_ENV", "development")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    env = {"RECALL_ENV": "development", "RECALL_INDEX_ROOT": str(root)}
    expected = "outside the directory"
    if boundary == "production":
        env = {"RECALL_ENV": "production", "RECALL_INDEX_ROOT": str(tmp_path)}
        expected = "development-only"
    store = make_store(64)
    memory = RecallAgentMemory.from_env(env, store=store, embedder=make_embedder("hashing"))
    refusal = None
    try:
        asyncio.run(memory._recall_index({"path": str(target)}))
    except IndexPreflightError as exc:
        refusal = str(exc)
    assert refusal is not None, f"the instance {boundary} boundary was ignored"
    assert expected in refusal
    assert store.count() == 0
    if boundary == "root":
        permitted = root / "inside.md"
        permitted.write_text("an allowed source", encoding="utf-8")
        asyncio.run(memory._recall_index({"path": str(permitted)}))
        assert store.count() > 0


def test_cancelled_hook_keeps_worker_serialization(monkeypatch) -> None:
    ready = threading.Event()
    release = threading.Event()
    second_created = threading.Event()
    stores = []

    class Store:
        closed = False

        def check_schema(self):
            if len(stores) == 1:
                ready.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release the first worker")
            else:
                second_created.set()

        def close(self):
            self.closed = True

    def make_store():
        store = Store()
        stores.append(store)
        return store

    memory = RecallAgentMemory(embedder=make_embedder("hashing"))
    monkeypatch.setattr(memory, "_make_store", make_store)
    monkeypatch.setattr(memory_mod, "memory_stats", lambda store: SimpleNamespace(chunks=0))

    async def exercise():
        first = asyncio.create_task(memory._session_start({}, None, None))
        second = None
        try:
            assert await asyncio.to_thread(ready.wait, 5), "first worker did not enter"
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(memory._session_start({}, None, None))
            overlapped = await asyncio.to_thread(second_created.wait, 0.25)
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        assert not overlapped, "a second store was created while the cancelled worker was active"
        assert len(stores) == 1

    try:
        asyncio.run(exercise())
    finally:
        memory.close()
    assert all(store.closed for store in stores)
