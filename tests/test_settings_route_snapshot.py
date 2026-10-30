"""PR610 route consumers must use the same immutable environment as their configuration.

Red proof: on 3ada1afd0, both lifespan cases select the ambient route and both indexing
cases either write through a supplied generation refusal or reject supplied legacy ingestion.
The unchanged nodes pass after the two resolve_runtime_route calls receive their snapshots.
"""
from __future__ import annotations

import asyncio

import pytest

from recall.embeddings import HashingEmbedder
from recall.lite import LiteStore
from recall_mcp import server
from recall_mcp.indexing import IndexPreflightError, index_memory
from recall_mcp.settings import Settings


@pytest.mark.parametrize("mode", ["legacy", "generation"])
def test_lifespan_route_uses_supplied_snapshot(tmp_path, monkeypatch, mode) -> None:
    monkeypatch.setenv("RECALL_ENV", "development")
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation" if mode == "legacy" else "legacy")
    env = {
        "RECALL_DSN": "sqlite:///" + (tmp_path / "memory.db").as_posix(),
        "RECALL_ENV": "development",
        "RECALL_INDEX_MODE": mode,
        "RECALL_EMBEDDER": "hashing",
        "RECALL_MCP_TOOLS": "search",
        "RECALL_LITE_AUTO_REFRESH": "0",
        "RECALL_INDEX_ROOT": str(tmp_path),
        "RECALL_EMBED_CACHE": "off",
    }
    observed = []

    async def drive() -> None:
        async with server._make_lifespan(None, settings=Settings.from_env(env))(None) as state:
            observed.append(state["route"].mode)
            assert isinstance(state["store"], LiteStore)

    error = None
    try:
        asyncio.run(drive())
    except RuntimeError as exc:
        error = str(exc)
    if mode == "generation":
        assert error and "generation" in error, "the supplied generation route must refuse SQLite"
        assert observed == []
    else:
        assert error is None, "the ambient generation route must not override the legacy snapshot"
        assert observed == ["legacy"]


@pytest.mark.parametrize("mode", ["legacy", "generation"])
def test_indexing_route_uses_supplied_snapshot(tmp_path, monkeypatch, mode) -> None:
    monkeypatch.setenv("RECALL_ENV", "development")
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation" if mode == "legacy" else "legacy")
    memo = tmp_path / "memo.md"
    memo.write_text("The deployment decision uses an immutable configuration snapshot.", encoding="utf-8")
    store = LiteStore(tmp_path / "memory.db", dim=64)
    result = None
    error = None
    try:
        try:
            result = index_memory(
                store, HashingEmbedder(dim=64), str(memo),
                env={"RECALL_ENV": "development", "RECALL_INDEX_MODE": mode,
                     "RECALL_INDEX_ROOT": str(tmp_path), "RECALL_EMBED_CACHE": "off"},
            )
        except IndexPreflightError as exc:
            error = str(exc)
        if mode == "generation":
            assert error and "generation" in error, "the supplied generation route must refuse legacy indexing"
            assert store.count() == 0
        else:
            assert error is None, "ambient generation mode must not reject a legacy snapshot"
            assert result is not None and result.chunks == 1
            assert store.count() == 1
    finally:
        store.close()
