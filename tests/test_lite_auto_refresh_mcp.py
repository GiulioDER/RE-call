"""The MCP server refreshes a lite store on its own (`recall_mcp.lite_refresh`).

- R9 `before_retrieval` indexes a memo written since the last check before it returns, and a second
  call inside the interval does not look at the folder at all.
- R9b a failing check never fails the retrieval that asked for it.
- R10 the real lifespan runs a catch-up refresh at start, a search refreshes first, and
  `recall_stats` reports the last refresh.

Red proof, 2026-10-12, on Linux: each mutation alone, the named test failing in its own assertion,
then restored and green (3 passed):
- M9a (R9) `before_retrieval`: the interval check made `if False:`: "a check inside the interval walked the folder"
- M9b (R9) `before_retrieval`: no bounded wait (`self._kick()` without `join`): "the retrieval ran before
  the new memo was indexed"
- M9c (R9b) `before_retrieval`: catches only ValueError: "the check raised into the retrieval: OSError"
- M10a (R10) lifespan: `lite_refresher.start()` made `pass`: "no catch-up refresh at start"
- M10b (R10) `_run_retrieval_tool.execute`: `before_retrieval()` made `pass`: "the search did not refresh first"
- M10c (R10) `recall_stats`: the `refresh` field not filled: "recall_stats has no refresh"
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import recall_mcp.lite_refresh as auto
from recall.lite import LiteStore
from recall_mcp.lite_refresh import LiteAutoRefresher
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_mcp import _serve, lite_env  # noqa: F401  # the fixture is used by name


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[LiteStore, Path, Clock]]:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "memory"
    _write_memos(root, 3)
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as store:
        yield store, root, Clock()


def _refresher(store: LiteStore, root: Path, clock: Clock) -> LiteAutoRefresher:
    return LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=30, wait_s=30, clock=clock)  # type: ignore[arg-type]


def _sources(store: LiteStore) -> set[str]:
    return {Path(c.source).name for c in store.iter_chunks()}


def test_a_new_memo_is_indexed_before_the_retrieval_and_checks_are_debounced(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9."""
    store, root, clock = setup
    refresher = _refresher(store, root, clock)
    refresher.start()
    refresher._kick().join(30)
    assert "memo-002.md" in _sources(store)

    _write_memos(root, 1, start=3)
    refresher.before_retrieval()
    assert "memo-003.md" in _sources(store), "the retrieval ran before the new memo was indexed"

    walks: list[int] = []
    real = auto.snapshot
    monkeypatch.setattr(auto, "snapshot", lambda roots: walks.append(1) or real(roots))
    _write_memos(root, 1, start=4)
    clock.now += 10
    refresher.before_retrieval()
    assert walks == [], "a check inside the interval walked the folder"
    assert "memo-004.md" not in _sources(store)
    clock.now += 30
    refresher.before_retrieval()
    assert walks and "memo-004.md" in _sources(store)


def test_a_failing_check_never_fails_the_retrieval(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9b."""
    store, root, clock = setup

    def broken(roots: object) -> None:
        raise OSError("the disk went away")

    monkeypatch.setattr(auto, "snapshot", broken)
    try:
        _refresher(store, root, clock).before_retrieval()
        raised = ""
    except Exception as exc:  # noqa: BLE001  # asserted below
        raised = f"{type(exc).__name__}: {exc}"
    assert raised == "", f"the check raised into the retrieval: {raised}"


def test_the_server_refreshes_at_start_and_before_a_search(lite_env: dict[str, str]) -> None:  # noqa: F811
    """R10: RECALL_INDEX_ROOT is the folder; the store sits in a subfolder of it."""
    root = Path(lite_env["RECALL_INDEX_ROOT"])
    _write_memos(root, 3)
    env = {**lite_env, "RECALL_LITE_REFRESH_INTERVAL": "0", "RECALL_LITE_REFRESH_WAIT": "30"}

    async def body(state: dict[str, Any], call: Any) -> None:
        refresher = state["lite_refresher"]
        assert refresher is not None, "a lite server with a memo folder has no refresher"
        if refresher._thread is not None:  # never `_kick()` here: that would start the catch-up itself
            refresher._thread.join(30)
        stats = json.loads(await call("recall_stats"))
        assert stats["chunks"] > 0, f"no catch-up refresh at start: {stats}"
        assert (stats.get("refresh") or {}).get("outcome") == "refreshed", f"recall_stats has no refresh: {stats}"

        _write_memos(root, 1, start=3)
        await call("recall_search", query="serviced every spring")
        stats = json.loads(await call("recall_stats"))
        assert stats["refresh"]["indexed_files"] == 1, f"the search did not refresh first: {stats['refresh']}"

    _serve(env, body)
