"""A forget on a lite store erases what the opt-in atomic rescue keeps of the forgotten memo.

Invariant: after `forget_memory` or `withdraw_memory` (the funnel `recall forget`, `recall withdraw`
and the MCP `recall_forget` and `recall_withdraw` tools share) on a lite store whose
RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is set, `<root>/micro-views.sqlite` holds no view row of a
forgotten chunk's text under any profile, no byte of those vectors survives in that file or its
write-ahead log, and no artifact directory built from a corpus that held the memo is left (each
one keeps the memo's view vectors in `matrix.npy` and its source in `views.json`). The views of
memos that were not forgotten stay, and the root is erased whether or not the rescue is still
active, since a rescue turned off keeps what it built.

Failure mode it catches: the view store is keyed by chunk text and had no delete path, and
nothing in the forget path touched the root, so a forgotten memo's views and the artifact built
from them stayed on disk after every store-side erasure had run.

Negative controls: each test first asserts that the rows, bytes and artifact it expects gone are
present. The byte scan turns `secure_delete` off at every connect, because Debian and Ubuntu build
SQLite with it on by default (the test host and CI), where a plain DELETE would zero the cells and
the scan could not fail. It probes distinctive values written into the memo's own view rows
(`_mark`): a first version scanned for the stored blob and passed with `secure_delete` removed,
because that blob is mostly zero bytes split across overflow pages and was absent before too.

Red proof, on the test host (Ubuntu, Python 3.12), against this commit's parent `04c949e25` (the
head of `claude/forget-erases-embedding-cache`, PR 1307, which has no `erase_forgotten`), so with
the forget path as it was: `::test_forget_erases_the_memos_rescue_views_and_artifacts[forget]`,
`[withdraw]` and `::test_a_rescue_turned_off_still_has_its_views_erased` fail at "the forgotten
memo's views are still in the view store"; `::test_a_rescue_that_cannot_be_erased_is_reported`
fails at its message assertion, the message having no warning. Mutations of the fix, each red:
the `secure_delete = ON` of `recall.atomic_view_store.AtomicViewStore.erase_digests` removed
(alone, or with its checkpoint) fails both parametrized cases at "a forgotten view vector's bytes
survive"; the `rmtree` loop of `recall.lite.atomic.erase_forgotten` removed fails the turned-off
case at "an artifact built with the memo is left" and the parametrized cases at their no-WARNING
assertion, since the left-behind check then raises; `rescue_views_erased = -1` in the `except` of
`recall_mcp.lifecycle._forget_memory_unlocked` replaced by `pass` fails the reporting test.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from recall.atomic_rescue import clear_atomic_rescue_artifact_cache
from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
from recall.lite import LiteStore
from recall.lite.atomic import VIEW_STORE_NAME, ensure_rescue_artifact
from recall_mcp.lifecycle import forget_memory, withdraw_memory
from tests.test_lite_atomic_rescue import _certified, _env

ROOT_ENV = "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"


@pytest.fixture(autouse=True)
def _fresh_artifact_cache() -> None:
    clear_atomic_rescue_artifact_cache()


@pytest.fixture
def no_default_secure_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = sqlite3.connect

    def connect_without_secure_delete(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        conn = connect(*args, **kwargs)
        conn.execute("PRAGMA secure_delete = OFF")
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect_without_secure_delete)


def _built(tmp_path: Path) -> tuple[LiteStore, Path, str, list[str]]:
    """A certified store with its artifact built; the needle memo's source and chunk texts."""
    root = tmp_path / "atomic"
    store, embedder = _certified(tmp_path)
    report = ensure_rescue_artifact(store, embedder, env=_env(root))
    assert report is not None and report.manifest is not None, report
    chunks = [chunk for chunk in store.iter_chunks() if Path(chunk.source).name == "ops-log.md"]
    assert chunks, "the fixture has no needle memo"
    return store, root, chunks[0].source, [chunk.text for chunk in chunks]


def _views(root: Path) -> dict[str, bytes]:
    with closing(sqlite3.connect(root / VIEW_STORE_NAME)) as conn:
        return {str(digest): bytes(blob) for digest, blob in conn.execute("SELECT digest, vectors FROM micro_views")}


def _file_bytes(path: Path) -> bytes:
    return b"".join(p.read_bytes() for p in (path, Path(f"{path}-wal")) if p.exists())


def _mark(root: Path, digests: set[str]) -> list[bytes]:
    """Rewrite these view rows with distinctive values of the same shape; 16-byte probes of them.

    The test embedder's views are sparse 2048-wide vectors, mostly zero bytes and spread over
    overflow pages, so a scan for the stored blob itself finds nothing before the forget either.
    The rows keep their profile and digest key, which is all the erasure reads.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    probes: list[bytes] = []
    with closing(sqlite3.connect(root / VIEW_STORE_NAME)) as conn:
        rows = conn.execute(
            "SELECT profile, digest, view_count, dimension FROM micro_views WHERE digest IN "
            f"({', '.join('?' * len(digests))})",
            sorted(digests),
        ).fetchall()
    assert rows, "negative control: no view row to mark"
    with AtomicViewStore(root / VIEW_STORE_NAME) as views:
        for profile, digest, count, dimension in rows:
            blob = rng.uniform(1.0, 2.0, (count, dimension)).astype(np.float32)
            views.put(profile, digest, blob)
            data = blob.tobytes()
            probes.extend(data[i : i + 16] for i in range(0, len(data) - 16, 512))
        views.commit()
    return probes


def _artifacts(root: Path) -> list[Path]:
    return [entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith("lite-")]


@pytest.mark.parametrize("erase", [forget_memory, withdraw_memory], ids=["forget", "withdraw"])
def test_forget_erases_the_memos_rescue_views_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_default_secure_delete: None, erase
) -> None:
    store, root, source, texts = _built(tmp_path)
    for name, value in _env(root).items():
        monkeypatch.setenv(name, value)
    gone = {chunk_view_digest(text) for text in texts}
    before = _views(root)
    assert gone <= set(before), "negative control: the memo's views were never stored"
    kept = set(before) - gone
    assert kept, "negative control: no other memo has views"
    fragments = _mark(root, gone)
    present = sum(fragment in _file_bytes(root / VIEW_STORE_NAME) for fragment in fragments)
    assert present >= 0.9 * len(fragments), f"negative control: only {present} of {len(fragments)} fragments found"
    built = _artifacts(root)
    assert built and any("ops-log.md" in (entry / "views.json").read_text(encoding="utf-8") for entry in built), (
        "negative control: the artifact does not name the memo"
    )

    result = erase(store, [source])

    assert result.chunks_removed > 0, result.message
    assert "WARNING" not in result.message, result.message
    after = _views(root)
    assert gone & set(after) == set(), "the forgotten memo's views are still in the view store"
    assert kept <= set(after), "a memo that was not forgotten lost its views"
    raw = _file_bytes(root / VIEW_STORE_NAME)
    assert [f for f in fragments if f in raw] == [], "a forgotten view vector's bytes survive"
    assert [entry for entry in built if entry.exists()] == [], "an artifact built with the memo is left"
    store.close()


def test_a_rescue_turned_off_still_has_its_views_erased(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, root, source, texts = _built(tmp_path)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.setenv(ROOT_ENV, str(root))
    gone = {chunk_view_digest(text) for text in texts}
    assert gone <= set(_views(root)), "negative control: the memo's views were never stored"

    result = forget_memory(store, [source])

    assert result.chunks_removed > 0, result.message
    assert gone & set(_views(root)) == set(), "the forgotten memo's views are still in the view store"
    assert _artifacts(root) == [], "an artifact built with the memo is left"
    store.close()


def test_a_rescue_that_cannot_be_erased_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, root, source, _texts = _built(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(root))

    def refuse(self, digests):  # noqa: ANN001, ANN202
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(AtomicViewStore, "erase_digests", refuse, raising=False)

    result = forget_memory(store, [source])

    assert result.chunks_removed > 0, result.message
    assert "erasing them from the atomic rescue failed" in result.message, result.message
    store.close()
