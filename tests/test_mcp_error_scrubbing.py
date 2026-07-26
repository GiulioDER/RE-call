"""Errors raised deep in the library must not carry server paths out through the MCP boundary.

The confinement refusal was fixed earlier, but two errors reach an MCP client from further down
and still embedded the resolved root: `PruneGuardTripped` names the directory it refused to prune,
and the all-candidates-vanished `FileNotFoundError` names the root it scanned. Both are excellent
CLI diagnostics — the operator wants that path — and both are a filesystem map when the caller is
a remote tenant.

So the scrubbing belongs at the boundary, not in the library: `recall/index.py` keeps saying
exactly what it means, and `recall_mcp.service` redacts the server-side path on its way out while
logging the untouched message for the operator. That way the CLI loses nothing and the fix cannot
be undone by someone improving an error message later.

What must SURVIVE scrubbing is the actionable part. A refusal that hides both the cause and the
remedy is worse than the disclosure it prevents.
"""
from __future__ import annotations

import pytest

from recall.index import PruneGuardTripped
from recall_mcp.service import index_memory


class _VanishedCorpusStore:
    """A store that believes several sources are indexed which are not on disk.

    That is the exact state the prune guard exists for — an unmounted volume or an interrupted
    sync, indistinguishable at the filesystem level from "the author deleted everything".
    """

    def __init__(self, sources: list[str]) -> None:
        self._sources = sources
        self.deleted: list[str] = []

    def source_content_hashes(self):
        return {s: "deadbeef" for s in self._sources}

    def delete_sources(self, sources):
        self.deleted.extend(sources)
        return len(sources)

    def replace_sources(self, sources, chunks, embeddings):
        return len(chunks)

    def analyze_if_stale(self, modified):
        return True


class _NullEmbedder:
    dim = 2
    name = "null"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _corpus_with_vanished_sources(tmp_path):
    """One real memo plus six sources the store thinks exist — a >50% disappearance."""
    root = tmp_path / "server-side-corpus-directory"
    root.mkdir()
    (root / "still_here.md").write_text("# a memo\n")
    gone = [str(root / f"gone_{i}.md") for i in range(6)]
    return root, _VanishedCorpusStore([str(root / "still_here.md"), *gone])


def test_the_prune_guard_still_fires_through_the_mcp_boundary(tmp_path, monkeypatch):
    """Scrubbing must not swallow the refusal — the guard is the point."""
    root, store = _corpus_with_vanished_sources(tmp_path)
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))

    with pytest.raises((PruneGuardTripped, ValueError)):
        index_memory(store, _NullEmbedder(), str(root))

    assert store.deleted == [], "nothing may be deleted when the guard trips"


def test_the_prune_refusal_does_not_leak_the_server_path(tmp_path, monkeypatch):
    root, store = _corpus_with_vanished_sources(tmp_path)
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))

    with pytest.raises(Exception) as exc:
        index_memory(store, _NullEmbedder(), str(root))

    message = str(exc.value)
    assert "server-side-corpus-directory" not in message, "the server's path leaked to the client"
    assert str(root) not in message


def test_the_remedy_survives_scrubbing(tmp_path, monkeypatch):
    """A refusal that hides the cause AND the fix is worse than the disclosure it prevents."""
    root, store = _corpus_with_vanished_sources(tmp_path)
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))

    with pytest.raises(Exception) as exc:
        index_memory(store, _NullEmbedder(), str(root))

    message = str(exc.value)
    assert "6 of 7" in message, "the operator must still see the scale of the disappearance"
    assert "allow_prune" in message or "--allow-prune" in message, "the remedy must survive"


def test_the_operator_still_gets_the_full_path_in_the_log(tmp_path, monkeypatch, caplog):
    """Redacted for the client, intact for whoever runs the server."""
    root, store = _corpus_with_vanished_sources(tmp_path)
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))

    with caplog.at_level("WARNING", logger="recall.mcp.service"):
        with pytest.raises(Exception):
            index_memory(store, _NullEmbedder(), str(root))

    assert "server-side-corpus-directory" in caplog.text, "the operator lost the diagnosis"
