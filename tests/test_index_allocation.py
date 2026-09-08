from __future__ import annotations

import pytest

from recall import index
from recall.embeddings import HashingEmbedder
from recall.types import Chunk


def test_allocation_failure_names_the_batch_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("onnxruntime failed to allocate memory for Attention_0")

    monkeypatch.setattr(index, "embed_with_cache", fail)
    indexer = index.Indexer(object(), HashingEmbedder(dim=64))

    with pytest.raises(RuntimeError, match=r"RECALL_INDEX_BATCH_CHUNKS \(currently 64\)"):
        indexer._flush(["notes.md"], [Chunk("id", "notes.md", "body")])


def test_indexer_uses_the_explicit_environment_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indexing limits must stay attached to the startup settings snapshot.

    Red proof: mutating ``Indexer.__init__`` to call ``_batch_chunks_from_env()`` produced 8
    instead of 2, proving the test observes the production constructor boundary.
    """
    monkeypatch.setenv("RECALL_INDEX_BATCH_CHUNKS", "8")
    monkeypatch.setenv("RECALL_MAX_PRUNE_FRACTION", "0.9")

    indexer = index.Indexer(
        object(),
        HashingEmbedder(dim=64),
        env={
            "RECALL_INDEX_BATCH_CHUNKS": "2",
            "RECALL_MAX_PRUNE_FRACTION": "0.2",
        },
    )

    assert indexer._batch_chunks == 2
    assert indexer._max_prune_fraction == 0.2
