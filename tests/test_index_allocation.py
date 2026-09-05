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
