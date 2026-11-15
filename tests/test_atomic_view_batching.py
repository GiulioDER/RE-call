"""Micro views reach an embedder without a group method in one call per `assemble` batch.

- V1 a lite rescue build embeds its views in one call per `EMBED_BATCH_CHUNKS` chunks, not one per
  chunk, and every artifact row is still the vector of its own view text.
- V2 an embedder that returns fewer vectors than texts is refused, rather than shifting the later
  vectors into the wrong group.
- V3 no texts means no call: an empty request is not sent to the embedder.

Their own file, not `tests/test_lite_atomic_rescue.py`, because a concurrent branch extends that
file's docstring and tail.

Red proof, run on the testbench and green again with `recall.embedding_core.embed_document_groups` restored:
- Against master `d2f0eeb9b`'s fallback (`[embed_passages(embedder, group) for group in groups]`)
  all three fail in their own assertion: V1 at "views were embedded in 55 calls, expected 1 for 55
  chunks", V2 at "DID NOT RAISE RuntimeError", V3 at "an empty request reached the embedder"
  (calls `[0, 0]`).
- V1's row check, with `vectors = vectors[::-1]` inserted before the split (lengths kept, order
  broken): fails at "artifact row 0 is not the vector of its own view". A shifted slice is not
  used as proof, since it trips `assemble`'s own per-chunk length check first.
- V2 with the length check deleted: "DID NOT RAISE RuntimeError".
- V3 with the `if not texts` return deleted: "an empty request reached the embedder".
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from recall.atomic_rescue import load_atomic_rescue_artifact
from recall.atomic_view_store import EMBED_BATCH_CHUNKS
from recall.atomizer import chunk_micro_views
from recall.embedding_core import embed_document_groups
from recall.lite.atomic import ensure_rescue_artifact
from tests.test_lite_atomic_rescue import _certified, _env
from tests.test_lite_calibration import ContentWordEmbedder


class _Counting:
    """Wraps an embedder that has no group method and records the size of every call."""

    def __init__(self, inner: ContentWordEmbedder, drop: int = 0) -> None:
        self._inner = inner
        self._drop = drop
        self.name = inner.name
        self.dim = inner.dim
        self.calls: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        vectors = self._inner.embed(texts)
        return vectors[: len(vectors) - self._drop]


def test_a_lite_build_embeds_views_in_one_call_per_batch(tmp_path: Path) -> None:
    store, embedder = _certified(tmp_path)
    # Same name and width, so the same legacy profile: the build sees the embedder it calibrated with.
    counting = _Counting(embedder)
    report = ensure_rescue_artifact(store, counting, env=_env(tmp_path / "atomic"))
    assert report is not None and report.manifest is not None, report

    chunks = sorted(store.iter_chunks(), key=lambda chunk: (chunk.source, chunk.metadata["ord"]))
    groups = [views for views in (chunk_micro_views(chunk.text) for chunk in chunks) if views]
    assert len(groups) > EMBED_BATCH_CHUNKS // 2, "the fixture has too few chunks to tell the two shapes apart"
    expected_calls = math.ceil(len(groups) / EMBED_BATCH_CHUNKS)
    assert len(counting.calls) == expected_calls, (
        f"views were embedded in {len(counting.calls)} calls, expected {expected_calls} for {len(groups)} chunks"
    )
    assert sum(counting.calls) == sum(len(views) for views in groups)

    matrix = load_atomic_rescue_artifact(report.manifest).matrix
    texts = [view for views in groups for view in views]
    assert len(texts) == len(matrix)
    for row, text in enumerate(texts):
        alone = embedder.embed([text])[0]
        norm = math.sqrt(sum(value * value for value in alone))
        assert [float(x) for x in matrix[row]] == pytest.approx([value / norm for value in alone], abs=1e-6), (
            f"artifact row {row} is not the vector of its own view"
        )


def test_a_short_answer_is_refused_not_shifted(tmp_path: Path) -> None:
    short = _Counting(ContentWordEmbedder(), drop=1)
    with pytest.raises(RuntimeError, match="returned 2 vectors for 3 texts"):
        embed_document_groups(short, [["alpha beta gamma"], ["delta epsilon"], ["zeta eta theta"]])


def test_no_texts_sends_no_request() -> None:
    counting = _Counting(ContentWordEmbedder())
    assert embed_document_groups(counting, [[], []]) == [[], []]
    assert embed_document_groups(counting, []) == []
    assert not counting.calls, "an empty request reached the embedder"
