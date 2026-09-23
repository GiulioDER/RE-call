"""The view-level admission gate for atomic rescue.

Each view vector is built so that its cosine with the query (axis 0) is exactly the score named
in the test, which makes the reference arithmetic visible.

Red proofs (2026-09-23, each mutation applied to ``select_view_gated_atomic_rescue`` in
``recall/atomic_rescue.py`` and reverted):

* ``test_reference_is_the_weakest_protected_parents_best_view`` failed on its admitted case when
  the reference was the strongest protected parent's best view (``min(protected_best)`` changed
  to ``max(protected_best)``), and on its margin assertion (0.4 instead of 0.1) when the
  reference was the weakest single protected view (per-parent ``np.max`` changed to ``np.min``).
* ``test_no_protected_view_means_no_reference_and_no_rescue`` failed when an empty reference list
  admitted instead of refusing.
* ``test_protected_parents_are_never_rescued`` failed when protected parents' views were left
  selectable.
* ``test_an_admitted_view_gated_rescue_lands_at_dense_rank_six`` failed when the insertion placed
  the rescue at index 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from recall.atomic_rescue import (  # noqa: E402
    clear_atomic_rescue_artifact_cache,
    insert_view_gated_atomic_rescue_dense,
    load_atomic_rescue_artifact,
    select_view_gated_atomic_rescue,
    write_atomic_rescue_artifact,
)
from recall.types import Chunk, ScoredChunk  # noqa: E402

DIM = 4
QUERY = [1.0, 0.0, 0.0, 0.0]


def _scoring(score: float, axis: int = 1) -> Any:
    """A unit vector whose cosine with QUERY is exactly ``score``."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = score
    vector[axis] = (1.0 - score * score) ** 0.5
    return vector


def _chunk(name: str) -> Chunk:
    return Chunk(id=name, source=f"src/{name}", text=name, metadata={})


DENSE = [ScoredChunk(_chunk(name), 0.9 - 0.1 * i) for i, name in enumerate("abcdefg")]


def _artifact(tmp_path: Path, views: list[tuple[str, float]]) -> Any:
    clear_atomic_rescue_artifact_cache()
    manifest = write_atomic_rescue_artifact(
        tmp_path / f"artifact-{len(list(tmp_path.iterdir()))}",
        matrix=np.stack([_scoring(score) for _, score in views]),
        views=[
            {"chunk_id": name, "source": f"src/{name}", "parent_ordinal": 0, "view_ordinal": index}
            for index, (name, _) in enumerate(views)
        ],
        generation_id="g",
        calibration_id="c",
        pipeline_fingerprint="p",
        corpus_fingerprint="0" * 64,
        embedding_profile="test",
        embedding_fingerprint="f",
        ordinary_chunk_count=10,
        source_commit="test",
    )
    return load_atomic_rescue_artifact(manifest)


def test_reference_is_the_weakest_protected_parents_best_view(tmp_path: Path) -> None:
    # Protected parent a: best view 0.9 (and a weak 0.2). Protected parent b: best view 0.5.
    # The reference is min(0.9, 0.5) = 0.5.
    protected = [("a", 0.9), ("a", 0.2), ("b", 0.5)]
    admitted = select_view_gated_atomic_rescue(_artifact(tmp_path, [*protected, ("x", 0.6)]), QUERY, DENSE)
    assert admitted is not None
    assert admitted.selection.chunk_id == "x"
    assert admitted.margin == pytest.approx(0.1, abs=1e-5)
    refused = select_view_gated_atomic_rescue(_artifact(tmp_path, [*protected, ("x", 0.4)]), QUERY, DENSE)
    assert refused is None


def test_no_protected_view_means_no_reference_and_no_rescue(tmp_path: Path) -> None:
    assert select_view_gated_atomic_rescue(_artifact(tmp_path, [("x", 0.99)]), QUERY, DENSE) is None


def test_protected_parents_are_never_rescued(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, [("a", 0.95), ("b", 0.3), ("y", 0.4)])
    gated = select_view_gated_atomic_rescue(artifact, QUERY, DENSE)
    assert gated is not None
    assert gated.selection.chunk_id == "y"


def test_an_admitted_view_gated_rescue_lands_at_dense_rank_six(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, [("b", 0.3), ("x", 0.7)])
    ranked, gated = insert_view_gated_atomic_rescue_dense(
        artifact, QUERY, DENSE, lambda chunk_id, score: ScoredChunk(_chunk(chunk_id), score)
    )
    assert gated is not None
    assert [hit.chunk.id for hit in ranked] == ["a", "b", "c", "d", "e", "x", "f", "g"]


def test_a_refused_view_gate_leaves_the_ranking_unchanged(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, [("b", 0.8), ("x", 0.7)])
    ranked, gated = insert_view_gated_atomic_rescue_dense(
        artifact, QUERY, DENSE, lambda chunk_id, score: ScoredChunk(_chunk(chunk_id), score)
    )
    assert gated is None
    assert [hit.chunk.id for hit in ranked] == [hit.chunk.id for hit in DENSE]
