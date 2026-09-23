"""The dense-rank admission gate for atomic rescue.

Red proofs (2026-09-22, each mutation applied to ``recall/atomic_rescue.py`` and reverted):

* ``test_a_view_weaker_than_the_fifth_window_is_refused`` failed when the gate accepted any
  margin (``if margin < 0.0`` changed to ``if False``).
* ``test_each_probe_is_gated_against_its_own_dense_ranking`` failed when every probe was compared
  with the served query's rank-five window (``dense[gate_rank - 1]`` changed to
  ``protected[gate_rank - 1]``).
* ``test_protected_parents_stay_excluded_for_every_probe`` failed when exclusion used each probe's
  own dense top five instead of the served query's.
* ``test_an_admitted_rescue_lands_at_dense_rank_six`` failed when the insertion placed the rescue
  at index 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from recall.atomic_rescue import (  # noqa: E402
    clear_atomic_rescue_artifact_cache,
    insert_gated_atomic_rescue_dense,
    load_atomic_rescue_artifact,
    select_gated_atomic_rescue,
    write_atomic_rescue_artifact,
)
from recall.types import Chunk, ScoredChunk  # noqa: E402

DIM = 8


def _unit(values: list[float]) -> Any:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def _axis(index: int, lean: float = 0.0, towards: int = 7) -> Any:
    vector = [0.0] * DIM
    vector[index] = 1.0
    vector[towards] += lean
    return _unit(vector)


def _chunk(name: str) -> Chunk:
    return Chunk(id=name, source=f"src/{name}", text=name, metadata={})


def _dense(names: list[str], scores: list[float]) -> list[ScoredChunk]:
    return [ScoredChunk(_chunk(name), score) for name, score in zip(names, scores, strict=True)]


def _artifact(tmp_path: Path, views: list[tuple[str, Any]]) -> Any:
    clear_atomic_rescue_artifact_cache()
    manifest = write_atomic_rescue_artifact(
        tmp_path / "artifact",
        matrix=np.stack([vector for _, vector in views]),
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


SERVED = _dense(["a", "b", "c", "d", "e", "f", "g"], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])


def test_a_view_weaker_than_the_fifth_window_is_refused(tmp_path: Path) -> None:
    query = _axis(0)
    # The only candidate view scores cos(query, view) = 0.3 < the fifth window's 0.5.
    view = _unit([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, (1 - 0.09) ** 0.5, 0.0])
    artifact = _artifact(tmp_path, [("x", view)])
    assert select_gated_atomic_rescue(artifact, [(query, SERVED)], SERVED) is None
    ranked, gated = insert_gated_atomic_rescue_dense(
        artifact, [(query, SERVED)], SERVED, lambda chunk_id, score: ScoredChunk(_chunk(chunk_id), score)
    )
    assert gated is None
    assert [hit.chunk.id for hit in ranked] == [hit.chunk.id for hit in SERVED]


def test_an_admitted_rescue_lands_at_dense_rank_six(tmp_path: Path) -> None:
    query = _axis(0)
    view = _unit([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0])
    artifact = _artifact(tmp_path, [("x", view)])
    ranked, gated = insert_gated_atomic_rescue_dense(
        artifact, [(query, SERVED)], SERVED, lambda chunk_id, score: ScoredChunk(_chunk(chunk_id), score)
    )
    assert gated is not None and gated.probe_index == 0
    assert gated.margin == pytest.approx(0.8 - 0.5, abs=1e-5)
    assert [hit.chunk.id for hit in ranked] == ["a", "b", "c", "d", "e", "x", "f", "g"]


def test_each_probe_is_gated_against_its_own_dense_ranking(tmp_path: Path) -> None:
    query = _axis(0)
    sub_question = _axis(1)
    # Under the sub-question the view scores 0.6, which beats the sub-question's own fifth window
    # (0.2) but not the served query's fifth window (0.5 is below 0.6 too, so make it 0.65).
    served = _dense(["a", "b", "c", "d", "e", "f"], [0.9, 0.85, 0.8, 0.75, 0.65, 0.3])
    sub_dense = _dense(["p", "q", "r", "s", "t"], [0.5, 0.4, 0.3, 0.25, 0.2])
    view = _unit([0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0])
    artifact = _artifact(tmp_path, [("x", view)])
    gated = select_gated_atomic_rescue(artifact, [(query, served), (sub_question, sub_dense)], served)
    assert gated is not None
    assert gated.probe_index == 1
    assert gated.margin == pytest.approx(0.6 - 0.2, abs=1e-5)


def test_protected_parents_stay_excluded_for_every_probe(tmp_path: Path) -> None:
    query = _axis(0)
    sub_question = _axis(1)
    sub_dense = _dense(["p", "q", "r", "s", "t"], [0.5, 0.4, 0.3, 0.25, 0.2])
    # "a" is in the served top five, so its view must never be rescued, even though it is the
    # sub-question's best view. The weaker "y" view is the only admissible candidate.
    strong = _unit([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.43589, 0.0])
    weak = _unit([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.95394, 0.0])
    artifact = _artifact(tmp_path, [("a", strong), ("y", weak)])
    gated = select_gated_atomic_rescue(artifact, [(query, SERVED), (sub_question, sub_dense)], SERVED)
    assert gated is not None
    assert gated.selection.chunk_id == "y"


def test_a_probe_without_its_gate_window_is_refused(tmp_path: Path) -> None:
    from recall.atomic_rescue import AtomicRescueSelectionError

    artifact = _artifact(tmp_path, [("x", _axis(2))])
    with pytest.raises(AtomicRescueSelectionError, match="gate window"):
        select_gated_atomic_rescue(artifact, [(_axis(0), SERVED[:3])], SERVED)
