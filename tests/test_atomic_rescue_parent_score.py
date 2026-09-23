"""An atomic rescue changes rank only: its parent is scored as a chunk, and its matrix product is
bounded.

Red proofs (2026-09-23, each mutation applied and reverted):

* ``test_a_view_score_cannot_turn_an_abstention_into_an_answer`` failed when the loader in
  ``recall.trust`` (``parent_loader.load``) passed ``view_score`` to ``scored_chunk_by_id`` instead
  of the parent's own cosine: ``abstained`` was False, because the 0.99 view score cleared the
  0.1 threshold that no chunk had cleared.
* ``test_a_parent_the_generation_lacks_is_not_invented`` failed with ``DID NOT RAISE`` when
  ``load`` read ``parent_cosines(...).get(chunk_id, view_score)``, falling back to the view score
  for a parent ``cosines_for`` did not return, so the rescue was served instead of refused.
* ``test_the_view_product_runs_inside_a_bounded_blas_pool`` failed when ``_view_scores`` in
  ``recall.atomic_rescue`` computed ``artifact.matrix @ query`` outside the ``limit`` context.
"""

from __future__ import annotations

import numpy as np
import pytest

from recall import atomic_rescue
from recall.atomic_rescue import AtomicRescueSelectionError, select_atomic_rescue
from recall.calibration import Calibration
from recall.trust import TrustPolicy, trusted_search
from recall.types import Chunk, ScoredChunk
from tests.test_atomic_rescue_placement import _artifact
from tests.test_atomic_rescue_production_shadow import _ActiveStore, _Embedder


class _WeakStore(_ActiveStore):
    """Every chunk scores below the 0.1 threshold, so an honest search abstains."""

    def __init__(self, parent_cosines: dict[str, float]) -> None:
        super().__init__()
        self.parent_cosines = parent_cosines

    def query_dense(self, vector, k, source=None, scope=None):
        del vector, source, scope
        return [
            ScoredChunk(Chunk(f"dense-{index}", f"s{index}.md", str(index), {}), 0.05)
            for index in range(1, 8)
        ][:k]

    def cosines_for(self, ids, vec):
        del vec
        return {i: self.parent_cosines[i] for i in ids if i in self.parent_cosines}


def _search(monkeypatch, tmp_path, store: _WeakStore, placement: str):
    from recall import trust

    class Artifact:
        def assert_lineage(self, **kwargs):
            del kwargs

    def dense_insert(artifact, query_vector, dense, loader):
        rescue = loader("rescued", 0.99)
        if rescue is None:
            raise AtomicRescueSelectionError("atomic rescue selected parent is unavailable")
        return [*dense[:5], rescue, *dense[5:]]

    def fused_insert(artifact, query_vector, dense, ranked, loader):
        rescue = loader("rescued", 0.99)
        if rescue is None:
            raise AtomicRescueSelectionError("atomic rescue selected parent is unavailable")
        return [*ranked[:5], rescue, *ranked[5:]]

    monkeypatch.setattr(trust, "load_atomic_rescue_artifact", lambda path: Artifact())
    monkeypatch.setattr(trust, "insert_atomic_rescue_dense", dense_insert)
    monkeypatch.setattr(trust, "insert_atomic_rescue_fused", fused_insert)
    return trusted_search(
        store,
        _Embedder(),
        "query",
        k=6,
        candidate_k=7,
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
            "RECALL_ATOMIC_RESCUE_PLACEMENT": placement,
        },
    )


@pytest.mark.parametrize("placement", ["dense", "fused"])
def test_a_view_score_cannot_turn_an_abstention_into_an_answer(
    monkeypatch, tmp_path, placement
) -> None:
    store = _WeakStore({"rescued": 0.04})
    result = _search(monkeypatch, tmp_path, store, placement)
    assert result.abstained is True
    assert store.loaded == [("rescued", 0.04)]
    rescued = [hit for hit in result.hits if hit.chunk.id == "rescued"]
    assert [(hit.cosine, hit.verdict == "ok") for hit in rescued] == [(pytest.approx(0.04), False)]


def test_a_parent_the_generation_lacks_is_not_invented(monkeypatch, tmp_path) -> None:
    store = _WeakStore({})
    with pytest.raises(AtomicRescueSelectionError, match="unavailable"):
        _search(monkeypatch, tmp_path, store, "fused")
    assert store.loaded == []


def test_the_view_product_runs_inside_a_bounded_blas_pool(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path, "winner")
    state = {"inside": False}
    seen: list[tuple[int, str, bool]] = []

    class Limit:
        def __init__(self, limits, user_api):
            self.args = (limits, user_api)

        def __enter__(self):
            state["inside"] = True

        def __exit__(self, *exc):
            state["inside"] = False

    class Controller:
        def limit(self, *, limits, user_api):
            return Limit(limits, user_api)

    class Matrix:
        def __init__(self, inner):
            self.inner = inner

        def __matmul__(self, other):
            seen.append((atomic_rescue.ATOMIC_RESCUE_BLAS_THREADS, "blas", state["inside"]))
            return self.inner @ other

    monkeypatch.setattr(atomic_rescue, "_THREADPOOL_CONTROLLER", Controller())
    object.__setattr__(artifact, "matrix", Matrix(np.asarray(artifact.matrix)))
    dense = [ScoredChunk(Chunk(f"d{index}", "d.md", "d", {}), 1.0) for index in range(1, 8)]
    selection = select_atomic_rescue(artifact, [1.0, 0.0], dense)
    assert selection.chunk_id == "winner"
    assert seen == [(2, "blas", True)]
