"""Finite float32 directions retain their cosine under extreme positive scaling.

NUM-002 red proof uses the public assembler and selector before their float64 normalization
repair. Numeric refusals and invalid normalized rows reach the invariant assertions below.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifact,
    AtomicRescueSelectionError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    select_atomic_rescue,
    write_atomic_rescue_artifact,
)
from recall.atomic_view_store import AtomicViewStore, assemble
from recall.types import Chunk, ScoredChunk


@pytest.fixture
def artifact(tmp_path: Path) -> Iterator[AtomicRescueArtifact]:
    manifest = write_atomic_rescue_artifact(
        tmp_path / "artifact", matrix=np.asarray([[1.0, 0.0]] * 6, dtype=np.float32),
        views=[{"chunk_id": str(i), "source": f"{i}.md", "parent_ordinal": 0, "view_ordinal": 0} for i in range(6)],
        generation_id="generation", calibration_id="calibration", pipeline_fingerprint="pipeline",
        corpus_fingerprint="corpus", embedding_profile="profile", embedding_fingerprint="fingerprint",
        ordinary_chunk_count=6, source_commit="synthetic",
    )
    try:
        yield load_atomic_rescue_artifact(manifest)
    finally:
        clear_atomic_rescue_artifact_cache()


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_assembly_preserves_finite_float32_directions(tmp_path: Path, scale: float) -> None:
    parents = [("parent", "memo.md", 0, "alpha bravo charlie delta echo foxtrot golf hotel")]
    with AtomicViewStore(tmp_path / "views.sqlite") as store:
        try:
            actual, _, _ = assemble(
                parents, store, "profile", 2,
                lambda groups: [[[scale, scale] for _ in group] for group in groups],
            )
        except (RuntimeError, ValueError) as exc:
            actual = type(exc).__name__
    assert (
        isinstance(actual, np.ndarray) and actual.dtype == np.float32
        and np.allclose(actual, [[2**-0.5, 2**-0.5]], rtol=0, atol=1e-6)
    ), "assembly lost a finite nonzero float32 direction"


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_selection_preserves_finite_float32_query_directions(artifact: AtomicRescueArtifact, scale: float) -> None:
    dense = [ScoredChunk(Chunk(str(i), f"{i}.md", "synthetic", {}), 1.0) for i in range(5)]
    try:
        selection = select_atomic_rescue(artifact, [scale, scale], dense)
        actual = (selection.chunk_id, selection.score)
    except AtomicRescueSelectionError as exc:
        actual = (type(exc).__name__, 0.0)
    assert actual[0] == "5" and actual[1] == pytest.approx(2**-0.5, abs=1e-6), (
        "selection lost a finite nonzero float32 query direction"
    )


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")], ids=["zero", "nan", "infinity"])
def test_selection_still_refuses_zero_and_nonfinite_queries(artifact: AtomicRescueArtifact, value: float) -> None:
    """The repair retains the query guard. A guard-bypass mutation must fail this assertion."""
    dense = [ScoredChunk(Chunk(str(i), f"{i}.md", "synthetic", {}), 1.0) for i in range(5)]
    with pytest.raises(AtomicRescueSelectionError, match="nonfinite or zero norm"):
        select_atomic_rescue(artifact, [value, value], dense)


def test_assembly_still_refuses_zero_directions(tmp_path: Path) -> None:
    """The repair retains the zero guard. A zero-to-unit-denominator mutation must fail it."""
    parents = [("parent", "memo.md", 0, "alpha bravo charlie delta echo foxtrot golf hotel")]
    with AtomicViewStore(tmp_path / "views.sqlite") as store:
        with pytest.raises(RuntimeError, match="zero norm"):
            assemble(parents, store, "profile", 2, lambda groups: [[[0.0, 0.0] for _ in group] for group in groups])
