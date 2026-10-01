"""The atomic rescue loader validates its matrix in row blocks, and still rejects every bad row.

The loader runs in every serving process on its first search. Validating the whole memory-mapped
matrix at once built two full-size temporaries (`np.isfinite` and the row norm), which measured a
328 MB temporary on the 2026-10-01 memory corpus and an 817 MB first-search peak per process.
These tests pin both halves of the blocked replacement: the bound, and that no row escapes it.
"""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from recall import atomic_rescue
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from recall.embeddings import EmbeddingProfile


def _artifact(root: Path, rows: int, dimension: int) -> Path:
    rng = np.random.default_rng(20261001)
    matrix = rng.standard_normal((rows, dimension)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    views = [
        {"chunk_id": f"chunk-{i}", "source": f"s{i}.md", "parent_ordinal": 0, "view_ordinal": 0}
        for i in range(rows)
    ]
    return write_atomic_rescue_artifact(
        root,
        matrix=matrix,
        views=views,
        generation_id="generation",
        calibration_id="calibration",
        pipeline_fingerprint="pipeline",
        corpus_fingerprint="corpus",
        embedding_profile="test-profile",
        embedding_fingerprint=EmbeddingProfile(
            profile_id="test-profile",
            model_name="test-model",
            artifact_digest="test-digest",
            dimension=dimension,
            query_mode="embed",
            passage_mode="embed",
        ).fingerprint(),
        ordinary_chunk_count=rows,
        source_commit="0123456789abcdef",
    )


def _tamper(manifest: Path, edit) -> None:
    """Rewrite the matrix on disk and re-sign the manifest, so only the row checks can refuse it."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    matrix_path = manifest.with_name(payload["matrix_file"])
    matrix = np.load(matrix_path)
    edit(matrix)
    np.save(matrix_path, matrix)
    payload["matrix_sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    clear_atomic_rescue_artifact_cache()


@pytest.fixture
def two_rows_per_block(monkeypatch):
    """Seven rows of dimension 4 in blocks of two: 0-1, 2-3, 4-5, and a partial block, 6."""
    monkeypatch.setattr(atomic_rescue, "_VALIDATION_BLOCK_BYTES", 2 * 4 * 4)


def test_a_nonfinite_value_in_the_last_partial_block_is_rejected(tmp_path, two_rows_per_block):
    """A blocked loop that drops its trailing block would accept this matrix.

    Red proof (2026-10-01, Linux), node
    `tests/test_atomic_rescue_load_validation.py::test_a_nonfinite_value_in_the_last_partial_block_is_rejected`:
    mutating the finite pass in `recall.atomic_rescue._load_atomic_rescue_artifact` to
    `range(0, matrix.shape[0] - rows_per_block, rows_per_block)` fails the `match`: the skipped NaN
    reaches the norm pass, so the load still raises, but as "rows are not normalized". The file is
    refused for the wrong reason, which is what this test exists to see.
    """
    manifest = _artifact(tmp_path / "artifact", rows=7, dimension=4)
    _tamper(manifest, lambda m: m.__setitem__((6, 2), np.nan))

    with pytest.raises(AtomicRescueArtifactError, match="nonfinite"):
        load_atomic_rescue_artifact(manifest)


def test_a_non_unit_row_in_the_last_partial_block_is_rejected(tmp_path, two_rows_per_block):
    """The same tail, for the norm pass.

    Red proof (2026-10-01, Linux), node
    `tests/test_atomic_rescue_load_validation.py::test_a_non_unit_row_in_the_last_partial_block_is_rejected`:
    the same trailing-block mutation in the norm pass fails with "DID NOT RAISE".
    """
    manifest = _artifact(tmp_path / "artifact", rows=7, dimension=4)
    _tamper(manifest, lambda m: m.__setitem__(6, m[6] * 2.0))

    with pytest.raises(AtomicRescueArtifactError, match="not normalized"):
        load_atomic_rescue_artifact(manifest)


def test_a_matrix_with_both_faults_reports_the_nonfinite_one(tmp_path, two_rows_per_block):
    """The unblocked checks ran the finite test over the whole matrix first. Kept.

    Red proof (2026-10-01, Linux), node
    `tests/test_atomic_rescue_load_validation.py::test_a_matrix_with_both_faults_reports_the_nonfinite_one`:
    merging the two passes into one loop, which checks block 0's norm before block 3's values,
    fails with `Regex pattern did not match`, the error naming "not normalized".
    """
    manifest = _artifact(tmp_path / "artifact", rows=7, dimension=4)

    def both(m):
        m[0] = m[0] * 2.0
        m[6, 1] = np.inf

    _tamper(manifest, both)

    with pytest.raises(AtomicRescueArtifactError, match="nonfinite"):
        load_atomic_rescue_artifact(manifest)


def test_load_time_validation_allocates_a_block_not_the_matrix(tmp_path, monkeypatch):
    """The point of the change: what a load allocates is bounded by the block, not the matrix.

    A 32 MiB matrix (4,096 x 2,048 float32) validated in 1 MiB blocks. The mapping itself is not
    a traced allocation, so the traced peak is the temporaries plus the metadata, and it must stay
    under a quarter of the matrix. The whole-matrix checks allocate at least the matrix again.

    Red proof (2026-10-01, Linux), node
    `tests/test_atomic_rescue_load_validation.py::test_load_time_validation_allocates_a_block_not_the_matrix`:
    restoring the unblocked `np.isfinite(matrix)` and `np.linalg.norm(matrix, axis=1)` in
    `recall.atomic_rescue._load_atomic_rescue_artifact` fails the peak assertion, "load traced a
    34.2 MiB peak for a 32 MiB matrix".
    """
    rows, dimension = 4096, 2048
    manifest = _artifact(tmp_path / "artifact", rows=rows, dimension=dimension)
    monkeypatch.setattr(atomic_rescue, "_VALIDATION_BLOCK_BYTES", 1024 * 1024)
    clear_atomic_rescue_artifact_cache()
    matrix_bytes = rows * dimension * 4

    tracemalloc.start()
    try:
        artifact = load_atomic_rescue_artifact(manifest)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        clear_atomic_rescue_artifact_cache()

    assert artifact.matrix.shape == (rows, dimension)
    assert peak < matrix_bytes // 4, (
        f"load traced a {peak / 2**20:.1f} MiB peak for a {matrix_bytes / 2**20:.0f} MiB matrix; "
        "a whole-matrix temporary is back"
    )
