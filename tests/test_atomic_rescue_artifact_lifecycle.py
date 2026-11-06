"""The atomic rescue artifact writer, loader, cache and dense placement (second CCA hunt).

Run 20261106T105829Z, a second DEEP hunt of recall/atomic_rescue.py over the first hunt's
deferred pool. Red proofs, 2026-11-06, each against the pre-fix module (the first hunt's
version, master 31320c8d2) with this file unchanged; each test failed in the assertion its
docstring names and passed on the fixed module.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from recall import atomic_rescue
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueSelection,
    atomic_rescue_expectation_parity,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    place_atomic_selection_dense,
    write_atomic_rescue_artifact,
)
from recall.types import Chunk, ScoredChunk


def _views(count: int) -> list[dict[str, object]]:
    return [
        {"chunk_id": f"chunk-{i}", "source": f"s{i}.md", "parent_ordinal": 0, "view_ordinal": 0}
        for i in range(count)
    ]


def _unit_rows(count: int, dimension: int = 4) -> np.ndarray:
    matrix = np.eye(max(count, dimension), dimension, dtype=np.float32)[:count]
    return np.ascontiguousarray(matrix)


def _write(root: Path, *, matrix=None, views=None, **overrides) -> Path:
    arguments: dict[str, object] = {
        "generation_id": "generation",
        "calibration_id": "calibration",
        "pipeline_fingerprint": "pipeline",
        "corpus_fingerprint": "corpus",
        "embedding_profile": "profile",
        "embedding_fingerprint": "fingerprint",
        "ordinary_chunk_count": 3,
        "source_commit": "0123456789abcdef",
    }
    arguments.update(overrides)
    return write_atomic_rescue_artifact(
        root,
        matrix=_unit_rows(3) if matrix is None else matrix,
        views=_views(3) if views is None else views,
        **arguments,
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_atomic_rescue_artifact_cache()
    yield
    clear_atomic_rescue_artifact_cache()


def test_a_matrix_that_cannot_be_converted_leaves_no_staging_directory(tmp_path: Path) -> None:
    """DAT-005: the conversion runs inside the cleanup's try.

    Red proof: ``assert ['.artifact.xxxx'] == []``, the staging directory left behind.
    """
    with pytest.raises((ValueError, TypeError)):
        _write(tmp_path / "artifact", matrix=[[1.0, 0.0], [1.0]], views=_views(2))
    assert [entry.name for entry in tmp_path.iterdir()] == []


@pytest.mark.parametrize(
    ("matrix", "views", "overrides"),
    [
        (np.zeros((0, 4), dtype=np.float32), [], {}),
        (None, None, {"ordinary_chunk_count": 0}),
        (None, None, {"source_commit": ""}),
        (
            None,
            [
                {"chunk_id": 7, "source": "a.md", "parent_ordinal": 0, "view_ordinal": 0},
                *_views(3)[1:],
            ],
            {},
        ),
        (
            None,
            [
                {"chunk_id": "x", "source": "a.md", "parent_ordinal": 0, "view_ordinal": 0},
                {"chunk_id": "y", "source": "a.md", "parent_ordinal": 0, "view_ordinal": 1},
                {"chunk_id": "z", "source": "b.md", "parent_ordinal": 0, "view_ordinal": 0},
            ],
            {},
        ),
    ],
    ids=["no-views", "no-ordinary-chunks", "empty-commit", "int-chunk-id", "shared-identity"],
)
def test_the_writer_never_publishes_what_the_loader_refuses(
    tmp_path: Path, matrix, views, overrides
) -> None:
    """DOC-005, BUG-103: the staged directory is loaded before it is renamed into place.

    Invariant: a published artifact always loads, and a refused one leaves the path free.
    Failure mode: the writer checked only the matrix, published an immutable directory every
    serving load refused, and that directory then blocked a rebuild. Red proof: every parameter
    failed with ``Failed: DID NOT RAISE``.
    """
    root = tmp_path / "artifact"
    with pytest.raises(AtomicRescueArtifactError):
        _write(root, matrix=matrix, views=views, **overrides)
    assert not root.exists()
    assert [entry.name for entry in tmp_path.iterdir()] == []


def test_the_writer_fsyncs_every_member(tmp_path: Path, monkeypatch) -> None:
    """DAT-104: members are durable before the rename publishes them.

    Red proof: ``assert 0 >= 3``, no fsync at all.
    """
    calls: list[int] = []
    real = os.fsync

    def spy(fd: int) -> None:
        calls.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    _write(tmp_path / "artifact")
    assert len(calls) >= 3


def test_a_manifest_refusal_costs_no_hash_of_the_matrix(tmp_path: Path, monkeypatch) -> None:
    """PERF-101: manifest scalars are checked before the members are hashed.

    Invariant: an artifact refused for its manifest (here the intended refusal of a missing
    embedding_fingerprint) is refused without reading the matrix. Failure mode: a refused
    artifact is never cached, so the full hash and validation ran again on every query. Red
    proof: ``assert 2 == 0`` hash calls.
    """
    manifest = _write(tmp_path / "artifact")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["embedding_fingerprint"]
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    hashed: list[Path] = []
    real = atomic_rescue._sha256

    def counting(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(atomic_rescue, "_sha256", counting)
    with pytest.raises(AtomicRescueArtifactError, match="embedding_fingerprint"):
        load_atomic_rescue_artifact(manifest)
    assert len(hashed) == 0


def test_an_npz_matrix_member_is_refused_as_unreadable(tmp_path: Path) -> None:
    """BUG-101: an archive member is refused with the module's own error.

    Red proof: ``assert isinstance(AttributeError("'NpzFile' object has no attribute 'dtype'"),
    AtomicRescueArtifactError)`` failed.
    """
    manifest = _write(tmp_path / "artifact")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    archive = manifest.with_name("matrix.npz")
    np.savez(archive, matrix=_unit_rows(3))
    payload["matrix_file"] = archive.name
    payload["matrix_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(Exception) as refused:
        load_atomic_rescue_artifact(manifest)
    assert isinstance(refused.value, AtomicRescueArtifactError)


def test_a_missing_atomic_extra_is_refused_at_load(tmp_path: Path, monkeypatch) -> None:
    """DEPLOY-002: threadpoolctl is probed with numpy, at load, not first at query time.

    Red proof: ``Failed: DID NOT RAISE``; the artifact loaded and only searches failed.
    """
    manifest = _write(tmp_path / "artifact")
    clear_atomic_rescue_artifact_cache()
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    with pytest.raises(AtomicRescueArtifactError, match=r"recall-rag\[atomic\]"):
        load_atomic_rescue_artifact(manifest)


def test_a_cached_artifact_is_served_while_another_loads(tmp_path: Path, monkeypatch) -> None:
    """PERF-002: a cold load holds no lock a cache hit needs.

    Invariant: a hit on a cached artifact returns while a different artifact is loading.
    Failure mode: the whole cold load ran under the process cache lock. Red proof:
    ``assert not True`` (the hit's thread was still alive after 5 s).
    """
    cached = _write(tmp_path / "cached")
    slow = _write(tmp_path / "slow")
    load_atomic_rescue_artifact(cached)
    entered = threading.Event()
    release = threading.Event()
    real = atomic_rescue._load_atomic_rescue_artifact

    def blocking(path: Path):
        if path == slow.resolve():
            entered.set()
            release.wait(10)
        return real(path)

    monkeypatch.setattr(atomic_rescue, "_load_atomic_rescue_artifact", blocking)
    loader = threading.Thread(target=load_atomic_rescue_artifact, args=(slow,))
    loader.start()
    try:
        assert entered.wait(5)
        hit = threading.Thread(target=load_atomic_rescue_artifact, args=(cached,))
        hit.start()
        hit.join(5)
        assert not hit.is_alive()
    finally:
        release.set()
        loader.join(10)


def _hit(name: str) -> ScoredChunk:
    return ScoredChunk(Chunk(id=name, source=f"{name}.md", text=name, metadata={}), 1.0)


def test_dense_placement_never_lists_a_protected_parent_twice() -> None:
    """STAKES-005: a selection already in the top five leaves the ranking as it was.

    Red proof: ``assert ['a', 'b', 'c', 'd', 'e', 'c', 'f'] == ['a', 'b', 'c', 'd', 'e', 'f']``.
    """
    dense = [_hit(name) for name in "abcdef"]
    selection = AtomicRescueSelection("c", "c.md", 0, 0, 0.9)

    ranked = place_atomic_selection_dense(selection, dense, lambda chunk_id, score: _hit(chunk_id))

    assert [hit.chunk.id for hit in ranked] == ["a", "b", "c", "d", "e", "f"]


def test_an_expectation_key_that_is_not_lower_case_hex_is_malformed(tmp_path: Path) -> None:
    """CODE-006: a row key that can never match is refused as malformed, not as a missing query.

    Red proof: ``Regex pattern did not match``; the error was 'atomic rescue expectation lacks
    the query'.
    """
    receipt = tmp_path / "expected.json"
    key = hashlib.sha256(b"query").hexdigest().upper()
    receipt.write_text(
        json.dumps(
            {"schema_version": 1, "rows": {key: {"source": "a.md", "parent_ordinal": 0, "score": 0.5}}}
        ),
        encoding="utf-8",
    )
    selection = AtomicRescueSelection("a", "a.md", 0, 0, 0.5)
    with pytest.raises(AtomicRescueArtifactError, match="malformed"):
        atomic_rescue_expectation_parity(receipt, query="query", selection=selection)


# Owner decisions of 2026-11-06 on the second hunt's open items. Red proofs, 2026-11-06, against
# the second hunt's module (a77611f8a) with this file otherwise unchanged.


def test_a_receipt_rewritten_in_place_is_read_again(tmp_path: Path) -> None:
    """DAT-102: the expectation cache follows the receipt file, not only its path.

    Red proof: ``assert True is False``; the second call still saw the old score.
    """
    receipt = tmp_path / "expected.json"
    key = hashlib.sha256(b"query").hexdigest()

    def write(score: float) -> None:
        row = {"source": "a.md", "parent_ordinal": 0, "score": score}
        receipt.write_text(
            json.dumps({"schema_version": 1, "rows": {key: row}}), encoding="utf-8"
        )

    selection = AtomicRescueSelection("a", "a.md", 0, 0, 0.5)
    write(0.5)
    assert atomic_rescue_expectation_parity(receipt, query="query", selection=selection) == (True, True)
    write(0.25)
    assert atomic_rescue_expectation_parity(receipt, query="query", selection=selection)[1] is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory modes")
def test_a_published_artifact_directory_takes_the_umask_mode(tmp_path: Path) -> None:
    """DAT-103: the published directory is not left at mkdtemp's 0700.

    Red proof: ``assert 448 == 493`` (0o700 against 0o755 under umask 022).
    """
    import stat

    previous = os.umask(0o022)
    try:
        manifest = _write(tmp_path / "artifact")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(manifest.parent.stat().st_mode) == 0o755
