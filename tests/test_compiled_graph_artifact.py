"""An artifact serves only the lineage it was built for, and a damaged one never serves.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.artifact``):

- ``test_round_trip``: ``edges`` not written fails on the loaded record.
- ``test_other_lineage_is_refused``: the lineage comparison removed fails with no exception.
- ``test_damaged_member_is_refused``: the checksum comparison removed fails with no exception.
- ``test_two_tenants_share_a_root``: a directory named by generation alone fails on the second
  tenant's records (RF4).
- ``test_missing_is_reported``: a missing manifest read as an empty artifact fails.
- ``test_a_verified_artifact_is_not_hashed_again`` (CCA PERF-001): against the pre-fix
  ``load_artifact`` (no cache) fails on the hash count, 4 against 2.
- ``test_an_erased_artifact_is_never_served_from_memory`` (CCA PERF-001): the manifest stamp
  check removed from the cache lookup fails on the rebuilt artifact's source, ``a.md`` served.
- ``test_a_remnant_without_a_manifest_is_replaced`` (CCA FA-2-1): against the pass-2
  ``write_artifact``, which kept any existing directory, fails with ``ArtifactMissing``.
- ``test_a_read_error_is_classified`` (CCA BUG-001): against the pre-fix ``load_artifact`` fails
  with the raw ``PermissionError`` and ``FileNotFoundError``.
"""

from __future__ import annotations

from pathlib import Path

import shutil

import numpy as np
import pytest

from recall.compiled_graph import artifact as artifact_module
from recall.compiled_graph.artifact import (
    ArtifactDamaged,
    ArtifactLineageMismatch,
    ArtifactMissing,
    ArtifactRecord,
    CompiledGraphArtifact,
    artifact_dir,
    load_artifact,
    write_artifact,
)

LINEAGE = {"tenant": "t1", "generation": "g1", "calibration": "c1", "embedder": "e", "model": "m"}


def _artifact(lineage: dict[str, str], source: str = "a.md") -> CompiledGraphArtifact:
    record = ArtifactRecord(key=f"{source}#0", source=source, kind="fact",
                            fields=("Alice owns it.", ""), entities=("Alice",), edges=(f"{source}:0",))
    return CompiledGraphArtifact(lineage=lineage, records=(record,), vectors=np.ones((1, 4)) / 2)


def test_round_trip(tmp_path: Path) -> None:
    write_artifact(tmp_path, _artifact(LINEAGE))
    loaded = load_artifact(tmp_path, LINEAGE)
    assert loaded.records == _artifact(LINEAGE).records
    assert loaded.vectors.dtype == np.float32 and loaded.vectors.shape == (1, 4)


def test_other_lineage_is_refused(tmp_path: Path) -> None:
    target = write_artifact(tmp_path, _artifact(LINEAGE))
    other = {**LINEAGE, "calibration": "c2"}
    # The same directory read under another lineage: the manifest must refuse it.
    target.rename(artifact_dir(tmp_path, other))
    with pytest.raises(ArtifactLineageMismatch):
        load_artifact(tmp_path, other)


def test_damaged_member_is_refused(tmp_path: Path) -> None:
    target = write_artifact(tmp_path, _artifact(LINEAGE))
    records = target / "records.jsonl"
    records.write_text(records.read_text(encoding="utf-8").replace("Alice", "Mallory"), encoding="utf-8")
    with pytest.raises(ArtifactDamaged):
        load_artifact(tmp_path, LINEAGE)


def test_two_tenants_share_a_root(tmp_path: Path) -> None:
    other = {**LINEAGE, "tenant": "t2"}
    write_artifact(tmp_path, _artifact(LINEAGE, "a.md"))
    write_artifact(tmp_path, _artifact(other, "b.md"))
    assert load_artifact(tmp_path, LINEAGE).records[0].source == "a.md"
    assert load_artifact(tmp_path, other).records[0].source == "b.md"


def test_missing_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ArtifactMissing):
        load_artifact(tmp_path, LINEAGE)


def test_empty_artifact_round_trips(tmp_path: Path) -> None:
    empty = CompiledGraphArtifact(lineage=LINEAGE, records=(), vectors=np.zeros((0, 4)))
    write_artifact(tmp_path, empty)
    assert load_artifact(tmp_path, LINEAGE).records == ()


def test_a_verified_artifact_is_not_hashed_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_artifact(tmp_path, _artifact(LINEAGE))
    real, hashed = artifact_module._sha256, []

    def counting(path: Path) -> str:
        hashed.append(path.name)
        return real(path)

    monkeypatch.setattr(artifact_module, "_sha256", counting)
    first = load_artifact(tmp_path, LINEAGE)
    second = load_artifact(tmp_path, LINEAGE)
    assert len(hashed) == 2
    assert second.records == first.records


def test_an_erased_artifact_is_never_served_from_memory(tmp_path: Path) -> None:
    target = write_artifact(tmp_path, _artifact(LINEAGE))
    load_artifact(tmp_path, LINEAGE)
    shutil.rmtree(target)
    with pytest.raises(ArtifactMissing):
        load_artifact(tmp_path, LINEAGE)
    write_artifact(tmp_path, _artifact(LINEAGE, "b.md"))
    assert load_artifact(tmp_path, LINEAGE).records[0].source == "b.md"


@pytest.mark.parametrize(
    ("error", "expected"), [(PermissionError, ArtifactDamaged), (FileNotFoundError, ArtifactMissing)]
)
def test_a_read_error_is_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError], expected: type[Exception]
) -> None:
    write_artifact(tmp_path, _artifact(LINEAGE))

    def failing(path: Path) -> str:
        raise error(str(path))

    monkeypatch.setattr(artifact_module, "_sha256", failing)
    with pytest.raises(expected):
        load_artifact(tmp_path, LINEAGE)


def test_a_remnant_without_a_manifest_is_replaced(tmp_path: Path) -> None:
    target = write_artifact(tmp_path, _artifact(LINEAGE))
    (target / "manifest.json").unlink()
    (target / "records.jsonl").unlink()
    assert write_artifact(tmp_path, _artifact(LINEAGE)) == target
    assert load_artifact(tmp_path, LINEAGE).records == _artifact(LINEAGE).records
