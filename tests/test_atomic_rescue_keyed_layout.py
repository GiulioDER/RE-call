"""PostgreSQL rescue artifacts keyed by generation AND calibration (docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md).

A recalibration used to replace `<root>/<generation_id>/` in place, which Windows refuses while any
other server still has the old matrix mapped, so a desktop install would never recover from a
recalibration until every server restarted. A build now writes `<root>/<generation_id>.<calibration_id>/`
and never touches another calibration's directory; readers fall back to the old key, so artifacts
built before the change keep serving.

No database: `build_artifact`'s store, calibration repository and manager are the fakes of
`tests/test_atomic_rescue_default_on_cca.py`.

Red proofs (mutations of the new code, each run on Linux, failing in its own assertion):
- K1 with `build_artifact` writing `resolve_atomic_rescue_manifest(root, generation_id)` again:
  "the second calibration's artifact is not beside the first".
- K2 with `artifact_manifest` dropping its legacy fallback: "a legacy artifact no longer serves".
- K3 with `remove_artifacts` resolving only the legacy directory: "removed 1 of 2".
- K4 with `generation_artifact_dirs` globbing `<generation_id>*`: "another generation's directory
  was taken for this one's".
"""

from __future__ import annotations

import shutil
from pathlib import Path

from recall.atomic_rescue import clear_atomic_rescue_artifact_cache, load_atomic_rescue_artifact
from recall.lite.atomic import ROOT_ENV
from tests.test_atomic_rescue_default_on_cca import _Corpus, _publish, corpus  # noqa: F401  # fixture by name
from tests.test_calibration_carry_forward import _CarryEmbedder


def test_a_recalibration_builds_beside_the_old_artifact(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """K1: the build for a new calibration leaves the old calibration's directory alone."""
    import recall.generation_rescue as rescue

    embedder = _CarryEmbedder()
    assert rescue.build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    corpus.repository.calibration_id = "cal_2"  # type: ignore[attr-defined]
    removed: list[Path] = []
    real = shutil.rmtree
    monkeypatch.setattr(rescue.shutil, "rmtree", lambda path, **kw: (removed.append(Path(path)), real(path, **kw)))

    assert rescue.build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")

    assert (tmp_path / "gen_1.cal_1" / "manifest.json").is_file(), "the first calibration's artifact was removed"
    assert (tmp_path / "gen_1.cal_2" / "manifest.json").is_file(), "the second calibration's artifact is not beside the first"
    assert tmp_path / "gen_1.cal_1" not in removed
    clear_atomic_rescue_artifact_cache()
    assert load_atomic_rescue_artifact(tmp_path / "gen_1.cal_2" / "manifest.json").calibration_id == "cal_2"


def test_a_legacy_artifact_still_serves(corpus: _Corpus, tmp_path: Path) -> None:  # noqa: F811
    """K2: an artifact at the old `<root>/<generation_id>/` key, built before the change, is the one
    readers resolve and a build keeps in force."""
    from recall.generation_rescue import artifact_manifest, build_artifact

    embedder = _CarryEmbedder()
    build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)
    (tmp_path / "gen_1.cal_1").rename(tmp_path / "gen_1")
    clear_atomic_rescue_artifact_cache()

    assert artifact_manifest(tmp_path, "gen_1", "cal_1") == (tmp_path / "gen_1" / "manifest.json").resolve(), (
        "a legacy artifact no longer serves"
    )
    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path) == "in force for gen_1"
    assert not (tmp_path / "gen_1.cal_1").exists()


def test_remove_artifacts_removes_both_layouts(tmp_path: Path) -> None:
    """K3: gc and forget remove a generation's artifacts under either key."""
    from recall.generation_rescue import remove_artifacts

    _publish(tmp_path / "gen_1", "cal_1")
    _publish(tmp_path / "gen_1.cal_2", "cal_2")

    removed = remove_artifacts(["gen_1"], env={ROOT_ENV: str(tmp_path)})

    assert removed == 2, f"removed {removed} of 2"
    assert not (tmp_path / "gen_1").exists() and not (tmp_path / "gen_1.cal_2").exists()


def test_generation_artifact_dirs_ignores_other_generations(tmp_path: Path) -> None:
    """K4: `gen_1` must not claim `gen_10` or `gen_1x`."""
    from recall.generation_rescue import generation_artifact_dirs

    for name in ("gen_1", "gen_1.cal_1", "gen_10", "gen_10.cal_1", "gen_1x"):
        (tmp_path / name).mkdir()

    found = [path.name for path in generation_artifact_dirs(tmp_path, "gen_1")]

    assert found == ["gen_1", "gen_1.cal_1"], f"another generation's directory was taken for this one's: {found}"
