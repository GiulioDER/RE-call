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
  "the first calibration's artifact was removed" (the build wrote the old key, so no keyed
  directory exists). Also red against the parent commit `9efff64f7`, at the same assertion.
- K3 is also red against the parent commit, "removed 1 of 2".
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


def test_prune_keeps_the_active_and_the_rollback_target(tmp_path: Path, monkeypatch) -> None:
    """K5: after a host build, the artifacts of generations that cannot serve again are removed (a
    desktop install runs no gc), while the active generation, the newest retired one (a rollback's
    target), a ready one (about to be promoted) and another tenant's generation keep theirs.

    Red proof: with `prune_superseded` keeping every generation, "removed 0, expected the older
    retired, the failed and the collected generations' 3".
    """
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import prune_superseded
    from recall.lineage import GenerationState

    start = datetime(2026, 11, 1, tzinfo=UTC)
    records = [
        SimpleNamespace(generation_id="gen_active", state=GenerationState.ACTIVE, retired_at=None),
        SimpleNamespace(generation_id="gen_old", state=GenerationState.RETIRED, retired_at=start),
        SimpleNamespace(generation_id="gen_prev", state=GenerationState.RETIRED, retired_at=start + timedelta(days=1)),
        SimpleNamespace(generation_id="gen_ready", state=GenerationState.READY, retired_at=None),
        SimpleNamespace(generation_id="gen_failed", state=GenerationState.FAILED, retired_at=None),
    ]

    class _Manager:
        def __init__(self, dsn: str, tenant: str, **_: object) -> None:
            pass

        def rescue_prune_snapshot(self):  # noqa: ANN202
            return tuple(records), {"gen_active", "gen_prev"}, {"gen_collected"}

    monkeypatch.setattr(generations, "GenerationManager", _Manager)
    for name in ("gen_active.cal_1", "gen_old.cal_1", "gen_prev", "gen_ready.cal_1", "gen_failed.cal_1",
                 "gen_collected.cal_1", "gen_other_tenant.cal_1"):
        _publish(tmp_path / name, "cal_1")

    removed = prune_superseded("postgresql://x", "t", tmp_path)

    left = sorted(path.name for path in tmp_path.iterdir() if path.is_dir())
    assert removed == 3, f"removed {removed}, expected the older retired, the failed and the collected generations' 3"
    assert left == ["gen_active.cal_1", "gen_other_tenant.cal_1", "gen_prev", "gen_ready.cal_1"], left


def test_prune_removes_the_staging_left_by_a_killed_build(tmp_path: Path, monkeypatch) -> None:
    """K6 (final review, Minor 4 re-graded Important): a server killed mid-write leaves a full-size
    `.<generation_id>.<calibration_id>.<hex>` staging directory that no glob of `<gen>.*` matches.
    A prune removes those of the generations it removes; a kept generation's staging may belong to a
    build still running elsewhere, so it stays.

    Red proof: against `898eaf748`, the doomed generation's staging directory was still there.
    """
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import prune_superseded
    from recall.lineage import GenerationState

    class _Manager:
        def __init__(self, dsn: str, tenant: str, **_: object) -> None:
            pass

        def rescue_prune_snapshot(self):  # noqa: ANN202
            records = (
                SimpleNamespace(generation_id="gen_active", state=GenerationState.ACTIVE, retired_at=None),
                SimpleNamespace(generation_id="gen_failed", state=GenerationState.FAILED, retired_at=None),
            )
            return records, {"gen_active"}, set()

    monkeypatch.setattr(generations, "GenerationManager", _Manager)
    (tmp_path / ".gen_failed.cal_1.0123456789abcdef").mkdir()
    (tmp_path / ".gen_active.cal_1.fedcba9876543210").mkdir()

    prune_superseded("postgresql://x", "t", tmp_path)

    assert not (tmp_path / ".gen_failed.cal_1.0123456789abcdef").exists(), "a killed build's staging was kept"
    assert (tmp_path / ".gen_active.cal_1.fedcba9876543210").exists(), "a kept generation's staging was removed"


def test_prune_does_not_remove_staging_while_a_builder_holds_the_lock(tmp_path: Path, monkeypatch) -> None:
    """STAKES-001. On 31e6841d, prune_superseded removed staging under a held build lock."""
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import build_lock, prune_superseded

    manager = SimpleNamespace(
        # The older candidate uses the separate inventory methods.
        list_generations=lambda: (),
        collected_generation_ids=lambda: {"gen_failed"},
        rescue_prune_snapshot=lambda: ((), set(), {"gen_failed"}),
    )
    monkeypatch.setattr(generations, "GenerationManager", lambda *a, **k: manager)
    stage = tmp_path / ".gen_failed.cal_1.0123456789abcdef"
    stage.mkdir()
    with build_lock(tmp_path, "gen_failed") as held:
        assert held
        prune_superseded("postgresql://x", "t", tmp_path)
        assert stage.is_dir(), "prune removed a stage owned by a running worker"
    assert prune_superseded("postgresql://x", "t", tmp_path) == 1
    assert not stage.exists()


def test_prune_enumerates_the_root_once_for_many_obsolete_generations(tmp_path: Path, monkeypatch) -> None:
    """PERF-002. The signed parent enumerated the root twice for every obsolete generation."""
    import os
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import prune_superseded
    from recall.lineage import GenerationState

    obsolete = "gen_old.custom"
    active = f"{obsolete}.keep"
    rollback = f"{obsolete}.previous"
    missing = {f"gen_gone_{index}" for index in range(16)}
    collected = missing | {obsolete, "gen_backoff"}
    records = (
        SimpleNamespace(generation_id=active, state=GenerationState.ACTIVE, retired_at=None),
        SimpleNamespace(generation_id=rollback, state=GenerationState.RETIRED, retired_at=None),
    )
    manager = SimpleNamespace(rescue_prune_snapshot=lambda: (records, {active, rollback}, collected))
    monkeypatch.setattr(generations, "GenerationManager", lambda *args, **kwargs: manager)
    remove = (obsolete, f"{obsolete}.cal_2", f".{obsolete}.0123456789abcdef",
              f".{obsolete}.legacy_cal_3.0123456789abcdef")
    keep = (active, f"{active}.cal_4", f".{active}.cal_4.0123456789abcdef", f"{rollback}.cal_5",
            "gen_other_tenant.cal_3", f".{obsolete}.other.0123456789abcdef")
    for name in (*remove, *keep):
        (tmp_path / name).mkdir()
    backoff = tmp_path / ".rescue-gen_backoff.backoff.json"
    backoff.write_text("{}", encoding="utf-8")
    scandir = os.scandir
    scans = 0

    def count_root(path="."):  # noqa: ANN001, ANN202
        nonlocal scans
        if not isinstance(path, int) and Path(path) == tmp_path:
            scans += 1
        return scandir(path)

    monkeypatch.setattr(os, "scandir", count_root)
    removed = prune_superseded("postgresql://x", "t", tmp_path)

    assert scans == 1, f"prune enumerated the same artifact root {scans} times"
    assert removed == len(remove)
    assert all(not (tmp_path / name).exists() for name in remove)
    assert all((tmp_path / name).is_dir() for name in keep)
    assert not backoff.exists()
    assert all(not (tmp_path / f".rescue-{generation_id}.lock").exists() for generation_id in missing)


def test_prune_keeps_live_artifacts_that_also_match_a_collected_id(
    corpus: _Corpus, tmp_path: Path, monkeypatch  # noqa: F811
) -> None:
    """A calibrated key can also be a different generation's literal legacy key."""
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import build_artifact, build_lock, prune_superseded
    from recall.lineage import GenerationState

    build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    published = tmp_path / "gen_1.cal_1"
    stage = tmp_path / ".gen_1.cal_1.0123456789abcdef"
    stage.mkdir()
    obsolete = tmp_path / "gen_obsolete"
    obsolete.mkdir()
    active = SimpleNamespace(generation_id="gen_1", state=GenerationState.ACTIVE, retired_at=None)
    manager = SimpleNamespace(
        rescue_prune_snapshot=lambda: ((active,), {"gen_1"}, {"gen_1.cal_1", "gen_obsolete"}),
    )
    monkeypatch.setattr(generations, "GenerationManager", lambda *args, **kwargs: manager)

    with build_lock(tmp_path, "gen_1") as held:
        assert held
        prune_superseded("postgresql://x", "t", tmp_path)
        assert (published / "manifest.json").is_file(), "a collected literal key deleted a live calibrated artifact"
        assert stage.is_dir(), "prune took the collected generation's lock to remove a live stage"
    assert not obsolete.exists()
    assert load_atomic_rescue_artifact(published / "manifest.json").generation_id == "gen_1"


def test_prune_preserves_the_rollback_pointer_after_a_clock_step(tmp_path: Path, monkeypatch) -> None:
    """STAKES-002. On 31e6841d, prune_superseded removed the actual rollback target."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import recall.generations as generations
    from recall.generation_rescue import prune_superseded
    from recall.lineage import GenerationState

    previous = tmp_path / "gen_previous.cal_1"
    obsolete = tmp_path / "gen_obsolete.cal_1"
    previous.mkdir()
    obsolete.mkdir()
    records = (
        SimpleNamespace(generation_id="gen_previous", state=GenerationState.RETIRED,
                        retired_at=datetime(2026, 11, 1, tzinfo=UTC)),
        SimpleNamespace(generation_id="gen_obsolete", state=GenerationState.RETIRED,
                        retired_at=datetime(2026, 11, 2, tzinfo=UTC)),
    )
    manager = SimpleNamespace(
        # The DAT-001 parent uses the separate inventory methods.
        list_generations=lambda: records,
        collected_generation_ids=set,
        rescue_prune_snapshot=lambda: (records, {"gen_previous"}, set()),
    )
    monkeypatch.setattr(generations, "GenerationManager", lambda *a, **k: manager)
    prune_superseded("postgresql://x", "t", tmp_path)
    assert previous.is_dir(), "prune removed the actual rollback target"
