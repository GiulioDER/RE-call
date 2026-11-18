"""Generation rescue cleanup and reuse at their normal builder entry point."""

from contextlib import ExitStack
from pathlib import Path

import pytest

from recall.generation_types import NoActiveGeneration
from recall.generation_rescue import build_artifact
from tests.test_atomic_rescue_default_on_cca import _forget, _view_rows
from tests.test_atomic_rescue_default_on_cca import corpus  # noqa: F401
from tests.test_calibration_carry_forward import _CarryEmbedder


def test_a_concurrent_build_cannot_publish_between_the_early_inspection_and_removal(
    corpus, tmp_path: Path, monkeypatch  # noqa: F811
) -> None:
    """BUG-002. The original candidate removed a valid artifact published after the early check."""
    from recall import generation_rescue
    from recall.atomic_rescue import AtomicRescueArtifactError

    serves = generation_rescue.artifact_serves
    remove = generation_rescue._remove_damaged
    overtaking = False
    removed: list[Path] = []
    embedder = _CarryEmbedder()

    def inspect_then_compete(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal overtaking
        result = serves(*args, **kwargs)
        if not overtaking:
            overtaking = True
            try:
                build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)
            except AtomicRescueArtifactError as exc:
                assert "already being built" in str(exc)
        return result

    def observe_removal(manifest: Path) -> None:
        if manifest.is_file():
            removed.append(manifest)
        remove(manifest)

    monkeypatch.setattr(generation_rescue, "artifact_serves", inspect_then_compete)
    monkeypatch.setattr(generation_rescue, "_remove_damaged", observe_removal)
    build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)

    assert overtaking
    assert not removed, "early damage cleanup deleted the competing build's valid artifact"
    assert list(tmp_path.glob("gen_1.*/manifest.json"))


def test_collection_during_publication_removes_this_builds_artifact(corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    from recall import generation_rescue, generation_store

    publish = generation_rescue.write_atomic_rescue_artifact
    select = generation_store.GenerationStore.set_fixed_generation
    collected = False

    def selected(store, generation_id):
        if collected:
            raise NoActiveGeneration("synthetic generation collection")
        select(store, generation_id)

    def publish_then_collect(*args, **kwargs):
        nonlocal collected
        result = publish(*args, **kwargs)
        collected = True
        return result

    monkeypatch.setattr(generation_store.GenerationStore, "set_fixed_generation", selected)
    monkeypatch.setattr(generation_rescue, "write_atomic_rescue_artifact", publish_then_collect)
    with pytest.raises(NoActiveGeneration, match="synthetic generation collection"):
        build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    assert not list(tmp_path.glob("gen_1.*/manifest.json")), "collection left a newly published orphan artifact"


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "host"])
def test_dat_001_forget_preserves_a_live_stage(corpus, tmp_path: Path, monkeypatch, background: bool) -> None:  # noqa: F811
    """Red on 31e6841d plus prior local fixes: erase_forgotten did not report the live stage."""
    from recall import atomic_rescue, generation_rescue
    from recall.lite.atomic import ROOT_ENV

    write = atomic_rescue._write_durable
    inspected: list[Path] = []

    def write_then_forget(path: Path, text: str) -> None:
        write(path, text)
        if path.name == "views.json":
            corpus.forget("forgotten.md")
            with pytest.raises(atomic_rescue.AtomicRescueArtifactError, match="still being built"):
                generation_rescue.erase_forgotten(
                    ["gen_1"], [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)},
                )
            assert path.is_file(), "forget removed a live writer's metadata"
            assert (path.parent / "matrix.npy").is_file(), "forget removed a live writer's matrix"
            inspected.append(path.parent)

    monkeypatch.setattr(atomic_rescue, "_write_durable", write_then_forget)
    if background:
        from types import SimpleNamespace

        from recall_mcp.rescue_builder import RescueBuilder

        store = SimpleNamespace(_dsn="postgresql://x", tenant="t")
        RescueBuilder(store, _CarryEmbedder(), tmp_path)._run(("gen_1", "cal_1"))  # type: ignore[arg-type]
    else:
        with pytest.raises(atomic_rescue.AtomicRescueArtifactError, match="changed while"):
            build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    assert inspected and not inspected[0].exists()
    assert not (tmp_path / "gen_1.cal_1").exists(), "the writer kept an artifact with stale lineage"


def test_dat_001_forget_reports_a_writer_before_its_stage_is_visible(corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """Red on 31e6841d: erase_forgotten reported success while publication was starting."""
    from recall import atomic_rescue, generation_rescue
    from recall.lite.atomic import ROOT_ENV

    publish = generation_rescue.write_atomic_rescue_artifact

    def forget_before_stage(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert not list(tmp_path.glob(".gen_1.*")), "publication already started staging"
        corpus.forget("forgotten.md")
        with pytest.raises(atomic_rescue.AtomicRescueArtifactError, match="still being built"):
            generation_rescue.erase_forgotten(
                ["gen_1"], [], ["forgotten.md"], env={ROOT_ENV: str(tmp_path)},
            )
        return publish(*args, **kwargs)

    monkeypatch.setattr(generation_rescue, "write_atomic_rescue_artifact", forget_before_stage)
    with pytest.raises(atomic_rescue.AtomicRescueArtifactError, match="changed while"):
        build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    assert not list(tmp_path.glob("gen_1.*/manifest.json"))


def test_dat_001_contended_publication_erases_obsolete_cached_views(corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """A foreground build can finish assembly after forget while a competitor owns the lock."""
    from recall import atomic_rescue, atomic_view_store, generation_rescue

    assemble = atomic_view_store.assemble
    with ExitStack() as competing_writer:
        def assemble_after_forget(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            _forget(corpus, tmp_path, "forgotten.md")
            assert competing_writer.enter_context(generation_rescue.build_lock(tmp_path, "gen_1"))
            result = assemble(*args, **kwargs)
            assert _view_rows(tmp_path) == 3, "assembly must commit the obsolete cache row"
            return result

        monkeypatch.setattr(atomic_view_store, "assemble", assemble_after_forget)
        with pytest.raises(atomic_rescue.AtomicRescueArtifactError):
            build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
        assert _view_rows(tmp_path) == 2, "a contended publication left obsolete cached views"
        assert not list(tmp_path.glob("gen_1.*/manifest.json"))


@pytest.mark.parametrize("reuse", [False, True], ids=["assembled", "recalibrated"])
def test_dat_001_writer_rechecks_corpus_after_forget_before_lock(corpus, tmp_path: Path, monkeypatch, reuse: bool) -> None:  # noqa: F811
    """Red with the pre-write _refuse_if_corpus_moved call removed: the writer tried to stage stale data."""
    from recall import atomic_rescue, generation_rescue
    from recall.lite.atomic import ROOT_ENV

    embedder = _CarryEmbedder()
    if reuse:
        build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)
        corpus.repository.calibration_id = "cal_2"
    lock = generation_rescue.build_lock
    lock_calls = 0

    def forget_before_lock(root: Path, generation_id: str):  # noqa: ANN202
        nonlocal lock_calls
        lock_calls += 1
        if lock_calls == 2:  # the publication lock, after the read/assembly
            corpus.forget("forgotten.md")
            generation_rescue.erase_forgotten(
                [generation_id], [], ["forgotten.md"], env={ROOT_ENV: str(root)},
            )
        return lock(root, generation_id)

    def stage_stale_data(*args: object, **kwargs: object) -> None:
        raise AssertionError("the writer tried to stage stale data after forget succeeded")

    monkeypatch.setattr(generation_rescue, "build_lock", forget_before_lock)
    monkeypatch.setattr(generation_rescue, "write_atomic_rescue_artifact", stage_stale_data)
    with pytest.raises(atomic_rescue.AtomicRescueArtifactError, match="changed while"):
        build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)
    assert lock_calls == 3  # the initial check, publication, and the intervening forget
    assert not list(tmp_path.glob(".gen_1.*"))
