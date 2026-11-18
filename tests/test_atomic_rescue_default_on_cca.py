"""CCA run 20261116T104135Z over the atomic rescue turned on at first install (#1323 to #1329).

No database: the PostgreSQL store and calibration repository `build_artifact` reads are replaced by
fakes holding a corpus fingerprint that a forget changes, which is the one thing the race fixes
read. Each test names the finding it pins; its red proof is in the pull request.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from recall.lite.atomic import ROOT_ENV
from recall.types import Chunk
from tests.test_calibration_carry_forward import _CarryEmbedder

_TEXTS = {
    "kept-a.md": "the harbour office keeps the tide tables for every pier along the eastern quay",
    "kept-b.md": "the archive clerk balances the ledgers each evening before the doors are locked",
    "forgotten.md": "the kingfisher protocol reconciles drifting ledger totals with the morning count",
}


class _Corpus:
    """One generation's rows, and the fingerprint a forget rewrites (`GenerationManager.forget`)."""

    def __init__(self) -> None:
        self.chunks = [
            Chunk(id=f"c{index}", source=source, text=text, metadata={"ord": 0})
            for index, (source, text) in enumerate(sorted(_TEXTS.items()))
        ]
        self.fingerprint = "corpus-1"
        self.recorded: list[tuple[str, str]] = []  # (generation, root) a build recorded

    def forget(self, source: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.source != source]
        self.fingerprint = "corpus-2"


@pytest.fixture
def corpus(monkeypatch) -> _Corpus:
    import recall.calibration_v2 as calibration
    import recall.generation_store as generation_store
    import recall.generations as generations
    from recall.calibration_v2 import CalibrationStatus

    db = _Corpus()
    embedder = _CarryEmbedder()

    class _Store:
        def __init__(self, dsn: str, dim: int, *, tenant: str) -> None:
            self.generation_id = ""

        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def set_fixed_generation(self, generation_id: str) -> None:
            self.generation_id = generation_id

        def generation_binding(self) -> dict[str, str]:
            return {
                "generation_id": self.generation_id,
                "pipeline_fingerprint": "pipe-1",
                "corpus_fingerprint": db.fingerprint,
                "embedder_model": embedder.name,
                "embedder_dimension": str(embedder.dim),
            }

        def iter_chunks(self):  # noqa: ANN202
            return iter(list(db.chunks))

    class _Repository:
        calibration_id = "cal_1"

        def __init__(self, dsn: str, tenant: str, **_: object) -> None:
            pass

        def resolve(self, generation_id: str) -> Any:
            return SimpleNamespace(
                status=CalibrationStatus.CERTIFIED,
                artifact=SimpleNamespace(calibration_id=_Repository.calibration_id),
            )

    monkeypatch.setattr(generation_store, "GenerationStore", _Store)
    class _Manager:
        def __init__(self, dsn: str, tenant: str, **_: object) -> None:
            pass

        def record_rescue_root(self, generation_id: str, root: str) -> None:
            db.recorded.append((generation_id, root))

    monkeypatch.setattr(calibration, "CalibrationRepository", _Repository)
    monkeypatch.setattr(generations, "GenerationManager", _Manager)
    db.repository = _Repository  # type: ignore[attr-defined]
    clear_atomic_rescue_artifact_cache()
    yield db
    clear_atomic_rescue_artifact_cache()


def _forget(db: _Corpus, root: Path, source: str) -> None:
    """A forget as `recall_mcp.lifecycle` runs it: rows deleted, then the rescue erasure."""
    from recall.generation_rescue import erase_forgotten

    db.forget(source)
    erase_forgotten(["gen_1"], [_TEXTS[source]], [source], env={ROOT_ENV: str(root)})


def _view_rows(root: Path) -> int:
    from recall.atomic_view_store import AtomicViewStore

    with AtomicViewStore(root / "micro-views.sqlite") as views:
        return views.count()


def test_a_forget_during_the_assembly_is_not_undone(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:
    """BUG-004 (P1): a forget whose erasure ran after the build read the chunks and before it
    stored their views found nothing to erase; the build then stored the forgotten text's views and
    published an artifact naming its source, which no later forget can reach.

    Invariant: the build refuses, and neither the view store nor the root keeps the forgotten text.
    """
    import recall.atomic_view_store as view_store
    from recall.generation_rescue import build_artifact

    real = view_store.assemble

    def forgotten_meanwhile(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        _forget(corpus, tmp_path, "forgotten.md")
        return real(*args, **kwargs)

    monkeypatch.setattr(view_store, "assemble", forgotten_meanwhile)

    with pytest.raises(AtomicRescueArtifactError, match="changed while"):
        build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)

    assert _view_rows(tmp_path) == 2, "the forgotten text's views were stored after its erasure"
    assert not (tmp_path / "gen_1.cal_1").exists()


def test_a_forget_during_the_write_is_not_undone(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:
    """BUG-004 (P1), the second window: a forget that committed after the assembly and scanned the
    root before the artifact existed. Invariant: the artifact this build wrote is removed."""
    import recall.generation_rescue as rescue

    real = rescue.write_atomic_rescue_artifact

    def forgotten_meanwhile(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        try:
            _forget(corpus, tmp_path, "forgotten.md")
        except AtomicRescueArtifactError as exc:
            # The production forget reports this root as incomplete and lets the writer finish.
            assert "still being built" in str(exc)
        return real(*args, **kwargs)

    monkeypatch.setattr(rescue, "write_atomic_rescue_artifact", forgotten_meanwhile)

    with pytest.raises(AtomicRescueArtifactError, match="changed while"):
        rescue.build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)

    assert not (tmp_path / "gen_1.cal_1").exists(), "an artifact naming the forgotten source was kept"


def test_a_failed_assembly_clears_views_written_after_forget(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:
    """A late assembly error must clean views written after a generation's corpus changed."""
    import recall.atomic_view_store as view_store
    from recall.generation_rescue import build_artifact

    assemble = view_store.assemble

    def forget_then_fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        _forget(corpus, tmp_path, "forgotten.md")
        assemble(*args, **kwargs)
        assert _view_rows(tmp_path) == 3, "negative control: no forgotten views were committed"
        raise AtomicRescueArtifactError("synthetic later assembly failure")

    monkeypatch.setattr(view_store, "assemble", forget_then_fail)
    with pytest.raises(AtomicRescueArtifactError, match="synthetic later assembly failure"):
        build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    assert _view_rows(tmp_path) == 2, "forgotten views survived the failed build"


def _publish(directory: Path, calibration_id: str, *, pipeline: str = "p") -> None:
    write_atomic_rescue_artifact(
        directory,
        matrix=np.eye(1, 4, dtype=np.float32),
        views=[{"chunk_id": "c1", "source": "a.md", "parent_ordinal": 0, "view_ordinal": 0}],
        generation_id=directory.name,
        calibration_id=calibration_id,
        pipeline_fingerprint=pipeline,
        corpus_fingerprint="c",
        embedding_profile="hashing",
        embedding_fingerprint="f",
        ordinary_chunk_count=1,
        source_commit="test",
    )


def test_an_artifact_rebuilt_in_place_is_reloaded(tmp_path: Path) -> None:
    """BUG-001: a PostgreSQL artifact is rebuilt at the same path for a new calibration, and the
    loader's cache, keyed by path alone, kept serving every other process the replaced one, whose
    lineage then refused each unscoped search until a restart.

    Invariant: a load after the files were replaced (without this process clearing its cache, as
    another process's rebuild never does) returns the new artifact.
    """
    clear_atomic_rescue_artifact_cache()
    directory = tmp_path / "gen_1"
    _publish(directory, "cal_1")
    assert load_atomic_rescue_artifact(directory / "manifest.json").calibration_id == "cal_1"

    shutil.rmtree(directory)
    _publish(directory, "cal_2")

    assert load_atomic_rescue_artifact(directory / "manifest.json").calibration_id == "cal_2"
    clear_atomic_rescue_artifact_cache()


def test_a_build_keeps_the_artifact_another_process_published(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:
    """BUG-002: the re-check after the assembly read the stale artifact this build had loaded
    before, from the cache, so it deleted the one a concurrent build in another process published.

    Invariant: the winner's artifact is kept and reported in force.
    """
    import recall.atomic_view_store as view_store
    from recall.embeddings import embedding_profile, embedding_profile_id
    from recall.generation_rescue import build_artifact

    embedder = _CarryEmbedder()
    _publish(tmp_path / "gen_1", "cal_0", pipeline="pipe-0")  # stale: another pipeline
    corpus.repository.calibration_id = "cal_2"  # type: ignore[attr-defined]
    real = view_store.assemble
    winner: list[str] = []

    def published_elsewhere(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        matrix, metadata, counts = real(*args, **kwargs)
        # Another process publishes the same key first, and cannot clear this process's cache.
        write_atomic_rescue_artifact(
            tmp_path / "gen_1.cal_2",
            matrix=matrix,
            views=metadata,
            generation_id="gen_1",
            calibration_id="cal_2",
            pipeline_fingerprint="pipe-1",
            corpus_fingerprint="corpus-1",
            embedding_profile=embedding_profile_id(embedder),
            embedding_fingerprint=embedding_profile(embedder).fingerprint(),
            ordinary_chunk_count=len(args[0]),
            source_commit="winner",
        )
        winner.append(json.loads((tmp_path / "gen_1.cal_2" / "manifest.json").read_text(encoding="utf-8"))["constructed_at"])
        return matrix, metadata, counts

    monkeypatch.setattr(view_store, "assemble", published_elsewhere)

    report = build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)

    assert report == "in force for gen_1", report
    manifest = json.loads((tmp_path / "gen_1.cal_2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["constructed_at"] == winner[0]


def test_a_recalibration_reuses_the_views_in_place(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:
    """PERF-001: a new calibration of an unchanged corpus re-scanned every chunk and assembled every
    view again. Invariant: the rebuild copies the artifact in place, and serves the new calibration.
    """
    import recall.atomic_view_store as view_store
    from recall.generation_rescue import build_artifact

    embedder = _CarryEmbedder()
    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    corpus.repository.calibration_id = "cal_2"  # type: ignore[attr-defined]

    def assembled(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("the views of an unchanged corpus were assembled again")

    monkeypatch.setattr(view_store, "assemble", assembled)

    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    clear_atomic_rescue_artifact_cache()
    assert load_atomic_rescue_artifact(tmp_path / "gen_1.cal_2" / "manifest.json").calibration_id == "cal_2"


def test_an_active_rescue_without_a_root_refuses_the_promotion() -> None:
    """DOC-007: with the mode active and no root, `ensure_for_promotion` returned None and the
    promotion went ahead with no artifact, so the generation refused every unscoped search."""
    from recall.generation_rescue import ensure_for_promotion

    def embedder():  # noqa: ANN202
        raise AssertionError("no model is loaded to refuse")

    with pytest.raises(AtomicRescueArtifactError, match="not set"):
        ensure_for_promotion("postgresql://x", "t", "gen", embedder, env={"RECALL_ATOMIC_RESCUE_MODE": "active"})
    by_tenant = {"RECALL_ATOMIC_RESCUE_MODE_BY_TENANT": "t:active"}
    with pytest.raises(AtomicRescueArtifactError, match="not set"):
        ensure_for_promotion("postgresql://x", "t", "gen", embedder, env=by_tenant)


class _GcManager:
    def __init__(self, dsn: str, tenant: str, **_: object) -> None:
        self.last_gc_facts_retracted = 0

    def gc(self, **_: object) -> tuple[str, ...]:
        return ("gen_old",)

    def collected_generation_ids(self) -> set[str]:
        # `gen_older` was collected by an earlier gc that could not remove its artifact.
        return {"gen_old", "gen_older"}


def test_gc_retries_an_artifact_an_earlier_gc_left(tmp_path: Path, monkeypatch) -> None:
    """BUG-003: gc removed only the artifacts of the generations it collected itself, so one an
    earlier gc could not remove (Windows, a mapped matrix) was never retried, although its warning
    said the next gc would, and no forget lists a collected generation."""
    import recall.generations as generations
    from recall.cli_commands.generation_cmd import _cmd_generation

    monkeypatch.setattr(generations, "GenerationManager", _GcManager)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    for generation_id in ("gen_old", "gen_older", "gen_live"):
        _publish(tmp_path / generation_id, "cal_1")

    _cmd_generation(argparse.Namespace(
        dsn="postgresql://x", tenant="t", embedder="hashing", generation_cmd="gc",
        retention_days=7, retain_previous=2,
    ))

    assert not (tmp_path / "gen_older").exists(), "the artifact an earlier gc left was not retried"
    assert not (tmp_path / "gen_old").exists() and (tmp_path / "gen_live").exists()


def test_a_cli_forget_on_a_default_lite_install_erases_the_rescue(tmp_path: Path) -> None:
    """ENV-001: `recall setup --lite` registers the root on the MCP server only, so a CLI forget,
    whose environment has no root, erased nothing from `<store>.atomic` and said nothing."""
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite.atomic import erase_forgotten

    text = _TEXTS["forgotten.md"]
    store_path = tmp_path / "memory.lite"
    default = tmp_path / "memory.lite.atomic"
    default.mkdir()
    with AtomicViewStore(default / "micro-views.sqlite") as views:
        views.put("profile", chunk_view_digest(text), [[1.0, 0.0, 0.0, 0.0]])
        views.commit()
    store = SimpleNamespace(
        path=store_path,
        tenant="t",
        resolve_calibration=lambda: SimpleNamespace(status=CalibrationStatus.CERTIFIED, artifact=None),
    )

    erased, _removed = erase_forgotten(store, [text], ["forgotten.md"], env={})  # type: ignore[arg-type]

    assert erased == 1


def test_a_rescue_left_off_keeps_the_root_an_earlier_run_built_under(tmp_path: Path) -> None:
    """ENV-002: a re-run that leaves the rescue off (missing dependencies, the Docker stack) dropped
    the root, and the wizard then removed it from `.env`, so later forgets skipped what the earlier
    run built there. Invariant: the root is kept, with the mode off."""
    from recall.lite.atomic import rescue_settings

    built = tmp_path / "atomic"
    built.mkdir()

    settings, note = rescue_settings(built, off_reason="the Docker services cannot reach it", env={})

    assert settings == {"RECALL_ATOMIC_RESCUE_MODE": "off", ROOT_ENV: str(built)}
    assert "off" in note
    assert rescue_settings(tmp_path / "never-built", off_reason="x", env={})[0] == {}


def test_a_resumed_wizard_run_builds_the_rescue_of_a_reused_corpus(tmp_path: Path, monkeypatch) -> None:
    """DAT-004: a reused corpus kept the `atomic` its earlier run recorded and no builder ran, so a
    run with another root registered it active over a root with no artifact, and one after
    installing the extra (as the note advises) left it pinned off.

    Invariant: a reused corpus is built again under this run's settings.
    """
    from recall.wizard.headless import _RealServices
    from tests.test_wizard_state import _CountingSpy, _config

    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE_BY_TENANT", raising=False)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path / "atomic"))
    config = _config(tmp_path)
    state_path = tmp_path / "s.json"
    from recall.wizard.headless import run_headless

    run_headless(config, services=_CountingSpy(), state_path=state_path)

    class _Resumed(_CountingSpy, _RealServices):
        def __init__(self) -> None:
            _CountingSpy.__init__(self)
            self.config = config
            self._embedder = None
            self.atomic_env = {}
            self.rescued: list[str] = []

        def configure_fact_boundary(self, *args: object, **kwargs: object) -> None:
            pass

        def has_legacy_rows(self, *args: object) -> bool:
            return False

        def smoke(self, block: Any) -> Any:
            raise RuntimeError("no database in this test")

        def _rescue_builder(self, tenant: str):  # noqa: ANN202
            if self.atomic_env.get("RECALL_ATOMIC_RESCUE_MODE", "").lower() != "active":
                return None
            return lambda generation_id: self.rescued.append(generation_id) or f"in force for {generation_id}"

    resumed = _Resumed()
    report = run_headless(config, services=resumed, state_path=state_path)

    assert resumed.built == [], "the corpora were reused, not rebuilt"
    assert set(report.reused) >= {"default-docs", "default-code"}
    assert sorted(resumed.rescued) == ["gen_default-code", "gen_default-docs"]

    class _Unbuildable(_Resumed):
        def _rescue_builder(self, tenant: str):  # noqa: ANN202
            def refuse(generation_id: str) -> str:
                # Not an AtomicRescueArtifactError: a provider or database error reaches here too.
                raise ValueError("the embedding provider refused the request")

            return refuse

    # A reused corpus is already serving: a rebuild that fails leaves it served, rescue off.
    unbuildable = _Unbuildable()
    report = run_headless(config, services=unbuildable, state_path=state_path)

    assert not report.failures, report.failures
    reused = {outcome.tenant: outcome for outcome in report.outcomes}
    assert {"default-docs", "default-code"} <= set(reused)
    assert reused["default-docs"].atomic == "" and reused["default-code"].atomic == ""
