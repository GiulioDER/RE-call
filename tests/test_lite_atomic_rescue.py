"""Atomic rescue on a lite store (`recall.lite.atomic`): built after each write, served as on PostgreSQL.

- A1 an active lite store, refreshed, moves the parent `select_atomic_rescue` picks (the selector the
  PostgreSQL path runs) to final rank six, scored by its own chunk cosine; the top five are those
  of the same search with the rescue off, and without the rescue that parent is not in the top six.
- A2 with the rescue off (the default) a refresh builds nothing and creates no artifact root.
- A3 a second build after one memo is added embeds only that memo's views.
- A4 a background refit that lands builds the artifact its new calibration names.
- A5 `scored_chunk_for_query` and `scored_chunk_by_id` load a parent with the score dense search
  gives it, and nothing for an id the store lacks.
- A6 a build that loses the race to publish the same artifact (a refresh beside a background refit)
  accepts the winner's instead of failing its write.
- A7 a build overtaken by a background refit (its own lane) leaves the refit's artifact, the one
  the store now resolves, in place, and publishes nothing (CCA STAKES-001).
- A8 a build that loses the race at the rename (not at the up-front check) accepts the winner;
  A8b, a control: a publish that fails with no winner still fails (CCA STAKES-002).
- A9 a refresh whose build fails still reconciles the stated facts of the write it follows, and
  reports the failure (CCA STAKES-003).
- A10 an artifact directory a removal left partly behind is rebuilt, not reported in force from
  the process cache (CCA STAKES-004).
- A10b the rebuild is not blocked by this process's own cached mapping of that matrix, which
  Windows refuses to delete (CCA STAKES-004).
- A11 a relative artifact root is refused by the builder and the search path (CCA STAKES-005).
- A12 the artifact root carries a `.gitignore` of `*` (CCA SEC-001).
- A13 `recall search` on a lite store resolves the tenant's rescue mode as the builder does (CCA
  ENV-001).
- A14 `recall_calibration_run` on a lite store builds the artifact its new fit names (CCA CODE-001).
- A15 two stores sharing one artifact root: a build for the second leaves the first's artifact,
  and the first's active search still succeeds.
- A16 an artifact at the top of the root, where builds put it before per-store folders, still
  serves the search whose key it names.
- A17 a build leaves such top-level artifact folders in place, since their owner is unknown, and
  its report says how many remain.

Red proof: A1 to A6 in the pull request that added this file (mutations, test node ids and the
failing assertion of each). A7 to A14 in the commit that added them: each run against that
commit's parent, `5a10e6e03`, and failing in its own assertion, except the two below, recorded
there too. A8b is a control that passes on the parent by design; it is red with
`if not manifest.is_file(): raise` removed from `ensure_rescue_artifact`. A10b is red with the
`clear_atomic_rescue_artifact_cache()` call before the removal deleted.

A15 to A17 are red as follows; each is green with the fix. A15
(`test_two_stores_sharing_a_root_keep_each_others_artifacts`) run against `195435c19`, the
commit before per-store folders, fails at "a build for one store removed the artifact of another
store sharing its root": `_remove_others` scanned the whole root. A16
(`test_a_flat_layout_artifact_still_serves`) is red with `serving_manifest` returning
`manifest_path` alone, at "an artifact built before per-store folders no longer serves". A17
(`test_a_build_leaves_flat_layout_artifacts_and_reports_them`) is red with
`_remove_others(root, ...)` in place of the store's folder, at "a build removed an artifact
folder whose owner it cannot know".
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    select_atomic_rescue,
)
from recall.calibration_v2 import CalibrationStatus
from recall.embeddings import embed_query
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite import atomic
from recall.lite.atomic import VIEW_STORE_NAME, ensure_rescue_artifact, manifest_path
from recall.lite.calibration import ensure_calibrated, wait_for_refit
from recall.lite.refresh import Outcome, refresh
from recall.trust import trusted_search
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos

QUERY = "quasar beacon recalibration"
TERMS = QUERY.split()


@pytest.fixture(autouse=True)
def _fresh_artifact_cache() -> None:
    clear_atomic_rescue_artifact_cache()


def _decoys(root: Path) -> None:
    """Six memos with every query word, far apart: high whole-chunk cosine, low view cosine.

    Two of the words twice, so the keyword leg ranks every decoy above the needle as dense search
    does: with one occurrence each the seven memos tie there, and a tie falls to chunk ids, which
    depend on the folder the test runs in. No 24-word window holds all three words.
    """
    first, second, third = TERMS
    for n in range(6):
        words = [f"d{n}x{k:02d}" for k in range(60)]
        words[0] = words[30] = first
        words[15] = words[45] = second
        words[59] = third
        (root / f"decoy-{n}.md").write_text(" ".join(words) + "\n", encoding="utf-8")


def _needle(root: Path) -> None:
    """One memo whose three query words sit together in a long chunk: a view finds what the chunk hides."""
    words = [f"lgw{k:03d}" for k in range(80)]
    words[40:40] = TERMS
    (root / "ops-log.md").write_text(" ".join(words) + "\n", encoding="utf-8")


def _corpus(tmp_path: Path) -> Path:
    memos = tmp_path / "m"
    _write_memos(memos, 48)
    _decoys(memos)
    _needle(memos)
    return memos


def _env(root: Path) -> dict[str, str]:
    return {
        "RECALL_ATOMIC_RESCUE_MODE": "active",
        "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(root),
        "RECALL_ATOMIC_RESCUE_PLACEMENT": "fused",
    }


def _ids(result) -> list[str]:  # noqa: ANN001
    return [hit.chunk.id for hit in result.hits]


def test_an_active_lite_store_rescues_a_parent_as_the_postgres_path_does(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "atomic"
    env = _env(root)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    report = refresh(store, [_corpus(tmp_path)], embedder)
    assert report.outcome is Outcome.REFRESHED, report.to_dict()
    assert report.calibration == CalibrationStatus.CERTIFIED.value, report.calibration_reason
    assert report.atomic.startswith("built for "), f"the refresh built no artifact: {report.atomic!r}"

    resolution = store.resolve_calibration()
    assert resolution.artifact is not None
    manifest = manifest_path(root, store, store.lite_generation_id(), resolution.artifact.calibration_id)
    artifact = load_atomic_rescue_artifact(manifest)
    vector = embed_query(embedder, QUERY)
    selection = select_atomic_rescue(artifact, vector, store.query_dense(vector, 20))
    assert Path(selection.source).name == "ops-log.md", "the fixture no longer hides its needle from the chunk"

    off = trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={})
    on = trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env=env)
    assert selection.chunk_id not in _ids(off)[:6], "the parent ranked in the top six without the rescue"
    assert _ids(on)[:5] == _ids(off)[:5], "the rescue moved the top five"
    assert _ids(on)[5] == selection.chunk_id, "the selected parent is not at rank six"
    own = store.cosines_for([selection.chunk_id], vector)[selection.chunk_id]
    assert on.hits[5].cosine == pytest.approx(own), "the rescued parent was not scored by its own cosine"


def test_the_rescue_is_off_by_default(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "atomic"
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", str(root))
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    report = refresh(store, [_corpus(tmp_path)], ContentWordEmbedder())
    assert report.outcome is Outcome.REFRESHED, report.to_dict()
    assert report.atomic == "" and not root.exists(), "a refresh with the rescue off built an artifact"


def test_a_second_build_embeds_only_the_new_memo(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    env = _env(root)
    memos = _corpus(tmp_path)
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(memos)
    assert ensure_calibrated(store, embedder).status is CalibrationStatus.CERTIFIED
    first = ensure_rescue_artifact(store, embedder, env=env)
    assert first is not None and first.manifest is not None, first
    (memos / "added.md").write_text(" ".join(f"add{k:02d}" for k in range(30)) + "\n", encoding="utf-8")
    Indexer(store, embedder, env={}).index_path(memos)
    assert ensure_calibrated(store, embedder).status is CalibrationStatus.CERTIFIED
    # The first build's own check left its matrix mapped in this process's artifact cache, and
    # Windows refuses to delete a mapped file; a serving process releases it on eviction.
    clear_atomic_rescue_artifact_cache()
    second = ensure_rescue_artifact(store, embedder, env=env)
    assert second is not None and second.manifest is not None, second
    added = second.counts["views"] - first.counts["views"]
    assert added > 0 and second.counts["embedded_views"] == added, (
        f"the second build embedded {second.counts['embedded_views']} view(s), the new memo has {added}"
    )
    assert not first.manifest.parent.exists(), "the first corpus's artifact was left behind"
    assert (root / VIEW_STORE_NAME).exists()


def test_a_background_refit_builds_the_artifact_its_calibration_names(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "atomic"
    for name, value in _env(root).items():
        monkeypatch.setenv(name, value)
    memos = _corpus(tmp_path)
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(memos)
    fitted = ensure_calibrated(store, embedder)
    assert fitted.status is CalibrationStatus.CERTIFIED and fitted.artifact is not None
    # Past the bridge a carried threshold may span, so the refit is needed and runs in the background.
    _write_memos(memos, 120, start=48)
    Indexer(store, embedder, env={}).index_path(memos)
    ensure_calibrated(store, embedder, background=True)
    landed = wait_for_refit(store, timeout=120)
    assert landed is not None and landed.status is CalibrationStatus.CERTIFIED, landed
    resolution = store.resolve_calibration()
    assert resolution.artifact is not None and resolution.artifact.calibration_id != fitted.artifact.calibration_id
    manifest = manifest_path(root, store, store.lite_generation_id(), resolution.artifact.calibration_id)
    assert manifest.exists(), "the refit's calibration has no atomic rescue artifact"


def test_a_parent_is_loaded_with_its_dense_score(tmp_path: Path) -> None:
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(_corpus(tmp_path))
    # Not unit length, as a query vector need not be: dense search normalises it, and so must a loader.
    vector = [3.0 * x for x in embed_query(embedder, QUERY)]
    for hit in store.query_dense(vector, 10):
        loaded = store.scored_chunk_for_query(hit.chunk.id, vector)
        assert loaded is not None and loaded.chunk.id == hit.chunk.id
        assert loaded.score == pytest.approx(hit.score, abs=1e-6), "a loaded parent scored unlike dense search"
        by_id = store.scored_chunk_by_id(hit.chunk.id, 0.25)
        assert by_id is not None and by_id.score == 0.25 and by_id.chunk.text == hit.chunk.text
    assert store.scored_chunk_for_query("absent", vector) is None
    assert store.scored_chunk_by_id("absent", 0.5) is None


def test_a_build_that_loses_the_race_to_publish_accepts_the_winner(tmp_path: Path, monkeypatch) -> None:
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(_corpus(tmp_path))
    assert ensure_calibrated(store, embedder).status is CalibrationStatus.CERTIFIED
    real = atomic.write_atomic_rescue_artifact

    def racing(directory, **fields):  # noqa: ANN001, ANN202
        real(directory, **fields)  # the other build publishes first
        return real(directory, **fields)

    monkeypatch.setattr(atomic, "write_atomic_rescue_artifact", racing)
    try:
        report = ensure_rescue_artifact(store, embedder, env=_env(tmp_path / "atomic"))
    except FileExistsError:
        pytest.fail("a build that lost the race to publish failed its write")
    assert report is not None and report.manifest is not None and report.manifest.exists(), report


def _certified(tmp_path: Path) -> tuple[LiteStore, ContentWordEmbedder]:
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(_corpus(tmp_path))
    assert ensure_calibrated(store, embedder).status is CalibrationStatus.CERTIFIED
    return store, embedder


def _in_force(store: LiteStore, root: Path) -> Path:
    resolution = store.resolve_calibration()
    assert resolution.artifact is not None
    return manifest_path(root, store, store.lite_generation_id(), resolution.artifact.calibration_id)


def test_a_build_overtaken_by_a_refit_leaves_the_refit_artifact_in_force(tmp_path: Path, monkeypatch) -> None:
    import recall.atomic_view_store as view_store
    from recall.lite.calibration import auto_calibrate

    root = tmp_path / "atomic"
    env = _env(root)
    store, embedder = _certified(tmp_path)
    real = view_store.assemble
    raced: list[bool] = []

    def overtaken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        built = real(*args, **kwargs)
        if not raced:
            raced.append(True)
            # A background refit stores a new fit and publishes its artifact meanwhile.
            auto_calibrate(store, embedder)
            refit = ensure_rescue_artifact(store, embedder, env=env)
            assert refit is not None and refit.manifest is not None, refit
        return built

    monkeypatch.setattr(view_store, "assemble", overtaken)
    report = ensure_rescue_artifact(store, embedder, env=env)
    assert _in_force(store, root).is_file(), "a build overtaken by a refit removed the artifact the store now needs"
    assert report is not None and report.manifest is None, report


def test_a_build_that_loses_the_race_at_the_rename_accepts_the_winner(tmp_path: Path, monkeypatch) -> None:
    store, embedder = _certified(tmp_path)
    real = atomic.write_atomic_rescue_artifact

    def collides(directory, **fields):  # noqa: ANN001, ANN003, ANN202
        real(directory, **fields)  # the other build renamed its directory into place first
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(directory))

    monkeypatch.setattr(atomic, "write_atomic_rescue_artifact", collides)
    try:
        report = ensure_rescue_artifact(store, embedder, env=_env(tmp_path / "atomic"))
    except OSError:
        pytest.fail("a build that lost the race at the rename failed its write")
    assert report is not None and report.manifest is not None and report.manifest.is_file(), report


def test_a_publish_that_fails_with_no_winner_still_fails(tmp_path: Path, monkeypatch) -> None:
    store, embedder = _certified(tmp_path)

    def full(directory, **fields):  # noqa: ANN001, ANN003, ANN202
        raise OSError(errno.ENOSPC, "No space left on device", str(directory))

    monkeypatch.setattr(atomic, "write_atomic_rescue_artifact", full)
    with pytest.raises(OSError, match="No space left"):
        ensure_rescue_artifact(store, embedder, env=_env(tmp_path / "atomic"))


def test_a_failed_build_still_records_the_write_it_follows(tmp_path: Path, monkeypatch) -> None:
    import recall.stated_facts as stated_facts

    for name, value in _env(tmp_path / "atomic").items():
        monkeypatch.setenv(name, value)
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    reconciled: list[bool] = []
    real = stated_facts.reconcile_stated_facts

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        reconciled.append(True)
        return real(*args, **kwargs)

    def broken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("view embedding failed")

    monkeypatch.setattr(stated_facts, "reconcile_stated_facts", counted)
    monkeypatch.setattr(atomic, "ensure_rescue_artifact", broken)
    report = refresh(store, [_corpus(tmp_path)], ContentWordEmbedder())
    assert reconciled, "a failed atomic rescue build skipped the stated facts of the write it follows"
    assert report.outcome is Outcome.FAILED and "atomic rescue" in report.reason, report.to_dict()


def test_a_partly_removed_artifact_is_rebuilt_not_reported_in_force(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    store, embedder = _certified(tmp_path)
    first = ensure_rescue_artifact(store, embedder, env=_env(root))
    assert first is not None and first.manifest is not None, first
    # What a removal leaves on Windows while a server has the matrix mapped; this process's
    # artifact cache still holds the key.
    first.manifest.unlink()
    (first.manifest.parent / "views.json").unlink()
    again = ensure_rescue_artifact(store, embedder, env=_env(root))
    assert again is not None
    assert first.manifest.is_file(), f"a partly removed artifact was reported {again.reason!r} without its manifest"


def test_a_relative_artifact_root_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store, embedder = _certified(tmp_path)
    with pytest.raises(AtomicRescueArtifactError, match="absolute"):
        ensure_rescue_artifact(store, embedder, env=_env(Path("relative-root")))
    assert not (tmp_path / "relative-root").exists(), "a relative root was built under the working directory"
    with pytest.raises(AtomicRescueArtifactError, match="absolute"):
        store.atomic_rescue_manifest("relative-root", store.lite_generation_id(), "cal_0")


def test_the_artifact_root_ignores_itself(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    store, embedder = _certified(tmp_path)
    ensure_rescue_artifact(store, embedder, env=_env(root))
    ignore = root / ".gitignore"
    assert ignore.is_file() and ignore.read_text(encoding="utf-8").splitlines()[-1] == "*", (
        "the artifact root, which holds memo-derived vectors and sources, is not ignored by git"
    )


def test_the_cli_search_resolves_the_tenant_rescue_mode_as_the_builder_does(tmp_path: Path, monkeypatch) -> None:
    from recall.cli_commands import index_search
    from tests.lite_cli_support import run_cli

    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    memos = tmp_path / "mem"
    memos.mkdir()
    (memos / "rate.md").write_text("# Rate limit\n\nThe public API rate limit is 100 per second.\n", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", "sqlite:///" + (tmp_path / "store" / "memory.db").as_posix()]
    indexed = run_cli([*base, "index", str(memos)])
    assert "indexed" in indexed, indexed
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", str(tmp_path / "atomic"))
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE_BY_TENANT", "default:off")
    seen: dict[str, object] = {}

    def captured(store, embedder, query, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen["env"] = kwargs.get("env")
        raise SystemExit("search captured")

    monkeypatch.setattr(index_search, "trusted_search", captured)
    out = run_cli([*base, "search", "rate limit"])
    assert "search captured" in out, out
    assert atomic.configured_root("default") is None, "the fixture no longer turns the tenant off for the builder"
    searched = seen["env"] if seen["env"] is not None else os.environ
    assert searched.get("RECALL_ATOMIC_RESCUE_MODE") == "off", (  # type: ignore[union-attr]
        "the CLI search ignored the tenant's rescue mode the builder applies"
    )


def test_a_calibration_run_builds_the_artifact_its_fit_names(tmp_path: Path, monkeypatch) -> None:
    from recall.lite.calibration import run_report

    root = tmp_path / "atomic"
    for name, value in _env(root).items():
        monkeypatch.setenv(name, value)
    store, embedder = _certified(tmp_path)
    before = store.resolve_calibration().artifact
    run_report(store, embedder)
    after = store.resolve_calibration().artifact
    assert before is not None and after is not None and after.calibration_id != before.calibration_id
    assert _in_force(store, root).is_file(), "a calibration run stored a fit with no atomic rescue artifact"


def test_the_rebuild_is_not_blocked_by_this_process_own_cached_mapping(tmp_path: Path, monkeypatch) -> None:
    import shutil

    import recall.atomic_rescue as rescue

    root = tmp_path / "atomic"
    store, embedder = _certified(tmp_path)
    first = ensure_rescue_artifact(store, embedder, env=_env(root))
    assert first is not None and first.manifest is not None, first
    manifest = first.manifest
    real = shutil.rmtree

    def windows(path, ignore_errors=False, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        # Windows refuses to delete a matrix a cached artifact of this process still maps.
        if (Path(path) / "manifest.json").resolve() in rescue._ARTIFACT_CACHE:
            for entry in Path(path).iterdir():
                if entry.name != "matrix.npy":
                    entry.unlink()
            if not ignore_errors:
                raise PermissionError(f"{path}/matrix.npy is mapped")
            return None
        return real(path, ignore_errors, *args, **kwargs)

    monkeypatch.setattr(atomic.shutil, "rmtree", windows)
    manifest.unlink()  # what an earlier removal left of this key, which now comes back into force
    try:
        again = ensure_rescue_artifact(store, embedder, env=_env(root))
    except AtomicRescueArtifactError as exc:
        pytest.fail(f"the rebuild was blocked by this process's own cached mapping: {exc}")
    assert again is not None and manifest.is_file(), f"no artifact was rebuilt: {again}"


def test_two_stores_sharing_a_root_keep_each_others_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    env = _env(root)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, embedder = _certified(tmp_path / "a")
    second, _ = _certified(tmp_path / "b")
    built_first = ensure_rescue_artifact(first, embedder, env=env)
    assert built_first is not None and built_first.manifest is not None, built_first
    built_second = ensure_rescue_artifact(second, embedder, env=env)
    assert built_second is not None and built_second.manifest is not None, built_second
    assert built_second.manifest.parent != built_first.manifest.parent, "the fixture gave both stores one artifact"
    assert built_first.manifest.is_file(), "a build for one store removed the artifact of another store sharing its root"
    try:
        trusted_search(first, embedder, QUERY, k=8, candidate_k=20, env=env)
    except AtomicRescueArtifactError as exc:
        pytest.fail(f"a store's active search failed after another store sharing its root built: {exc}")


def test_a_flat_layout_artifact_still_serves(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    env = _env(root)
    store, embedder = _certified(tmp_path)
    built = ensure_rescue_artifact(store, embedder, env=env)
    assert built is not None and built.manifest is not None, built
    # Where a build put it before per-store folders: the top of the root.
    flat = root / built.manifest.parent.name
    clear_atomic_rescue_artifact_cache()
    built.manifest.parent.rename(flat)
    calibration = store.resolve_calibration().artifact
    assert calibration is not None
    served = store.atomic_rescue_manifest(root, store.lite_generation_id(), calibration.calibration_id)
    assert served == (flat / "manifest.json").resolve(), "an artifact built before per-store folders no longer serves"
    try:
        trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env=env)
    except AtomicRescueArtifactError as exc:
        pytest.fail(f"an artifact built before per-store folders no longer serves: {exc}")


def test_a_build_leaves_flat_layout_artifacts_and_reports_them(tmp_path: Path) -> None:
    root = tmp_path / "atomic"
    unknown = root / "lite-0123456789abcdef.cal_00"
    unknown.mkdir(parents=True)
    (unknown / "manifest.json").write_text("{}", encoding="utf-8")
    store, embedder = _certified(tmp_path)
    built = ensure_rescue_artifact(store, embedder, env=_env(root))
    assert built is not None and built.manifest is not None, built
    assert unknown.is_dir(), "a build removed an artifact folder whose owner it cannot know"
    assert "1 artifact folder(s) from before per-store folders remain" in built.reason, built.reason
