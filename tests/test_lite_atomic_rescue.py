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

Red proof: recorded in the pull request that added this file (mutations, test node ids and the
failing assertion of each).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.atomic_rescue import (
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
    manifest = manifest_path(root, store.lite_generation_id(), resolution.artifact.calibration_id)
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
    manifest = manifest_path(root, store.lite_generation_id(), resolution.artifact.calibration_id)
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
