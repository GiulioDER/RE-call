"""Rescue cleanup retains concurrent and unattributed artifacts."""

import json
from pathlib import Path

from recall.atomic_rescue import AtomicRescueArtifactError
from recall.lite import LiteStore
from recall.lite.atomic import ROOT_ENV
from tests.test_lite_atomic_rescue import _certified, _env


def test_publication_and_cleanup_hold_the_same_file_mutex(tmp_path: Path, monkeypatch) -> None:
    from recall.lite import atomic
    from recall.lite.writer_lock import _open, _try_lock, _unlock

    store, embedder = _certified(tmp_path)
    observed = []

    def held(folder):
        folder.mkdir(parents=True, exist_ok=True)
        with _open(folder / "publish.mutex") as handle:
            acquired = _try_lock(handle)
            if acquired:
                _unlock(handle)
            return not acquired

    publish, cleanup = atomic.write_atomic_rescue_artifact, atomic._remove_others

    def published(directory, **kwargs):
        observed.append(held(directory.parent))
        return publish(directory, **kwargs)

    def cleaned(folder, **kwargs):
        observed.append(held(folder))
        return cleanup(folder, **kwargs)

    monkeypatch.setattr(atomic, "write_atomic_rescue_artifact", published)
    monkeypatch.setattr(atomic, "_remove_others", cleaned)
    try:
        atomic.ensure_rescue_artifact(store, embedder, env=_env(tmp_path / "atomic"))
    finally:
        store.close()
    assert observed == [True, True], "publication and cleanup did not share one exclusive file mutex"


def test_forget_preserves_a_legacy_artifact_without_proven_store_ownership(tmp_path: Path) -> None:
    from recall.lite.atomic import erase_forgotten

    root = tmp_path / "atomic"
    legacy = root / ("lite-" + "a" * 16 + ".cal_" + "b" * 64)
    legacy.mkdir(parents=True)
    (legacy / "views.json").write_text(json.dumps({"views": [{"source": "shared.md"}]}), encoding="utf-8")
    warning = ""
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        try:
            erase_forgotten(store, [], ["shared.md"], env={ROOT_ENV: str(root)})
        except AtomicRescueArtifactError as exc:
            warning = str(exc)
    assert legacy.is_dir(), "an unattributed shared artifact was removed"
    assert "owner" in warning, "incomplete erasure did not report unknown ownership"


def test_forget_pruning_holds_the_publication_mutex(tmp_path: Path, monkeypatch) -> None:
    from recall.lite import atomic
    from recall.lite.writer_lock import _open, _try_lock, _unlock

    root = tmp_path / "atomic"
    observed = []
    remove = atomic.shutil.rmtree

    def removed(entry, **kwargs):
        with _open(entry.parent / "publish.mutex") as handle:
            acquired = _try_lock(handle)
            if acquired:
                _unlock(handle)
            observed.append(not acquired)
        return remove(entry, **kwargs)

    monkeypatch.setattr(atomic.shutil, "rmtree", removed)
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        old = atomic.manifest_path(root, store, "lite-" + "a" * 16, "cal_" + "b" * 64).parent
        old.mkdir(parents=True)
        atomic.erase_forgotten(store, [], [], env={ROOT_ENV: str(root)})
    assert not old.exists(), "the stale artifact was not pruned"
    assert observed == [True], "forget pruning raced an unlocked artifact publication"


def test_damaged_artifact_replacement_holds_the_publication_mutex(tmp_path: Path, monkeypatch) -> None:
    from recall.lite import atomic
    from recall.lite.writer_lock import _open, _try_lock, _unlock
    from tests.test_lite_atomic_rescue import _in_force

    store, embedder = _certified(tmp_path)
    root = tmp_path / "atomic"
    manifest = _in_force(store, root)
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    observed = []
    remove = atomic.shutil.rmtree

    def removed(entry, **kwargs):
        if entry == manifest.parent:
            with _open(entry.parent / "publish.mutex") as handle:
                acquired = _try_lock(handle)
                if acquired:
                    _unlock(handle)
                observed.append(not acquired)
        return remove(entry, **kwargs)

    monkeypatch.setattr(atomic.shutil, "rmtree", removed)
    try:
        result = atomic.ensure_rescue_artifact(store, embedder, env=_env(root))
        assert result is not None and result.manifest == manifest
    finally:
        store.close()
    assert observed == [True], "damaged-artifact cleanup could remove a concurrent published winner"
