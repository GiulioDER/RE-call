"""`trusted_search` must READ a calibration on disk, not just accept one passed in.

Before this, the threshold resolved as `calibration or _UNCALIBRATED` and `load_for()` was called
only by `recall.cli`. A user who ran `recall calibrate`, saw calibration.json appear, and then
used the library API silently got the uncalibrated default — while `recall.trust`'s own docstring
promised the threshold "comes from recall.calibration when available".

The cost was measured on a real corpus: `DEFAULT_GAP_THRESHOLD` is 0.50, right for bge-small,
but `text-embedding-3-small` returns top-1 cosines between 0.41 and 0.76, so the floor sat inside
the distribution and returned EMPTY retrieval for 18 of 300 benchmark questions — a guaranteed
zero with no error anywhere.

These tests are written against the DETECTION path: each one is constructed so that it fails if
the auto-load is removed, rather than merely passing while the feature happens to be present.
"""
from __future__ import annotations

import os

import pytest

from recall.calibration import ENV_VAR, Calibration, load_for, save
from recall.embeddings import HashingEmbedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.index import Indexer
from recall.trust import trusted_search

from tests.conftest import requires_db

DOC = "The API rate limit is one hundred requests per second per client key.\n"


@pytest.fixture(autouse=True)
def _calibration_in_tmp(tmp_path, monkeypatch):
    """Point the loader at a per-test file and clear its cache.

    The env var is the documented override, so using it here also exercises the path a real
    deployment uses rather than a test-only seam. The cache is cleared on both sides because it
    is keyed by (path, mtime) and tmp_path reuse could otherwise serve a stale entry.
    """
    from recall.calibration import _LOAD_CACHE

    monkeypatch.setenv(ENV_VAR, str(tmp_path / "calibration.json"))
    _LOAD_CACHE.clear()
    yield
    _LOAD_CACHE.clear()


def _index(tmp_path, store):
    (tmp_path / "rate.md").write_text(DOC, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)


@requires_db
def test_calibration_on_disk_is_applied_without_being_passed(tmp_path, make_store):
    """A saved calibration reaches the result even though the caller passes none.

    `calibrated` is the observable: it is `True` only when a Calibration object was resolved, so
    this fails closed if the auto-load is reverted — the search would still succeed, but come back
    flagged uncalibrated.
    """
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)
    save(Calibration(embedder=embedder.name, threshold=0.11, scale=0.05))

    res = trusted_search(store, embedder, "API rate limit", k=5)

    assert res.calibrated is True, "a calibration on disk was ignored — auto-load is not wired"


@requires_db
def test_explicit_calibration_still_wins_over_the_file(tmp_path, make_store):
    """An argument must beat the file, or a caller loses the ability to override deliberately."""
    store = make_store(64)
    _index(tmp_path, store)
    embedder = HashingEmbedder(dim=64)
    save(Calibration(embedder=embedder.name, threshold=0.99, scale=0.05))
    explicit = Calibration(embedder=embedder.name, threshold=0.01, scale=0.05)

    res = trusted_search(store, embedder, "API rate limit", k=5, calibration=explicit)

    # The file's 0.99 would abstain on everything; the explicit 0.01 must be what is used.
    assert res.calibrated is True
    assert res.hits, "the file's threshold was applied instead of the explicit argument"


@requires_db
def test_no_calibration_file_behaves_exactly_as_before(tmp_path, make_store):
    """The change must not alter behaviour for anyone who never calibrated.

    This is the compatibility guarantee that makes the fix safe to ship: no file, no change.
    """
    store = make_store(64)
    _index(tmp_path, store)

    res = trusted_search(store, HashingEmbedder(dim=64), "API rate limit", k=5)

    assert res.calibrated is False


def test_loader_cache_invalidates_when_the_file_is_rewritten(tmp_path):
    """A re-calibration must be picked up, or the cache turns a fix into a stale read.

    `trusted_search` calls the loader on every query, so the file is cached rather than read and
    parsed each time. A load-once cache would be faster still and WRONG: a `recall calibrate` run
    by a concurrent process would not be seen until restart.

    Written twice in quick succession this used to pass by luck, and that is the whole defect: the
    cache was invalidated on `st_mtime_ns`, which is not a version — the smallest non-zero delta
    between consecutive writes is 1,000,001 ns on ext4, so two writes inside one tick carry the
    SAME mtime and the second was never seen. Measured against the loader itself, the old key
    served the stale threshold on 223 of 300 rewrites (74 %) on ext4 and 2 of 500 on NTFS; the
    difference in rate is why this reached CI as a flake instead of a red build, since a collision
    needs the two writes close together and that depended on which test file warmed the imports.

    A stale threshold gates abstention and never self-corrects, so it must be impossible, not
    improbable — hence the collision is forced below rather than waited for.
    """
    path = tmp_path / "calibration.json"
    save(Calibration(embedder="e1", threshold=0.42, scale=0.05), path)
    assert load_for("e1", path).threshold == pytest.approx(0.42)
    first = path.stat()

    save(Calibration(embedder="e1", threshold=0.31, scale=0.05), path)
    # Reinstating the first write's timestamp does not simulate anything the filesystem cannot
    # do on its own — it is exactly the state left behind when both writes land in one tick. It
    # only makes that state reachable on every run instead of a fraction of them.
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    assert path.stat().st_mtime_ns == first.st_mtime_ns, "the mtime collision was not reproduced"
    # Same byte length either side, so an invalidation keyed on (size, mtime) is no fix at all
    # and would still fail here. Only the CONTENT distinguishes these two files.
    assert path.stat().st_size == first.st_size, "the two payloads must be the same size"

    assert load_for("e1", path).threshold == pytest.approx(0.31), "cache served a stale threshold"


def test_calibration_for_a_different_embedder_is_not_applied(tmp_path):
    """A threshold from another model's cosine regime must never be used — cached or not.

    This is the property that makes auto-loading safe at all: picking up whatever file happens to
    be on disk would be dangerous if it were not embedder-checked first.
    """
    path = tmp_path / "calibration.json"
    save(Calibration(embedder="bge-small", threshold=0.50, scale=0.05), path)

    assert load_for("openai:text-embedding-3-small", path) is None
    assert DEFAULT_GAP_THRESHOLD == 0.50  # the fallback the caller gets instead
