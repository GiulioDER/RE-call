"""`RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip`: answer without a rescue artifact that is not there
(docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md, owner decision 2026-11-16).

Under `skip`, an active rescue whose artifact is missing, fails its lineage check, or is damaged
answers exactly as the same search with the rescue off, and says why in
`diagnostics.atomic_rescue_skipped`. The default, `refuse`, keeps today's refusal, and a
configuration error refuses whatever the setting. A Lite store stands in for PostgreSQL: the
trust gate's code path is the same.

Red proofs (each run on Linux, failing in its own assertion):
- W1, W3, W4 against the parent commit, where `skip` is ignored: the search raises
  `AtomicRescueArtifactError` instead of answering.
- W2 is a control that passes on the parent by design; it is red with the default changed to
  `skip`: "DID NOT RAISE".
- W5 with the setting's validation removed: an unknown value answers instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.atomic_rescue import AtomicRescueArtifactError, clear_atomic_rescue_artifact_cache
from recall.lite import LiteStore
from recall.lite.atomic import manifest_path
from recall.lite.refresh import Outcome, refresh
from recall.trust import trusted_search
from tests.test_lite_atomic_rescue import QUERY, _corpus, _env, _ids
from tests.test_lite_calibration import DIM, ContentWordEmbedder
from tests.test_lite_mcp import lite_env  # noqa: F401  # the fixture is used by name

SKIP = {"RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip"}


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_atomic_rescue_artifact_cache()


def _store(tmp_path: Path, monkeypatch, *, build: bool) -> tuple[LiteStore, ContentWordEmbedder, Path]:
    for name in ("RECALL_ATOMIC_RESCUE_MODE", "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "atomic"
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    report = refresh(store, [_corpus(tmp_path)], embedder, refusal_env=_env(root) if build else None)
    assert report.outcome is Outcome.REFRESHED, report.to_dict()
    return store, embedder, root


def _manifest(store: LiteStore, root: Path) -> Path:
    resolution = store.resolve_calibration()
    assert resolution.artifact is not None
    return manifest_path(root, store, store.lite_generation_id(), resolution.artifact.calibration_id)


def _off(store: LiteStore, embedder: ContentWordEmbedder) -> list[str]:
    return _ids(trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={}))


def test_skip_answers_like_off_when_the_artifact_is_missing(tmp_path: Path, monkeypatch) -> None:
    """W1."""
    store, embedder, root = _store(tmp_path, monkeypatch, build=False)

    result = trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={**_env(root), **SKIP})

    assert _ids(result) == _off(store, embedder), "a skipped rescue changed the answer"
    assert result.diagnostics.atomic_rescue_skipped == "missing"


def test_refuse_is_the_default(tmp_path: Path, monkeypatch) -> None:
    """W2, a control: without the setting a missing artifact refuses, as it always has."""
    store, embedder, root = _store(tmp_path, monkeypatch, build=False)

    with pytest.raises(AtomicRescueArtifactError):
        trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env=_env(root))


def test_a_lineage_mismatch_is_skipped_as_lineage(tmp_path: Path, monkeypatch) -> None:
    """W3: an artifact for another calibration (a recalibration not yet rebuilt) is skipped."""
    store, embedder, root = _store(tmp_path, monkeypatch, build=True)
    manifest = _manifest(store, root)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["calibration_id"] = "cal_" + "0" * 32
    manifest.write_text(json.dumps(document), encoding="utf-8")

    result = trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={**_env(root), **SKIP})

    assert result.diagnostics.atomic_rescue_skipped == "lineage"
    assert _ids(result) == _off(store, embedder)


@pytest.mark.parametrize("damage", ["manifest-gone", "manifest-unreadable"])
def test_a_damaged_artifact_is_skipped_as_damaged(tmp_path: Path, monkeypatch, damage: str) -> None:
    """W4: a partial directory (the matrix without its manifest) or an unreadable manifest is
    `damaged`, not `missing`, since a directory is there."""
    store, embedder, root = _store(tmp_path, monkeypatch, build=True)
    manifest = _manifest(store, root)
    if damage == "manifest-gone":
        manifest.unlink()
    else:
        manifest.write_text("{not json", encoding="utf-8")

    result = trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={**_env(root), **SKIP})

    assert result.diagnostics.atomic_rescue_skipped == "damaged"
    assert _ids(result) == _off(store, embedder)


@pytest.mark.parametrize(
    "change",
    [
        {"RECALL_ATOMIC_RESCUE_WHEN_MISSING": "sometimes"},
        {"RECALL_ATOMIC_RESCUE_MODE": "actve"},
        {"RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": "relative/root"},
    ],
)
def test_configuration_errors_still_raise_under_skip(tmp_path: Path, monkeypatch, change: dict[str, str]) -> None:
    """W5: a build cannot fix a configuration error, so `skip` does not hide one. The artifact is
    built and valid, so the configuration is the only reason a search could raise."""
    store, embedder, root = _store(tmp_path, monkeypatch, build=True)

    with pytest.raises(AtomicRescueArtifactError):
        trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={**_env(root), **SKIP, **change})


def test_recall_search_reports_a_skipped_rescue(lite_env: dict[str, str], tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """W6: the reason reaches the client, in `recall_search`'s response, not only the result.

    Red proof: with `atomic_rescue_skipped=` dropped from `recall_mcp.retrieval`'s search payload,
    the field reads "" and the assertion fails.
    """
    import shutil

    from tests.test_lite_calibration import _write_memos
    from tests.test_lite_mcp import QUESTION, _serve

    root = tmp_path / "atomic"
    rescue = {"RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(root), **SKIP}
    for name, value in rescue.items():
        monkeypatch.setenv(name, value)
    env = {**lite_env, **rescue}
    _write_memos(tmp_path / "mem", 48)

    async def body(state, call):  # noqa: ANN001, ANN202
        indexed = json.loads(await call("recall_index", path=str(tmp_path / "mem")))
        assert indexed["chunks"] == 48, indexed
        shutil.rmtree(root)  # the artifact the index built is gone, as after a container's promotion
        clear_atomic_rescue_artifact_cache()
        found = json.loads(await call("recall_search", query=QUESTION))
        assert found["atomic_rescue_skipped"] == "missing", found

    _serve(env, body)


def test_a_missing_atomic_extra_still_raises_under_skip(tmp_path: Path, monkeypatch) -> None:
    """W7 (final review, Important 1): without the `atomic` packages a build can never succeed, so
    `skip` must not report the artifact `missing` and leave the builder retrying forever.

    Red proof: against `898eaf748`, under skip the search answered with "missing" instead of raising.
    """
    import sys

    store, embedder, root = _store(tmp_path, monkeypatch, build=False)
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)  # import raises ImportError

    with pytest.raises(AtomicRescueArtifactError, match=r"recall-rag\[atomic\]"):
        trusted_search(store, embedder, QUERY, k=8, candidate_k=20, env={**_env(root), **SKIP})


def test_an_uncalibrated_search_answers_as_off_under_skip(tmp_path: Path, monkeypatch) -> None:
    """W8 (final review, Important 2): a development-trust search with no resolved calibration (a
    forget left it STALE, or none was ever certified) answered degraded with the rescue off; under
    `skip` it must answer the same, not raise on the manifest needing a calibration.

    Red proof: against `898eaf748`, the search raised AtomicRescueArtifactError ("requires a
    certified calibration").
    """
    from recall.lite.refresh import refresh as lite_refresh
    from recall.trust_policy import TrustPolicy
    from tests.test_lite_calibration import _write_memos

    for name in ("RECALL_ATOMIC_RESCUE_MODE", "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT"):
        monkeypatch.delenv(name, raising=False)
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    _write_memos(tmp_path / "few", 3)  # too few to certify
    lite_refresh(store, [tmp_path / "few"], embedder)
    assert store.resolve_calibration().artifact is None, "the fixture calibrated"
    development = TrustPolicy.from_env({"RECALL_TRUST_MODE": "development"})
    root = tmp_path / "atomic"

    off = trusted_search(store, embedder, "crew", k=3, policy=development, env={})
    skipped = trusted_search(store, embedder, "crew", k=3, policy=development, env={**_env(root), **SKIP})

    assert _ids(skipped) == _ids(off)
    assert skipped.diagnostics.atomic_rescue_skipped == "uncalibrated"
