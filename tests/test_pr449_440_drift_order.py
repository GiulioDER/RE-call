"""Drift follows publication history without changing calibration-list history."""

from __future__ import annotations

import psycopg
import pytest

from recall.calibration_v2 import CalibrationArtifactV2, CalibrationError, CalibrationRepository
from recall.drift import DriftVerdict, evaluate_drift
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import (  # noqa: F401 - shared DB fixture
    _bodies,
    _CarryEmbedder,
    _labels,
    _ready,
    carry_tenant,
)


@requires_db
def test_drift_uses_last_publication_while_list_keeps_creation_order(
    carry_tenant, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """VERIFY-001, baseline 160a23a3f: a later publication of an older fit is ignored.

    Build and fit two real generations, then publish in the opposite order to creation. The
    intended red assertion observes evaluate_drift choosing the newer-created calibration rather
    than the last published one. The list_records assertion protects its existing history API.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    older_generation = _ready(manager, embedder, _bodies(), "older")
    older = repository.calibrate(older_generation, _labels(), embedder)
    newer_generation = _ready(manager, embedder, _bodies(added=2), "newer")
    newer = repository.calibrate(newer_generation, _labels(), embedder)

    repository.publish(newer.calibration_id)
    repository.publish(older.calibration_id)
    assert [row["calibration_id"] for row in repository.list_records()] == [
        newer.calibration_id, older.calibration_id,
    ]

    report = evaluate_drift(repository, generation_id=older_generation, probe=False)

    assert report.calibration_id == older.calibration_id, (
        "the last published calibration is the baseline, even when it was created earlier"
    )
    assert report.baseline_generation_id == older_generation
    assert report.verdict is DriftVerdict.STABLE
    assert report.delta["corpus_delta"] == 0.0
    assert report.probe is None

    # Controls stay in this same baseline-red node: correcting the ordering must retain the
    # existing lifecycle, loadability and equal-publication-time behavior.
    for rejected in (False, True):
        _unpublished_fit_does_not_replace_baseline(carry_tenant, rejected)
    _unloadable_publication_falls_back(carry_tenant, monkeypatch)
    _equal_publication_times_use_creation_order(carry_tenant)


def _unpublished_fit_does_not_replace_baseline(
    tenant_fixture, rejected: bool
) -> None:
    """VERIFY-001 control for the existing draft and rejected-fit exclusions.

    Publication-order repair must retain the existing exclusion of a newer draft or rejected fit.
    """
    tenant, manager = tenant_fixture
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    embedder = _CarryEmbedder()
    baseline = _ready(manager, embedder, _bodies(), "baseline")
    published = repository.publish(
        repository.calibrate(baseline, _labels(), embedder).calibration_id
    )
    # A distinct identity forces a rebuild: reusing the baseline model would
    # retain its unpoisoned vectors and could still produce a certified fit.
    candidate_embedder = _CarryEmbedder(
        poisoned=rejected, model="rejected-model" if rejected else "draft-model"
    )
    candidate = _ready(manager, candidate_embedder, _bodies(added=2), "candidate")
    unpublished = repository.calibrate(candidate, _labels(), candidate_embedder)
    assert unpublished.certified is not rejected

    report = evaluate_drift(repository, generation_id=baseline, probe=False)

    assert report.calibration_id == published.calibration_id
    assert report.verdict is DriftVerdict.STABLE


def _unloadable_publication_falls_back(
    tenant_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VERIFY-001 control for the existing fallback after a failed artifact load.

    A failed artifact load must retain the existing fallback to the next published calibration.
    Only the read failure is injected; publication ordering and generation comparison use the DB.
    """
    tenant, manager = tenant_fixture
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    embedder = _CarryEmbedder()
    baseline = _ready(manager, embedder, _bodies(), "baseline")
    usable = repository.publish(repository.calibrate(baseline, _labels(), embedder).calibration_id)
    latest_generation = _ready(manager, embedder, _bodies(added=2), "latest")
    latest = repository.publish(
        repository.calibrate(latest_generation, _labels(), embedder).calibration_id
    )
    original_get = repository.get

    def get(calibration_id: str) -> CalibrationArtifactV2:
        if calibration_id == latest.calibration_id:
            raise CalibrationError("the published artifact could not be loaded")
        return original_get(calibration_id)

    monkeypatch.setattr(repository, "get", get)
    report = evaluate_drift(repository, generation_id=baseline, probe=False)

    assert report.calibration_id == usable.calibration_id
    assert report.verdict is DriftVerdict.STABLE


def _equal_publication_times_use_creation_order(tenant_fixture) -> None:
    """VERIFY-001 control for the existing creation-order secondary ordering.

    Publication timestamps can tie. Preserve the existing creation-order secondary ordering,
    matching carry_forward's parent lookup; only lifecycle timestamps are adjusted in the fixture.
    """
    tenant, manager = tenant_fixture
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    embedder = _CarryEmbedder()
    older_generation = _ready(manager, embedder, _bodies(), "older")
    repository.publish(repository.calibrate(older_generation, _labels(), embedder).calibration_id)
    newer_generation = _ready(manager, embedder, _bodies(added=2), "newer")
    newer = repository.publish(
        repository.calibrate(newer_generation, _labels(), embedder).calibration_id
    )
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "UPDATE recall_calibrations SET published_at = statement_timestamp() "
            "WHERE tenant_id = %s AND lifecycle_state = 'published'",
            (tenant,),
        )

    report = evaluate_drift(repository, generation_id=newer_generation, probe=False)

    assert report.calibration_id == newer.calibration_id
    assert report.verdict is DriftVerdict.STABLE
