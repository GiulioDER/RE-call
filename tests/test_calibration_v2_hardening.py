"""Second-pass hardening of calibration v2: import trust, chain drift, and boundary errors.

Each test here pins one of the deferred audit fixes:

* a bundle's unkeyed checksum proves internal consistency only, so the recorded statistics and
  the certification verdict must be re-derived from the scores the bundle itself carries;
* a chain of carry-forwards must be bounded by its CUMULATIVE corpus delta, not only by each
  step's delta against its immediate parent;
* boundary errors (missing bundle keys, re-imported ids, out-of-range legacy thresholds,
  non-numeric provenance) must surface as `CalibrationBindingError`, never as a bare KeyError,
  TypeError, ValueError, or raw psycopg error;
* `show_record` must render every timestamptz in the canonical UTC form, not just `created_at`.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from recall.calibration_v2 import (
    CalibrationArtifactV2,
    CalibrationBindingError,
    CalibrationRepository,
    CalibrationStatus,
    _require_carry_forward,
    canonical_sha256,
)
from recall.generations import GenerationManager
from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import (
    CORPUS_SIZE,
    _bodies,
    _CarryEmbedder,
)
from tests.test_calibration_carry_forward import _labels as _carry_labels
from tests.test_calibration_carry_forward import _ready as _carry_ready
from tests.test_calibration_v2 import (
    _CalibrationEmbedder,
    _dsn_in_timezone,
    _labels,
    _ready,
)


@pytest.fixture
def hardening_tenant():
    tenant = "hard-test-" + uuid.uuid4().hex[:10]
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield tenant, manager
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_calibrations WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_calibration_query_sets WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


def _resigned(bundle: dict, artifact_overrides: dict) -> dict:
    """The bundle with its artifact tampered and BOTH digests recomputed over the tampering.

    The forgery being modelled is exactly this: `bundle_checksum` and the artifact `checksum`
    are unkeyed, so anyone can rewrite the content and recompute them. A test that did not
    re-sign would only exercise tamper detection, which is a different (already tested) check.
    """
    raw = dict(bundle["artifact"], **artifact_overrides)
    immutable = {
        key: raw[key] for key in CalibrationArtifactV2.__dataclass_fields__ if key in raw
    }
    immutable.pop("lifecycle_state", None)
    immutable.pop("checksum", None)
    raw["checksum"] = canonical_sha256(immutable)
    rebuilt = dict(bundle, artifact=raw)
    rebuilt["bundle_checksum"] = canonical_sha256(
        {key: value for key, value in rebuilt.items() if key != "bundle_checksum"}
    )
    return rebuilt


def _delete_calibration(tenant: str, calibration_id: str) -> None:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "DELETE FROM recall_calibrations WHERE tenant_id = %s AND calibration_id = %s",
            (tenant, calibration_id),
        )


# --------------------------------------------------------------------------------------------
# SEC-002 / STAKES-004: import re-derives the certification from the bundle's own scores.
# --------------------------------------------------------------------------------------------


@requires_db
def test_a_bundle_whose_scores_do_not_support_its_separability_is_refused(
    hardening_tenant, tmp_path: Path
) -> None:
    """Certified stats beside scores that do not produce them are a forgery, not an artifact."""
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    assert artifact.certified is True
    bundle = json.loads(
        repository.export_bundle(artifact.calibration_id, tmp_path / "b.json").read_text(
            encoding="utf-8"
        )
    )
    _delete_calibration(tenant, artifact.calibration_id)

    # Overlapping classes: separability 0.5, while the recorded fields still claim ~1.0 and
    # certified=true. Counts stay 20/20 so the count check is not what trips.
    forged = _resigned(
        bundle,
        {"scores": {"answerable": [0.5] * 20, "unanswerable": [0.5] * 20}},
    )
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(CalibrationBindingError, match="separability"):
        repository.import_bundle(path)


@requires_db
def test_a_bundle_whose_certified_flag_its_scores_do_not_earn_is_refused(
    hardening_tenant, tmp_path: Path
) -> None:
    """The publish-laundering vector: certified=true imports as a draft, and publish promotes
    any certified draft, so a flipped flag alone must be caught at the import boundary."""
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    rejected = repository.calibrate(generation_id, _labels(2), embedder)
    assert rejected.certified is False, "two samples per class cannot certify"
    bundle = json.loads(
        repository.export_bundle(rejected.calibration_id, tmp_path / "b.json").read_text(
            encoding="utf-8"
        )
    )
    _delete_calibration(tenant, rejected.calibration_id)

    forged = _resigned(bundle, {"certified": True, "lifecycle_state": "draft"})
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(CalibrationBindingError, match="certified=True"):
        repository.import_bundle(path)


def test_a_bundle_whose_threshold_its_own_scores_do_not_fit_is_refused(
    hardening_tenant, tmp_path: Path
) -> None:
    """The threshold is the one served field that `certified` never looks at.

    Re-deriving only the verdict leaves the attack open: `Calibration.certified` reads
    separability, its interval and the sample counts, so a forger picks trivially separable
    scores (which certify honestly) and writes ANY threshold beside them. That threshold is
    what `publish` puts into serving, and a value near 1.0 refuses every query while one near
    -1.0 confirms every query. So a fitted artifact's threshold and scale are re-derived from
    `from_samples` over its own scores, which a legitimate round trip reproduces exactly.
    """
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    honest = repository.calibrate(generation_id, _labels(2), embedder)
    bundle = json.loads(
        repository.export_bundle(honest.calibration_id, tmp_path / "honest.json").read_text(
            encoding="utf-8"
        )
    )
    _delete_calibration(tenant, honest.calibration_id)

    forged = _resigned(bundle, {"threshold": 0.999})
    path = tmp_path / "forged-threshold.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(CalibrationBindingError, match="threshold"):
        repository.import_bundle(path)


def test_an_untampered_bundle_still_round_trips(hardening_tenant, tmp_path: Path) -> None:
    """The control for the two forgery tests: re-derivation must not refuse an honest export."""
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    honest = repository.calibrate(generation_id, _labels(2), embedder)
    path = repository.export_bundle(honest.calibration_id, tmp_path / "honest.json")
    _delete_calibration(tenant, honest.calibration_id)

    imported_id = repository.import_bundle(path)

    assert imported_id == honest.calibration_id
    restored = repository.get(imported_id)
    assert restored.threshold == honest.threshold
    assert restored.scale == honest.scale


@requires_db
def test_a_bundle_whose_counts_disagree_with_its_scores_is_refused(
    hardening_tenant, tmp_path: Path
) -> None:
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    bundle = json.loads(
        repository.export_bundle(artifact.calibration_id, tmp_path / "b.json").read_text(
            encoding="utf-8"
        )
    )
    _delete_calibration(tenant, artifact.calibration_id)

    forged = _resigned(bundle, {"n_answerable": 50, "n_unanswerable": 50})
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(CalibrationBindingError, match="sample counts disagree"):
        repository.import_bundle(path)


# --------------------------------------------------------------------------------------------
# STAKES-003: cumulative drift across a carry-forward chain is bounded.
# --------------------------------------------------------------------------------------------


@requires_db
def test_a_chain_of_carry_forwards_is_refused_once_cumulative_delta_exceeds_the_bound(
    hardening_tenant,
) -> None:
    """Each step passes the per-step bound; the second must still refuse on the chain total."""
    tenant, manager = hardening_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _carry_ready(manager, embedder, _bodies(), "v1")
    repository.publish(
        repository.calibrate(parent_generation, _carry_labels(), embedder).calibration_id
    )

    # Step one: 5 added over a union of 25, delta 0.2 <= 0.25, and the chain starts at a fitted
    # parent, so cumulative 0.2 passes too.
    first_child = _carry_ready(manager, embedder, _bodies(added=5), "v2")
    carried = repository.carry_forward(first_child, embedder)
    assert carried.certified is True
    provenance = dict(carried.carry_forward or {})
    assert provenance["corpus_delta"] == pytest.approx(5 / (CORPUS_SIZE + 5))
    assert provenance["cumulative_corpus_delta"] == pytest.approx(5 / (CORPUS_SIZE + 5))
    repository.publish(carried.calibration_id)

    # Step two: 5 more added over a union of 30, delta ~0.167 <= 0.25 STEP-WISE, but the chain
    # has now drifted 0.2 + 0.167 = 0.367 from the corpus the threshold was fitted on.
    second_child = _carry_ready(manager, embedder, _bodies(added=10), "v3")
    with pytest.raises(CalibrationBindingError, match="cumulative corpus delta"):
        repository.carry_forward(second_child, embedder)

    # The refusal wrote nothing that anyone could publish.
    assert not [
        record
        for record in repository.list_records()
        if record["generation_id"] == second_child
    ]
    assert repository.resolve(second_child).status == CalibrationStatus.STALE


def test_cumulative_delta_reads_zero_for_fitted_and_step_delta_for_old_artifacts() -> None:
    """Pre-existing carried artifacts lack the key and read as their recorded STEP delta.

    That is the conservative floor of what their chain actually drifted: older steps are
    unknowable once generation gc has pruned their manifests, and reading 0.0 instead would
    let a long pre-existing chain keep drifting unbounded.
    """
    # Imported inside the test: this helper does not exist before the fix, and a module
    # level import would make the whole file uncollectable rather than red.
    from recall.calibration_v2 import _cumulative_corpus_delta

    assert _cumulative_corpus_delta(None) == 0.0
    assert _cumulative_corpus_delta({"corpus_delta": 0.2}) == pytest.approx(0.2)
    assert _cumulative_corpus_delta(
        {"corpus_delta": 0.1, "cumulative_corpus_delta": 0.5}
    ) == pytest.approx(0.5)
    with pytest.raises(CalibrationBindingError, match="not a number"):
        _cumulative_corpus_delta({"corpus_delta": "wide"})
    with pytest.raises(CalibrationBindingError, match="finite non-negative"):
        _cumulative_corpus_delta({"cumulative_corpus_delta": -0.1})


def test_provenance_validation_tolerates_both_shapes_and_names_bad_numbers() -> None:
    base = {"parent_calibration_id": "c", "parent_generation_id": "g", "corpus_delta": 0.1}
    # The pre-cumulative shape stays valid: artifacts written before the key existed must keep
    # verifying.
    _require_carry_forward(dict(base), 0.7, 0.05)
    _require_carry_forward(dict(base, cumulative_corpus_delta=0.3), 0.7, 0.05)
    with pytest.raises(CalibrationBindingError, match="cumulative_corpus_delta"):
        _require_carry_forward(dict(base, cumulative_corpus_delta="drift"), 0.7, 0.05)
    with pytest.raises(CalibrationBindingError, match="cumulative_corpus_delta"):
        _require_carry_forward(dict(base, cumulative_corpus_delta=True), 0.7, 0.05)
    # BUG-008: the inherited threshold and scale coercions must name the problem rather than
    # escape as a bare TypeError or ValueError.
    with pytest.raises(CalibrationBindingError, match="not a number"):
        _require_carry_forward(dict(base, inherited_threshold="high"), 0.7, 0.05)
    with pytest.raises(CalibrationBindingError, match="not a number"):
        _require_carry_forward(dict(base, inherited_scale={}), 0.7, 0.05)


# --------------------------------------------------------------------------------------------
# BUG-008 / DAT-007: boundary errors are named CalibrationBindingError, not raw exceptions.
# --------------------------------------------------------------------------------------------


@requires_db
def test_a_bundle_missing_a_required_key_is_named_not_crashed(
    hardening_tenant, tmp_path: Path
) -> None:
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    bundle = json.loads(
        repository.export_bundle(artifact.calibration_id, tmp_path / "b.json").read_text(
            encoding="utf-8"
        )
    )
    _delete_calibration(tenant, artifact.calibration_id)

    for missing in ("separability_ci", "scores", "created_by"):
        broken = dict(bundle, artifact={
            key: value for key, value in bundle["artifact"].items() if key != missing
        })
        broken["bundle_checksum"] = canonical_sha256(
            {key: value for key, value in broken.items() if key != "bundle_checksum"}
        )
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(CalibrationBindingError, match=f"missing '{missing}'"):
            repository.import_bundle(path)


@requires_db
def test_reimporting_an_existing_calibration_is_named_not_a_unique_violation(
    hardening_tenant, tmp_path: Path
) -> None:
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    bundle = repository.export_bundle(artifact.calibration_id, tmp_path / "bundle.json")

    # The row is still stored, so the import collides on the primary key.
    with pytest.raises(CalibrationBindingError, match="already imported"):
        repository.import_bundle(bundle)


@requires_db
def test_a_legacy_threshold_outside_the_column_range_is_refused_at_the_boundary(
    hardening_tenant, tmp_path: Path
) -> None:
    """The table CHECKs threshold BETWEEN -1.0 AND 1.0; the refusal must not be psycopg's."""
    tenant, _manager = hardening_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    for threshold in (1.5, -1.5):
        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps({"embedder": "m", "threshold": threshold, "scale": 0.05}),
            encoding="utf-8",
        )
        with pytest.raises(CalibrationBindingError, match="invalid numeric"):
            repository.import_bundle(path)


# --------------------------------------------------------------------------------------------
# DAT-006: show_record renders every timestamp in canonical UTC, not the session TimeZone.
# --------------------------------------------------------------------------------------------


@requires_db
def test_show_record_renders_published_and_superseded_at_in_utc(hardening_tenant) -> None:
    """`published_at` and `superseded_at` are set by clock_timestamp() and must round-trip
    through a non-UTC session exactly as `created_at` already does."""
    tenant, manager = hardening_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    rome = CalibrationRepository(
        _dsn_in_timezone(TEST_DSN, "Europe/Rome"), tenant, actor="pytest"
    )
    first = rome.calibrate(generation_id, _labels(), embedder)
    second = rome.calibrate(generation_id, _labels(), embedder)
    rome.publish(first.calibration_id)
    rome.publish(second.calibration_id)  # supersedes the first

    record = rome.show_record(first.calibration_id)
    assert record["lifecycle_state"] == "superseded"
    for key in ("created_at", "published_at", "superseded_at"):
        value = record[key]
        assert isinstance(value, str), f"{key} must be rendered"
        canonical = datetime.fromisoformat(value).astimezone(UTC).isoformat()
        assert value == canonical, (
            f"{key} must be the canonical UTC rendering, got {value!r}"
        )


# --------------------------------------------------------------------------------------------
# ENV-007: the connection timeout is a constructor parameter, as on ControlPlane.
# --------------------------------------------------------------------------------------------


def test_connect_timeout_is_a_constructor_parameter_defaulting_to_ten() -> None:
    assert CalibrationRepository("dsn", "t")._connect_timeout_s == 10
    assert CalibrationRepository("dsn", "t", connect_timeout_s=3)._connect_timeout_s == 3
