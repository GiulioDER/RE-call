"""A certified threshold may be re-verified against a rebuilt generation, never merely reused.

The distinction these tests exist to hold is the whole feature. Carrying a threshold forward is
allowed because the parent's own labelled query set is re-scored against the CHILD generation and
must still clear the certification bar there. It is not allowed because the corpus delta looked
small. `test_carry_forward_rejects_when_the_threshold_stops_separating` is the test that tells the
two apart: it keeps the delta inside the bound and breaks the separation, and the mechanism must
refuse.
"""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    DEFAULT_MAX_CORPUS_DELTA,
    CalibrationArtifactV2,
    CalibrationBindingError,
    CalibrationNotFound,
    CalibrationRepository,
    CalibrationStatus,
    CalibrationUncertified,
    _require_carry_forward,
    canonical_sha256,
    corpus_delta,
)
from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from tests.conftest import TEST_DSN, requires_db
from tests.test_generations import _S3, _pipeline

#: Enough sources that one file changing is a small delta rather than the whole corpus. With the
#: single-object manifest the other calibration tests use, every delta is 0.0 or 1.0 and the bound
#: could never be exercised at all.
CORPUS_SIZE = 20


class _CarryEmbedder:
    """Answerable queries land on axis 0, unanswerable on axis 1, documents on axis 0.

    `poisoned` moves the documents onto the diagonal, so unanswerable queries start scoring high
    against them. That is how a corpus change breaks a threshold without changing the query set,
    which is the failure carry-forward has to catch.
    """

    dim = 64

    def __init__(self, *, poisoned: bool = False, model: str = "carry-model") -> None:
        self.poisoned = poisoned
        self.model = model

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            if text.startswith("answer-"):
                vector[0] = 1.0
            elif text.startswith("missing-"):
                vector[1] = 1.0
            elif self.poisoned:
                # Cosine 1/sqrt(2) ~ 0.707 to BOTH query classes: above the inherited threshold
                # for the unanswerable ones, which is exactly the separation collapse.
                vector[0] = 1.0
                vector[1] = 1.0
            else:
                vector[0] = 1.0
            values.append(vector)
        return values


def _labels(count: int = 22) -> list[dict[str, object]]:
    return [
        *({"query": f"answer-{index}", "answerable": True} for index in range(count)),
        *({"query": f"missing-{index}", "answerable": False} for index in range(count)),
    ]


def _bodies(*, changed: int = 0, added: int = 0) -> dict[str, bytes]:
    """The corpus as `{name: bytes}`, with `changed` files edited and `added` files appended."""
    bodies = {f"memo-{index}.md": f"document body {index}".encode() for index in range(CORPUS_SIZE)}
    for index in range(changed):
        bodies[f"memo-{index}.md"] = f"document body {index} revised".encode()
    for index in range(added):
        bodies[f"extra-{index}.md"] = f"new document {index}".encode()
    return bodies


def _manifest(tenant: str, bodies: dict[str, bytes], version: str) -> IndexManifestV1:
    return IndexManifestV1(
        tenant,
        version,
        tuple(
            ManifestObjectV1(
                f"s3://approved/corpora/{tenant}/{name}",
                hashlib.sha256(data).hexdigest(),
                "text/markdown",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            for name, data in sorted(bodies.items())
        ),
    )


def _reader(manifest: IndexManifestV1, bodies: dict[str, bytes]) -> S3ObjectReader:
    objects = {}
    for entry in manifest.objects:
        name = entry.uri.rsplit("/", 1)[1]
        key = ("approved", f"corpora/{manifest.tenant_id}/{name}", entry.version_id)
        objects[key] = bodies[name]
    return S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))


@pytest.fixture
def carry_tenant():
    tenant = "carry-test-" + uuid.uuid4().hex[:10]
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


def _ready(manager, embedder, bodies: dict[str, bytes], version: str) -> str:
    manifest = _manifest(manager.tenant_id, bodies, version)
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(generation.generation_id, _reader(manifest, bodies), embedder, lambda t: [t])
    manager.validate(generation.generation_id)
    return generation.generation_id


# --------------------------------------------------------------------------------------------
# The delta itself, which decides whether the mechanism is even attempted.
# --------------------------------------------------------------------------------------------


def test_corpus_delta_counts_additions_removals_and_edits() -> None:
    parent = [{"uri": f"u{i}", "sha256": f"h{i}"} for i in range(10)]
    child = [
        *({"uri": f"u{i}", "sha256": f"h{i}"} for i in range(1, 9) if i != 5),
        {"uri": "u5", "sha256": "edited"},
        {"uri": "new", "sha256": "hn"},
    ]
    delta = corpus_delta(parent, child)
    assert delta["sources_removed"] == 2  # u0 and u9
    assert delta["sources_added"] == 1  # new
    assert delta["sources_modified"] == 1  # u5
    # Union of 11 URIs (u0..u9 plus "new"), 4 of them changed.
    assert delta["sources_union"] == 11
    assert delta["corpus_delta"] == pytest.approx(4 / 11)


def test_corpus_delta_denominator_is_the_union_so_a_deletion_cannot_look_small() -> None:
    """Over the CHILD alone this scores 0.0, which is the reading the union denominator prevents.

    Nine of ten sources deleted is the largest change a corpus can undergo short of replacement,
    and every surviving source still matches. A child-count denominator reports that as no change
    at all and would carry a threshold straight across it.
    """
    parent = [{"uri": f"u{i}", "sha256": f"h{i}"} for i in range(10)]
    child = [{"uri": "u0", "sha256": "h0"}]
    delta = corpus_delta(parent, child)
    assert delta["sources_modified"] == 0
    assert delta["corpus_delta"] == pytest.approx(9 / 10)


# --------------------------------------------------------------------------------------------
# The mechanism.
# --------------------------------------------------------------------------------------------


@requires_db
def test_carry_forward_rebinds_a_certified_threshold_to_a_rebuilt_generation(carry_tenant) -> None:
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    parent = repository.publish(
        repository.calibrate(parent_generation, _labels(), embedder).calibration_id
    )
    assert parent.certified is True
    assert parent.threshold_was_measured_here is True

    child_generation = _ready(manager, embedder, _bodies(added=2), "v2")
    # Before carrying anything, the rebuilt generation has NO calibration: this is the state the
    # feature exists to remove, and asserting it here stops a green carry-forward being claimed
    # against a generation that was never actually stale.
    assert repository.resolve(child_generation).status == CalibrationStatus.STALE

    carried = repository.carry_forward(child_generation, embedder)

    assert carried.certified is True
    assert carried.generation_id == child_generation
    assert carried.threshold == parent.threshold, "the threshold must be inherited, not refitted"
    assert carried.scale == parent.scale
    assert carried.threshold_was_measured_here is False
    provenance = dict(carried.carry_forward or {})
    assert provenance["parent_calibration_id"] == parent.calibration_id
    assert provenance["parent_generation_id"] == parent_generation
    assert provenance["sources_added"] == 2
    assert provenance["sources_removed"] == 0
    assert provenance["sources_modified"] == 0
    assert provenance["corpus_delta"] == pytest.approx(2 / (CORPUS_SIZE + 2))
    assert provenance["parent_separability"] == parent.separability

    published = repository.publish(carried.calibration_id)
    assert repository.resolve(child_generation).artifact == published
    assert repository.resolve(child_generation).status == CalibrationStatus.CERTIFIED


@requires_db
def test_carry_forward_rejects_when_the_threshold_stops_separating(carry_tenant) -> None:
    """A small delta is not sufficient. This one is inside the bound and still must refuse.

    Without this test the mechanism is indistinguishable from a tolerance that rubber-stamps any
    rebuild under 25%, which is the thing it must not be.
    """
    tenant, manager = carry_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    clean = _CarryEmbedder()
    parent_generation = _ready(manager, clean, _bodies(), "v1")
    parent = repository.publish(
        repository.calibrate(parent_generation, _labels(), clean).calibration_id
    )

    # Two added documents, the same delta as the passing test, but they embed onto the diagonal so
    # unanswerable queries now score ~0.707 against them.
    poisoned = _CarryEmbedder(poisoned=True)
    child_generation = _ready(manager, poisoned, _bodies(added=2), "v2")
    carried = repository.carry_forward(child_generation, poisoned)

    assert carried.certified is False
    assert carried.lifecycle_state == "rejected"
    assert carried.status is CalibrationStatus.REJECTED
    assert carried.threshold == parent.threshold
    provenance = dict(carried.carry_forward or {})
    assert provenance["corpus_delta"] <= DEFAULT_MAX_CORPUS_DELTA

    # THE POINT OF THIS TEST. Separability is untouched — the classes are still perfectly ordered
    # — so the certification bar that `calibrate` uses would have passed this. What failed is the
    # error of the FIXED threshold, and only a check that looks at the cut can see it. If this
    # assertion ever starts failing because separability dropped, the fixture stopped reproducing
    # the hole and the test is no longer evidence for the check it guards.
    assert carried.separability == pytest.approx(1.0)
    assert carried.separability_ci[0] >= 0.90
    assert provenance["false_confirm_rate"] > DEFAULT_MAX_CARRY_FORWARD_ERROR
    assert provenance["false_abstain_rate"] == pytest.approx(0.0)
    assert "only the boundary has moved" in carried.certification_reason

    # The evidence survives the rejection, so an operator can see WHY it failed.
    assert carried.n_answerable == 22 and carried.n_unanswerable == 22
    with pytest.raises(CalibrationUncertified):
        repository.publish(carried.calibration_id)
    assert repository.resolve(child_generation).status == CalibrationStatus.UNCERTIFIED


@requires_db
def test_carry_forward_refuses_a_delta_beyond_the_bound(carry_tenant) -> None:
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent_generation, _labels(), embedder).calibration_id)

    child_generation = _ready(manager, embedder, _bodies(changed=12), "v2")
    with pytest.raises(CalibrationBindingError, match="exceeds the carry-forward bound"):
        repository.carry_forward(child_generation, embedder, max_corpus_delta=0.25)

    # Nothing was written: a refusal must not leave a draft behind that someone can publish.
    assert repository.resolve(child_generation).status == CalibrationStatus.STALE
    assert not [
        record for record in repository.list_records() if record["generation_id"] == child_generation
    ]


@requires_db
def test_carry_forward_refuses_a_different_pipeline_at_any_delta(carry_tenant) -> None:
    """A threshold is a property of an embedder's cosine regime, so no delta makes this safe."""
    tenant, manager = carry_tenant
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    embedder = _CarryEmbedder()
    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent_generation, _labels(), embedder).calibration_id)

    other = _CarryEmbedder(model="carry-model-2")
    child_generation = _ready(manager, other, _bodies(added=1), "v2")
    with pytest.raises(CalibrationBindingError, match="different pipeline fingerprint"):
        repository.carry_forward(child_generation, other, max_corpus_delta=1.0)


@requires_db
def test_carry_forward_refuses_an_unchanged_corpus(carry_tenant) -> None:
    """An identical corpus fingerprint means the caller named the generation it is already on."""
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent_generation, _labels(), embedder).calibration_id)

    twin = _ready(manager, embedder, _bodies(), "v1")
    with pytest.raises(CalibrationBindingError, match="same corpus fingerprint"):
        repository.carry_forward(twin, embedder)


@requires_db
def test_carry_forward_refuses_an_uncertified_parent(carry_tenant) -> None:
    """A draft threshold has no published certification, and must not gain one by being copied."""
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    draft = repository.calibrate(parent_generation, _labels(), embedder)
    assert draft.status is CalibrationStatus.DRAFT  # certified but never published

    child_generation = _ready(manager, embedder, _bodies(added=1), "v2")
    with pytest.raises(CalibrationUncertified):
        repository.carry_forward(
            child_generation, embedder, parent_calibration_id=draft.calibration_id
        )


@requires_db
def test_carry_forward_without_a_parent_says_so(carry_tenant) -> None:
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    generation = _ready(manager, embedder, _bodies(), "v1")
    with pytest.raises(CalibrationNotFound, match="no published calibration"):
        repository.carry_forward(generation, embedder)


@requires_db
def test_a_carried_artifact_round_trips_through_export_and_import(carry_tenant, tmp_path) -> None:
    """The provenance is inside the checksum, so dropping it on import corrupts the artifact.

    `import_bundle` builds its INSERT column by column, and an artifact whose `carry_forward` were
    silently discarded would fail `verify_checksum` the next time anyone read it, presenting as
    corruption rather than as a lost field.
    """
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    repository.publish(repository.calibrate(parent_generation, _labels(), embedder).calibration_id)
    child_generation = _ready(manager, embedder, _bodies(added=2), "v2")
    carried = repository.carry_forward(child_generation, embedder)

    bundle = repository.export_bundle(carried.calibration_id, tmp_path / "bundle.json")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "DELETE FROM recall_calibrations WHERE tenant_id = %s AND calibration_id = %s",
            (tenant, carried.calibration_id),
        )

    imported_id = repository.import_bundle(bundle)
    imported = repository.get(imported_id)  # raises on a checksum mismatch
    assert imported.carry_forward is not None
    assert dict(imported.carry_forward) == dict(carried.carry_forward or {})
    assert imported.checksum == carried.checksum


@requires_db
def test_list_records_says_which_thresholds_were_measured_here(carry_tenant) -> None:
    tenant, manager = carry_tenant
    embedder = _CarryEmbedder()
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")

    parent_generation = _ready(manager, embedder, _bodies(), "v1")
    parent = repository.publish(
        repository.calibrate(parent_generation, _labels(), embedder).calibration_id
    )
    child_generation = _ready(manager, embedder, _bodies(added=2), "v2")
    carried = repository.carry_forward(child_generation, embedder)

    records = {record["calibration_id"]: record for record in repository.list_records()}
    assert records[parent.calibration_id]["threshold_was_measured_here"] is True
    assert records[parent.calibration_id]["carried_forward_from"] is None
    assert records[carried.calibration_id]["threshold_was_measured_here"] is False
    assert records[carried.calibration_id]["carried_forward_from"] == parent.calibration_id
    assert records[carried.calibration_id]["corpus_delta"] == pytest.approx(2 / (CORPUS_SIZE + 2))


def test_provenance_cannot_name_a_threshold_the_artifact_does_not_carry() -> None:
    """Provenance that disagrees with the numbers beside it is decoration, and is refused."""
    payload = {
        "artifact_version": 2,
        "calibration_id": "cal_x",
        "tenant_id": "t",
        "generation_id": "gen_x",
        "embedder_identity": {"model": "m"},
        "pipeline_fingerprint": "a" * 64,
        "corpus_fingerprint": "b" * 64,
        "query_set_digest": "c" * 64,
        "threshold": 0.70,
        "scale": 0.05,
        "separability": 0.99,
        "separability_ci": [0.95, 1.0],
        "n_answerable": 22,
        "n_unanswerable": 22,
        "certified": True,
        "certification_reason": "ok",
        "created_at": "2026-08-20T00:00:00+00:00",
        "created_by": "pytest",
        "scores": {},
        "carry_forward": {
            "parent_calibration_id": "cal_parent",
            "parent_generation_id": "gen_parent",
            "corpus_delta": 0.1,
            "inherited_threshold": 0.55,  # not 0.70
        },
    }
    values = dict(payload)
    values["separability_ci"] = tuple(payload["separability_ci"])
    with pytest.raises(CalibrationBindingError, match="does not carry"):
        CalibrationArtifactV2(**values, lifecycle_state="draft", checksum=canonical_sha256(payload))


def test_carry_forward_provenance_requires_a_parent_and_a_delta() -> None:
    with pytest.raises(CalibrationBindingError, match="parent_calibration_id"):
        _require_carry_forward({"parent_generation_id": "g", "corpus_delta": 0.1}, 0.7, 0.05)
    with pytest.raises(CalibrationBindingError, match="corpus_delta"):
        _require_carry_forward({"parent_calibration_id": "c", "parent_generation_id": "g"}, 0.7, 0.05)
    with pytest.raises(CalibrationBindingError, match="fraction in"):
        _require_carry_forward(
            {"parent_calibration_id": "c", "parent_generation_id": "g", "corpus_delta": 1.5},
            0.7,
            0.05,
        )
    # A bool is an int in Python, and `True` would otherwise pass a numeric check and store as a
    # delta of 1.0.
    with pytest.raises(CalibrationBindingError, match="finite number"):
        _require_carry_forward(
            {"parent_calibration_id": "c", "parent_generation_id": "g", "corpus_delta": True},
            0.7,
            0.05,
        )
