"""Calibration v2 is exact, tenant scoped, immutable, and history preserving."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb

from recall.calibration_v2 import (
    CalibrationBindingError,
    CalibrationNotFound,
    CalibrationRepository,
    CalibrationStatus,
    CalibrationUncertified,
    canonical_query_set,
)
from recall.cli import main as cli_main
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.trust import trusted_search
from recall_mcp.service import search_memory
from tests.conftest import TEST_DSN, requires_db
from tests.test_generations import _manifest, _pipeline, _reader


class _CalibrationEmbedder:
    model = "cal-model"
    dim = 64

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        values: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            vector[1 if text.startswith("missing-") else 0] = 1.0
            values.append(vector)
        return values


def _labels(count: int = 20) -> list[dict[str, object]]:
    return [
        *({"query": f"answer-{index}", "answerable": True} for index in range(count)),
        *({"query": f"missing-{index}", "answerable": False} for index in range(count)),
    ]


@pytest.fixture
def calibration_tenant():
    tenant = "cal-test-" + uuid.uuid4().hex[:10]
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


def _ready(manager: GenerationManager, embedder: _CalibrationEmbedder, data: bytes, version: str):
    manifest = _manifest(manager.tenant_id, data, version=version)
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(
        generation.generation_id,
        _reader(manifest, data),
        embedder,
        lambda text: [text],
    )
    manager.validate(generation.generation_id)
    return generation.generation_id


@requires_db
def test_exact_published_binding_drives_search_and_exposes_identities(calibration_tenant) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)

    assert artifact.certified is True
    assert repository.resolve(generation_id).status == CalibrationStatus.DRAFT
    artifact = repository.publish(artifact.calibration_id)
    assert repository.resolve(generation_id).artifact == artifact
    manager.promote(generation_id, unsafe_development=True)

    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        result = trusted_search(store, embedder, "answer-live", k=1)
    finally:
        store.close()

    assert result.calibrated is True
    assert result.calibration_status == CalibrationStatus.CERTIFIED
    assert result.calibration_id == artifact.calibration_id
    assert result.tenant_id == tenant
    assert result.generation_id == generation_id
    assert result.pipeline_fingerprint == artifact.pipeline_fingerprint
    assert result.corpus_fingerprint == artifact.corpus_fingerprint
    assert result.query_set_digest == artifact.query_set_digest

    store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
    try:
        mcp_result = search_memory(store, embedder, "answer-live", k=1)
    finally:
        store.close()
    assert mcp_result.calibrated is True
    assert mcp_result.calibration_id == artifact.calibration_id
    assert mcp_result.generation_id == generation_id
    assert mcp_result.pipeline_fingerprint == artifact.pipeline_fingerprint


@requires_db
def test_cross_tenant_access_and_import_are_rejected(calibration_tenant, tmp_path: Path) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    owner = CalibrationRepository(TEST_DSN, tenant)
    artifact = owner.publish(owner.calibrate(generation_id, _labels(), embedder).calibration_id)
    bundle = owner.export_bundle(artifact.calibration_id, tmp_path / "artifact.json")

    other = CalibrationRepository(TEST_DSN, "other-" + uuid.uuid4().hex[:10])
    with pytest.raises(CalibrationNotFound):
        other.get(artifact.calibration_id)
    with pytest.raises(CalibrationBindingError, match="another tenant"):
        other.import_bundle(bundle)


@requires_db
def test_new_corpus_is_stale_and_reusing_labels_recomputes_every_score(calibration_tenant) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    first = _ready(manager, embedder, b"answer corpus v1", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant)
    labels = _labels()
    first_artifact = repository.publish(
        repository.calibrate(first, labels, embedder).calibration_id
    )
    manager.promote(first, unsafe_development=True)

    second = _ready(manager, embedder, b"answer corpus v2", "v2")
    manager.promote(second, unsafe_development=True)
    assert repository.resolve(second).status == CalibrationStatus.STALE

    calls_before = embedder.calls
    second_artifact = repository.calibrate(second, labels, embedder)
    assert embedder.calls - calls_before == len(labels)
    assert second_artifact.calibration_id != first_artifact.calibration_id
    assert second_artifact.query_set_digest == first_artifact.query_set_digest
    assert second_artifact.corpus_fingerprint != first_artifact.corpus_fingerprint


@requires_db
def test_privacy_erasure_changes_effective_corpus_and_stales_calibration(
    calibration_tenant,
) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    data = b"answer corpus"
    manifest = _manifest(tenant, data, version="v1")
    generation_id = _ready(manager, embedder, data, "v1")
    repository = CalibrationRepository(TEST_DSN, tenant)
    artifact = repository.publish(
        repository.calibrate(generation_id, _labels(), embedder).calibration_id
    )

    manager.forget(manifest.objects[0].uri)

    changed_fingerprint = manager.get(generation_id).corpus_fingerprint
    assert changed_fingerprint != artifact.corpus_fingerprint
    assert repository.resolve(generation_id).status == CalibrationStatus.STALE
    with pytest.raises(CalibrationBindingError, match="lineage changed"):
        repository.publish(artifact.calibration_id)
    replacement = manager.create(manifest, _pipeline(embedder.name))
    assert replacement.corpus_fingerprint == changed_fingerprint


@requires_db
def test_generation_pipeline_and_query_set_mismatches_are_stale(calibration_tenant) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    data = b"answer corpus"
    manifest = _manifest(tenant, data, version="v1")
    first = manager.create(manifest, _pipeline(embedder.name, overlap=20))
    manager.build(first.generation_id, _reader(manifest, data), embedder, lambda text: [text])
    manager.validate(first.generation_id)
    repository = CalibrationRepository(TEST_DSN, tenant)
    artifact = repository.publish(
        repository.calibrate(first.generation_id, _labels(), embedder).calibration_id
    )

    second = manager.create(manifest, _pipeline(embedder.name, overlap=40))
    manager.build(second.generation_id, _reader(manifest, data), embedder, lambda text: [text])
    manager.validate(second.generation_id)
    assert repository.resolve(second.generation_id).status == CalibrationStatus.STALE

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        with pytest.raises(psycopg.errors.RaiseException, match="query sets are immutable"):
            conn.execute(
                "UPDATE recall_calibration_query_sets SET queries = %s "
                "WHERE tenant_id = %s AND query_set_digest = %s",
                (
                    Jsonb([{"query": "tampered", "answerable": True}]),
                    tenant,
                    artifact.query_set_digest,
                ),
            )
        with pytest.raises(psycopg.errors.RaiseException, match="payloads are immutable"):
            conn.execute(
                "UPDATE recall_calibrations SET threshold = 0.12 "
                "WHERE tenant_id = %s AND calibration_id = %s",
                (tenant, artifact.calibration_id),
            )
    assert repository.resolve(first.generation_id).status == CalibrationStatus.CERTIFIED


@requires_db
def test_export_import_is_checksum_verified_and_imported_history_is_not_active(
    calibration_tenant, tmp_path: Path
) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant)
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    bundle = repository.export_bundle(artifact.calibration_id, tmp_path / "bundle.json")

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "DELETE FROM recall_calibrations WHERE tenant_id = %s AND calibration_id = %s",
            (tenant, artifact.calibration_id),
        )
    imported_id = repository.import_bundle(bundle)
    imported = repository.get(imported_id)
    assert imported.lifecycle_state == "draft"
    assert repository.resolve(generation_id).status == CalibrationStatus.DRAFT
    with pytest.raises(TypeError):
        imported.embedder_identity["model"] = "mutated"  # type: ignore[index]

    corrupted = json.loads(bundle.read_text(encoding="utf-8"))
    corrupted["artifact"]["threshold"] = 0.12
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(CalibrationBindingError, match="bundle checksum"):
        repository.import_bundle(corrupt_path)


@requires_db
def test_uncertified_and_legacy_evidence_cannot_be_published(
    calibration_tenant, tmp_path: Path
) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant)

    legacy_path = tmp_path / "calibration.json"
    legacy_path.write_text(
        json.dumps({"embedder": embedder.name, "threshold": 0.61, "scale": 0.05}),
        encoding="utf-8",
    )
    legacy_id = repository.import_bundle(legacy_path)
    record = repository.show_record(legacy_id)
    assert record["lifecycle_state"] == CalibrationStatus.LEGACY_UNBOUND
    assert record["generation_id"] is None
    assert repository.resolve(generation_id).status == CalibrationStatus.MISSING

    rejected = repository.calibrate(generation_id, _labels(2), embedder)
    assert rejected.status == CalibrationStatus.REJECTED
    with pytest.raises(CalibrationUncertified):
        repository.publish(rejected.calibration_id)
    assert repository.get(rejected.calibration_id).lifecycle_state == "rejected"
    assert repository.resolve(generation_id).status == CalibrationStatus.UNCERTIFIED


@requires_db
def test_concurrent_publication_keeps_one_active_artifact_and_immutable_history(
    calibration_tenant,
) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant)
    first = repository.calibrate(generation_id, _labels(), embedder)
    second = repository.calibrate(generation_id, _labels(), embedder)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(repository.publish, (first.calibration_id, second.calibration_id)))

    states = {row["calibration_id"]: row["lifecycle_state"] for row in repository.list_records()}
    assert sorted(states.values()) == ["published", "superseded"]
    assert repository.resolve(generation_id).artifact is not None
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        events = {
            row[0]
            for row in conn.execute(
                "SELECT event_type FROM recall_audit_events WHERE tenant_id = %s",
                (tenant,),
            ).fetchall()
        }
    assert {"calibration_created", "calibration_published", "calibration_superseded"} <= events


@requires_db
def test_calibration_cli_create_publish_inspect_and_export(
    calibration_tenant, tmp_path: Path, monkeypatch, capsys
) -> None:
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(_labels()), encoding="utf-8")
    monkeypatch.setattr("recall.cli._make_embedder", lambda _name: embedder)
    base = ["--serving-dsn", TEST_DSN, "--tenant", tenant]

    cli_main(
        [
            *base,
            "calibrate",
            "--generation",
            generation_id,
            "--queries",
            str(query_path),
            "--publish",
        ]
    )
    created = capsys.readouterr().out
    calibration_id = next(
        line.split(":", 1)[1].strip()
        for line in created.splitlines()
        if line.startswith("calibration:")
    )
    assert "status: certified" in created

    cli_main([*base, "calibration", "show", calibration_id])
    assert generation_id in capsys.readouterr().out
    cli_main([*base, "calibration", "list"])
    assert calibration_id in capsys.readouterr().out
    output = tmp_path / "export.json"
    cli_main(
        [*base, "calibration", "export", calibration_id, "--output", str(output)]
    )
    capsys.readouterr()
    assert json.loads(output.read_text(encoding="utf-8"))["artifact"]["calibration_id"] == calibration_id


def test_query_set_digest_is_canonical_and_rejects_duplicates() -> None:
    first, first_digest = canonical_query_set(
        [
            {"query": "b", "answerable": False, "relevant_ids": ["2", "1", "1"]},
            {"query": "a", "answerable": True},
        ]
    )
    second, second_digest = canonical_query_set(list(reversed(first)))
    assert first == second
    assert first_digest == second_digest
    with pytest.raises(CalibrationBindingError, match="duplicate"):
        canonical_query_set([{"query": "a", "answerable": True}] * 2)
