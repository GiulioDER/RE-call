"""Calibration v2 is exact, tenant scoped, immutable, and history preserving."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg.types.json import Jsonb

from recall.calibration_v2 import (
    CalibrationArtifactV2,
    CalibrationBindingError,
    CalibrationNotFound,
    CalibrationRepository,
    CalibrationStatus,
    CalibrationUncertified,
    canonical_query_set,
    canonical_sha256,
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


def _dsn_in_timezone(dsn: str, timezone: str) -> str:
    """The same DSN, but the session runs in `timezone` instead of the server default."""
    parts = urlsplit(dsn)
    query = dict(parse_qsl(parts.query))
    query["options"] = f"-c TimeZone={timezone}"
    # quote_via=quote, not the default quote_plus: libpq reads a "+" literally, so a
    # plus-encoded space turns the option into a parameter named "+TimeZone".
    return urlunsplit(parts._replace(query=urlencode(query, quote_via=quote)))


@requires_db
def test_a_calibration_reads_back_under_a_non_utc_session_timezone(calibration_tenant) -> None:
    """`created_at` is checksummed as a string but stored as `timestamptz`.

    psycopg renders a `timestamptz` in the *connection's* TimeZone, so a calibration
    written by one session and read by another whose TimeZone differs recomputes a
    different digest and raises `CalibrationBindingError` on every read. The suite's
    own database runs UTC, which is exactly why this needs an explicit non-UTC
    session: under UTC the defect is invisible.
    """
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")

    written = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = written.calibrate(generation_id, _labels(), embedder)
    published = written.publish(artifact.calibration_id)

    rome = CalibrationRepository(_dsn_in_timezone(TEST_DSN, "Europe/Rome"), tenant, actor="pytest")

    # The stored instant must survive the round trip as the same string.
    assert rome.get(published.calibration_id).created_at == published.created_at
    # And the serving path must still resolve it, not fail closed on a digest mismatch.
    assert rome.resolve(generation_id).status == CalibrationStatus.CERTIFIED
    # The listing renders the same instant too.
    listed = {row["calibration_id"]: row["created_at"] for row in rome.list_records()}
    assert listed[published.calibration_id] == published.created_at


@requires_db
def test_a_bundle_whose_timestamp_is_not_canonical_utc_is_refused_at_import(
    calibration_tenant, tmp_path: Path
) -> None:
    """An artifact is only storable if its `created_at` survives the timestamptz round trip.

    The column keeps the instant and discards the rendering, and `created_at` is inside the
    checksum as a *string*. So a bundle carrying any other valid ISO-8601 spelling of the
    same instant (a non-UTC offset, or the `Z`-plus-milliseconds form a JavaScript producer
    emits) would import cleanly and then be unreadable by every later `get`/`publish`/
    `resolve`. Refuse it at the boundary instead of committing a row nothing can read.
    """
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(generation_id, _labels(), embedder)
    bundle = json.loads(repository.export_bundle(artifact.calibration_id, tmp_path / "b.json").read_text(encoding="utf-8"))

    instant = datetime.fromisoformat(bundle["artifact"]["created_at"])
    for spelling in (
        instant.astimezone(timezone(timedelta(hours=2))).isoformat(),
        instant.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    ):
        raw = dict(bundle["artifact"], created_at=spelling)
        # Re-sign honestly, so this tests the rendering rule and not tamper detection.
        immutable = {key: raw[key] for key in CalibrationArtifactV2.__dataclass_fields__ if key in raw}
        immutable.pop("lifecycle_state", None)
        immutable.pop("checksum", None)
        raw["checksum"] = canonical_sha256(immutable)
        # Re-sign the envelope too, or import stops at the bundle checksum and never reaches
        # the artifact, which would test tamper detection instead of the rendering rule.
        rebuilt = dict(bundle, artifact=raw)
        rebuilt["bundle_checksum"] = canonical_sha256(
            {key: value for key, value in rebuilt.items() if key != "bundle_checksum"}
        )
        forged = tmp_path / "forged.json"
        forged.write_text(json.dumps(rebuilt), encoding="utf-8")
        with pytest.raises(CalibrationBindingError, match="created_at"):
            repository.import_bundle(forged)


@requires_db
def test_list_and_show_render_the_same_created_at(calibration_tenant) -> None:
    """`calibration list` and `calibration show` must not disagree about the same field.

    `show_record` reads the row through `to_jsonb`, which Postgres renders in the session
    TimeZone as a string, so the datetime-normalising loop never sees it.
    """
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    generation_id = _ready(manager, embedder, b"answer corpus", "v1")
    rome = CalibrationRepository(_dsn_in_timezone(TEST_DSN, "Europe/Rome"), tenant, actor="pytest")
    artifact = rome.calibrate(generation_id, _labels(), embedder)

    listed = {row["calibration_id"]: row["created_at"] for row in rome.list_records()}
    shown = rome.show_record(artifact.calibration_id)
    assert shown["created_at"] == listed[artifact.calibration_id] == artifact.created_at
