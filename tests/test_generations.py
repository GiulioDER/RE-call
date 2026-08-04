from __future__ import annotations

import hashlib
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO

import psycopg
import pytest

from recall.generations import (
    GenerationError,
    GenerationManager,
    NoActiveGeneration,
    UnsafePromotion,
)
from recall.generation_store import GenerationStore, ImmutableGenerationError
from recall.cli import main as cli_main
from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    GenerationState,
    IndexManifestV1,
    ManifestObjectV1,
    PipelineIdentity,
    UnverifiedPipelineError,
)
from recall.manifest import ManifestVerificationError, S3Allowlist, S3ObjectReader
from tests.conftest import TEST_DSN, requires_db


class _S3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])
        data = self.objects[key]
        return {
            "Body": BytesIO(data),
            "ContentLength": len(data),
            "VersionId": kwargs["VersionId"],
        }


class _Embedder:
    def __init__(self, salt: int, model: str = "model-a", dim: int = 64) -> None:
        self.salt = salt
        self.model = model
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [float((index + self.salt) % 7) for index in range(self._dim)] for _ in texts
        ]


def _pipeline(
    model: str, *, overlap: int = 80, fts_language: str = "english"
) -> PipelineIdentity:
    return PipelineIdentity(
        EmbedderIdentity("fixture", model, 64, revision=f"{model}-commit"),
        ChunkerIdentity(
            "paragraph-pack", 1, {"max_chars": 800, "overlap": overlap}
        ),
        fts_configuration={"language": fts_language, "schema_version": 1},
    )


def _manifest(tenant: str, data: bytes, *, version: str = "object-v1") -> IndexManifestV1:
    return IndexManifestV1(
        tenant,
        "corpus-v1",
        (
            ManifestObjectV1(
                f"s3://approved/corpora/{tenant}/memo.md",
                version,
                "text/markdown",
                len(data),
                hashlib.sha256(data).hexdigest(),
            ),
        ),
    )


def _reader(manifest: IndexManifestV1, data: bytes) -> S3ObjectReader:
    entry = manifest.objects[0]
    key = ("approved", f"corpora/{manifest.tenant_id}/memo.md", entry.version_id)
    return S3ObjectReader(_S3({key: data}), S3Allowlist.parse("approved/corpora/"))


@pytest.fixture
def manager():
    tenant = "gen-test-" + uuid.uuid4().hex[:10]
    value = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield value
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


def _ready(manager: GenerationManager, manifest, pipeline, reader, embedder) -> str:
    generation = manager.create(manifest, pipeline)
    manager.build(generation.generation_id, reader, embedder, lambda text: [text])
    manager.validate(generation.generation_id)
    return generation.generation_id


@requires_db
def test_same_dimensional_model_change_cannot_reuse_embeddings(manager) -> None:
    data = b"---\nstatus: current\n---\nfirst immutable memo"
    manifest = _manifest(manager.tenant_id, data)
    reader = _reader(manifest, data)
    first_embedder = _Embedder(1)
    first = _ready(manager, manifest, _pipeline("model-a"), reader, first_embedder)
    manager.promote(first, unsafe_development=True)

    incompatible = _Embedder(2, "model-b")
    second = manager.create(manifest, _pipeline("model-b"))
    stats = manager.build(
        second.generation_id, reader, incompatible, lambda text: [text]
    )

    assert stats.reused_objects == 0
    assert incompatible.calls == 1


@requires_db
def test_exact_pipeline_and_source_hash_reuses_chunks_without_embedding(manager) -> None:
    data = b"unchanged source"
    manifest = _manifest(manager.tenant_id, data)
    reader = _reader(manifest, data)
    pipeline = _pipeline("model-a")
    first = _ready(manager, manifest, pipeline, reader, _Embedder(1))
    manager.promote(first, unsafe_development=True)

    must_not_run = _Embedder(9)
    second = manager.create(manifest, pipeline)
    stats = manager.build(
        second.generation_id, reader, must_not_run, lambda text: [text]
    )

    assert stats.reused_objects == 1
    assert stats.reused_chunks == 1
    assert must_not_run.calls == 0


@requires_db
def test_failed_build_never_changes_the_active_generation(manager) -> None:
    data = b"known good"
    manifest = _manifest(manager.tenant_id, data)
    first = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1))
    manager.promote(first, unsafe_development=True)

    bad_manifest = _manifest(manager.tenant_id, b"manifest claims different bytes")
    failed = manager.create(bad_manifest, _pipeline("model-a"))
    with pytest.raises(ManifestVerificationError):
        manager.build(
            failed.generation_id,
            _reader(bad_manifest, b"mutated object bytes"),
            _Embedder(1),
            lambda text: [text],
        )

    assert manager.active_generation_id() == first
    assert manager.get(failed.generation_id).state == GenerationState.FAILED


@requires_db
def test_embedder_implementation_mismatch_fails_the_generation(manager) -> None:
    data = b"identity mismatch"
    manifest = _manifest(manager.tenant_id, data)
    generation = manager.create(manifest, _pipeline("model-a"))

    with pytest.raises(GenerationError, match="does not match pipeline model"):
        manager.build(
            generation.generation_id,
            _reader(manifest, data),
            _Embedder(1, "model-b"),
            lambda text: [text],
        )

    assert manager.get(generation.generation_id).state == GenerationState.FAILED


@requires_db
def test_sparse_search_uses_the_generation_fts_configuration(manager) -> None:
    data = b"the"
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(
        manager,
        manifest,
        _pipeline("model-a", fts_language="simple"),
        _reader(manifest, data),
        _Embedder(1),
    )
    manager.promote(generation, unsafe_development=True)

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        hits = store.query_sparse("the", 1)

    assert [hit.chunk.text for hit in hits] == ["the"]


@requires_db
def test_forget_survives_rollback_and_tombstone_survives_gc(manager) -> None:
    first_data = b"first revision"
    first_manifest = _manifest(manager.tenant_id, first_data, version="v1")
    pipeline = _pipeline("model-a")
    first = _ready(
        manager, first_manifest, pipeline, _reader(first_manifest, first_data), _Embedder(1)
    )
    manager.promote(first, unsafe_development=True)

    second_data = b"second revision"
    second_manifest = _manifest(manager.tenant_id, second_data, version="v2")
    second = _ready(
        manager, second_manifest, pipeline, _reader(second_manifest, second_data), _Embedder(1)
    )
    manager.promote(second, unsafe_development=True)
    source = second_manifest.objects[0].uri

    erased = manager.forget(source)
    assert erased.chunks_removed == 2
    assert manager.rollback() == first
    manager.gc(retention_days=0, retain_previous=0)

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(
            "SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,)
        )
        remaining = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, source),
        ).fetchone()[0]
        tombstone = conn.execute(
            "SELECT event_id FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, source),
        ).fetchone()
    assert remaining == 0
    assert tombstone == (erased.event_id,)


@requires_db
def test_gc_retains_two_previous_active_generations_not_failed_builds(manager) -> None:
    pipeline = _pipeline("model-a")
    generations: list[str] = []
    for ordinal in range(3):
        data = f"revision {ordinal}".encode()
        manifest = _manifest(manager.tenant_id, data, version=f"v{ordinal}")
        generation = _ready(
            manager, manifest, pipeline, _reader(manifest, data), _Embedder(1)
        )
        manager.promote(generation, unsafe_development=True)
        generations.append(generation)

    failed = manager.create(
        _manifest(manager.tenant_id, b"failed", version="failed"), pipeline
    )
    manager.fail(failed.generation_id, "fixture failure")
    collected = manager.gc(
        now=datetime.now(timezone.utc) + timedelta(days=1),
        retention_days=0,
        retain_previous=2,
    )

    assert collected == (failed.generation_id,)
    assert {record.generation_id for record in manager.list_generations()} == set(generations)


@requires_db
def test_promotion_is_explicitly_unsafe_and_unavailable_in_production(manager) -> None:
    data = b"ready generation"
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(
        manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1)
    )
    with pytest.raises(UnsafePromotion):
        manager.promote(generation)

    production = GenerationManager(
        TEST_DSN, manager.tenant_id, actor="pytest", environment="production"
    )
    with pytest.raises(UnsafePromotion, match="unavailable in production"):
        production.promote(generation, unsafe_development=True)


@requires_db
def test_search_snapshot_sees_one_generation_during_concurrent_promotion(manager) -> None:
    first_data = b"first generation searchable text"
    first_manifest = _manifest(manager.tenant_id, first_data, version="v1")
    pipeline = _pipeline("model-a")
    first = _ready(
        manager, first_manifest, pipeline, _reader(first_manifest, first_data), _Embedder(1)
    )
    manager.promote(first, unsafe_development=True)

    second_data = b"second generation replacement text"
    second_manifest = _manifest(manager.tenant_id, second_data, version="v2")
    second = _ready(
        manager, second_manifest, pipeline, _reader(second_manifest, second_data), _Embedder(1)
    )
    entered = threading.Event()
    promoted = threading.Event()

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        store.check_schema()

        def search_while_promoting() -> tuple[str, str]:
            with store.snapshot() as generation_id:
                entered.set()
                assert promoted.wait(timeout=10)
                dense = store.query_dense([1.0] * 64, 5)[0]
                sparse = store.query_sparse("generation text", 5, vec=[1.0] * 64)[0]
                assert generation_id == first
                return dense.chunk.text, sparse.chunk.text

        def promote() -> None:
            assert entered.wait(timeout=10)
            manager.promote(second, unsafe_development=True)
            promoted.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            search_future = executor.submit(search_while_promoting)
            promotion_future = executor.submit(promote)
            promotion_future.result(timeout=15)
            dense_text, sparse_text = search_future.result(timeout=15)

        assert dense_text == first_data.decode()
        assert sparse_text == first_data.decode()
        assert store.query_dense([1.0] * 64, 5)[0].chunk.text == second_data.decode()

        entered.clear()
        promoted.clear()

        def search_while_rolling_back() -> tuple[str, str]:
            with store.snapshot() as generation_id:
                entered.set()
                assert promoted.wait(timeout=10)
                dense = store.query_dense([1.0] * 64, 5)[0]
                sparse = store.query_sparse("generation text", 5, vec=[1.0] * 64)[0]
                assert generation_id == second
                return dense.chunk.text, sparse.chunk.text

        def rollback() -> None:
            assert entered.wait(timeout=10)
            assert manager.rollback() == first
            promoted.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            search_future = executor.submit(search_while_rolling_back)
            rollback_future = executor.submit(rollback)
            rollback_future.result(timeout=15)
            dense_text, sparse_text = search_future.result(timeout=15)

        assert dense_text == second_data.decode()
        assert sparse_text == second_data.decode()
        assert store.query_dense([1.0] * 64, 5)[0].chunk.text == first_data.decode()


@requires_db
def test_generation_store_refuses_mutation_and_legacy_is_never_an_active_fallback(manager) -> None:
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        with pytest.raises(NoActiveGeneration, match="no active generation"):
            store.query_dense([1.0] * 64, 1)
        with pytest.raises(ImmutableGenerationError):
            store.upsert([], [])


@requires_db
def test_production_rejects_an_unverified_embedder_identity(manager) -> None:
    data = b"unverified development source"
    manifest = _manifest(manager.tenant_id, data)
    unverified = PipelineIdentity(
        EmbedderIdentity(
            "fixture",
            "mutable-model-alias",
            64,
            unverified_reason="development-only test identity",
        ),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
    )
    production = GenerationManager(
        TEST_DSN, manager.tenant_id, actor="pytest", environment="production"
    )

    with pytest.raises(UnverifiedPipelineError):
        production.create(manifest, unverified)
    with pytest.raises(GenerationError, match="allow_unverified=True"):
        manager.create(manifest, unverified)
    generation = manager.create(manifest, unverified, allow_unverified=True)
    assert generation.state == GenerationState.BUILDING


@requires_db
def test_generation_cli_build_validate_promote_and_list(
    manager, tmp_path, monkeypatch, capsys
) -> None:
    data = b"CLI generation source"
    manifest = _manifest(manager.tenant_id, data)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    reader = _reader(manifest, data)
    monkeypatch.setattr(
        S3ObjectReader,
        "from_environment",
        classmethod(lambda cls: reader),
    )
    base = [
        "--serving-dsn",
        TEST_DSN,
        "--tenant",
        manager.tenant_id,
        "--embedder",
        "hashing",
    ]

    cli_main([*base, "generation", "build", str(manifest_path)])
    output = capsys.readouterr().out
    match = re.search(r"built (gen_[0-9a-f]+):", output)
    assert match is not None
    generation_id = match.group(1)

    cli_main([*base, "generation", "validate", generation_id])
    assert f"ready {generation_id}" in capsys.readouterr().out
    cli_main(
        [
            *base,
            "generation",
            "promote",
            generation_id,
            "--unsafe-development-promotion",
        ]
    )
    assert f"active generation: {generation_id}" in capsys.readouterr().out
    cli_main([*base, "generation", "list"])
    listing = capsys.readouterr().out
    assert generation_id in listing and "active" in listing


@requires_db
def test_forget_erases_a_source_that_has_left_the_active_generation(
    manager, monkeypatch, capsys
) -> None:
    """Erasure must not be filtered by what the ACTIVE generation can currently see.

    A source that dropped out of the newest build still has rows in the previous
    generation, and it is exactly that source whose erasure request most needs the
    tombstone: without one, `_is_tombstoned` returns False and the next build happily
    re-ingests it. Filtering the request through `source_content_hashes()` (which is
    scoped to one generation) classified such a source as "not found", so no tombstone
    was written, nothing was deleted, and the command reported success.
    """
    embedder = _Embedder(1)
    first_data = b"---\nstatus: current\n---\nthe memo to erase"
    first_manifest = _manifest(manager.tenant_id, first_data, version="v1")
    first = _ready(
        manager, first_manifest, _pipeline("model-a"), _reader(first_manifest, first_data), embedder
    )
    manager.promote(first, unsafe_development=True)
    memo_uri = first_manifest.objects[0].uri

    # A newer generation built from a corpus that no longer contains memo.md.
    second_data = b"a different document"
    second_manifest = IndexManifestV1(
        manager.tenant_id,
        "corpus-v2",
        (
            ManifestObjectV1(
                f"s3://approved/corpora/{manager.tenant_id}/notes.md",
                "v2",
                "text/markdown",
                len(second_data),
                hashlib.sha256(second_data).hexdigest(),
            ),
        ),
    )
    second_reader = S3ObjectReader(
        _S3({("approved", f"corpora/{manager.tenant_id}/notes.md", "v2"): second_data}),
        S3Allowlist.parse("approved/corpora/"),
    )
    second = _ready(manager, second_manifest, _pipeline("model-a"), second_reader, embedder)
    manager.promote(second, unsafe_development=True)

    store = GenerationStore(TEST_DSN, embedder.dim, tenant=manager.tenant_id)
    try:
        # Invisible to the active generation, which is what the old filter keyed on...
        assert memo_uri not in store.source_content_hashes()
    finally:
        store.close()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        # ...but its rows are still on disk in the previous generation.
        before = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, memo_uri),
        ).fetchone()[0]
    assert before > 0

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(
        ["--serving-dsn", TEST_DSN, "--tenant", manager.tenant_id, "--embedder", "hashing",
         "forget", memo_uri, "--yes"]
    )
    capsys.readouterr()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        after = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, memo_uri),
        ).fetchone()[0]
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, memo_uri),
        ).fetchone()[0]
    assert after == 0, "chunks in a non-active generation survived the erasure"
    assert tombstoned == 1, "no tombstone was written, so the next build will re-ingest it"
