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

from recall.derived_block import DerivedEntry, render_derived_block
from recall.generations import (
    GenerationError,
    GenerationManager,
    NoActiveGeneration,
    UnsafePromotion,
    _body_rule_changed,
)
from psycopg.pq import TransactionStatus

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
        conn.execute("DELETE FROM chunks WHERE tenant_id = %s", (tenant,))


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
def test_cosines_for_matches_query_dense_on_the_generation_store(manager) -> None:
    """`GenerationStore` overrides `_cosines_for`; the base implementation selects a column,

    `id`, that `recall_chunks_v1` does not have. Without this override the call raises
    `UndefinedColumn`, and `RECALL_ENV=production` selects `GenerationStore`.
    """
    data = b"alpha generation text"
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(
        manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1)
    )
    manager.promote(generation, unsafe_development=True)

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        vec = [1.0] * 64
        dense = store.query_dense(vec, k=1)
        chunk_id = dense[0].chunk.id

        rescored = store.cosines_for([chunk_id], vec)

    assert rescored[chunk_id] == pytest.approx(dense[0].score, abs=1e-6)


@requires_db
def test_cosines_for_omits_a_chunk_from_a_generation_that_is_no_longer_active(manager) -> None:
    """The row for a retired generation's chunk still physically exists in `recall_chunks_v1`

    (`LIVE_MANIFEST_STATES` keeps it until `gc`), so an unscoped query would find it. `cosines_for`
    must filter by the ACTIVE generation the same way `_query_dense` and `_query_sparse` do, or a
    rescore could report a cosine for a chunk that is not part of what is being searched.
    """
    pipeline = _pipeline("model-a")

    first_data = b"first generation text"
    first_manifest = _manifest(manager.tenant_id, first_data, version="v1")
    first = _ready(
        manager, first_manifest, pipeline, _reader(first_manifest, first_data), _Embedder(1)
    )
    manager.promote(first, unsafe_development=True)

    vec = [1.0] * 64
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        retired_chunk_id = store.query_dense(vec, k=1)[0].chunk.id

    second_data = b"second generation text"
    second_manifest = _manifest(manager.tenant_id, second_data, version="v2")
    second = _ready(
        manager, second_manifest, pipeline, _reader(second_manifest, second_data), _Embedder(1)
    )
    manager.promote(second, unsafe_development=True)

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        assert store.cosines_for([retired_chunk_id], vec) == {}


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


@requires_db
def test_forget_refuses_a_source_no_generation_ever_held(manager, monkeypatch, capsys) -> None:
    """A typo must not write a tombstone, because a tombstone cannot be undone.

    Nothing in the package deletes from `recall_source_tombstones`, there is no `unforget`
    command, and `build()` skips every manifest entry a tombstone matches. So tombstoning an
    unknown URI would silently and irreversibly bar it from every future build. Erasure must
    widen its existence check to all generations, not abandon it.
    """
    embedder = _Embedder(1)
    data = b"---\nstatus: current\n---\nkeep me"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    generation = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), embedder)
    manager.promote(generation, unsafe_development=True)
    typo = manifest.objects[0].uri + "d"

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(
        ["--serving-dsn", TEST_DSN, "--tenant", manager.tenant_id, "--embedder", "hashing",
         "forget", typo, "--yes"]
    )
    out = capsys.readouterr().out

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        tombstones = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert tombstones == 0, "a typo wrote an irreversible tombstone"
    assert "NOT tombstoned" in out
    # The real source is untouched and still erasable.
    assert manager.get(generation).state == "active"


@requires_db
def test_forget_refuses_a_blank_source_before_erasing_anything(
    manager, monkeypatch, capsys
) -> None:
    """`recall forget "$A" "$B" --yes` with one variable unset must erase nothing.

    delete_sources commits one transaction per source, so validating late meant the first
    source was erased, GenerationManager.forget raised on the empty string, the remaining
    sources were never reached, and the report line never printed because it sat inside the
    statement that raised.
    """
    embedder = _Embedder(1)
    data = b"---\nstatus: current\n---\nkeep me"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    generation = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), embedder)
    manager.promote(generation, unsafe_development=True)
    real = manifest.objects[0].uri

    monkeypatch.setenv("RECALL_ENV", "production")
    with pytest.raises(SystemExit, match="empty source argument"):
        cli_main(
            ["--serving-dsn", TEST_DSN, "--tenant", manager.tenant_id, "--embedder", "hashing",
             "forget", real, "", "--yes"]
        )
    capsys.readouterr()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        survived = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, real),
        ).fetchone()[0]
        tombstones = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert survived > 0, "the valid source was erased despite the malformed request"
    assert tombstones == 0


@requires_db
def test_forget_erases_a_source_that_chunked_to_nothing(manager, monkeypatch, capsys) -> None:
    """Chunk rows are not the corpus; the manifest is.

    An object whose body chunks to nothing is built as `empty_objects` and writes no row to
    recall_chunks_v1. Checking only chunk rows therefore called it a typo, refused the
    erasure, wrote no tombstone, and let the next build ingest it the moment its content
    changed: the same class STAKES-001 closed, reached from a different direction.
    """
    embedder = _Embedder(1)
    body = b"---\nstatus: current\n---\nreal content"
    blank = b"---\nstatus: current\n---\n"
    tenant = manager.tenant_id
    manifest = IndexManifestV1(
        tenant,
        "corpus-v1",
        (
            ManifestObjectV1(f"s3://approved/corpora/{tenant}/memo.md", "v1", "text/markdown",
                             len(body), hashlib.sha256(body).hexdigest()),
            ManifestObjectV1(f"s3://approved/corpora/{tenant}/blank.md", "v1", "text/markdown",
                             len(blank), hashlib.sha256(blank).hexdigest()),
        ),
    )
    reader = S3ObjectReader(
        _S3({
            ("approved", f"corpora/{tenant}/memo.md", "v1"): body,
            ("approved", f"corpora/{tenant}/blank.md", "v1"): blank,
        }),
        S3Allowlist.parse("approved/corpora/"),
    )
    # IndexManifestV1 sorts its objects by URI, so pick by name, not by position.
    blank_uri = next(o.uri for o in manifest.objects if o.uri.endswith("blank.md"))

    generation = manager.create(manifest, _pipeline("model-a"))
    # Mirrors the real chunker: an empty body yields no chunks at all.
    manager.build(generation.generation_id, reader, embedder,
                  lambda text: [] if not text.strip() else [text])
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (tenant, blank_uri),
        ).fetchone()[0]
    assert rows == 0, "fixture no longer produces a zero-chunk source"

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(["--serving-dsn", TEST_DSN, "--tenant", tenant, "--embedder", "hashing",
              "forget", blank_uri, "--yes"])
    out = capsys.readouterr().out

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (tenant, blank_uri),
        ).fetchone()[0]
    assert tombstoned == 1, "a source the corpus contains was refused as a typo"
    assert "NOT erased" not in out


@requires_db
def test_forget_refuses_a_url_only_a_failed_generation_names(manager, monkeypatch, capsys) -> None:
    """A FAILED generation's manifest entries must not be tombstonable.

    A failed build may name objects that never existed at the source, and a tombstone is
    permanent, so admitting them would let a URI that was never in the corpus be barred from
    every future build. `building` is deliberately NOT excluded: see the test below.
    """
    data = b"---\nstatus: current\n---\nnever built"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = manager.create(manifest, _pipeline("model-a"))
    manager.fail(generation.generation_id, "the source object could not be fetched")
    assert manager.get(generation.generation_id).state == "failed"

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(
        ["--serving-dsn", TEST_DSN, "--tenant", manager.tenant_id, "--embedder", "hashing",
         "forget", uri, "--yes"]
    )
    out = capsys.readouterr().out

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert tombstoned == 0, "a failed generation's manifest entry got a permanent tombstone"
    assert "NOT tombstoned" in out


@requires_db
def test_every_live_state_is_erasable_and_only_failed_is_not(manager) -> None:
    """Both directions of the state list, pinned per state.

    Admitting a state too many is a permanent tombstone on a URI that was never in the
    corpus. Dropping one is the mirror harm and the quieter one: a genuine right-to-erasure
    request answered "check for typos" while the data stays on disk. Only `building` was
    pinned, so `retired` could be dropped with the suite green.

    The iteration domain is EVERY state the schema allows, with the expectation written per
    state, and it is deliberately not derived from `LIVE_MANIFEST_STATES` nor from any
    literal that mirrors it. An earlier version looped over a copy of the live list, so
    dropping `retired` from both the constant and the copy (the edit its own failure message
    invited) removed that state's behavioural check in the same stroke and stayed green.
    Here, dropping a state from the constant leaves its `True` entry standing and turns this
    red; making it green again means writing `False` next to that state, which is an explicit,
    reviewable claim that such a source is not erasable.
    """
    erasable_by_state = {
        "building": True,
        "validating": True,
        "ready": True,
        "active": True,
        "retired": True,
        # A failed build may name objects that never existed at the source, and a tombstone
        # is permanent, so these must never resolve.
        "failed": False,
        # Migration 0008's adopted rows carry a `{"legacy_table": ...}` manifest with no
        # objects; `sources_in_legacy_table()` is what finds them, not this query.
        "legacy_unverified": False,
    }
    # A state added to EITHER domain must force a decision here rather than fall outside the
    # sweep unnoticed. Both are checked because they fail differently: the enum is what the
    # code can produce, the CHECK constraint is what the database will accept, and a state
    # can be added to one without the other.
    assert {s.value for s in GenerationState} == set(erasable_by_state), (
        "GenerationState and this sweep disagree; decide whether the new state is erasable "
        "and add it above"
    )
    # Found by the COLUMN it covers, not by name. `LIKE '%state%'` matched any constraint whose
    # text merely mentions the column; hardcoding `conname` traded that for a different
    # brittleness, since a later migration that drops and re-adds the constraint under another
    # name would report "found 0" and blame the wrong thing. `state = ANY(conkey)` (membership,
    # not equality) also keeps working if the constraint grows to cover a second column.
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'recall_generations'::regclass AND contype = 'c' "
            "AND (SELECT attnum FROM pg_attribute "
            "     WHERE attrelid = 'recall_generations'::regclass AND attname = 'state') "
            "    = ANY(conkey)"
        ).fetchall()
    assert len(rows) == 1, (
        "expected exactly one CHECK constraint covering recall_generations.state, found "
        f"{len(rows)}: {sorted(name for name, _ in rows)}"
    )
    # The definition's SHAPE is matched whole and its members compared as a set, rather than a
    # sub-expression being scraped out of it. Three separate defects came from reading a
    # fragment and treating it as the domain, and each failed OPEN, which is the direction that
    # leaves a live state's erasability undecided:
    #   * `state = ANY (ARRAY[` has no left boundary, so it also matched inside
    #     `prev_state = ANY (ARRAY[`, and `re.search` takes the FIRST hit. The same constraint
    #     with the same widened domain passed or failed on the order of its conjuncts alone.
    #   * a sub-expression is the domain only in a CONJUNCTIVE position. `... OR state =
    #     'archived_v2'` widened what the database accepts to eight states while the scrape
    #     still reported the seven inside the ARRAY.
    #   * the non-greedy `(.*?)\]\)` stopped at the first `])`, including one inside a state
    #     literal, silently truncating the member list.
    # Pinning the rendered SHAPE costs the tolerance for a benign tightening that the fragment
    # scrape was written to buy. That tolerance is what admitted all three, and this constraint
    # guards a PERMANENT tombstone, so any edit to it should be re-read against this map rather
    # than waved through. Narrowing was already caught; it is widening that must not slip.
    #
    # Three assertions, each doing one job, because pinning the definition BYTE for byte did one
    # job too many: it tied the verdict to this map's insertion order, so merely reordering the
    # map (or the migration's ARRAY) without changing either one's CONTENTS went red, carrying
    # the identical message a real widening carries. The only instruction that message offers is
    # "update the map", which is precisely the reflex that must never be applied to a widening.
    shape = re.fullmatch(r"CHECK \(\(state = ANY \(ARRAY\[(.*)\]\)\)\)", rows[0][1])
    assert shape is not None, (
        "the state constraint is no longer a plain membership test over `state` alone. It may "
        "have gained a clause, an OR, or another column, and a sub-expression of it is NOT the "
        f"domain. Re-read it against the map above before changing this.\n  {rows[0][1]}"
    )
    literals = re.findall(r"'((?:[^']|'')*)'::text", shape.group(1))
    # Proves the parse accounted for every byte of the member list rather than stopping early,
    # which is how a `]` inside a literal silently truncated it before.
    assert ", ".join(f"'{literal}'::text" for literal in literals) == shape.group(1), (
        f"could not read the state list as a plain sequence of quoted literals: {shape.group(1)}"
    )
    schema_states = {literal.replace("''", "'") for literal in literals}
    assert schema_states == set(erasable_by_state), (
        "the schema's state domain and this sweep disagree "
        f"(only in schema: {sorted(schema_states - set(erasable_by_state))}; "
        f"only in sweep: {sorted(set(erasable_by_state) - schema_states)}); decide whether "
        "each is erasable and update the map above"
    )

    for state, should_resolve in erasable_by_state.items():
        data = f"---\nstatus: current\n---\nnamed only by a {state} generation".encode()
        manifest = _manifest(manager.tenant_id, data, version="v1")
        uri = manifest.objects[0].uri
        generation = manager.create(manifest, _pipeline("model-a"))
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
            conn.execute(
                "UPDATE recall_generations SET state = %s "
                "WHERE tenant_id = %s AND generation_id = %s",
                (state, manager.tenant_id, generation.generation_id),
            )
        with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
            resolved = store.manifest_uris_matching([uri])
        # `_manifest` derives the URI from the TENANT, so every iteration names the same
        # source. Left in place, the live generations from earlier iterations would keep it
        # resolving and the `failed` case would pass for the wrong reason.
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
            conn.execute(
                "DELETE FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
                (manager.tenant_id, generation.generation_id),
            )
        if should_resolve:
            assert resolved == frozenset({uri}), (
                f"a source named only by a {state!r} generation did not resolve, so erasing "
                "it would be refused as a typo while the data stays on disk"
            )
        else:
            assert resolved == frozenset(), (
                f"a {state!r} generation's manifest entry resolved, and resolving is what "
                "earns a permanent tombstone"
            )


@requires_db
def test_the_store_itself_refuses_a_url_only_a_failed_generation_names(manager) -> None:
    """The same exclusion, asserted on the surface MCP erasures actually resolve through.

    The test above drives the CLI. When the resolver moved to its own SQL query with its own
    copy of the live-state list, adding `failed` to that copy left the whole forget suite
    green, because nothing drove `sources_for_identifiers` through a failed generation. A
    resolved identifier is exactly what `forget()` converts into a permanent tombstone, so
    the gap was one mutation away from barring a URI that was never in the corpus.
    """
    data = b"---\nstatus: current\n---\nnever built"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = manager.create(manifest, _pipeline("model-a"))
    manager.fail(generation.generation_id, "the source object could not be fetched")
    assert manager.get(generation.generation_id).state == "failed"

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        assert store.manifest_uris_matching([uri]) == frozenset()
        assert store.sources_for_identifiers([uri]) == {}


@requires_db
def test_a_non_string_manifest_uri_does_not_resolve(manager) -> None:
    """`->>` casts a JSON number to text; `isinstance(uri, str)` does not.

    The manifest column is raw JSONB and nothing revalidates its shape on read, so a
    manifest carrying `{"uri": 123}` made the identifier `123` resolve through SQL while the
    Python path rejected it. Resolving is what earns a permanent tombstone.
    """
    data = b"---\nstatus: current\n---\nshape check"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    generation = manager.create(manifest, _pipeline("model-a"))
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        conn.execute(
            "UPDATE recall_generations SET manifest = %s::jsonb, state = 'ready' "
            "WHERE tenant_id = %s AND generation_id = %s",
            ('{"objects": [{"uri": 123}]}', manager.tenant_id, generation.generation_id),
        )

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        assert store.manifest_uris_matching(["123"]) == frozenset()
        assert store.sources_for_identifiers(["123"]) == {}


@requires_db
@pytest.mark.parametrize("objects", ["null", "5", '"a string"', "true"])
def test_a_malformed_objects_value_does_not_abort_the_erasure(manager, objects) -> None:
    """`jsonb_array_elements` RAISES on a scalar, and this query is now on the CLI path too.

    The `jsonb_typeof(g.manifest->'objects') = 'array'` guard reads like the trap where a
    WHERE clause cannot protect a FROM clause, and nothing could see it: deleting it left the
    whole suite green. It is load-bearing. The manifest column is raw JSONB and nothing
    revalidates its shape on read, so one row like this would turn a right-to-erasure request
    into `InvalidParameterValue: cannot extract elements from a scalar`, aborting the request
    rather than answering it. A MISSING `objects` key is already safe (SQL NULL, zero rows)
    and is NOT what this guards.
    """
    data = b"---\nstatus: current\n---\nmalformed manifest"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = manager.create(manifest, _pipeline("model-a"))
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        conn.execute(
            "UPDATE recall_generations SET manifest = %s::jsonb, state = 'ready' "
            "WHERE tenant_id = %s AND generation_id = %s",
            (f'{{"objects": {objects}}}', manager.tenant_id, generation.generation_id),
        )

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        assert store.manifest_uris_matching([uri]) == frozenset()


@requires_db
def test_forget_during_a_build_is_honoured_by_that_build(manager, monkeypatch, capsys) -> None:
    """An erasure issued while a generation is BUILDING must land in that build.

    `building` is the state a generation occupies for the whole of its ingest, and build()
    re-checks `_is_tombstoned` per object exactly so a concurrent erasure takes effect.
    Excluding `building` from the existence check made the request answer "check for typos",
    write no tombstone, and let the build index the content the user asked to erase.
    """
    embedder = _Embedder(1)
    data = b"---\nstatus: current\n---\nthe user asked for this to be erased"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = manager.create(manifest, _pipeline("model-a"))
    assert manager.get(generation.generation_id).state == "building"

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(
        ["--serving-dsn", TEST_DSN, "--tenant", manager.tenant_id, "--embedder", "hashing",
         "forget", uri, "--yes"]
    )
    capsys.readouterr()

    stats = manager.build(
        generation.generation_id, _reader(manifest, data), embedder, lambda text: [text]
    )
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        surviving = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, uri),
        ).fetchone()[0]
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert tombstoned == 1, "the erasure was refused, so the build had nothing to honour"
    assert stats.tombstoned_objects == 1
    assert surviving == 0, "the build indexed content the user had asked to erase"


@requires_db
def test_forget_erases_rows_adopted_from_the_v08_table(manager, monkeypatch, capsys) -> None:
    """Migration 0008 adopts a v0.8 install's rows IN PLACE, and they must stay erasable.

    Those rows never enter recall_chunks_v1, and the `legacy_unverified` generation carries a
    `{"legacy_table": ...}` manifest with no `objects`, so neither the chunk probe nor the
    manifest probe can see them. Production `forget` answered an erasure request for one with
    "check for typos" about data the tenant demonstrably holds, and widening the check alone
    would have written the tombstone while leaving the rows on disk.
    """
    tenant = manager.tenant_id
    uri = "/legacy/adopted-note.md"
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "INSERT INTO chunks (tenant_id, id, source, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            # `chunks` is vector(64) here: the CLI below runs with --embedder hashing.
            (tenant, f"legacy-{uuid.uuid4().hex[:8]}", uri, "adopted body",
             "[" + ",".join(["0.0"] * 63) + ",1.0]"),
        )
        seeded = conn.execute(
            "SELECT count(*) FROM chunks WHERE tenant_id = %s AND source = %s", (tenant, uri)
        ).fetchone()[0]
    assert seeded == 1

    monkeypatch.setenv("RECALL_ENV", "production")
    cli_main(
        ["--serving-dsn", TEST_DSN, "--tenant", tenant, "--embedder", "hashing",
         "forget", uri, "--yes"]
    )
    out = capsys.readouterr().out

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        surviving = conn.execute(
            "SELECT count(*) FROM chunks WHERE tenant_id = %s AND source = %s", (tenant, uri)
        ).fetchone()[0]
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (tenant, uri),
        ).fetchone()[0]
    assert surviving == 0, "the adopted v0.8 rows survived an erasure that reported success"
    assert tombstoned == 1, "no tombstone, so a later build could reintroduce the source"
    assert "NOT erased" not in out


@requires_db
def test_mcp_and_cli_erasure_agree_on_an_adopted_v08_tenant(manager) -> None:
    """The two erasure surfaces must not disagree about the same tenant's data.

    Migration 0008 adopts a v0.8 install's rows IN PLACE, so a tenant can hold data with no
    active generation at all. `sources_for_identifiers` was scoped to the ACTIVE generation of
    recall_chunks_v1, so the MCP `recall_forget` raised NoActiveGeneration and left the rows on
    disk, while `recall forget` erased them. A hosted deployment exposes the MCP surface.
    """
    from recall_mcp.service import forget_memory

    tenant = manager.tenant_id
    uri = "/legacy/mcp-adopted.md"
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for suffix in ("a", "b"):
            conn.execute(
                "INSERT INTO chunks (tenant_id, id, source, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (tenant, f"legacy-{suffix}-{uuid.uuid4().hex[:8]}", uri, "adopted body",
                 "[" + ",".join(["0.0"] * 63) + ",1.0]"),
            )

    store = GenerationStore(TEST_DSN, 64, tenant=tenant)
    try:
        # The tenant genuinely has no active generation; this must resolve, not raise.
        assert store.sources_for_identifiers([uri]) == {uri: [uri]}
        result = forget_memory(store, [uri])
    finally:
        store.close()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        surviving = conn.execute(
            "SELECT count(*) FROM chunks WHERE tenant_id = %s AND source = %s", (tenant, uri)
        ).fetchone()[0]
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (tenant, uri),
        ).fetchone()[0]
    assert surviving == 0, "the MCP erasure surface left adopted v0.8 rows on disk"
    assert tombstoned == 1
    assert result.chunks_removed == 2
    assert uri in result.sources_removed


@requires_db
def test_mcp_forget_during_a_build_is_honoured_by_that_build(manager) -> None:
    """The MCP erasure surface must honour a mid-build erasure, as the CLI does.

    `build()` opens a transaction per manifest entry, so mid-build NOTHING has rows yet for an
    object it has not reached. Resolving erasure identifiers on chunk rows alone therefore
    answered "not found", wrote no tombstone, and let the build index the very content the user
    asked to erase -- while `recall forget`, which consults the manifest, erased it. Two
    erasure surfaces disagreeing on the same request.
    """
    from recall_mcp.service import forget_memory

    embedder = _Embedder(1)
    data = b"---\nstatus: current\n---\nthe user asked for this to be erased"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = manager.create(manifest, _pipeline("model-a"))
    assert manager.get(generation.generation_id).state == "building"

    store = GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id)
    try:
        # No chunk rows exist yet, so this only resolves via the manifest.
        assert store.sources_for_identifiers([uri]) == {uri: [uri]}
        result = forget_memory(store, [uri])
    finally:
        store.close()
    assert result.sources_removed == [uri]

    stats = manager.build(
        generation.generation_id, _reader(manifest, data), embedder, lambda text: [text]
    )
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        surviving = conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, uri),
        ).fetchone()[0]
        tombstoned = conn.execute(
            "SELECT count(*) FROM recall_source_tombstones WHERE tenant_id = %s",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert tombstoned == 1, "the MCP erasure wrote no tombstone, so the build had nothing to honour"
    assert stats.tombstoned_objects == 1
    assert surviving == 0, "the build indexed content the user had asked to erase via MCP"


@requires_db
def test_repeating_an_erasure_does_not_move_when_it_happened(manager) -> None:
    """`erased_at` records WHEN an irreversible action occurred; a repeat must not move it.

    The tombstone was written `ON CONFLICT DO UPDATE SET erased_at = EXCLUDED.erased_at`, so
    re-issuing a forget rewrote the timestamp and the event id. Once the manifest fallback
    made an already-erased source resolve again, any repeat silently drifted the recorded
    moment of a right-to-erasure action forward. The repeat is still recorded, as its own
    audit event.
    """
    embedder = _Embedder(1)
    data = b"---\nstatus: current\n---\nerase me"
    manifest = _manifest(manager.tenant_id, data, version="v1")
    uri = manifest.objects[0].uri
    generation = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), embedder)
    manager.promote(generation, unsafe_development=True)

    first = manager.forget(uri)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        original = conn.execute(
            "SELECT erased_at, event_id FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, uri),
        ).fetchone()
    assert first.chunks_removed > 0

    second = manager.forget(uri)
    assert second.chunks_removed == 0
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        after = conn.execute(
            "SELECT erased_at, event_id FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (manager.tenant_id, uri),
        ).fetchone()
        events = conn.execute(
            "SELECT count(*) FROM recall_audit_events "
            "WHERE tenant_id = %s AND event_type = 'source_forgotten'",
            (manager.tenant_id,),
        ).fetchone()[0]
    assert after == original, "repeating the erasure moved when it was recorded as happening"
    assert events == 2, "the repeat should still be audited, just not restamp the tombstone"


class _AsymmetricEmbedder(_Embedder):
    """Records which encoder the generation builder reached for.

    Indexing writes PASSAGES. A generation built with the query encoder stores vectors from the
    wrong side of an asymmetric model: same width, plausible cosines, quietly worse retrieval,
    and no error anywhere to say so.
    """

    def __init__(self) -> None:
        super().__init__(1)
        self.used: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.used.append("embed")
        return super().embed(texts)

    def embed_query(self, text: str) -> list[float]:
        self.used.append("embed_query")
        return super().embed([text])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.used.append("embed_passages")
        return super().embed(texts)


@requires_db
def test_a_generation_is_built_with_the_passage_encoder(manager) -> None:
    data = b"---\nstatus: current\n---\nan immutable memo indexed as a passage"
    manifest = _manifest(manager.tenant_id, data)
    embedder = _AsymmetricEmbedder()

    _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), embedder)

    assert embedder.used == ["embed_passages"]


# --- reuse must not carry a pre-fix body into a new generation --------------------------------
#
# `_reuse_source` is keyed on (tenant, uri, sha256, pipeline_fingerprint) and returns BEFORE
# `parse_frontmatter` runs. None of those four terms moved when the frontmatter pairing rule
# changed, so a markdown object whose leading `---` stopped being read as a fence would carry
# its truncated chunk set into every future generation. `recall.index`'s own trigger does not
# reach this path: it is a different freshness guard entirely.
#
# These three are pure. The call site is covered by the DB-gated test directly below them: the
# two pre-existing reuse tests do NOT reach it, because neither uses a payload for which
# `_body_rule_changed` is True, so without that test the guard could be deleted outright and the
# whole suite would stay green, database or no database.


def test_a_markdown_object_whose_pairing_changed_refuses_reuse() -> None:
    text = "---\n\n# Release notes\n\nProse the old rule deleted.\n\n---\n\nTail.\n"
    assert _body_rule_changed("text/markdown", text) is True


def test_an_ordinary_markdown_object_still_reuses() -> None:
    # The common case. A spurious True here rebuilds a whole corpus for nothing.
    assert _body_rule_changed("text/markdown", "---\nvalid_until: 2030-01-01\n---\nbody\n") is False
    assert _body_rule_changed("text/markdown", "# Title\n\nbody\n") is False


def test_a_non_markdown_object_is_never_reparsed_so_it_always_reuses() -> None:
    # `parse_frontmatter` is only applied to markdown media types, so a text/plain object
    # carrying the same bytes never had a frontmatter block to lose.
    text = "---\n\n# Release notes\n\nProse.\n\n---\n\nTail.\n"
    assert _body_rule_changed("text/plain", text) is False


@requires_db
def test_a_thematic_break_object_is_never_reused_into_a_new_generation(manager) -> None:
    """The assertion that dies if `_body_rule_changed` is removed from the build loop.

    Reuse is keyed on the object's bytes, and those did not change when the frontmatter pairing
    rule did. Without the guard this object would carry the chunk set built under the old rule,
    with its whole first section missing, into every generation built afterwards.
    """
    data = b"---\n\n# Release notes\n\nProse the old rule deleted.\n\n---\n\nTail.\n"
    manifest = _manifest(manager.tenant_id, data)
    reader = _reader(manifest, data)
    pipeline = _pipeline("model-a")
    first = _ready(manager, manifest, pipeline, reader, _Embedder(1))
    manager.promote(first, unsafe_development=True)
    with manager._connect() as conn, conn.transaction():
        conn.execute(
            "UPDATE recall_chunks_v1 SET metadata = metadata - %s "
            "WHERE tenant_id = %s AND generation_id = %s",
            ("body_rule_version", manager.tenant_id, first),
        )

    must_run = _Embedder(9)
    second = manager.create(manifest, pipeline)
    stats = manager.build(second.generation_id, reader, must_run, lambda text: [text])

    assert stats.reused_objects == 0, "an object whose body moved must not be reused"
    assert must_run.calls == 1, "it must be re-chunked and re-embedded, not copied"

    manager.validate(second.generation_id)
    manager.promote(second.generation_id, unsafe_development=True)

    reused_after_repair = _Embedder(11)
    third = manager.create(manifest, pipeline)
    third_stats = manager.build(third.generation_id, reader, reused_after_repair, lambda text: [text])

    assert third_stats.reused_objects == 1, "the repaired source should reuse on later generations"
    assert reused_after_repair.calls == 0, "reuse should avoid a second repair re-embed"


@requires_db
def test_markdown_build_strips_the_derived_block_before_chunking(manager) -> None:
    """`recall/generations.py:501` is the site the design doc calls "not optional": `recall
    index` is refused under RECALL_ENV=production (`recall/cli.py:1209`), so this S3 build path
    is the one build path that runs in production. Two markdown objects carry the SAME prose;
    one also carries a well-formed derived block. If the markdown branch regressed to
    `body = text`, or reordered to chunk before stripping, the blocked object's chunk would carry
    the fence lines and this comparison would diverge.
    """
    tenant = manager.tenant_id
    frontmatter = "---\nstatus: current\n---\n"
    prose = (
        "# Retention\n\n"
        "The 90 day retention window is fully documented in this paragraph, long enough to "
        "stand as a chunk of its own.\n"
    )
    block = render_derived_block(
        [
            DerivedEntry(
                head="status",
                value="adopted",
                proposal="a" * 64,
                provider="recall.deterministic@session3-v1",
                reviewer="giulio",
                at="2026-08-11T09:14:22Z",
            )
        ]
    )
    plain_data = (frontmatter + prose).encode("utf-8")
    blocked_data = (frontmatter + prose + "\n" + block).encode("utf-8")
    plain_uri = f"s3://approved/corpora/{tenant}/plain.md"
    blocked_uri = f"s3://approved/corpora/{tenant}/blocked.md"
    manifest = IndexManifestV1(
        tenant,
        "corpus-v1",
        (
            ManifestObjectV1(plain_uri, "v1", "text/markdown", len(plain_data),
                             hashlib.sha256(plain_data).hexdigest()),
            ManifestObjectV1(blocked_uri, "v1", "text/markdown", len(blocked_data),
                             hashlib.sha256(blocked_data).hexdigest()),
        ),
    )
    reader = S3ObjectReader(
        _S3({
            ("approved", f"corpora/{tenant}/plain.md", "v1"): plain_data,
            ("approved", f"corpora/{tenant}/blocked.md", "v1"): blocked_data,
        }),
        S3Allowlist.parse("approved/corpora/"),
    )
    embedder = _Embedder(1)

    generation_id = _ready(manager, manifest, _pipeline("model-a"), reader, embedder)

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT source_uri, chunk_ordinal, text FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s ORDER BY source_uri, chunk_ordinal",
            (tenant, generation_id),
        ).fetchall()

    plain_texts = [text for uri, _, text in rows if uri == plain_uri]
    blocked_texts = [text for uri, _, text in rows if uri == blocked_uri]
    assert plain_texts, "the plain source produced no chunks"
    assert blocked_texts == plain_texts


@requires_db
def test_iter_chunks_wraps_its_server_side_cursor_in_a_transaction(manager) -> None:
    """`GenerationStore.iter_chunks` overrides the base method and dropped its transaction.

    `PgVectorStore.iter_chunks` opens `conn.transaction()` around the named cursor, and says why
    four lines above it: "a server-side cursor is transaction-scoped, and under autocommit each
    FETCH would otherwise land in its own transaction, where the cursor no longer exists". The
    override kept the cursor and lost the transaction, so on the autocommit connections this store
    uses, `DECLARE CURSOR` fails outright.

    Measured on VPS2 before this test existed: **five of the ten MCP tools** were unusable under
    `RECALL_ENV=production` — `recall_reasoning_query`, `_projection`, `_proposals`, `_audit` and
    `recall_rewrite_plan` — every one of them reporting `DECLARE CURSOR can only be used in
    transaction blocks`. The same tools worked under `development`, because that path builds a
    `PgVectorStore` instead. Production is the ONLY mode that selects `GenerationStore`, so the
    defect was invisible to any test that did not ask for it.

    Sibling of `test_cosines_for_matches_query_dense_on_the_generation_store`: both are overrides
    that inherited a behaviour and silently discarded part of it.
    """
    data = b"alpha generation text"
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(
        manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1)
    )
    manager.promote(generation, unsafe_development=True)

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        # Consuming the iterator is the point: the failure is raised by DECLARE, and a lazy
        # generator that is never advanced would pass while the tool it backs still breaks.
        chunks = list(store.iter_chunks())

        # ⚠️ AND the transaction must be RELEASED, which is a separate claim from "one was open".
        # `conn.execute("BEGIN")` in place of `conn.transaction()` satisfies DECLARE and returns
        # every row, so the assertions above alone cannot tell a fix from a leak — it just leaves
        # the connection `idle in transaction` forever. This repository has already lost days to
        # exactly that shape: a 5-day idle-in-transaction backend pinned the xmin horizon of a
        # 205 GB production database and blocked every vacuum on it.
        assert store._direct.info.transaction_status == TransactionStatus.IDLE, (
            "iter_chunks left the connection in a transaction after the iterator was exhausted"
        )

    assert chunks, "iter_chunks yielded nothing for an active generation"
    assert all(c.text for c in chunks)
