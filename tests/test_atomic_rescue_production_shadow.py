from __future__ import annotations

import gc

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from recall import atomic_rescue
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueLineageError,
    AtomicRescueSelection,
    atomic_rescue_expectation_parity,
    atomic_rescue_reference_parity,
    clear_atomic_rescue_artifact_cache,
    insert_atomic_rescue_dense,
    load_atomic_rescue_artifact,
    resolve_atomic_rescue_manifest,
    select_atomic_rescue,
    write_atomic_rescue_artifact,
)
from recall.calibration import Calibration
from recall.embeddings import EmbeddingProfile
from recall.retriever import RetrievalCandidateTrace
from recall.scope import Scope
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import Chunk, RetrievalResult, ScoredChunk
from recall_mcp import service
from recall_mcp.settings import (
    ENVIRONMENT_SCHEMA,
    Settings,
    activate_runtime_settings,
    reset_runtime_settings,
)
from tests.test_source_conditioning import _trusted_result
from scripts.build_atomic_fact_production_artifact import _parents
from scripts.run_atomic_fact_active_live import _candidate_changes
from scripts.run_atomic_fact_production_shadow import build_expected_payload
from scripts.run_live_tty_graph_precision import _command


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _artifact(tmp_path, *, generation_id: str = "generation-new"):
    matrix = np.asarray(
        [
            _unit([1.0, 0.0]),
            _unit([0.99, 0.1]),
            _unit([0.8, 0.6]),
            _unit([0.0, 1.0]),
        ],
        dtype=np.float32,
    )
    views = [
        {"chunk_id": "atomic-a", "source": "a.md", "parent_ordinal": 0, "view_ordinal": 0},
        {"chunk_id": "atomic-a", "source": "a.md", "parent_ordinal": 0, "view_ordinal": 1},
        {"chunk_id": "atomic-b", "source": "b.md", "parent_ordinal": 2, "view_ordinal": 0},
        {"chunk_id": "atomic-c", "source": "c.md", "parent_ordinal": 1, "view_ordinal": 0},
    ]
    return write_atomic_rescue_artifact(
        tmp_path / f"artifact-{generation_id}",
        matrix=matrix,
        views=views,
        generation_id=generation_id,
        calibration_id="calibration",
        pipeline_fingerprint="pipeline",
        corpus_fingerprint="corpus-new",
        embedding_profile="test-profile",
        embedding_fingerprint=EmbeddingProfile(
            profile_id="test-profile",
            model_name="test-model",
            artifact_digest="test-digest",
            dimension=2,
            query_mode="embed",
            passage_mode="embed",
        ).fingerprint(),
        ordinary_chunk_count=6,
        source_commit="0123456789abcdef",
    )


def _dense() -> list[ScoredChunk]:
    ids = ["atomic-a", "dense-2", "dense-3", "dense-4", "dense-5", "atomic-c"]
    return [
        ScoredChunk(Chunk(value, f"stored/{value}", value, {"file": value, "ord": index}), 1.0)
        for index, value in enumerate(ids)
    ]


class _Embedder:
    dim = 2
    name = "test-profile"
    profile = EmbeddingProfile(
        profile_id="test-profile",
        model_name="test-model",
        artifact_digest="test-digest",
        dimension=2,
        query_mode="embed",
        passage_mode="embed",
    )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _ActiveStore:
    tenant = "memory"
    generation_id = "generation-new"

    def __init__(self) -> None:
        self.loaded: list[tuple[str, float]] = []

    def generation_binding(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant,
            "generation_id": self.generation_id,
            "pipeline_fingerprint": "pipeline",
            "corpus_fingerprint": "corpus-new",
        }

    def query_dense(self, vector, k, source=None, scope=None):
        del vector, source, scope
        return [
            ScoredChunk(Chunk(f"dense-{index}", f"s{index}.md", str(index), {}), 1.0)
            for index in range(1, 8)
        ][:k]

    def query_sparse(self, query, k, source=None, vec=None, scope=None):
        del query, k, source, vec, scope
        return []

    def scored_chunk_by_id(self, chunk_id: str, score: float) -> ScoredChunk:
        self.loaded.append((chunk_id, score))
        return ScoredChunk(Chunk(chunk_id, "rescued.md", "rescued", {}), score)

    def newest_indexed_at(self):
        return None

    def supersession(self):
        return {}, frozenset()


def test_artifact_builder_binds_each_parent_identity_to_one_chunk() -> None:
    """Artifact metadata cannot silently bind one source ordinal to two chunk identifiers.

    Red proof receipt ``atomic-shadow-parent-map-01`` targets ``_parents``. Removing its duplicate
    identity refusal makes the final assertion fail because the second chunk silently wins.
    """

    class Store:
        def iter_chunks(self):
            return iter(
                [
                    Chunk("first", "stored/a", "first text", {"file": "a.md", "ord": 0}),
                    Chunk("second", "stored/a", "second text", {"file": "a.md", "ord": 0}),
                ]
            )

    with pytest.raises(RuntimeError, match="duplicate pinned parent identity"):
        _parents(Store())


def test_artifact_validation_and_exact_masked_selection(tmp_path) -> None:
    """The selector excludes dense parents and returns the exact best remaining parent.

    Red proof receipt ``atomic-shadow-selector-01`` targets ``select_atomic_rescue``. Replacing
    the exclusion mask update with a no-op returns ``atomic-a`` and fails the intended winner
    assertion below. Restoring the mask makes this node green.
    """

    clear_atomic_rescue_artifact_cache()
    artifact = load_atomic_rescue_artifact(_artifact(tmp_path))
    selected = select_atomic_rescue(artifact, [1.0, 0.0], _dense())

    assert selected.chunk_id == "atomic-b"
    assert selected.source == "b.md"
    assert selected.parent_ordinal == 2
    assert selected.score == pytest.approx(0.8, abs=1e-6)
    assert artifact.view_count == 4
    assert artifact.parent_count == 3


def test_artifact_rejects_one_chunk_id_with_conflicting_parent_identities(tmp_path) -> None:
    """Every chunk id must map to exactly one source and parent ordinal."""
    manifest = _artifact(tmp_path)
    metadata_path = manifest.with_name("views.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["views"][1]["source"] = "b.md"
    metadata["views"][1]["parent_ordinal"] = 2
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8", newline="\n")
    clear_atomic_rescue_artifact_cache()

    with pytest.raises(AtomicRescueArtifactError, match="parent identities"):
        load_atomic_rescue_artifact(manifest)


def test_active_insertion_moves_winner_to_rank_six_and_deduplicates(tmp_path) -> None:
    """Active rescue fetches one winner, moves it to rank six, and keeps later order.

    Red proof receipt ``atomic-active-insertion-01`` targets ``insert_atomic_rescue_dense``.
    Appending the winner instead of inserting it at index five makes the rank assertion fail;
    retaining its old dense occurrence makes the uniqueness assertion fail.
    """

    clear_atomic_rescue_artifact_cache()
    artifact = load_atomic_rescue_artifact(_artifact(tmp_path))
    dense = _dense() + [
        ScoredChunk(Chunk("atomic-b", "stored/atomic-b", "late", {}), 0.7),
        ScoredChunk(Chunk("dense-8", "stored/dense-8", "last", {}), 0.6),
    ]
    calls: list[tuple[str, float]] = []

    def load(chunk_id: str, score: float) -> ScoredChunk:
        calls.append((chunk_id, score))
        return ScoredChunk(Chunk(chunk_id, "stored/atomic-b", "rescued", {}), score)

    result = insert_atomic_rescue_dense(artifact, [1.0, 0.0], dense, load)

    assert calls == [("atomic-b", pytest.approx(0.8, abs=1e-6))]
    assert [hit.chunk.id for hit in result] == [
        "atomic-a",
        "dense-2",
        "dense-3",
        "dense-4",
        "dense-5",
        "atomic-b",
        "atomic-c",
        "dense-8",
    ]
    assert result[5].chunk.text == "rescued"
    with pytest.raises(atomic_rescue.AtomicRescueSelectionError, match="unavailable"):
        insert_atomic_rescue_dense(artifact, [1.0, 0.0], dense, lambda _id, _score: None)


def test_active_manifest_resolution_is_generation_bound(tmp_path) -> None:
    """Both platform separator forms are rejected.

    Red proof receipt ``vps2-linux-preflight-2026-09-16`` failed on ``a\\b`` before the
    resolver rejected separators explicitly instead of relying on host ``Path`` semantics.
    """

    root = tmp_path / "registry"
    expected = (root / "generation-new" / "manifest.json").resolve()

    assert resolve_atomic_rescue_manifest(root, "generation-new") == expected
    fingerprint = "a" * 64
    scoped = (root / "aml_scope" / "generation-new" / fingerprint / "manifest.json").resolve()
    assert (
        resolve_atomic_rescue_manifest(
            root,
            "generation-new",
            scope_id="aml_scope",
            corpus_fingerprint=fingerprint,
        )
        == scoped
    )
    for invalid in ("", ".", "..", "../generation-new", "a/b", "a\\b"):
        with pytest.raises(AtomicRescueArtifactError):
            resolve_atomic_rescue_manifest(root, invalid)
        with pytest.raises(AtomicRescueArtifactError):
            resolve_atomic_rescue_manifest(
                root,
                "generation-new",
                scope_id=invalid,
                corpus_fingerprint=fingerprint,
            )
    for invalid in ("", "A" * 64, "a" * 63, "g" * 64):
        with pytest.raises(AtomicRescueArtifactError):
            resolve_atomic_rescue_manifest(
                root,
                "generation-new",
                scope_id="aml_scope",
                corpus_fingerprint=invalid,
            )
    with pytest.raises(AtomicRescueArtifactError):
        resolve_atomic_rescue_manifest(root, "generation-new", scope_id="aml_scope")
    with pytest.raises(AtomicRescueArtifactError):
        resolve_atomic_rescue_manifest(root, "generation-new", corpus_fingerprint=fingerprint)


def test_live_trust_explanation_uses_candidate_identity_not_score(tmp_path) -> None:
    """Reason-only trust changes require a changed rank-six parent receipt.

    Red proof receipt ``atomic-active-trust-explanation-01`` targets ``_candidate_changes``.
    Comparing scores instead of source and ordinal marks the first row changed and fails below.
    """

    rows = []
    for index in range(96):
        dense = [{"source": f"dense-{slot}.md", "ordinal": slot, "score": 0.5} for slot in range(6)]
        active = [*dense[:5], dict(dense[5])]
        active[5]["score"] = 0.8
        if index == 1:
            active[5] = {"source": "atomic.md", "ordinal": 9, "score": 0.8}
        rows.append(
            {
                "query_id": f"query-{index}",
                "dense": dense,
                "dense5_atomic1": active,
            }
        )
    path = tmp_path / "offline-private.json"
    path.write_text(
        json.dumps({"generation_id": "generation-new", "rows": rows}), encoding="utf-8"
    )

    changes = _candidate_changes(path, "generation-new")

    assert changes["query-0"] is False
    assert changes["query-1"] is True
    assert sum(changes.values()) == 1


def test_active_mode_reaches_real_fusion_and_trust(monkeypatch, tmp_path) -> None:
    """Unscoped active retrieval mutates the real dense leg before the trust pass.

    Red proof receipt ``atomic-active-trust-path-01`` targets the active branch in
    ``recall.trust._trusted_search``. Disabling that branch leaves ``rescued`` absent from the
    trusted result and makes the artifact-load assertion fail.
    """

    from recall import trust

    lineage: list[dict[str, object]] = []

    class Artifact:
        def assert_lineage(self, **kwargs):
            lineage.append(kwargs)

    inserted: list[list[str]] = []

    def insert(artifact, query_vector, dense, loader):
        del artifact, query_vector
        inserted.append([hit.chunk.id for hit in dense])
        return [*dense[:5], loader("rescued", 0.75), *dense[5:]]

    monkeypatch.setattr(trust, "load_atomic_rescue_artifact", lambda path: Artifact())
    monkeypatch.setattr(trust, "insert_atomic_rescue_dense", insert)
    store = _ActiveStore()
    result = trusted_search(
        store,
        _Embedder(),
        "query",
        k=6,
        candidate_k=7,
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
        },
    )

    assert inserted == [[f"dense-{index}" for index in range(1, 8)]]
    assert store.loaded == [("rescued", 0.75)]
    assert [hit.chunk.id for hit in result.hits] == [
        "dense-1",
        "dense-2",
        "dense-3",
        "dense-4",
        "dense-5",
        "rescued",
    ]
    assert lineage[0]["generation_id"] == "generation-new"


def test_active_mode_bypasses_scoped_queries_without_loading_artifact(monkeypatch, tmp_path) -> None:
    """A source-scoped query remains byte-for-byte on the baseline retrieval path."""

    from recall import trust

    monkeypatch.setattr(
        trust,
        "load_atomic_rescue_artifact",
        lambda path: pytest.fail(f"scoped query loaded active artifact: {path}"),
    )
    result = trusted_search(
        _ActiveStore(),
        _Embedder(),
        "query",
        k=2,
        candidate_k=7,
        source="s1.md",
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
        },
    )

    assert [hit.chunk.id for hit in result.hits] == ["dense-1", "dense-2"]


@pytest.mark.parametrize(
    "scope",
    [
        Scope(folder="notes"),
        Scope(facet="decision"),
        Scope(source_prefixes=("sentiment-agent",)),
    ],
)
def test_active_mode_bypasses_structural_scopes(monkeypatch, tmp_path, scope) -> None:
    """Folder, facet, and prefix scopes cannot enter the unscoped active selector."""

    from recall import trust

    monkeypatch.setattr(
        trust,
        "load_atomic_rescue_artifact",
        lambda path: pytest.fail(f"scoped query loaded active artifact: {path}"),
    )
    trusted_search(
        _ActiveStore(),
        _Embedder(),
        "query",
        k=2,
        candidate_k=7,
        scope=scope,
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
        },
    )


def test_selector_uses_matrix_kernel_and_matches_full_sort(tmp_path) -> None:
    """The production kernel remains the confirmed matrix operation and matches a full sort.

    Red proof receipt ``atomic-shadow-runtime-kernel-01`` targets ``select_atomic_rescue``.
    Replacing matrix multiplication with einsum makes the traced matrix call count zero.
    Returning the excluded first view makes reference parity fail.
    """

    clear_atomic_rescue_artifact_cache()
    artifact = load_atomic_rescue_artifact(_artifact(tmp_path))
    calls: list[tuple[int, int]] = []

    class RecordingMatrix(np.ndarray):
        def __matmul__(self, other):
            calls.append((self.shape[0], other.shape[0]))
            return super().__matmul__(other)

    instrumented = replace(artifact, matrix=artifact.matrix.view(RecordingMatrix))
    selected = select_atomic_rescue(instrumented, [1.0, 0.0], _dense())
    parity = atomic_rescue_reference_parity(
        instrumented, [1.0, 0.0], _dense(), selected
    )

    assert calls == [(4, 2), (4, 2)]
    assert parity == (True, True)
    excluded = AtomicRescueSelection("atomic-a", "a.md", 0, 0, 1.0)
    assert atomic_rescue_reference_parity(
        artifact, [1.0, 0.0], _dense(), excluded
    ) == (False, False)


def test_concurrent_selectors_do_not_overlap_matrix_work(tmp_path) -> None:
    """One process serializes selector CPU work while preserving concurrent request safety.

    Red proof receipt ``atomic-shadow-selector-serialization-01`` targets
    ``_SELECTION_LOCK``. Removing the lock lets the sleeping matrix operation overlap and makes
    ``maximum_active`` exceed one.
    """

    clear_atomic_rescue_artifact_cache()
    artifact = load_atomic_rescue_artifact(_artifact(tmp_path))
    counter_lock = threading.Lock()
    active = maximum_active = 0

    class SleepingMatrix(np.ndarray):
        def __matmul__(self, other):
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            try:
                return super().__matmul__(other)
            finally:
                with counter_lock:
                    active -= 1

    instrumented = replace(artifact, matrix=artifact.matrix.view(SleepingMatrix))
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = list(
            pool.map(
                lambda _index: select_atomic_rescue(
                    instrumented, [1.0, 0.0], _dense()
                ),
                range(16),
            )
        )

    assert maximum_active == 1
    assert {item.chunk_id for item in selected} == {"atomic-b"}


def test_artifact_digest_lineage_and_single_flight_loading(tmp_path, monkeypatch) -> None:
    """One process loads once, then refuses changed bytes and stale serving lineage.

    Red proof receipt ``atomic-shadow-artifact-01`` targets the cache lock and digest check.
    Moving the uncached load outside the lock makes the concurrent count exceed one; bypassing
    the matrix digest check makes the corrupted artifact assertion fail because no error occurs.
    """

    path = _artifact(tmp_path)
    clear_atomic_rescue_artifact_cache()
    original = atomic_rescue._load_atomic_rescue_artifact
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(atomic_rescue, "_load_atomic_rescue_artifact", counted)
    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts = list(pool.map(lambda _index: load_atomic_rescue_artifact(path), range(16)))
    assert calls == 1
    assert all(value is artifacts[0] for value in artifacts)

    with pytest.raises(AtomicRescueLineageError, match="generation_id"):
        artifacts[0].assert_compatible(
            result=_trusted_result().__class__(
                **{
                    **_trusted_result().__dict__,
                    "generation_id": "rolled-over-generation",
                }
            ),
            embedder=_Embedder(),
        )

    # Artifacts are memory-mapped and immutable by contract. Windows refuses to write a mapped file,
    # so release every mapping before simulating on-disk corruption; the reload below must still
    # refuse the changed bytes.
    artifacts.clear()
    clear_atomic_rescue_artifact_cache()
    gc.collect()
    matrix_path = path.parent / "matrix.npy"
    matrix_path.write_bytes(matrix_path.read_bytes() + b"corrupt")
    clear_atomic_rescue_artifact_cache()
    with pytest.raises(AtomicRescueArtifactError, match="digest mismatch"):
        load_atomic_rescue_artifact(path)


def test_artifact_refuses_a_different_embedding_fingerprint(tmp_path) -> None:
    """A matching profile id and dimension are insufficient lineage evidence."""
    artifact = load_atomic_rescue_artifact(_artifact(tmp_path))
    changed = _Embedder()
    changed.profile = replace(changed.profile, artifact_digest="different-digest")

    with pytest.raises(AtomicRescueLineageError, match="embedding_fingerprint"):
        artifact.assert_lineage(
            generation_id="generation-new",
            calibration_id="calibration",
            pipeline_fingerprint="pipeline",
            corpus_fingerprint="corpus-new",
            embedder=changed,
        )


def test_atomic_shadow_settings_and_sampling_are_off_by_default(tmp_path) -> None:
    """Shadow tracing stays explicit while active mode requires a generation registry.

    Red proof receipt ``atomic-shadow-settings-01`` targets ``_validate_runtime_options`` and
    ``_atomic_rescue_shadow_sampled``. Removing the active-root guard makes the missing-root
    refusal fail; treating active as shadow makes the default sampling assertions fail.
    """

    schema_names = {spec.name for spec in ENVIRONMENT_SCHEMA}
    assert {
        "RECALL_ATOMIC_RESCUE_MODE",
        "RECALL_ATOMIC_RESCUE_ARTIFACT",
        "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE",
    } <= schema_names
    assert service._atomic_rescue_shadow_sampled("query", {}) is False
    assert service._atomic_rescue_shadow_sampled(
        "query",
        {
            "RECALL_ATOMIC_RESCUE_MODE": "shadow",
            "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE": "1",
        },
    ) is True
    with pytest.raises(ValueError, match="RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"):
        Settings.from_env({"RECALL_ATOMIC_RESCUE_MODE": "active"})
    active = Settings.from_env(
        {
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
        }
    )
    assert active.values["RECALL_ATOMIC_RESCUE_MODE"] == "active"
    with pytest.raises(ValueError, match="RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE"):
        Settings.from_env({"RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE": "1.01"})


def test_private_expected_candidate_is_reported_only_as_parity_booleans(tmp_path) -> None:
    """Live validation compares identity and score without exposing either expected value.

    Red proof receipt ``atomic-shadow-private-parity-01`` targets
    ``atomic_rescue_expectation_parity``. Returning true for every score makes the second parity
    assertion fail because the frozen expected score deliberately differs.
    """

    query = "Which fact should atomic rescue recover?"
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": {
                    hashlib.sha256(query.encode("utf-8")).hexdigest(): {
                        "source": "b.md",
                        "parent_ordinal": 2,
                        "score": 0.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    clear_atomic_rescue_artifact_cache()
    parity = atomic_rescue_expectation_parity(
        path,
        query=query,
        selection=AtomicRescueSelection("private-id", "b.md", 2, 0, 0.8),
    )

    assert parity == (True, False)
    assert "private-id" not in json.dumps(parity)


def test_live_receipt_uses_the_appended_atomic_candidate() -> None:
    """The live parity receipt binds to slot six, not a preserved dense prefix candidate.

    Red proof receipt ``atomic-shadow-expected-slot-01`` targets ``build_expected_payload``.
    Selecting index four instead of five fails the intended source assertion below.
    """

    pool = {"queries": [{"id": f"q{index}", "query": f"query {index}"} for index in range(96)]}
    rows = []
    for index in range(96):
        rows.append(
            {
                "query_id": f"q{index}",
                "dense5_atomic1": [
                    {"source": f"dense-{slot}.md", "ordinal": slot, "score": 1.0 - slot / 10}
                    for slot in range(5)
                ]
                + [{"source": f"atomic-{index}.md", "ordinal": 9, "score": 0.42}],
            }
        )

    receipt = build_expected_payload(pool, {"rows": rows})
    query_digest = hashlib.sha256(b"query 0").hexdigest()
    selected = receipt["rows"][query_digest]

    assert selected == {"source": "atomic-0.md", "parent_ordinal": 9, "score": 0.42}


def test_tty_command_enables_atomic_shadow_only_when_requested(monkeypatch) -> None:
    """The live runner controls rescue while ordinary MCP launches remain unchanged.

    Red proof receipt ``atomic-shadow-command-01`` targets ``_command``. Omitting the atomic
    environment block makes the shadow command assertions fail while the ordinary command stays
    unchanged.

    Red proof receipt ``atomic-active-control-off-01`` targets the explicit off arm. Before the
    fix it inherited an active production environment, so the new off assertion failed.
    """

    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/srv/recall")
    ordinary = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
    )[-1]
    shadow = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10, "generation",
        atomic_rescue_mode="shadow",
        atomic_rescue_artifact="/private/manifest.json",
        atomic_rescue_sample_rate=1.0,
        atomic_rescue_expected="/private/expected.json",
    )[-1]
    active = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
        atomic_rescue_mode="active",
        atomic_rescue_artifact_root="/srv/atomic-registry",
    )[-1]
    explicit_off = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
        atomic_rescue_mode="off",
    )[-1]

    assert "RECALL_ATOMIC_RESCUE_MODE" not in ordinary
    assert "OPENBLAS_NUM_THREADS=1" not in ordinary
    assert "OPENBLAS_NUM_THREADS=1" in shadow
    assert "RECALL_ATOMIC_RESCUE_MODE=shadow" in shadow
    assert "RECALL_ATOMIC_RESCUE_ARTIFACT=/private/manifest.json" in shadow
    assert "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE=1.000000" in shadow
    assert "RECALL_BENCHMARK_ATOMIC_RESCUE_EXPECTED=/private/expected.json" in shadow
    assert "RECALL_ATOMIC_RESCUE_MODE=active" in active
    assert "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT=/srv/atomic-registry" in active
    assert "RECALL_ATOMIC_RESCUE_MODE=off" in explicit_off
    assert "OPENBLAS_NUM_THREADS=1" not in explicit_off


def test_shadow_reuses_main_trace_preserves_response_and_redacts_candidate(
    tmp_path, monkeypatch
) -> None:
    """The shadow adds no retrieval and cannot alter or expose the served response.

    Red proof receipt ``atomic-shadow-reuse-01`` targets the shadow consumer in
    ``_execute_reasoning_query``. Mutating it to overwrite ``executed.result`` with the atomic
    candidate changes the public evidence and fails the parity assertion below. Calling a second
    retrieval fails the injected assertion before a response is returned. Red proof receipt
    ``atomic-shadow-remediation-immutability-01`` forces the benchmark immutability field false;
    the final benchmark assertion fails.
    """

    path = _artifact(tmp_path)
    clear_atomic_rescue_artifact_cache()
    baseline = _trusted_result()
    dense = _dense()
    raw = RetrievalResult(baseline.query, dense, False, baseline.staleness)
    candidate_trace = (
        RetrievalCandidateTrace(raw, tuple(dense), tuple(), tuple()),
        baseline,
        Calibration("calibration", 0.5, 0.05),
    )
    capture_flags: list[bool] = []

    def fake_retrieve(*_args, **kwargs):
        capture = bool(kwargs.get("capture_candidate_trace"))
        capture_flags.append(capture)
        return SimpleNamespace(
            result=baseline,
            query_vector=[1.0, 0.0],
            profile=service.FAST_PROFILE,
            candidate_trace=candidate_trace if capture else None,
        )

    class Store:
        tenant = "memory"
        generation_id = "generation-new"

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(
        service,
        "_retrieval_leg_benchmark_audit_payload",
        lambda *_args, **_kwargs: pytest.fail("atomic shadow repeated database retrieval"),
    )

    off_settings = Settings.from_env({"RECALL_ATOMIC_RESCUE_MODE": "off"})
    token = activate_runtime_settings(off_settings)
    try:
        off = service.reasoning_query(
            Store(), _Embedder(), baseline.query, mode="retrieval_only", graph_expansion="off",
            policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)
    off_values = off.diagnostics.performance["values"]
    assert "atomic_rescue_shadow" not in off_values

    shadow_settings = Settings.from_env(
        {
            "RECALL_ATOMIC_RESCUE_MODE": "shadow",
            "RECALL_ATOMIC_RESCUE_ARTIFACT": str(path),
            "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE": "1",
        }
    )
    token = activate_runtime_settings(shadow_settings)
    try:
        shadow = service.reasoning_query(
            Store(), _Embedder(), baseline.query, mode="retrieval_only", graph_expansion="off",
            policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)

    assert capture_flags == [False, True]
    assert shadow.trusted_evidence == off.trusted_evidence
    values = shadow.diagnostics.performance["values"]
    payload = values["atomic_rescue_shadow"]
    assert payload["status"] == "ok"
    assert payload["selected_parent_equal_dense_rank_six"] is False
    assert payload["blas_threads"] is None
    assert "benchmark_public_result_unchanged" not in payload
    assert "benchmark_reference_identity_parity" not in payload
    serialized = json.dumps(payload)
    assert "atomic-b" not in serialized
    assert "b.md" not in serialized
    spans = shadow.diagnostics.performance["spans_ms"]
    assert spans["atomic_rescue_shadow_ms"] >= payload["selector_ms"]

    selected = select_atomic_rescue(
        load_atomic_rescue_artifact(path), [1.0, 0.0], dense
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": {
                    hashlib.sha256(baseline.query.encode("utf-8")).hexdigest(): {
                        "source": selected.source,
                        "parent_ordinal": selected.parent_ordinal,
                        "score": selected.score,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark_settings = Settings.from_env(
        {
            "RECALL_ATOMIC_RESCUE_MODE": "shadow",
            "RECALL_ATOMIC_RESCUE_ARTIFACT": str(path),
            "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE": "1",
            "RECALL_BENCHMARK_PIN": "1",
            "RECALL_PINNED_GENERATION_ID": "generation-new",
            "RECALL_BENCHMARK_ATOMIC_RESCUE_EXPECTED": str(expected_path),
        }
    )
    token = activate_runtime_settings(benchmark_settings)
    try:
        benchmark = service.reasoning_query(
            Store(), _Embedder(), baseline.query, mode="retrieval_only", graph_expansion="off",
            policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)
    benchmark_payload = benchmark.diagnostics.performance["values"]["atomic_rescue_shadow"]
    assert benchmark_payload["benchmark_public_result_unchanged"] is True
    assert benchmark_payload["benchmark_identity_parity"] is True
    assert benchmark_payload["benchmark_score_parity"] is True
    assert benchmark_payload["benchmark_reference_identity_parity"] is True
    assert benchmark_payload["benchmark_reference_score_parity"] is True


def test_source_scoped_shadow_skips_without_trace_or_artifact_load(tmp_path, monkeypatch) -> None:
    """A source scoped request remains baseline only until scope preserving rescue is measured.

    Red proof receipt ``atomic-shadow-source-scope-01`` targets ``capture_atomic_trace``. Removing
    the ``source is None`` guard makes the capture flag true and fails the intended assertion.
    """

    baseline = _trusted_result()
    capture_flags: list[bool] = []

    def fake_retrieve(*_args, **kwargs):
        capture_flags.append(bool(kwargs.get("capture_candidate_trace")))
        return SimpleNamespace(
            result=baseline,
            query_vector=[1.0, 0.0],
            profile=service.FAST_PROFILE,
            candidate_trace=None,
        )

    class Store:
        tenant = "memory"
        generation_id = "generation-new"

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(
        service,
        "load_atomic_rescue_artifact",
        lambda *_args, **_kwargs: pytest.fail("source scoped request opened the artifact"),
    )
    settings = Settings.from_env(
        {
            "RECALL_ATOMIC_RESCUE_MODE": "shadow",
            "RECALL_ATOMIC_RESCUE_ARTIFACT": str(tmp_path / "absent.json"),
            "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE": "1",
        }
    )
    token = activate_runtime_settings(settings)
    try:
        response = service.reasoning_query(
            Store(), _Embedder(), baseline.query, source="one.md", mode="retrieval_only",
            graph_expansion="off", policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)

    assert capture_flags == [False]
    payload = response.diagnostics.performance["values"]["atomic_rescue_shadow"]
    assert payload == {"status": "skipped", "reason_code": "source_scope"}
