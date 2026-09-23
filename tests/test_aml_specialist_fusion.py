"""Behavior proofs for the routed specialist corpus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
from pathlib import Path
import threading

from starlette.testclient import TestClient

from recall.types import Chunk, ScoredChunk
from recall_aml.app import create_app
from recall_aml.config import HostedSettings
from recall_aml.identity import specialist_tenant, tenant_for
from recall_aml.models import AddRequest, SearchRequest
from recall_aml.multimodal import media_tenant, multimodal_tenant
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService
from recall_aml.specialists import route_query
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import variant
from scripts.aml_release_manifest import (
    BOUND_REPOSITORY_ARTIFACTS,
    SPECIALIST_PREREGISTRATION,
    build_manifest,
)


class _Embedder:
    dim = 3

    def __init__(self, name: str) -> None:
        self.name = name
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _VectorEmbedder(_Embedder):
    def __init__(self, name: str, vector: list[float]) -> None:
        super().__init__(name)
        self.vector = vector
        self.passages: list[str] = []

    def embed_passages(self, texts):
        materialized = list(texts)
        self.passages.extend(materialized)
        return [list(self.vector) for _ in materialized]


class _UpsertStore:
    def __init__(self, root: "_BaseStore", tenant: str) -> None:
        self.root = root
        self.tenant = tenant

    def upsert(self, chunks, vectors):
        self.root.upserts[self.tenant] = (list(chunks), [list(vector) for vector in vectors])
        return len(chunks)


class _BaseStore:
    def __init__(self) -> None:
        self.upserts = {}

    def for_tenant(self, tenant):
        return _UpsertStore(self, tenant)


class _Reranker:
    def rerank(self, query, hits):
        return hits


class _MultimodalEmbedder:
    dim = 3
    profile = "test-multimodal"

    def __init__(self) -> None:
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, inputs):
        self.document_inputs.extend(inputs)
        return [[0.0, 1.0, 0.0] for _ in inputs]

    def embed_query(self, value):
        self.query_inputs.append(value)
        return [0.0, 1.0, 0.0]


class _Store:
    def __init__(self, repository: "_Repository", tenant: str) -> None:
        self.repository = repository
        self.tenant = tenant

    def query_dense_exact(self, vector, k):
        self.repository.dense_searches.append(self.tenant)
        chunks = sorted(self.repository.chunks[self.tenant].values(), key=lambda item: item.id)
        return [ScoredChunk(chunk, 0.9 - index / 1000) for index, chunk in enumerate(chunks[:k])]

    def query_dense(self, vector, k):
        self.repository.dense_searches.append(self.tenant)
        chunks = sorted(self.repository.chunks[self.tenant].values(), key=lambda item: item.id)
        return [ScoredChunk(chunk, 0.8 - index / 1000) for index, chunk in enumerate(chunks[:k])]

    def iter_chunks(self, batch_size=256):
        yield from sorted(self.repository.chunks[self.tenant].values(), key=lambda item: item.id)

    def explicit_superseded_chunk_ids(self):
        return frozenset()

    def chunks_by_ids(self, ids):
        return {
            chunk_id: self.repository.chunks[self.tenant][chunk_id]
            for chunk_id in ids
            if chunk_id in self.repository.chunks[self.tenant]
        }


class _Repository:
    def __init__(self, context_profile: str) -> None:
        self.context_profile = context_profile
        self.chunks = defaultdict(dict)
        self.receipts = {}
        self.locks = defaultdict(threading.Lock)
        self.dense_searches: list[str] = []
        self.multimodal_vectors = {}

    def tenant_store(self, tenant):
        return _Store(self, tenant)

    def specialist_store(self, tenant, profile):
        assert profile == self.context_profile
        return self.tenant_store(specialist_tenant(tenant, profile))

    def media_store(self, tenant):
        return self.tenant_store(media_tenant(tenant))

    def multimodal_store(self, tenant):
        return self.tenant_store(multimodal_tenant(tenant))

    def acquire_request_lock(self, tenant, request_id):
        lock = self.locks[(tenant, request_id)]
        lock.acquire()
        return lock

    def release_request_lock(self, handle):
        handle.release()

    def get_receipt(self, tenant, request_id, fingerprint):
        return self.receipts.get((tenant, request_id, fingerprint))

    def record_receipt(self, tenant, request_id, fingerprint, result):
        self.receipts[(tenant, request_id, fingerprint)] = result

    def prior_records(self, tenant, source, *, graph_sidecar=False):
        return []

    def persist(self, tenant, chunks):
        for chunk in chunks:
            self.chunks[tenant][chunk.id] = chunk
        return len(chunks)

    def persist_specialist(self, tenant, profile, chunks):
        target = specialist_tenant(tenant, profile)
        return self.persist(target, chunks)

    def persist_media(self, tenant, chunks):
        return self.persist(media_tenant(tenant), chunks)

    def persist_multimodal(self, tenant, chunks, vectors):
        target = multimodal_tenant(tenant)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[target][chunk.id] = chunk
            self.multimodal_vectors[chunk.id] = list(vector)
        return len(chunks)

    def verify_sparse_coverage(self, tenant):
        return {"sparse_ready": True}

    def corpus_status(self, tenant):
        return {"generation_id": "specialist-test", "corpus_sha256": "f" * 64}

    def health(self):
        return {"database_ready": True, "generation_id": "specialist-test"}

    def delete_tenant(self, tenant):
        targets = (
            tenant,
            specialist_tenant(tenant, self.context_profile),
            media_tenant(tenant),
            multimodal_tenant(tenant),
        )
        count = sum(len(self.chunks[target]) for target in targets)
        for target in targets:
            self.chunks.pop(target, None)
        return count


def _service(variant_name: str = "C7_routed_specialists"):
    behavior = variant(variant_name)
    code = _Embedder("code")
    context = _Embedder("context")
    multimodal = _MultimodalEmbedder()
    repository = _Repository(behavior.context_embedding_profile)
    service = HostedService(
        repository,
        object() if behavior.compiler else None,
        HostedRetriever(code, _Reranker()),
        behavior=behavior,
        multimodal_embedder=multimodal,
        specialist_retrievers={
            behavior.context_embedding_profile: HostedRetriever(context, _Reranker())
        },
    )
    return service, repository, code, context, multimodal


def test_router_never_compares_heterogeneous_embedding_scores() -> None:
    """The deterministic router must select one semantic space before retrieval.

    Red proof uses a deliberate mutation that returns ``context`` for code signals. The exact
    assertion for the pytest query then fails, proving this test observes the protected Code4
    routing boundary rather than an incidental response field.
    """
    assert route_query("Fix parser.py because pytest reports a traceback") == "code"
    assert route_query("What did we decide during yesterday's meeting?") == "context"
    assert route_query("What does this screenshot show?") == "multimodal"


def test_add_reuses_exact_chunk_identity_and_metadata_in_code_and_context_indexes() -> None:
    """A text Add must persist one logical chunk set into isolated Code4 and Context4 stores.

    Red proof uses a deliberate mutation that drops ``persist_specialist`` from the text Add
    path. The context id equality assertion fails with an empty set.
    """
    service, repository, _, _, _ = _service()
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "shared-text",
                    "user_id": "specialist-user",
                    "session_id": "specialist-session",
                    "messages": [{"role": "user", "content": "Fix the parser regression."}],
                }
            )
        )
    )

    tenant = tenant_for("specialist-user")
    context_tenant = specialist_tenant(tenant, "voyage-context-4-v1")
    assert set(repository.chunks[tenant]) == set(repository.chunks[context_tenant])
    assert repository.chunks[tenant] == repository.chunks[context_tenant]
    assert tenant != context_tenant


def test_repository_uses_the_specialist_embedder_only_inside_its_derived_tenant() -> None:
    """The Context4 namespace must receive Context4 vectors for the shared chunks.

    Red proof uses a deliberate mutation that embeds specialist chunks with ``self._embedder``.
    The context vector assertion fails with the primary Code4 vector.
    """
    base = _BaseStore()
    code = _VectorEmbedder("code", [1.0, 0.0, 0.0])
    context = _VectorEmbedder("context", [0.0, 1.0, 0.0])
    repository = PgHostedRepository(
        base,
        code,
        specialist_embedders={"voyage-context-4-v1": context},
    )
    chunk = Chunk(
        id="shared-id",
        source="aml://session/shared",
        text="one logical memory",
        metadata={"record_type": "raw", "shared": True},
    )

    repository.persist("logical-tenant", [chunk])
    repository.persist_specialist(
        "logical-tenant", "voyage-context-4-v1", [chunk]
    )

    context_tenant = specialist_tenant("logical-tenant", "voyage-context-4-v1")
    assert base.upserts["logical-tenant"] == ([chunk], [[1.0, 0.0, 0.0]])
    context_chunks, context_vectors = base.upserts[context_tenant]
    assert context_vectors == [[0.0, 1.0, 0.0]]
    assert context_chunks[0].id == chunk.id
    assert context_chunks[0].text == chunk.text
    assert context_chunks[0].metadata["embedding_profile"] == "voyage-context-4-v1"
    assert code.passages == [chunk.text]
    assert context.passages == [chunk.text]


def test_multimodal_add_populates_all_three_indexes_with_primary_mapping() -> None:
    """A visual Add must populate Code4, Context4 and multimodal namespaces.

    Red proof uses a deliberate mutation that skips ``persist_multimodal``. The multimodal
    mapping assertion fails because the visual namespace is empty.
    """
    service, repository, _, _, multimodal = _service()
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "shared-image",
                    "user_id": "visual-user",
                    "session_id": "visual-session",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "deployment panel"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            "data:image/png;base64,"
                                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
                                            "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                                        )
                                    },
                                },
                            ],
                        }
                    ],
                }
            )
        )
    )

    tenant = tenant_for("visual-user")
    context_tenant = specialist_tenant(tenant, "voyage-context-4-v1")
    visual_tenant = multimodal_tenant(tenant)
    assert repository.chunks[tenant] == repository.chunks[context_tenant]
    assert repository.chunks[visual_tenant]
    assert {
        chunk.metadata["primary_id"] for chunk in repository.chunks[visual_tenant].values()
    }.issubset(repository.chunks[tenant])
    assert len(multimodal.document_inputs) == 1


def test_search_routes_code_and_context_to_separate_physical_indexes() -> None:
    """Code and conversational queries must reach only their selected semantic namespace.

    Red proof uses a deliberate mutation that leaves ``store`` bound to the primary tenant on
    the context route. The final dense tenant assertion then reports the primary tenant twice.
    """
    service, repository, code, context, multimodal = _service()
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "route-text",
                    "user_id": "route-user",
                    "session_id": "route-session",
                    "messages": [{"role": "user", "content": "The team agreed on blue."}],
                }
            )
        )
    )
    tenant = tenant_for("route-user")
    repository.dense_searches.clear()

    code_response = asyncio.run(
        service.search(
            SearchRequest.model_validate(
                {"query": "Fix parser.py and run pytest", "user_id": "route-user"}
            )
        )
    )
    context_response = asyncio.run(
        service.search(
            SearchRequest.model_validate(
                {
                    "query": "What did we agree during yesterday's meeting?",
                    "user_id": "route-user",
                }
            )
        )
    )

    assert code_response.specialist_route == "code"
    assert context_response.specialist_route == "context"
    assert repository.dense_searches == [
        tenant,
        specialist_tenant(tenant, "voyage-context-4-v1"),
    ]
    assert code.queries == ["Fix parser.py and run pytest"]
    assert context.queries == ["What did we agree during yesterday's meeting?"]
    assert multimodal.query_inputs == []


def test_c8_context_search_does_not_apply_code4_window_ties_to_compiled_records() -> None:
    """Context4 records without a Code4 segment must remain searchable in C8.

    Red proof: passing C8's stable Code4 tie-breaker into Context4 makes
    ``rank_bm25_chunks`` reject this otherwise valid compiled record with the public
    ``stable Code4 ordering requires an integer segment`` 422.
    """
    service, repository, _, _, _ = _service("C8_routed_specialists_grounded_graph")
    tenant = tenant_for("c8-context-user")
    context_tenant = specialist_tenant(tenant, "voyage-context-4-v1")
    repository.chunks[context_tenant]["compiled-context"] = Chunk(
        id="compiled-context",
        source="aml://session/c8-context",
        text="The team agreed yesterday that Context4 uses the amber release marker.",
        metadata={
            "record_type": "compiled",
            "kind": "architectural decision",
            "source_session_id": "c8-context-session",
        },
    )

    response = asyncio.run(
        service.search(
            SearchRequest.model_validate(
                {
                    "query": "What did the team agree yesterday about the release marker?",
                    "user_id": "c8-context-user",
                }
            )
        )
    )

    assert response.specialist_route == "context"
    assert response.specialist_embedding_profile == "voyage-context-4-v1"
    assert [item.id for item in response.data] == ["compiled-context"]


def test_search_headers_expose_the_selected_specialist_route_and_profile() -> None:
    """Public Search diagnostics must make the selected embedding space auditable.

    Red proof receipt ``aml-specialist-route-headers-01`` targets the Search response header
    construction in ``recall_aml.app``. Before the repair, both header lookups raised ``KeyError``
    even though the internal response object carried the correct route and profile.
    """
    service, _, _, _, _ = _service()
    client = TestClient(
        create_app(
            HostedSettings(
                "postgresql://unused",
                "secret",
                "route-header-commit",
                variant_name="C7_routed_specialists",
            ),
            service,
        )
    )
    headers = {"X-Api-Key": "secret"}
    added = client.post(
        "/v1/add",
        headers=headers,
        json={
            "request_id": "route-header-add",
            "user_id": "route-header-user",
            "session_id": "route-header-session",
            "messages": [{"role": "user", "content": "Fix parser.py and run pytest."}],
        },
    )
    searched = client.post(
        "/v1/search",
        headers=headers,
        json={
            "query": "Fix parser.py and run pytest",
            "user_id": "route-header-user",
            "top_k": 10,
        },
    )

    assert added.status_code == 200
    assert searched.status_code == 200
    assert searched.headers["X-Recall-Specialist-Route"] == "code"
    assert searched.headers["X-Recall-Specialist-Embedding-Profile"] == "voyage-code-4-v1"
    assert list(searched.json()) == ["data"]


def test_c7_vps2_setup_uses_a_distinct_store_and_generation() -> None:
    """C7 must deploy without reusing C6 storage or generation identity.

    Red proof: this exact test fails against implementation commit ``3cc8f9a1`` because the C7
    launcher case and specialist checkout allowlist are absent.
    """
    script = (
        Path(__file__).parents[1] / "scripts" / "aml_experience_vps2_setup.sh"
    ).read_text(encoding="utf-8")

    assert "C7_routed_specialists)" in script
    assert 'readonly table="recall_aml_routed_specialists_chunks"' in script
    assert 'readonly generation="aml-routed-specialists-v1"' in script
    assert "/home/sentiment/recall-repos/aml-specialist-fusion-*" in script


def test_c8_binds_c7_routing_to_the_grounded_graph_sidecar() -> None:
    """The official-smoke candidate must not silently drop either C7 or graph behavior.

    Red proof: before C8 is registered, ``variant`` rejects this exact public configuration.
    """
    candidate = variant("C8_routed_specialists_grounded_graph")

    assert candidate.context_specialist is True
    assert candidate.graph_sidecar is True
    assert candidate.anchor_compiler is True
    assert candidate.anchor_compiler_version == 3


def test_c7_release_manifest_binds_every_specialist_implementation(tmp_path: Path) -> None:
    """The immutable receipt must bind routing, storage, multimodal code, and the preregistration.

    Red proof: this exact test fails against implementation commit ``3cc8f9a1`` because C7 uses
    the generic preregistration and omits all three specialist artifacts.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    required = {
        *BOUND_REPOSITORY_ARTIFACTS.values(),
        SPECIALIST_PREREGISTRATION,
        Path("recall_aml/code4.py"),
        Path("recall/store.py"),
        Path("recall_aml/retrieval.py"),
        Path("recall_aml/service.py"),
        Path("recall_aml/specialists.py"),
        Path("recall_aml/storage.py"),
        Path("recall_aml/multimodal.py"),
        Path("recall_aml/embedding_lock.py"),
        Path("scripts/aml_c7_qualification.py"),
    }
    for relative in required:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"artifact:{relative.as_posix()}".encode())
    wheel = repo / "dist/recall_rag.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"specialist wheel")

    manifest = build_manifest(
        repo_root=repo,
        wheel_path=wheel,
        commit="e" * 40,
        variant_name="C7_routed_specialists",
    )

    assert manifest["variant"]["context_specialist"] is True
    assert manifest["variant"]["context_embedding_profile"] == "voyage-context-4-v1"
    assert manifest["artifacts"]["preregistration"]["path"] == str(
        SPECIALIST_PREREGISTRATION
    )
    for artifact in (
        "specialist_qualification_source",
        "specialist_router_source",
        "specialist_storage_source",
        "multimodal_source",
    ):
        assert artifact in manifest["artifacts"]
        assert manifest["artifacts"][artifact]["sha256"] == hashlib.sha256(
            (repo / manifest["artifacts"][artifact]["path"]).read_bytes()
        ).hexdigest()


def test_a_visual_word_in_a_text_query_still_returns_text_memories() -> None:
    """A text query the router sends to the multimodal route must still see text-only memories.

    Found live on 2026-09-23: in C7 and C8 a text-only Add is stored as ordinary text windows with
    no ``multimodal_manifest``, and ``render_preserved`` skipped every hit without one. A query
    such as "create an encrypted container image" routed to ``multimodal`` because of the word
    "image" and returned an empty evidence list from a corpus holding the answer.

    Red proof: run against the pre-fix ``recall_aml/multimodal.py`` at master ``0365d30d``, it fails
    on ``assert response.data`` with ``data=[]`` and ``specialist_route='multimodal'``.
    """
    # C7 and C8 share the defect (both set multimodal_preserve and context_specialist); C7 is used
    # because this fixture cannot run C8's Add-time compiler and graph sidecar.
    service, _, _, _, _ = _service("C7_routed_specialists")
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "text-image",
                    "user_id": "visual-word-user",
                    "session_id": "registry-session",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Encrypt the container image with an RSA key pair before pushing it.",
                        }
                    ],
                }
            )
        )
    )

    response = asyncio.run(
        service.search(
            SearchRequest.model_validate(
                {
                    "query": "How do I create an encrypted image?",
                    "user_id": "visual-word-user",
                }
            )
        )
    )

    assert response.specialist_route == "multimodal"
    assert response.data, "text memories must survive the multimodal route"
    assert any("RSA key pair" in str(item.content) for item in response.data)
    assert all(isinstance(item.content, str) for item in response.data)
