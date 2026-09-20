"""AML Multimodal wire, persistence, and retrieval contracts."""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from starlette.testclient import TestClient

from recall_aml.app import create_app
import recall_aml.models as model_module
import recall_aml.multimodal as multimodal_module
from recall_aml.config import HostedSettings
from recall.types import ScoredChunk
from recall_aml.identity import tenant_for
from recall_aml.models import (
    AddRequest,
    AddResponse,
    Message,
    SearchRequest,
    SearchResponse,
)
from recall_aml.multimodal import (
    MULTIMODAL_EMBEDDING_MODEL,
    VoyageMultimodalEmbedder,
    media_tenant,
    multimodal_tenant,
    prepare_messages,
)
from recall_aml.readiness import verify_model_readiness
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService
from recall_aml.variants import variant


def _png_data_url(payload: bytes = b"pixel") -> str:
    data = b"\x89PNG\r\n\x1a\n" + payload
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class _CapturingService:
    variant_name = "MM1_preserve"

    def __init__(self) -> None:
        self.add_request = None
        self.search_request = None

    async def add(self, request):
        self.add_request = request
        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            raw_count=1,
            compiled_count=0,
        )

    async def search(self, request):
        self.search_request = request
        return SearchResponse(data=[])


def _client(service: _CapturingService) -> TestClient:
    settings = HostedSettings(
        database_url="postgresql://unused",
        api_key="secret",
        git_commit="f" * 40,
        variant_name="A0_raw",
    )
    return TestClient(create_app(settings, service))


def test_add_accepts_and_preserves_ordered_text_image_text_parts() -> None:
    """Pre-fix b95d34bf returns 422 because Message.content accepts only str.

    The intended assertion observes the real HTTP Add boundary. A parser that flattens, sorts, or
    drops a part makes the equality assertion fail after transport succeeds.
    """
    service = _CapturingService()
    parts = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": _png_data_url()}},
        {"type": "text", "text": "after"},
    ]

    response = _client(service).post(
        "/v1/add",
        headers={"Authorization": "Bearer secret"},
        json={
            "request_id": "mm-add-1",
            "messages": [{"role": "user", "content": parts}],
            "user_id": "mm-user",
            "session_id": "mm-session",
        },
    )

    assert response.status_code == 200
    assert service.add_request.messages[0].model_dump(mode="json")["content"] == parts


def test_search_accepts_and_preserves_an_ordered_multimodal_query() -> None:
    """Pre-fix b95d34bf returns 422 because SearchRequest.query accepts only str.

    This is the actual Search boundary. The field equality proves that the query image and its
    surrounding text reach retrieval in source order.
    """
    service = _CapturingService()
    parts = [
        {"type": "text", "text": "Which item is shown?"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"query")}},
    ]

    response = _client(service).post(
        "/v1/search",
        headers={"Authorization": "Bearer secret"},
        json={"query": parts, "user_id": "mm-user", "top_k": 100},
    )

    assert response.status_code == 200
    assert service.search_request.model_dump(mode="json")["query"] == parts


def test_http_body_limit_allows_a_valid_image_larger_than_the_old_two_megabyte_cap() -> None:
    """The real Add boundary must not reject an AML-valid image at the obsolete body cap.

    Red proof: setting ``recall_aml.app.MAX_BODY_BYTES`` back to 2,000,000 returned HTTP 422 at
    this assertion instead of the required 200.
    """
    service = _CapturingService()
    parts = [{"type": "image_url", "image_url": {"url": _png_data_url(b"x" * 2_100_000)}}]

    response = _client(service).post(
        "/v1/add",
        headers={"Authorization": "Bearer secret"},
        json={
            "request_id": "mm-large-body",
            "messages": [{"role": "user", "content": parts}],
            "user_id": "mm-user",
            "session_id": "mm-session",
        },
    )

    assert response.status_code == 200


def test_media_validation_rejects_remote_urls_and_mime_signature_mismatches() -> None:
    """Unsupported media must fail at the strict public model boundary.

    Red proof: bypassing the signature predicate in ``decode_image_data_url`` made the mismatched
    JPEG declaration reach the exact ``DID NOT RAISE ValidationError`` assertion.
    """
    base = {
        "request_id": "mm-invalid",
        "messages": [{"role": "user", "content": []}],
        "user_id": "mm-user",
        "session_id": "mm-session",
    }
    remote = [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]
    mismatch = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(b"\x89PNG\r\n\x1a\nwrong").decode("ascii")
            },
        }
    ]

    with pytest.raises(ValidationError, match="inline JPEG, PNG, or WebP"):
        AddRequest.model_validate({**base, "messages": [{"role": "user", "content": remote}]})
    with pytest.raises(ValidationError, match="do not match declared media type"):
        AddRequest.model_validate({**base, "messages": [{"role": "user", "content": mismatch}]})


def test_jpeg_png_and_webp_inline_images_are_accepted() -> None:
    """The public model must accept every AML-declared image media type.

    Red proof: removing ``webp`` from ``_DATA_URL`` made the WebP iteration fail validation.
    """
    values = [
        _png_data_url(),
        "data:image/jpeg;base64,"
        + base64.b64encode(b"\xff\xd8\xffjpeg").decode("ascii"),
        "data:image/webp;base64,"
        + base64.b64encode(b"RIFF\x04\x00\x00\x00WEBPdata").decode("ascii"),
    ]

    for index, value in enumerate(values):
        request = AddRequest.model_validate(
            {
                "request_id": f"media-{index}",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": value}}],
                    }
                ],
                "user_id": "mm-user",
                "session_id": "mm-session",
            }
        )
        assert request.messages[0].content[0].image_url.url == value


def test_decoded_per_image_and_aggregate_add_limits_are_enforced(monkeypatch) -> None:
    """Decoded bytes, not Base64 character counts, own both published AML limits.

    Red proof: bypassing both decoded-size predicates made the first ``pytest.raises`` report
    ``DID NOT RAISE ValidationError``.
    """
    payload = _png_data_url(b"12345678")
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": payload}}],
    }
    base = {
        "request_id": "mm-limits",
        "user_id": "mm-user",
        "session_id": "mm-session",
    }
    monkeypatch.setattr(model_module, "MAX_IMAGE_BYTES", 15)
    with pytest.raises(ValidationError, match="decoded image exceeds 15 bytes"):
        AddRequest.model_validate({**base, "messages": [message]})

    monkeypatch.setattr(model_module, "MAX_IMAGE_BYTES", 100)
    monkeypatch.setattr(model_module, "MAX_REQUEST_MEDIA_BYTES", 31)
    with pytest.raises(ValidationError, match="aggregate bytes"):
        AddRequest.model_validate({**base, "messages": [message, message]})


def test_preparation_content_addresses_media_once_and_keeps_base64_out_of_text_chunks() -> None:
    """A repeated source image must have one media object and reference-only text metadata.

    Red proof: keying the media map by SHA256 plus part position produced two media objects and
    failed ``len(prepared.media_chunks) == 1``.
    """
    image = _png_data_url(b"same-image")
    parts = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "between"},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    messages = [Message.model_validate({"role": "user", "content": parts})]

    prepared = prepare_messages(
        messages,
        request_id="dedup-request",
        session_id="dedup-session",
        source="aml://session/dedup",
    )

    assert len(prepared.media_chunks) == 1
    assert len(prepared.vector_chunks) == 1
    assert len(prepared.primary_chunks) == 1
    persisted_primary = json.dumps(
        {
            "text": prepared.primary_chunks[0].text,
            "metadata": prepared.primary_chunks[0].metadata,
        }
    )
    assert "base64" not in persisted_primary
    assert image not in persisted_primary
    manifest = prepared.primary_chunks[0].metadata["multimodal_manifest"]
    assert [entry["type"] for entry in manifest] == [
        "text",
        "image_ref",
        "text",
        "image_ref",
    ]
    assert manifest[1]["media_id"] == manifest[3]["media_id"]


def test_voyage_adapter_preserves_order_and_declares_document_and_query_modes() -> None:
    """The provider adapter must send the exact ordered evidence under the fixed Voyage model.

    Red proof: changing the document call's ``input_type`` to ``query`` failed the first exact
    provider-call assertion below.
    """
    calls = []

    class Client:
        def multimodal_embed(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in inputs])

    image = _png_data_url(b"voyage")
    value = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "after"},
            ],
        }
    ).content
    adapter = VoyageMultimodalEmbedder("unused", client=Client())
    expected = {
        "content": [
            {"type": "text", "text": "before"},
            {"type": "image_base64", "image_base64": image},
            {"type": "text", "text": "after"},
        ]
    }

    adapter.embed_documents([expected])
    adapter.embed_query(value)

    assert calls[0] == (
        [expected],
        {"model": MULTIMODAL_EMBEDDING_MODEL, "input_type": "document", "truncation": True},
    )
    assert calls[1] == (
        [expected],
        {"model": MULTIMODAL_EMBEDDING_MODEL, "input_type": "query", "truncation": True},
    )


class _TextEmbedder:
    dim = 3

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Reranker:
    def rerank(self, query, hits):
        return list(hits)


class _MMEmbedder:
    dim = 3
    profile = "test-mm"

    def __init__(self):
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, inputs):
        self.document_inputs.extend(inputs)
        return [[0.0, 1.0, 0.0] for _ in inputs]

    def embed_query(self, value):
        self.query_inputs.append(value)
        return [0.0, 1.0, 0.0]


class _Store:
    generation_id = "mm-test"

    def __init__(self, repository, tenant):
        self.repository = repository
        self.tenant = tenant

    def query_dense(self, vector, k):
        return [
            ScoredChunk(chunk, 0.9 - index / 1000)
            for index, chunk in enumerate(self.repository.chunks[self.tenant].values())
        ][:k]

    def query_sparse(self, query, k, vec=None):
        terms = set(query.casefold().split())
        chunks = list(self.repository.chunks[self.tenant].values())
        chunks.sort(
            key=lambda chunk: (
                -len(terms & set(chunk.text.casefold().split())),
                chunk.id,
            )
        )
        return [ScoredChunk(chunk, 0.8 - index / 1000) for index, chunk in enumerate(chunks)][:k]

    def explicit_superseded_chunk_ids(self):
        return frozenset()

    def chunks_by_ids(self, ids):
        return {
            chunk_id: self.repository.chunks[self.tenant][chunk_id]
            for chunk_id in ids
            if chunk_id in self.repository.chunks[self.tenant]
        }

    def iter_chunks(self, batch_size=256):
        yield from self.repository.chunks[self.tenant].values()

    def authored_graph_relation_count(self):
        return 0


class _Repository:
    def __init__(self):
        self.chunks = defaultdict(dict)
        self.receipts = {}
        self.locks = defaultdict(threading.Lock)
        self.multimodal_vectors = {}

    def tenant_store(self, tenant):
        return _Store(self, tenant)

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

    def prior_records(self, tenant, source):
        return []

    def persist(self, tenant, chunks):
        for chunk in chunks:
            self.chunks[tenant][chunk.id] = chunk
        return len(chunks)

    def persist_media(self, tenant, chunks):
        return self.persist(media_tenant(tenant), chunks)

    def persist_multimodal(self, tenant, chunks, vectors):
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[multimodal_tenant(tenant)][chunk.id] = chunk
            self.multimodal_vectors[(tenant, chunk.id)] = vector
        return len(chunks)

    def corpus_status(self, tenant):
        return {"generation_id": "mm-test", "corpus_sha256": "f" * 64}

    def health(self):
        return {"database_ready": True, "generation_id": "mm-test"}

    def delete_tenant(self, tenant):
        count = len(self.chunks[tenant])
        for target in (tenant, media_tenant(tenant), multimodal_tenant(tenant)):
            self.chunks.pop(target, None)
        return count


def _service(variant_name):
    repository = _Repository()
    mm_embedder = _MMEmbedder() if variant_name == "MM2_dual" else None
    service = HostedService(
        repository,
        None,
        HostedRetriever(_TextEmbedder(), _Reranker()),
        behavior=variant(variant_name),
        multimodal_embedder=mm_embedder,
    )
    return service, repository, mm_embedder


def _round_trip(variant_name):
    service, repository, mm_embedder = _service(variant_name)
    image = _png_data_url(b"round-trip")
    parts = [
        {"type": "text", "text": "the status panel"},
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "shows a green deployment"},
    ]
    request = AddRequest.model_validate(
        {
            "request_id": "mm-round-trip",
            "messages": [{"role": "user", "content": parts}],
            "user_id": "mm-user",
            "session_id": "mm-session",
        }
    )
    asyncio.run(service.add(request))
    response = asyncio.run(
        service.search(
            SearchRequest.model_validate(
                {"query": "deployment status", "user_id": "mm-user", "top_k": 10}
            )
        )
    )
    return service, repository, mm_embedder, parts, response


def test_mm2_startup_readiness_probes_the_multimodal_provider() -> None:
    """MM2 must fail startup unless its visual retrieval dependency answers correctly.

    Red proof: bypassing the multimodal readiness block omitted ``multimodal_ready`` and failed
    the exact status equality below.
    """
    multimodal = _MMEmbedder()

    status = verify_model_readiness(
        embedder=_TextEmbedder(),
        compiler=None,
        reranker=_Reranker(),
        multimodal_embedder=multimodal,
        behavior=variant("MM2_dual"),
    )

    assert status == {
        "embedder_ready": True,
        "compiler_ready": False,
        "reranker_ready": False,
        "multimodal_ready": True,
    }
    assert len(multimodal.query_inputs) == 1


def test_mm1_round_trips_exact_ordered_source_evidence() -> None:
    """MM1 must reconstruct the original text-image-text sequence, byte for byte.

    Red proof: iterating the persisted manifest in reverse failed the first content element,
    returning the trailing text instead of the leading text.
    """
    _, repository, _, parts, response = _round_trip("MM1_preserve")
    tenant = tenant_for("mm-user")

    assert response.model_dump(mode="json")["data"][0]["content"] == parts
    assert len(repository.chunks[media_tenant(tenant)]) == 1
    assert not repository.chunks[multimodal_tenant(tenant)]


def test_mm0_uses_only_supplied_text_and_does_not_retain_image_bytes() -> None:
    """The control arm must remain a genuine text-only storage and retrieval path.

    Red proof: adding image Data URIs to ``content_text`` placed ``data:image`` in the primary
    chunk and failed the exact persisted-corpus assertion below.
    """
    service, repository, _ = _service("MM0_caption")
    image = _png_data_url(b"control-image")
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "mm-control",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "supplied visual summary"},
                                {"type": "image_url", "image_url": {"url": image}},
                            ],
                        }
                    ],
                    "user_id": "mm-control-user",
                    "session_id": "mm-control-session",
                }
            )
        )
    )
    tenant = tenant_for("mm-control-user")
    corpus = json.dumps(
        [
            {"text": chunk.text, "metadata": chunk.metadata}
            for chunk in repository.chunks[tenant].values()
        ]
    )

    assert "supplied visual summary" in corpus
    assert "data:image" not in corpus
    assert not repository.chunks[media_tenant(tenant)]
    assert not repository.chunks[multimodal_tenant(tenant)]

    response = asyncio.run(
        service.search(
            SearchRequest(query="visual summary", user_id="mm-control-user", top_k=10)
        )
    )
    assert "supplied visual summary" in response.data[0].content


@pytest.mark.parametrize("variant_name", ["MM1_preserve", "MM2_dual"])
def test_preservation_variants_keep_text_only_messages_searchable_and_scalar(
    variant_name,
) -> None:
    """A preservation arm must not lose or reshape an ordinary text-only memory.

    Pre-fix red proof: the MM1 text record used the legacy raw path, so preservation rendering
    found no manifest and ``response.data`` was empty at the exact assertion below.
    """
    service, _, _ = _service(variant_name)
    asyncio.run(
        service.add(
            AddRequest(
                request_id="mm-text-only",
                messages=[Message(role="user", content="plain deployment note")],
                user_id="mm-text-user",
                session_id="mm-text-session",
            )
        )
    )

    response = asyncio.run(
        service.search(
            SearchRequest(query="deployment note", user_id="mm-text-user", top_k=10)
        )
    )

    assert response.data[0].content == "plain deployment note"


def test_mm2_uses_voyage_inputs_and_deletes_all_three_tenant_namespaces() -> None:
    """MM2 must execute both visual embedding boundaries and erase every sidecar.

    Red proof: replacing document embedding with zero vectors left ``document_inputs`` empty and
    failed its exact call-count assertion. The green path then exercises deletion assertions.
    """
    service, repository, mm_embedder, parts, response = _round_trip("MM2_dual")
    tenant = tenant_for("mm-user")

    assert response.model_dump(mode="json")["data"][0]["content"] == parts
    assert mm_embedder is not None
    assert len(mm_embedder.document_inputs) == 1
    assert len(mm_embedder.query_inputs) == 1
    assert repository.chunks[multimodal_tenant(tenant)]

    deleted = asyncio.run(service.delete_user("mm-user"))

    assert deleted == 1
    assert not repository.chunks[tenant]
    assert not repository.chunks[media_tenant(tenant)]
    assert not repository.chunks[multimodal_tenant(tenant)]


def test_mm1_keeps_tenants_isolated_and_caps_returned_decoded_media(monkeypatch) -> None:
    """Search may neither cross user boundaries nor exceed the decoded response budget.

    Red proof: reversing the response-budget comparison returned one item above the monkeypatched
    one-byte cap and failed ``capped.data == []``. The same run kept the outsider assertion green.
    """
    service, _, _ = _service("MM1_preserve")
    image = _png_data_url(b"budget")
    request = AddRequest.model_validate(
        {
            "request_id": "mm-isolation",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "private panel"},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ],
            "user_id": "owner",
            "session_id": "owner-session",
        }
    )
    asyncio.run(service.add(request))

    outsider = asyncio.run(
        service.search(SearchRequest(query="private panel", user_id="outsider", top_k=10))
    )
    assert outsider.data == []

    monkeypatch.setattr(multimodal_module, "MAX_RESPONSE_MEDIA_BYTES", 1)
    capped = asyncio.run(
        service.search(SearchRequest(query="private panel", user_id="owner", top_k=10))
    )
    assert capped.data == []
