"""MM-4 (machine-read image text sidecars) behaviour proofs.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-image-text-sidecar.md, step 1 of "How it
will be measured". The routed-specialist fixture is ``tests/test_aml_specialist_fusion.py``'s, with
an in-memory sidecar namespace and a fake extractor, so nothing here needs a network or a model.

Red proofs, each run on 2026-09-25 against a deliberate mutation of the production line named,
failing at the assertion named, then green after restoring it:

* ``test_build_writes_one_sidecar_per_image_with_its_parent``: ``refs[:MAX_IMAGES_PER_MESSAGE]``
  in ``sidecar_chunks`` cut to ``refs[:1]``; fails at the sidecar-id equality.
* ``test_the_leg_adds_the_parent_never_the_sidecar_itself``: two mutations, each failing at the
  parent-present assertion: ``fuse_hits`` given the sidecar hits instead of their parents, and the
  leg switched off (``if self.image_text_leg`` to ``if False``). The first version of this test used
  a fake store whose text leg returned every chunk, so the parent was always present and the first
  mutation stayed green (``fuse_hits`` folds a hit into a parent it already holds); the store now
  hides image messages from the text leg, which is the case the leg exists for.
* ``test_a_tenant_without_images_is_identical_with_every_mm4_flag``: both guards of the leg
  (``if sidecar_hits`` and ``if parent_hits``) forced true, so an empty leg still re-scores the
  ranking; fails at the dump equality.
* ``test_shown_text_is_appended_once_labelled_to_the_image_message``: the ``shown_items`` call
  removed from ``HostedService.search``; fails at the label-count assertion.
* ``test_the_extractor_is_bounded_and_fails_closed``: ``max_tokens=MAX_OUTPUT_TOKENS`` raised to
  2000; fails at the ``max_tokens`` equality.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from recall.types import ScoredChunk
from recall_aml.image_text import (
    MAX_OUTPUT_TOKENS,
    SHOWN_LABEL,
    ImageTextExtractor,
    image_text_tenant,
    sidecar_id,
)
from recall_aml.models import AddRequest, SearchRequest
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService
from recall_aml.variants import variant
from tests.test_aml_multimodal import _png_data_url
import tests.test_aml_specialist_fusion as fusion

READING = "Text: TOTAL 14.20 EUR\nObjects: 1 paper receipt"


class _SidecarRepository(fusion._Repository):
    def persist_image_text(self, tenant, chunks):
        return self.persist(image_text_tenant(tenant), chunks)

    def query_image_text(self, tenant, query, k):
        terms = set(query.casefold().split())
        chunks = sorted(
            self.chunks[image_text_tenant(tenant)].values(),
            key=lambda chunk: (-len(terms & set(chunk.text.casefold().split())), chunk.id),
        )
        return [ScoredChunk(chunk, 0.9 - index / 100) for index, chunk in enumerate(chunks[:k])]

    def image_text_by_parent(self, tenant, parent_ids):
        output: dict[str, list[str]] = {}
        wanted = set(parent_ids)
        for chunk in sorted(self.chunks[image_text_tenant(tenant)].values(), key=lambda c: c.id):
            parent = str(chunk.metadata.get("primary_id"))
            if parent in wanted:
                output.setdefault(parent, []).append(chunk.text)
        return output


class _Extractor:
    model = "fake-vision"

    def __init__(self):
        self.calls: list[str] = []

    def extract(self, data_url: str) -> str | None:
        self.calls.append(data_url)
        return READING


def _service(extractor=None):
    behavior = variant("C7_routed_specialists")
    repository = _SidecarRepository(behavior.context_embedding_profile)
    service = HostedService(
        repository,
        None,
        HostedRetriever(fusion._Embedder("code"), fusion._Reranker()),
        behavior=behavior,
        multimodal_embedder=fusion._MultimodalEmbedder(),
        specialist_retrievers={
            behavior.context_embedding_profile: HostedRetriever(fusion._Embedder("context"), fusion._Reranker())
        },
        image_text_extractor=extractor,
    )
    return service, repository


def _add(service, request_id, user_id, content, timestamp=1_726_133_200_000):
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": request_id,
                    "messages": [{"role": "user", "content": content, "timestamp": timestamp}],
                }
            )
        )
    )


def _image_message(*payloads: bytes):
    parts = [{"type": "text", "text": "a photo from the mall"}]
    parts += [{"type": "image_url", "image_url": {"url": _png_data_url(p)}} for p in payloads]
    return parts


def _search(service, user_id, query="receipt total euros"):
    return asyncio.run(
        service.search(SearchRequest.model_validate({"query": query, "user_id": user_id}))
    )


def test_build_writes_one_sidecar_per_image_with_its_parent(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_BUILD", "1")
    extractor = _Extractor()
    service, repository = _service(extractor)
    _add(service, "img-add", "mm4-user", _image_message(b"one", b"two"))

    tenant = next(t for t in repository.chunks if t.endswith("_imgtext"))
    sidecars = repository.chunks[tenant]
    parents = {chunk.metadata["primary_id"] for chunk in sidecars.values()}
    assert len(parents) == 1
    parent = parents.pop()
    assert sorted(sidecars) == [sidecar_id(parent, 0), sidecar_id(parent, 1)]
    assert all(chunk.text == READING for chunk in sidecars.values())
    assert len(extractor.calls) == 2


class _TextLegMissesImages(fusion._Store):
    """The text leg never finds an image message (the case the leg exists for); ids still resolve."""

    @staticmethod
    def _visible(chunks):
        return [chunk for chunk in chunks if "multimodal_parent_id" not in chunk.metadata]

    def query_dense_exact(self, vector, k):
        return [hit for hit in super().query_dense_exact(vector, 10_000) if "multimodal_parent_id" not in hit.chunk.metadata][:k]

    def query_dense(self, vector, k):
        return [hit for hit in super().query_dense(vector, 10_000) if "multimodal_parent_id" not in hit.chunk.metadata][:k]

    def iter_chunks(self, batch_size=256):
        yield from self._visible(super().iter_chunks(batch_size))


class _HidingRepository(_SidecarRepository):
    def tenant_store(self, tenant):
        if tenant.endswith("_imgtext") or tenant.endswith("_media") or tenant.endswith("_mm"):
            return fusion._Store(self, tenant)
        return _TextLegMissesImages(self, tenant)


def test_the_leg_adds_the_parent_never_the_sidecar_itself(monkeypatch) -> None:
    """The text leg misses the image message; the sidecar leg must bring in its PARENT."""
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_BUILD", "1")
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_LEG", "1")
    service, repository = _service(_Extractor())
    hiding = _HidingRepository(repository.context_profile)
    service._repository = hiding
    _add(service, "img-leg", "mm4-leg", _image_message(b"receipt"))
    for index in range(3):
        _add(service, f"txt-{index}", "mm4-leg", f"an unrelated note number {index}")
    parent = next(iter(hiding.chunks[next(t for t in hiding.chunks if t.endswith("_imgtext"))].values())).metadata["primary_id"]

    response = _search(service, "mm4-leg")
    ids = [item.id for item in response.data]

    assert response.specialist_route == "code"
    assert parent in ids, "the leg must add the image message the text leg missed"
    assert response.image_text_leg
    assert not any("_imgtext_" in item_id for item_id in ids)
    assert not any(SHOWN_LABEL in str(item.content) for item in response.data)


@pytest.mark.parametrize("flags", [("1", "0", "0"), ("1", "1", "0"), ("1", "1", "1")])
def test_a_tenant_without_images_is_identical_with_every_mm4_flag(monkeypatch, flags) -> None:
    def dump(build: str, leg: str, shown: str) -> list[str]:
        monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_BUILD", build)
        monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_LEG", leg)
        monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_SHOWN", shown)
        service, _ = _service(_Extractor())
        for index in range(4):
            _add(service, f"plain-{index}", "text-only", f"plain text memory about receipts {index}")
        return [item.model_dump_json() for item in _search(service, "text-only").data]

    assert dump(*flags) == dump("1", "0", "0")


def test_shown_text_is_appended_once_labelled_to_the_image_message(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_BUILD", "1")
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_SHOWN", "1")
    service, _ = _service(_Extractor())
    _add(service, "img-shown", "mm4-shown", _image_message(b"receipt"))
    _add(service, "txt-shown", "mm4-shown", "an unrelated note")

    items = _search(service, "mm4-shown").data
    labelled = [item for item in items if SHOWN_LABEL in str(item.content)]

    assert len(labelled) == 1
    assert str(labelled[0].content).count(SHOWN_LABEL) == 1
    assert READING in str(labelled[0].content)
    assert all(SHOWN_LABEL not in str(item.content) for item in items if item is not labelled[0])


def test_the_extractor_is_bounded_and_fails_closed() -> None:
    seen: dict[str, object] = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" Text: OPEN "))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    extractor = ImageTextExtractor(client, "vision-model", provider="OneProvider")

    assert extractor.extract("data:image/png;base64,AAAA") == "Text: OPEN"
    assert seen["max_tokens"] == MAX_OUTPUT_TOKENS == 300
    assert seen["temperature"] == 0
    assert seen["extra_body"] == {
        "reasoning": {"enabled": False},
        "provider": {"order": ["OneProvider"], "allow_fallbacks": False},
    }

    def boom(**kwargs):
        raise RuntimeError("provider down")

    failing = ImageTextExtractor(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom))), "m")
    assert failing.extract("data:image/png;base64,AAAA") is None


def test_building_without_an_extractor_stops_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_IMAGE_TEXT_BUILD", "1")
    with pytest.raises(ValueError, match="image text extractor"):
        _service(None)
