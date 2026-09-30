"""The core Voyage multimodal embedder respects Voyage's request and image limits.

`recall.multimodal.VoyageMultimodalEmbedder` cut requests by count (32) alone, never checked an
image against Voyage's 16M-pixel and 20 MB per-image limits (while admitting media up to 30 MiB),
and let an input over the 32K-token context be truncated silently. Voyage refuses a request over
1,000 inputs or 320,000 tokens, or holding an image over either image limit, with HTTP 400.

The fake client numbers every vector in the order it was produced, so concatenation order is
checked, not assumed; the fake token counter charges one token per character.

Red proof, run 2026-09-30, each a mutation of `recall/multimodal.py` with this file unchanged;
every test failed in its intended assertion and passed once the line was restored:

* ``test_passages_are_packed_under_the_request_token_cap``: ``or used + tokens > budget``
  deleted from ``plan_voyage_requests``. One request of 360,000 tokens (``[360000]`` against ``[]``).
* ``test_batch_size_still_caps_a_request``: ``max_inputs`` ignored in ``plan_voyage_requests``
  (``limit = VOYAGE_MAX_REQUEST_INPUTS``). ``[70]`` against ``[32, 32, 6]``.
* ``test_a_call_under_every_cap_is_sent_as_one_unchanged_request``: ``plan_voyage_requests``
  closing a request before every input after the first. Six requests where one was expected.
* ``test_a_query_image_over_the_pixel_limit_is_fitted``: ``fit_voyage_image`` returning every
  input unchanged. The sent image kept its 200 pixels against a limit of 100.
* ``test_an_image_over_the_byte_limit_is_fitted``: the byte loop in ``fit_voyage_image`` broken
  after the first pass. The sent image stayed over the byte limit.
* ``test_an_image_within_limits_is_sent_byte_for_byte``: ``fit_voyage_image`` re-encoding every
  image. The sent JPEG differed from the original.
* ``test_a_document_over_the_context_is_sent_whole_and_logged``: the ``_log_input`` call removed
  from ``embed_multimodal``. The log assertion failed.
* ``test_a_query_over_the_context_is_logged``: the ``_log_input`` call removed from
  ``embed_query``. The log assertion failed.
* ``test_an_ordinary_query_never_loads_the_tokenizer``: ``_estimate`` always taking the exact
  path. The loader was called for a five-character query.
* ``test_an_unreadable_image_passes_through_unless_strict``: the ``if strict:`` branch of
  ``fit_voyage_image`` inverted (``if not strict:``). The default call raised
  ``UnidentifiedImageError`` where it must return the input unchanged (added 2026-09-30, with
  ``strict``).
"""

from __future__ import annotations

import base64
from io import BytesIO
import logging
import random
from types import SimpleNamespace

from PIL import Image

import recall.multimodal as multimodal_module
from recall.multimodal import (
    VOYAGE_MAX_INPUT_TOKENS,
    VOYAGE_MAX_REQUEST_TOKENS,
    VOYAGE_REQUEST_TOKEN_HEADROOM,
    MultimodalQuery,
    VoyageMultimodalEmbedder,
)

BUDGET = int(VOYAGE_MAX_REQUEST_TOKENS * VOYAGE_REQUEST_TOKEN_HEADROOM)


def _one_token_per_char(texts: list[str]) -> list[int]:
    return [len(text) for text in texts]


class _Client:
    def __init__(self) -> None:
        self.requests: list[tuple[str, list[dict]]] = []
        self.produced = 0

    def multimodal_embed(self, *, inputs, model, input_type, **_kwargs):
        self.requests.append((input_type, [dict(item) for item in inputs]))
        vectors = []
        for _ in inputs:
            vectors.append([float(self.produced)] + [0.0] * 1023)
            self.produced += 1
        return SimpleNamespace(embeddings=vectors)

    def embed(self, texts, model, **_kwargs):  # pragma: no cover - the protocol's text fallback
        raise AssertionError("the multimodal path must be used")


def _embedder(batch_size: int = 32, counter=_one_token_per_char):
    client = _Client()
    embedder = VoyageMultimodalEmbedder(client=client, batch_size=batch_size, token_counter=counter)
    client.requests.clear()  # drop the construction probe
    client.produced = 0
    return embedder, client


def _documents(client: _Client) -> list[list[dict]]:
    return [inputs for kind, inputs in client.requests if kind == "document"]


def _text(chars: int, tag: str = "t") -> dict:
    return {"content": [{"type": "text", "text": (tag * chars)[:chars]}]}


def _encoded(image: Image.Image, image_format: str, **save: object) -> bytes:
    output = BytesIO()
    image.save(output, format=image_format, **save)
    return output.getvalue()


def _sent_image(client: _Client) -> bytes:
    [(_, [item])] = client.requests
    [part] = [part for part in item["content"] if part["type"] == "image_base64"]
    return base64.b64decode(part["image_base64"].split(",", 1)[1])


def _warned(caplog) -> list[tuple[int, str]]:
    return [
        (record.input_index, record.input_type)
        for record in caplog.records
        if record.getMessage() == "voyage_multimodal_input_over_context"
    ]


def test_passages_are_packed_under_the_request_token_cap() -> None:
    """Invariant: no request's estimated tokens exceed the budget, and order survives."""
    embedder, client = _embedder(batch_size=1_000)
    texts = [(f"{index}" * 30_000)[:30_000] for index in range(12)]  # 360,000 tokens

    vectors = embedder.embed_passages(texts)

    totals = [
        sum(len(item["content"][0]["text"]) for item in request) for request in _documents(client)
    ]
    assert [total for total in totals if total > BUDGET] == []
    assert [item["content"][0]["text"] for request in _documents(client) for item in request] == texts
    assert [vector[0] for vector in vectors] == [float(index) for index in range(12)]


def test_batch_size_still_caps_a_request() -> None:
    """Invariant: ``batch_size`` still bounds the inputs in one request."""
    embedder, client = _embedder(batch_size=32)

    embedder.embed_passages(["short"] * 70)

    assert [len(request) for request in _documents(client)] == [32, 32, 6]


def test_a_call_under_every_cap_is_sent_as_one_unchanged_request() -> None:
    """Invariant: a call under every cap goes out exactly as before, as one request."""
    embedder, client = _embedder()
    small = "data:image/png;base64," + base64.b64encode(
        _encoded(Image.new("RGB", (8, 8), (1, 2, 3)), "PNG")
    ).decode("ascii")
    inputs = [_text(40, f"{index}") for index in range(5)]
    inputs.append({"content": [{"type": "image_base64", "image_base64": small}]})

    embedder.embed_multimodal(inputs)

    assert _documents(client) == [inputs]


def test_a_query_image_over_the_pixel_limit_is_fitted(monkeypatch) -> None:
    """Invariant: an image over the pixel limit is scaled under it before it is sent."""
    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_PIXELS", 100)
    embedder, client = _embedder()
    payload = _encoded(Image.new("RGB", (20, 10), (200, 10, 10)), "PNG")

    embedder.embed_query(MultimodalQuery(image_bytes=payload, media_type="image/png"))

    with Image.open(BytesIO(_sent_image(client))) as sent:
        assert sent.width * sent.height <= 100
        assert sent.width > sent.height  # the aspect ratio survives


def test_an_image_over_the_byte_limit_is_fitted(monkeypatch) -> None:
    """Invariant: an image under the pixel limit but over the byte limit is shrunk until it fits."""
    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_BYTES", 4_000)
    noise = random.Random(0).randbytes(100 * 100 * 3)
    payload = _encoded(Image.frombytes("RGB", (100, 100), noise), "PNG")
    assert len(payload) > 4_000
    embedder, client = _embedder()

    embedder.embed_query(MultimodalQuery(image_bytes=payload, media_type="image/png"))

    assert len(_sent_image(client)) <= 4_000


def test_an_image_within_limits_is_sent_byte_for_byte() -> None:
    """Invariant: an image Voyage already accepts is never re-encoded."""
    payload = _encoded(Image.new("RGB", (64, 48), (30, 60, 90)), "JPEG", quality=95)
    embedder, client = _embedder()

    embedder.embed_query(MultimodalQuery(image_bytes=payload, media_type="image/jpeg"))

    assert _sent_image(client) == payload


def test_a_document_over_the_context_is_sent_whole_and_logged(caplog) -> None:
    """Invariant: an input over the 32K context is sent unchanged, and a warning names it."""
    embedder, client = _embedder()
    big = "b" * (VOYAGE_MAX_INPUT_TOKENS + 8_000)

    with caplog.at_level(logging.WARNING, logger="recall.multimodal"):
        embedder.embed_passages(["small", big, "small"])

    assert [item["content"][0]["text"] for item in _documents(client)[0]] == ["small", big, "small"]
    assert _warned(caplog) == [(1, "document")]


def test_a_query_over_the_context_is_logged(caplog) -> None:
    """Invariant: a query over the 32K context is sent unchanged and logged as a query."""
    embedder, client = _embedder()
    query = "q" * (VOYAGE_MAX_INPUT_TOKENS + 1)

    with caplog.at_level(logging.WARNING, logger="recall.multimodal"):
        embedder.embed_query(query)

    assert client.requests == [("query", [{"content": [{"type": "text", "text": query}]}])]
    assert _warned(caplog) == [(0, "query")]


def test_an_ordinary_query_never_loads_the_tokenizer(monkeypatch) -> None:
    """Invariant: a query the byte bound already clears never loads Voyage's tokenizer."""
    loads: list[str] = []

    def loader(model: str):
        loads.append(model)
        return None

    monkeypatch.setattr("recall.embeddings._voyage_token_counter", loader)
    embedder, client = _embedder(counter=None)

    embedder.embed_query("hello")

    assert loads == []
    assert len(client.requests) == 1


def test_an_unreadable_image_passes_through_unless_strict() -> None:
    """Invariant: by default an image Pillow cannot read is sent unchanged for the provider to
    judge; with ``strict`` it raises, which is how the AML adapter keeps refusing it."""
    import pytest
    from PIL import UnidentifiedImageError

    from recall.multimodal import fit_voyage_image

    corrupt = b"\x89PNG\r\n\x1a\n" + b"not an image body" * 8
    url = "data:image/png;base64," + base64.b64encode(corrupt).decode("ascii")

    assert fit_voyage_image(url) == (url, False)
    with pytest.raises(UnidentifiedImageError):
        fit_voyage_image(url, strict=True)
