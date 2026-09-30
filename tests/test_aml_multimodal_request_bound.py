"""The AML multimodal embedder keeps every request under Voyage's caps, and logs truncation.

`recall_aml.multimodal.VoyageMultimodalEmbedder.embed_documents` sent every input of an Add in ONE
request. Voyage caps a multimodal request at 1,000 inputs and 320,000 tokens, counting an image as
one token per 560 pixels, so about twelve images at the 16M-pixel fit made the whole Add a 400, on
every resend. Over the 32,000-token per-input context the provider truncates instead (the adapter
sends `truncation=True`), which is logged and otherwise left as it was, by the owner's decision of
2026-09-30.

The fake client records each request and numbers every vector in the order it was produced, so
the concatenation order is checked, not assumed. The fake token counter charges one token per
character, which makes each request's size exact.

Red proof, run 2026-09-30, each a mutation with this file unchanged; every test failed in its
own assertion and passed once the line was restored. The planning and estimating moved to shared
helpers in `recall.multimodal` the same day (`plan_voyage_requests`, `estimate_voyage_tokens`,
`estimate_voyage_inputs`); the first four and the last were re-run against those helpers after the
move, with the same failures, and the two warning mutations still target this adapter:

* ``test_a_large_add_is_split_under_the_request_token_cap``: ``or used + tokens > budget``
  deleted from ``plan_voyage_requests``. One request of 360,000 tokens went out; the per-request
  assertion failed (``[360000]`` against ``[]``).
* ``test_images_count_by_their_pixels``: the image term in ``estimate_voyage_tokens`` set to zero. All
  three images went in one request; the layout assertion failed (``[3]`` against ``[2, 1]``).
* ``test_no_request_carries_more_than_a_thousand_inputs``: the input-count condition deleted from
  ``plan_voyage_requests``. The layout assertion failed (``[1001]`` against ``[1000, 1]``).
* ``test_an_add_that_fits_is_sent_as_one_unchanged_request``: ``plan_voyage_requests`` mutated to close
  a request before every input after the first. The assertion that the provider saw exactly one
  request failed.
* ``test_a_document_over_the_context_is_sent_whole_and_logged``: the ``_warn_if_over_context``
  call removed from ``embed_documents``. The log assertion failed.
* ``test_a_query_over_the_context_is_logged``: the same call removed from ``embed_query``. The
  log assertion failed.
* ``test_an_ordinary_query_never_loads_the_tokenizer``: ``estimate_voyage_inputs`` always taking the exact
  path. The loader was called for a five-character query.
"""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from types import SimpleNamespace

from PIL import Image

# The limits and the planner are shared with the core tenant, so they are patched where they live.
import recall.multimodal as shared_module
from recall.multimodal import (
    VOYAGE_MAX_INPUT_TOKENS as MAX_VOYAGE_INPUT_TOKENS,
    VOYAGE_MAX_REQUEST_TOKENS as MAX_VOYAGE_REQUEST_TOKENS,
    VOYAGE_PIXELS_PER_TOKEN,
    VOYAGE_REQUEST_TOKEN_HEADROOM,
)
from recall_aml.multimodal import VoyageMultimodalEmbedder


def _one_token_per_char(texts: list[str]) -> list[int]:
    return [len(text) for text in texts]


class _Client:
    def __init__(self) -> None:
        self.requests: list[list[dict]] = []
        self.produced = 0

    def multimodal_embed(self, inputs, **_kwargs):
        self.requests.append(list(inputs))
        vectors = []
        for _ in inputs:
            vectors.append([float(self.produced)] + [0.0] * 1023)
            self.produced += 1
        return SimpleNamespace(embeddings=vectors)


def _embedder(client: _Client) -> VoyageMultimodalEmbedder:
    return VoyageMultimodalEmbedder("unused", client=client, token_counter=_one_token_per_char)


def _text(chars: int, tag: str = "t") -> dict:
    return {"content": [{"type": "text", "text": (tag * chars)[:chars]}]}


def _image(width: int, height: int) -> dict:
    output = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return {"content": [{"type": "image_base64", "image_base64": url}]}


def _budget() -> int:
    return int(shared_module.VOYAGE_MAX_REQUEST_TOKENS * VOYAGE_REQUEST_TOKEN_HEADROOM)


def _assert_order(vectors: list[list[float]], count: int) -> None:
    assert [vector[0] for vector in vectors] == [float(index) for index in range(count)]


def test_a_large_add_is_split_under_the_request_token_cap() -> None:
    """Invariant: no request's estimated tokens exceed the budget, and order survives."""
    client = _Client()
    inputs = [_text(30_000, tag=f"{index}") for index in range(12)]  # 360,000 tokens in all

    vectors = _embedder(client).embed_documents(inputs)

    totals = [sum(len(item["content"][0]["text"]) for item in request) for request in client.requests]
    assert [total for total in totals if total > _budget()] == []
    assert [item for request in client.requests for item in request] == inputs
    _assert_order(vectors, len(inputs))


def test_images_count_by_their_pixels(monkeypatch) -> None:
    """Invariant: an image costs its pixels over 560 against the request budget."""
    monkeypatch.setattr(shared_module, "VOYAGE_MAX_REQUEST_TOKENS", 12_000)  # budget 10,800
    client = _Client()
    image = _image(1_120, 2_500)  # 2.8M pixels, 5,000 tokens
    assert 1_120 * 2_500 // VOYAGE_PIXELS_PER_TOKEN == 5_000
    inputs = [image, image, image]

    vectors = _embedder(client).embed_documents(inputs)

    assert [len(request) for request in client.requests] == [2, 1]
    _assert_order(vectors, 3)


def test_no_request_carries_more_than_a_thousand_inputs() -> None:
    """Invariant: no request carries more than Voyage's 1,000 inputs."""
    client = _Client()
    inputs = [_text(3) for _ in range(1_001)]

    vectors = _embedder(client).embed_documents(inputs)

    assert [len(request) for request in client.requests] == [1_000, 1]
    _assert_order(vectors, 1_001)


def test_an_add_that_fits_is_sent_as_one_unchanged_request() -> None:
    """Invariant: an Add under every cap goes out exactly as before, as one identical request."""
    client = _Client()
    inputs = [_text(40, tag=f"{index}") for index in range(5)] + [_image(8, 8)]

    _embedder(client).embed_documents(inputs)

    assert client.requests == [inputs]


def test_a_document_over_the_context_is_sent_whole_and_logged(caplog) -> None:
    """Invariant: an input over the 32K context is sent unchanged and a warning names it."""
    client = _Client()
    big = _text(MAX_VOYAGE_INPUT_TOKENS + 8_000, tag="b")
    inputs = [_text(10), big, _text(10)]
    assert MAX_VOYAGE_INPUT_TOKENS + 8_000 < MAX_VOYAGE_REQUEST_TOKENS

    with caplog.at_level(logging.WARNING, logger="recall_aml"):
        vectors = _embedder(client).embed_documents(inputs)

    assert client.requests == [inputs]
    _assert_order(vectors, 3)
    warned = [
        record for record in caplog.records
        if record.getMessage() == "voyage_multimodal_input_over_context"
    ]
    assert [(record.input_index, record.input_type) for record in warned] == [(1, "document")]


def test_a_query_over_the_context_is_logged(caplog) -> None:
    """Invariant: a query over the 32K context is sent unchanged and logged as a query."""
    client = _Client()
    query = "q" * (MAX_VOYAGE_INPUT_TOKENS + 1)

    with caplog.at_level(logging.WARNING, logger="recall_aml"):
        _embedder(client).embed_query(query)

    assert client.requests == [[{"content": [{"type": "text", "text": query}]}]]
    warned = [
        record for record in caplog.records
        if record.getMessage() == "voyage_multimodal_input_over_context"
    ]
    assert [(record.input_index, record.input_type) for record in warned] == [(0, "query")]


def test_an_ordinary_query_never_loads_the_tokenizer(monkeypatch) -> None:
    """Invariant: a query the byte bound already clears never loads Voyage's tokenizer."""
    loads: list[str] = []

    def loader(model: str):
        loads.append(model)
        return None

    monkeypatch.setattr("recall.embeddings._voyage_token_counter", loader)
    client = _Client()

    VoyageMultimodalEmbedder("unused", client=client).embed_query("hello")

    assert loads == []
    assert len(client.requests) == 1
