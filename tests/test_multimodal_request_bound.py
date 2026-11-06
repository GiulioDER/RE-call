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


def test_fitted_jpeg_keeps_display_orientation(monkeypatch):
    """BUG-001: preserve display orientation; red at fa9d503f on 2026-10-27.

    embed_query sent a 14x7 landscape image instead of portrait before the fix.
    """
    from PIL import ImageOps

    image = Image.new("RGB", (20, 10), (200, 10, 10))
    exif = Image.Exif()
    exif[274] = 6
    payload = _encoded(image, "JPEG", exif=exif)
    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_PIXELS", 100)
    embedder, client = _embedder()
    embedder.embed_query(MultimodalQuery(image_bytes=payload, media_type="image/jpeg"))
    with Image.open(BytesIO(_sent_image(client))) as sent:
        displayed = ImageOps.exif_transpose(sent)
        assert displayed.height > displayed.width, "fitting must preserve the displayed orientation"


def test_header_readable_corrupt_image_passes_through(monkeypatch):
    """BUG-002: decoder errors pass through by default; strict mode still raises.

    Red at fa9d503f on 2026-10-27: the intended error-is-None assertion
    failed with OSError after a successful image-header dimension check.
    """
    import pytest

    payload = _encoded(Image.new("RGB", (20, 10), (200, 10, 10)), "PNG")
    # Signature, IHDR, IDAT header remain; compressed pixel body is truncated.
    start = payload.index(b"IDAT") + 4
    corrupt = payload[:start + 2]
    assert multimodal_module._image_pixels(corrupt, "PNG") == 200
    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_PIXELS", 100)
    embedder, client = _embedder()
    error = None
    try:
        embedder.embed_query(MultimodalQuery(image_bytes=corrupt, media_type="image/png"))
    except OSError as exc:
        error = exc
    assert error is None, f"default fitting must pass an undecodable image through: {error}"
    assert _sent_image(client) == corrupt
    url = "data:image/png;base64," + base64.b64encode(corrupt).decode("ascii")
    with pytest.raises(OSError):
        multimodal_module.fit_voyage_image(url, strict=True)


def test_an_image_in_pillows_warning_band_is_refused_before_it_is_decoded(monkeypatch) -> None:
    """Invariant: in strict mode an image declaring more pixels than Pillow's `MAX_IMAGE_PIXELS`
    is refused before it is decoded.

    Pillow only warns between that limit and twice it, so a small compressed file declaring about
    149M pixels used to be opened and decoded in full by the fitter's resize. Strict mode (the
    AML adapter) now raises `DecompressionBombError`, which AML turns into a per-image skip on
    Add and a 422 on a query. The default mode still fits it (control): passing it through would
    make Voyage refuse the whole batch. Scaled down here by lowering both limits; the 900x900
    image sits between the patched `MAX_IMAGE_PIXELS` and twice it, the band Pillow only warns in.
    Red proof (`fit_voyage_image` before the ceiling): the strict assert fails with DID NOT RAISE.
    """
    import pytest
    from PIL.Image import DecompressionBombError, DecompressionBombWarning

    from recall.multimodal import fit_voyage_image

    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_PIXELS", 100_000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 500_000)
    payload = _encoded(Image.new("L", (900, 900), 0), "PNG")
    url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")

    with pytest.warns(DecompressionBombWarning):
        fitted, changed = fit_voyage_image(url)
    assert changed and fitted != url
    with pytest.warns(DecompressionBombWarning), pytest.raises(DecompressionBombError):
        fit_voyage_image(url, strict=True)


class _CountingLimit(int):
    """A pixel limit that counts how often the fitter tests a size against it."""

    tests = 0

    def __lt__(self, other):  # `width * height > limit` reaches here first, int being its base
        type(self).tests += 1
        return int(self) < other


def test_a_strip_image_is_fitted_in_a_few_steps(monkeypatch) -> None:
    """A 1 x N strip is fitted without a step per surplus pixel (CCA hunt PERF-001).

    Invariant: fitting tests a size against the pixel limit a bounded number of times. Failure
    mode: the short side clamps to 1 and the long side shrank one pixel per loop step, about 36M
    steps (about 19 s) for a 1 x 170M PNG that is under the byte cap. Scaled down here: a 1 x 5000
    image against a limit of 100 took 607 steps.

    Red proof, 2026-11-06, against the pre-fix one-pixel loop in ``_fit_voyage_image``:
    ``assert 608 <= 10`` (the count), with the fitted size the same 1 x 100 before and after.
    """
    limit = _CountingLimit(100)
    _CountingLimit.tests = 0
    monkeypatch.setattr(multimodal_module, "VOYAGE_MAX_IMAGE_PIXELS", limit)
    embedder, client = _embedder()
    payload = _encoded(Image.new("L", (1, 5000), 0), "PNG")

    embedder.embed_query(MultimodalQuery(image_bytes=payload, media_type="image/png"))

    with Image.open(BytesIO(_sent_image(client))) as sent:
        assert (sent.width, sent.height) == (1, 100)
    assert _CountingLimit.tests <= 10


class _TextOnlyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts, model, input_type, **_kwargs):
        self.calls.append((list(texts), input_type))
        return SimpleNamespace(embeddings=[[1.0] + [0.0] * 1023 for _ in texts])


def test_a_text_only_client_answers_queries_as_it_embeds_passages() -> None:
    """embed_query uses the same provider path as the documents (CCA hunt CODE-011).

    Invariant: a client the constructor and the document path accept also embeds a text query,
    as a query. Failure mode: embed_query called ``multimodal_embed`` unconditionally, so a
    client with only ``embed`` built, embedded passages, then failed every query.

    Red proof, 2026-11-06, against the pre-fix ``embed_query``: ``AttributeError:
    '_TextOnlyClient' object has no attribute 'multimodal_embed'`` raised at the call under test.
    """
    client = _TextOnlyClient()
    embedder = VoyageMultimodalEmbedder(client=client, token_counter=_one_token_per_char)
    client.calls.clear()

    assert embedder.embed_query("where is the cat") == [1.0] + [0.0] * 1023
    assert client.calls == [(["where is the cat"], "query")]


def test_a_missing_pillow_is_reported_when_the_embedder_is_built(monkeypatch, caplog) -> None:
    """Without Pillow no image can be fitted, and that is said once, up front (DEPLOY-001).

    Invariant: an embedder that will send over-limit images unfitted says so when built.
    Failure mode: the install hint named an extra without Pillow, and the fitter then passed
    every image through silently, so Voyage refused whole requests with no local trace.

    Red proof, 2026-11-06, against the pre-fix constructor: the assertion on the
    ``voyage_multimodal_images_unfitted`` record failed (no such record).
    """
    import sys

    monkeypatch.setitem(sys.modules, "PIL", None)
    with caplog.at_level(logging.WARNING, logger="recall.multimodal"):
        VoyageMultimodalEmbedder(client=_Client(), token_counter=_one_token_per_char)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage() == "voyage_multimodal_images_unfitted"
    ] == ["voyage_multimodal_images_unfitted"]


def test_bytes_labelled_png_reach_only_the_png_decoder() -> None:
    """A declared image type limits Pillow to that one decoder (second hunt, SEC-002).

    Invariant: bytes labelled image/png are never opened by another Pillow plugin. Failure mode:
    ``Image.open`` without ``formats=`` read an EPS header behind a PNG label, so an over-limit
    EPS reached the resize and with it Ghostscript. Observed through the estimator: an image no
    allowed decoder reads counts as a whole context. Red proof, 2026-11-06, against the first
    hunt's module: ``assert 44643 == 32000``, the EPS header's 25M pixels at 560 per token.
    """
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 5000 5000\n%%EndComments\nshowpage\n%%EOF\n"
    item = {
        "content": [
            {
                "type": "image_base64",
                "image_base64": "data:image/png;base64," + base64.b64encode(eps).decode("ascii"),
            }
        ]
    }

    [(tokens, images)] = multimodal_module.estimate_voyage_inputs([item], _one_token_per_char)

    assert (tokens, images) == (VOYAGE_MAX_INPUT_TOKENS, 1)


class _SlowClient(_Client):
    """Records how many requests are in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        import threading

        self._lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0

    def multimodal_embed(self, *, inputs, model, input_type, **_kwargs):
        import time

        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.3)
        with self._lock:
            self.in_flight -= 1
            vectors = []
            for item in inputs:
                text = item["content"][0]["text"]
                vectors.append([float(len(text))] + [0.0] * 1023)
        return SimpleNamespace(embeddings=vectors)


def test_parallel_requests_keep_more_than_one_request_in_flight_and_the_order() -> None:
    """RECALL_VOYAGE_PARALLEL_REQUESTS reaches the multimodal embedder too (PERF-003, ENV-102).

    Invariant: with ``max_parallel_requests=2`` two planned requests are in flight together and
    the vectors still come back in input order. Failure mode: requests were always sent one
    after another, whatever the setting. Red proof, 2026-11-06, a mutation of the new
    ``embed_multimodal`` (the parallel branch disabled, ``if False and ...``): ``assert 1 == 2``
    on the peak.
    """
    client = _SlowClient()
    embedder = VoyageMultimodalEmbedder(
        client=client, batch_size=1, token_counter=_one_token_per_char, max_parallel_requests=2
    )
    client.peak = 0

    vectors = embedder.embed_passages(["a", "bb", "ccc"])

    assert client.peak == 2
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


# Owner decisions of 2026-11-06 on the second hunt's open items. Red proofs, 2026-11-06, against
# the second hunt's module (a77611f8a) with this file otherwise unchanged.


def test_an_unsupported_image_type_is_never_opened(monkeypatch) -> None:
    """SEC-002 residual: a type outside JPEG, PNG and WebP reaches no Pillow decoder at all.

    Red proof: ``assert ['open'] == []``; the strict probe opened the GIF with every decoder.
    """
    opened: list[str] = []
    real_open = Image.open

    def spy(*args, **kwargs):
        opened.append("open")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", spy)
    url = "data:image/gif;base64," + base64.b64encode(_encoded(Image.new("L", (4, 4)), "GIF")).decode()

    fitted, changed = multimodal_module.fit_voyage_image(url, strict=True)
    multimodal_module.estimate_voyage_inputs(
        [{"content": [{"type": "image_base64", "image_base64": url}]}], _one_token_per_char
    )

    assert (fitted, changed) == (url, False)
    assert opened == []


def test_the_unused_public_wrappers_are_deprecated_and_the_used_helpers_exported() -> None:
    """CODE-015: fit_voyage_input and estimate_voyage_tokens warn; the helpers callers import
    are in __all__.

    Red proof: ``Failed: DID NOT WARN``.
    """
    import pytest

    item = _text(3)
    with pytest.warns(DeprecationWarning, match="fit_voyage_input"):
        multimodal_module.fit_voyage_input(item)
    with pytest.warns(DeprecationWarning, match="estimate_voyage_inputs"):
        multimodal_module.estimate_voyage_tokens(item, _one_token_per_char)
    assert {
        "DEFAULT_MAX_ITEMS",
        "VOYAGE_MAX_INPUT_TOKENS",
        "estimate_voyage_inputs",
        "fit_voyage_image",
        "lazy_voyage_text_counter",
        "plan_voyage_requests",
    } <= set(multimodal_module.__all__)


class _NarrowingClient(_Client):
    """Answers the construction probe at 1024 and later requests at 512."""

    def multimodal_embed(self, *, inputs, model, input_type, **_kwargs):
        width = 1024 if not self.requests else 512
        self.requests.append((input_type, [dict(item) for item in inputs]))
        return SimpleNamespace(embeddings=[[0.5] * width for _ in inputs])


def test_a_response_narrower_than_the_probe_is_refused() -> None:
    """BUG-006: every vector must have the width the construction probe measured.

    Red proof: ``Failed: DID NOT RAISE``; 512-wide vectors were returned beside a 1024 dim.
    """
    import pytest

    embedder = VoyageMultimodalEmbedder(client=_NarrowingClient(), token_counter=_one_token_per_char)
    assert embedder.dim == 1024
    with pytest.raises(RuntimeError, match="not all 1024"):
        embedder.embed_passages(["narrow"])


class _EmptyVectorClient(_Client):
    def multimodal_embed(self, *, inputs, model, input_type, **_kwargs):
        return SimpleNamespace(embeddings=[[] for _ in inputs])


def test_an_empty_vector_is_reported_as_empty() -> None:
    """CCA run 20261106T141216Z DOC-004: the refusal names the fault.

    Red proof, 2026-11-06, against the owner-decision diff before this fix: ``Regex pattern did not
    match``; the message was 'Voyage multimodal response vectors are not all 0 wide'.
    """
    import pytest

    with pytest.raises(RuntimeError, match="empty vector"):
        VoyageMultimodalEmbedder(client=_EmptyVectorClient(), token_counter=_one_token_per_char)


def test_an_sdk_without_the_multimodal_endpoint_is_refused_with_the_install_hint(monkeypatch) -> None:
    """DEPLOY-101, DEPLOY-102: an old voyageai is refused at construction, naming the versions.

    Red proof: ``Failed: DID NOT RAISE``; the old SDK's text fallback answered the probe.
    """
    import sys
    import types

    import pytest

    class _OldClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def embed(self, texts, model, **_kwargs):
            return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in texts])

    old_sdk = types.ModuleType("voyageai")
    old_sdk.Client = _OldClient
    monkeypatch.setitem(sys.modules, "voyageai", old_sdk)
    with pytest.raises(ImportError, match=r'voyageai>=0\.3.*"pillow>=10"'):
        VoyageMultimodalEmbedder(api_key="test-key", token_counter=_one_token_per_char)
