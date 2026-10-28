"""Request-local image measurements avoid duplicate decode and header work.

PERF-001 red proof, 2026-10-28, baseline 935f224a: each parametrized test
failed its measurement-count assertion through the public embedding/estimation
entry point. Image fitting and exact text counting must reuse scalar image
measurements without keeping them across calls. No latency claim is made.
"""

import base64
from io import BytesIO
import math

from PIL import Image
import pytest

import recall.multimodal as multimodal
from tests.test_multimodal_request_bound import _embedder, _encoded, _sent_image


def _measurements(monkeypatch):
    calls = {"decode": 0, "pixels": 0}
    split = multimodal._split_data_url
    pixels = multimodal._image_pixels

    def decode(url):
        calls["decode"] += 1
        return split(url)

    def probe(payload):
        calls["pixels"] += 1
        return pixels(payload)

    monkeypatch.setattr(multimodal, "_split_data_url", decode)
    monkeypatch.setattr(multimodal, "_image_pixels", probe)
    return calls


@pytest.mark.parametrize("fit_pixels", [None, 100])
def test_core_query_reuses_prepared_image_measurements(monkeypatch, fit_pixels):
    """Fitting metadata reaches both text passes; each new call measures anew."""
    if fit_pixels is not None:
        monkeypatch.setattr(multimodal, "VOYAGE_MAX_IMAGE_PIXELS", fit_pixels)
    payload = _encoded(Image.new("RGB", (20, 10), (200, 10, 10)), "PNG")
    text = "x" * (multimodal.VOYAGE_MAX_INPUT_TOKENS + 1)
    text_calls = []

    def count_text(texts):
        text_calls.append(texts)
        return [len(value) for value in texts]

    embedder, client = _embedder(counter=count_text)
    calls = _measurements(monkeypatch)
    query = multimodal.MultimodalQuery(text=text, image_bytes=payload, media_type="image/png")

    embedder.embed_query(query)

    assert calls == {"decode": 1, "pixels": 1}
    assert text_calls == [[text]]
    sent = _sent_image(client)
    if fit_pixels is None:
        assert sent == payload
    else:
        with Image.open(BytesIO(sent)) as image:
            assert image.width * image.height <= fit_pixels
    client.requests.clear()

    embedder.embed_query(query)

    assert calls == {"decode": 2, "pixels": 2}
    assert text_calls == [[text], [text]]
    assert _sent_image(client) == sent


@pytest.mark.parametrize("readable", [True, False])
def test_shared_estimation_reuses_pixels_between_text_passes(monkeypatch, readable):
    """A failed image probe is also reused; no image buffers enter a persistent cache."""
    payload = (
        _encoded(Image.new("RGB", (20, 10), (1, 2, 3)), "PNG")
        if readable else b"not an image"
    )
    url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    text = "x" * (multimodal.VOYAGE_MAX_INPUT_TOKENS + 1)
    item = {"content": [
        {"type": "text", "text": text},
        {"type": "image_base64", "image_base64": url},
    ]}
    calls = _measurements(monkeypatch)
    text_calls = []

    def count_text(texts):
        text_calls.append(texts)
        return [len(value) for value in texts]

    image_tokens = (
        math.ceil(200 / multimodal.VOYAGE_PIXELS_PER_TOKEN)
        if readable else multimodal.VOYAGE_MAX_INPUT_TOKENS
    )
    expected = [(len(text) + image_tokens, 1)]

    assert multimodal.estimate_voyage_inputs([item], count_text) == expected
    assert calls == {"decode": 1, "pixels": 1}
    assert text_calls == [[text]]
    assert multimodal.estimate_voyage_inputs([item], count_text) == expected
    assert calls == {"decode": 2, "pixels": 2}
    assert text_calls == [[text], [text]]
