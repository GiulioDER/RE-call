"""Behavioral proofs for the historical PR689 through680 repairs."""

import asyncio
from types import SimpleNamespace

import pytest

from recall.types import Chunk, ScoredChunk
from recall_aml.app import create_app
from recall_aml.config import HostedSettings
from recall_aml.models import ImageContentPart, ImageURLValue, content_media_bytes
from recall_aml.multimodal import LazyMedia, render_preserved
from tests.test_aml_hosted import FakeEmbedder
from tests.test_aml_multimodal import _png_data_url
from tests.test_aml_multimodal_memeye import _arm
from tests.test_aml_promotion_gate import passing_artifact
from tests.test_aml_search_deadline import _image_parent
from scripts.aml_promotion_gate import decide as promotion
from scripts.select_aml_multimodal_memeye import ARMS, decide as admission


def test_authenticated_stream_stops_at_body_limit(monkeypatch):
    """SEC-001: baseline8c296fd31 consumes the sentinel third chunk before rejecting.

    The exact public endpoint test must fail its consumed-chunk assertion before repair.
    """
    import recall_aml.app as module

    monkeypatch.setattr(module, "MAX_BODY_BYTES", 4)

    class Service:
        variant_name = "A0_raw"

        async def add(self, request):
            raise AssertionError("oversized request reached the service")

    for path in ("/v1/add", "/v1/delete", "/v1/corpus/status"):
        app = create_app(
            HostedSettings("postgresql://unused", "secret", "abc123", variant_name="A0_raw"),
            Service(),
        )
        consumed = []
        sent = []
        chunks = [b"123", b"456", b"sentinel"]

        async def receive():
            index = len(consumed)
            consumed.append(index)
            return {"type": "http.request", "body": chunks[index], "more_body": index < 2}

        async def send(message):
            sent.append(message)

        asyncio.run(app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [(b"authorization", b"Bearer secret")],
            "client": ("127.0.0.1", 1), "server": ("testserver", 80),
        }, receive, send))
        assert next(m for m in sent if m["type"] == "http.response.start")["status"] == 422
        assert consumed == [0, 1], "oversized stream was buffered past its limit"


@pytest.mark.parametrize("failure_stage", ["compiler", "app"])
def test_failed_app_construction_closes_pool(monkeypatch, tmp_path, failure_stage):
    """BUG-001: baseline8c296fd31 fails the close-count assertion at both stages."""
    import recall_aml.__main__ as module

    closed = []
    pool = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(module, "_resolve_hosted_embedders", lambda *_: (FakeEmbedder(), {}))
    monkeypatch.setattr(module, "SharedPool", lambda *_args, **_kwargs: pool)
    monkeypatch.setattr(module, "PgVectorStore", lambda *_args, **_kwargs: SimpleNamespace(check_schema=lambda: None))
    monkeypatch.setattr(module, "PgHostedRepository", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "VoyageReranker", lambda **_kwargs: object())
    monkeypatch.setattr(module, "HostedRetriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "HostedService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "build_image_text_extractor", lambda *_args: None)
    monkeypatch.setattr(module, "verify_model_readiness", lambda **_kwargs: {"embedder_ready": True})

    def fail(*_args, **_kwargs):
        raise RuntimeError("construction sentinel")

    monkeypatch.setattr(module, "build_compiler", fail if failure_stage == "compiler" else lambda *_: None)
    monkeypatch.setattr(module, "create_app", fail)
    settings = HostedSettings(
        "postgresql://unused", "secret", "abc123", variant_name="A0_raw",
        voyage_api_key="offline-key", embedding_lock_path=tmp_path / "embed.lock",
    )
    with pytest.raises(RuntimeError, match="construction sentinel"):
        module.build_app(settings)
    assert closed == [True], "unsuccessful app construction must close its pool once"


def _media_case(count):
    triples = [_image_parent(index) for index in range(count)]
    return (
        [hit for hit, _, _ in triples],
        {parent.id: parent for _, parent, _ in triples},
        {row.id: row for _, _, row in triples},
    )


def _image_size(url):
    return content_media_bytes([ImageContentPart(type="image_url", image_url=ImageURLValue(url=url))])


def test_exhausted_media_budget_keeps_text_without_loading_tail(monkeypatch):
    """PERF-001: baseline8c296fd31 fetches every rejected image rather than one batch."""
    import recall_aml.multimodal as module

    hits, primary, media = _media_case(40)
    monkeypatch.setattr(module, "MAX_RESPONSE_MEDIA_BYTES", _image_size(media["media_000"].metadata["data_url"]))
    text = Chunk(id="plain", source="s", text="later text", metadata={})
    hits.append(ScoredChunk(text, .1))
    primary[text.id] = text
    reads = []

    def fetch(ids):
        reads.extend(ids)
        return {key: media[key] for key in ids}

    rows = render_preserved(hits, primary_by_id=primary, media_by_id=LazyMedia(list(media), fetch, batch=4), top_k=3)
    assert [row.id for row in rows] == ["raw_000", "plain"]
    assert len(reads) <= 4, "exhausted budget must not fetch rejected media tail"


def test_media_cache_bounds_rejected_tail_and_refetches_evicted_images(monkeypatch):
    """PERF-001: baseline8c296fd31 retains all rejected payload rows in its cache."""
    import recall_aml.multimodal as module

    hits, primary, media = _media_case(40)
    small = _png_data_url(b"small")
    size = _image_size(small)
    # Valid PNG bytes may have harmless trailing bytes; these parents cannot fit.
    import base64
    large = "data:image/png;base64," + base64.b64encode(base64.b64decode(small.split(",")[1]) + b"x" * 100).decode()
    for row in media.values():
        row.metadata["data_url"] = large
    media["media_039"].metadata["data_url"] = small
    monkeypatch.setattr(module, "MAX_RESPONSE_MEDIA_BYTES", size + 1)
    reads = []

    def fetch(ids):
        reads.extend(ids)
        return {key: media[key] for key in ids}

    lazy = LazyMedia(list(media), fetch, batch=4)
    rows = render_preserved(hits, primary_by_id=primary, media_by_id=lazy, top_k=2)
    assert [row.id for row in rows] == ["raw_039"]
    assert len(lazy) <= 4, "rejected media payloads must not accumulate"
    assert lazy["media_000"].id == "media_000", "an eviction must not turn an existing row into a miss"
    assert len(lazy) <= 4


def test_promotion_refuses_nonfinite_or_negative_metrics():
    """NUM-001: baseline8c296fd31 promotes the first NaN artifact instead of refusing."""
    for key in ("add_p95_seconds", "search_p95_seconds", "provider_spend_usd"):
        for invalid in (float("nan"), float("-inf"), float("inf"), -1, "NaN"):
            artifact = passing_artifact()
            artifact[key] = invalid
            accepted, reasons = promotion(artifact)
            assert accepted is False, (key, invalid)
            assert reasons


def test_memeye_refuses_nonfinite_cost_in_every_arm():
    """NUM-002: baseline8c296fd31 admits NaN and skips its aggregate spend ceiling."""
    for name in ARMS:
        for cost in (float("nan"), float("inf"), float("-inf")):
            arms = {
                "MM0_caption": _arm("MM0_caption", .4, .4),
                "MM1_preserve": _arm("MM1_preserve", .5, .4),
                "MM2_dual": _arm("MM2_dual", .55, .5),
            }
            for arm in arms.values():
                arm["provider_spend_usd"] = 20
            arms[name]["provider_spend_usd"] = cost
            assert admission(arms)["verdict"] == "INVALID", (name, cost)

