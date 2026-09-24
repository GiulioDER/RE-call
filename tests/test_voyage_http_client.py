"""`recall._voyage_http.Client` sends what the `voyageai` SDK sends and reads what it reads.

Invariant: for every call RE-call makes (`embed`, `contextualized_embed`, `rerank`), the HTTP
client produces the SDK's request (URL, JSON body, the two headers the SDK keeps, timeout), decodes
the SDK's base64 vectors to the identical Python floats, and raises errors the retry layer reads
identically (class name, `http_status`, case-insensitive `headers`). A vector that differs by one
bit from the SDK's is a different corpus key, so these compare against the SDK itself wherever it
is installed, and against a fixed expectation where it is not (CI installs only `dev`).

Failure mode caught: a hand-written client that drifts from the SDK in a way no unit test of the
client alone would notice: a missing `encoding_format`, a dropped `null`, JSON floats instead of
base64, a different host, or an error the retry classifier no longer recognises as transient.

Red proof, recorded 2026-09-23 by mutating `recall/_voyage_http.py` one property at a time and
restoring it after each run (the pre-fix code has no such module, so a mutation is the baseline):

* `Client.embed` without `"encoding_format": "base64"`: `test_the_request_bodies_are_fixed[0,1]`
  and `test_the_request_matches_the_sdk[0,1]` failed at the body comparison, the SDK's dict
  containing one more item.
* `Client.embed` dropping `None` values from its body: the same four failed at the body
  comparison, two items fewer.
* `decode_embedding` reading `array("d")`: `test_decoding_matches_the_sdk` failed at
  `assert 512 == 1024`.
* HTTP 429 mapped to `APIError`: `test_error_mapping_matches_the_sdk[3]` failed at
  `assert 'APIError' == 'RateLimitError'`. (The retry verdict itself survives that mutation,
  because the numeric status still decides it, which is why the class-name check is separate.)
"""

from __future__ import annotations

import base64
import json
import random
import struct
from array import array
from typing import Any

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from recall import _voyage_http
from recall.embeddings import _is_transient, _retry_after_seconds

#: Placeholders, never credentials: the transport is replaced in every test below.
PLACEHOLDER_KEY = "pa-CHANGEME-placeholder"
ATLAS_PLACEHOLDER_KEY = "al-CHANGEME-placeholder"


class _Response:
    def __init__(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        raw = body if isinstance(body, str) else json.dumps(body)
        self.content = raw.encode("utf-8")
        self.headers = CaseInsensitiveDict(
            {"Content-Type": "application/json", **(headers or {})}
        )


def _vector_b64(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _record(monkeypatch: pytest.MonkeyPatch, response: _Response) -> list[dict[str, Any]]:
    """Capture every request made through ANY `requests.Session`, SDK or ours."""
    seen: list[dict[str, Any]] = []

    def request(self, method, url, headers=None, data=None, timeout=None, **_kwargs):  # type: ignore[no-untyped-def]
        seen.append(
            {
                "method": method.lower(),
                "url": url,
                "headers": dict(headers or {}),
                "body": json.loads(data),
                "timeout": timeout,
            }
        )
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    return seen


EMBED_OK = {"data": [{"embedding": _vector_b64([0.25, -1.5])}], "usage": {"total_tokens": 3}}
CONTEXT_OK = {
    "data": [
        {"data": [{"embedding": _vector_b64([1.0, 2.0])}, {"embedding": _vector_b64([3.0, 4.0])}]}
    ],
    "usage": {"total_tokens": 5},
}
RERANK_OK = {
    "data": [{"index": 1, "relevance_score": 0.75}, {"index": 0, "relevance_score": 0.5}],
    "usage": {"total_tokens": 7},
}

CALLS = [
    ("embed", {"texts": ["a", "b"], "model": "voyage-code-3"}, EMBED_OK),
    ("embed", {"texts": ["a"], "model": "voyage-code-3", "input_type": "document"}, EMBED_OK),
    (
        "contextualized_embed",
        {"inputs": [["x", "y"]], "model": "voyage-context-4", "input_type": "document",
         "output_dimension": 1024, "output_dtype": "float"},
        CONTEXT_OK,
    ),
    (
        "contextualized_embed",
        {"inputs": ["q"], "model": "voyage-context-4", "input_type": "query",
         "output_dimension": 1024, "output_dtype": "float"},
        CONTEXT_OK,
    ),
    (
        "rerank",
        {"query": "q", "documents": ["d0", "d1"], "model": "rerank-2.5", "top_k": 2},
        RERANK_OK,
    ),
]

#: The request bodies the SDK sends at voyageai 0.5.0 for `CALLS`, as
#: `test_the_request_matches_the_sdk` observes them, so that CI, which has no SDK, still pins them.
EXPECTED_BODIES = [
    {"input": ["a", "b"], "model": "voyage-code-3", "input_type": None, "truncation": True,
     "output_dtype": None, "output_dimension": None, "encoding_format": "base64"},
    {"input": ["a"], "model": "voyage-code-3", "input_type": "document", "truncation": True,
     "output_dtype": None, "output_dimension": None, "encoding_format": "base64"},
    {"inputs": [["x", "y"]], "model": "voyage-context-4", "input_type": "document",
     "output_dtype": "float", "output_dimension": 1024, "encoding_format": "base64"},
    {"inputs": [["q"]], "model": "voyage-context-4", "input_type": "query",
     "output_dtype": "float", "output_dimension": 1024, "encoding_format": "base64"},
    {"query": "q", "documents": ["d0", "d1"], "model": "rerank-2.5", "top_k": 2,
     "truncation": True},
]


def _payload(result: Any) -> Any:
    """What RE-call reads off a response, in a comparable shape."""
    if hasattr(result, "embeddings"):
        return result.embeddings
    return [
        getattr(item, "embeddings", None) or (item.index, item.relevance_score)
        for item in result.results
    ]


@pytest.mark.parametrize("case", range(len(CALLS)))
def test_the_request_bodies_are_fixed(monkeypatch, case) -> None:
    method, kwargs, reply = CALLS[case]
    seen = _record(monkeypatch, _Response(200, reply))
    client = _voyage_http.Client(api_key=PLACEHOLDER_KEY, timeout=60.0)

    getattr(client, method)(**kwargs)

    assert seen[0]["body"] == EXPECTED_BODIES[case]
    assert list(seen[0]["body"]) == list(EXPECTED_BODIES[case])  # key order too
    assert seen[0]["headers"] == {
        "Authorization": f"Bearer {PLACEHOLDER_KEY}",
        "Content-Type": "application/json",
    }
    assert seen[0]["timeout"] == 60.0


@pytest.mark.timeout(300)  # the `voyageai_sdk` import, if this is the first to ask
@pytest.mark.parametrize("case", range(len(CALLS)))
def test_the_request_matches_the_sdk(monkeypatch, voyageai_sdk, case) -> None:
    method, kwargs, reply = CALLS[case]
    sdk_seen = _record(monkeypatch, _Response(200, reply))
    sdk_client = voyageai_sdk.Client(api_key=PLACEHOLDER_KEY, max_retries=0, timeout=60.0)
    sdk_result = getattr(sdk_client, method)(**kwargs)
    ours_seen = _record(monkeypatch, _Response(200, reply))
    ours_result = getattr(_voyage_http.Client(api_key=PLACEHOLDER_KEY, timeout=60.0), method)(
        **kwargs
    )

    assert ours_seen[0]["url"] == sdk_seen[0]["url"]
    assert ours_seen[0]["body"] == sdk_seen[0]["body"]
    assert list(ours_seen[0]["body"]) == list(sdk_seen[0]["body"])
    assert ours_seen[0]["headers"] == sdk_seen[0]["headers"]
    assert ours_seen[0]["timeout"] == sdk_seen[0]["timeout"]
    assert _payload(ours_result) == _payload(sdk_result)


@pytest.mark.timeout(300)
def test_the_base_url_follows_the_sdk_for_an_atlas_key(monkeypatch, voyageai_sdk) -> None:
    sdk_seen = _record(monkeypatch, _Response(200, EMBED_OK))
    voyageai_sdk.Client(api_key=ATLAS_PLACEHOLDER_KEY, max_retries=0).embed(["a"], model="voyage-4")
    ours_seen = _record(monkeypatch, _Response(200, EMBED_OK))
    _voyage_http.Client(api_key=ATLAS_PLACEHOLDER_KEY).embed(["a"], model="voyage-4")

    assert ours_seen[0]["url"] == sdk_seen[0]["url"] == "https://ai.mongodb.com/v1/embeddings"
    assert ours_seen[0]["timeout"] == sdk_seen[0]["timeout"] == 600


@pytest.mark.timeout(300)
def test_decoding_matches_the_sdk(voyageai_sdk) -> None:
    rng = random.Random(20260923)
    for width in (1024, 3, 1):  # 1024 first: a float64 misread of it is well formed
        values = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        encoded = base64.b64encode(array("f", values).tobytes()).decode()
        for dtype in (None, "float"):
            ours = _voyage_http.decode_embedding(encoded, dtype)
            sdk = voyageai_sdk.util.decode_base64_embedding(encoded, dtype)
            assert len(ours) == len(sdk) == width
            assert ours == sdk
            assert all(type(value) is float for value in ours)


ERROR_CASES = [
    (400, {"detail": "bad input"}, {}),
    (401, {"detail": "Provided API key is invalid."}, {}),
    (422, {"detail": "malformed"}, {}),
    (429, {"detail": "rate limit exceeded"}, {"Retry-After": "7"}),
    (404, {"detail": "not found"}, {}),
    (400, {"message": "no detail key"}, {}),
    (500, "internal", {}),
    (503, "overloaded", {"retry-after": "3"}),
    (418, "not json at all", {}),
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", range(len(ERROR_CASES)))
def test_error_mapping_matches_the_sdk(monkeypatch, voyageai_sdk, case) -> None:
    status, body, headers = ERROR_CASES[case]
    _record(monkeypatch, _Response(status, body, headers))
    with pytest.raises(Exception) as sdk_raised:
        voyageai_sdk.Client(api_key=PLACEHOLDER_KEY, max_retries=0).embed(["a"], model="voyage-4")
    with pytest.raises(_voyage_http.VoyageError) as ours_raised:
        _voyage_http.Client(api_key=PLACEHOLDER_KEY).embed(["a"], model="voyage-4")
    sdk, ours = sdk_raised.value, ours_raised.value

    assert type(ours).__name__ == type(sdk).__name__
    assert ours.http_status == sdk.http_status
    assert str(ours) == str(sdk)
    assert _is_transient(ours) == _is_transient(sdk)
    assert _retry_after_seconds(ours) == _retry_after_seconds(sdk)


@pytest.mark.parametrize(
    ("status", "transient"),
    [(429, True), (500, True), (503, True), (400, False), (401, False)],
)
def test_the_retry_layer_classifies_our_errors(monkeypatch, status, transient) -> None:
    """The CI-side twin of the SDK comparison: the verdict the retry layer reaches."""
    body = "x" if status >= 500 else {"detail": "d"}
    _record(monkeypatch, _Response(status, body, {"Retry-After": "2"}))
    with pytest.raises(_voyage_http.VoyageError) as raised:
        _voyage_http.Client(api_key=PLACEHOLDER_KEY).embed(["a"], model="voyage-4")

    assert raised.value.http_status == status
    assert _is_transient(raised.value) is transient


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (requests.exceptions.ConnectTimeout("slow"), _voyage_http.Timeout),
        (requests.exceptions.ConnectionError("reset"), _voyage_http.APIConnectionError),
    ],
)
def test_transport_failures_are_classified_as_transient(monkeypatch, error, expected) -> None:
    def request(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(requests.Session, "request", request)
    with pytest.raises(expected) as raised:
        _voyage_http.Client(api_key=PLACEHOLDER_KEY).embed(["a"], model="voyage-4")

    assert _is_transient(raised.value) is True


def test_the_client_refuses_its_own_retry_layer() -> None:
    with pytest.raises(ValueError, match="retry_with_backoff"):
        _voyage_http.Client(api_key=PLACEHOLDER_KEY, max_retries=2)


_CONSTRUCT_IN_A_FRESH_PROCESS = r'''
import base64, json, struct, sys
import requests
from requests.structures import CaseInsensitiveDict

class _Response:
    status_code = 200
    headers = CaseInsensitiveDict({"Content-Type": "application/json"})
    content = json.dumps({
        "data": [{"embedding": base64.b64encode(struct.pack("<2f", 1.0, 0.0)).decode()}],
        "usage": {"total_tokens": 1},
    }).encode()

requests.Session.request = lambda self, *a, **k: _Response()

from recall.embeddings import VoyageEmbedder
from recall.rerank import VoyageReranker

VoyageEmbedder(api_key="pa-CHANGEME-placeholder")
VoyageReranker(api_key="pa-CHANGEME-placeholder")._voyage_client()
print(json.dumps({name: name in sys.modules for name in ("voyageai", "torch", "transformers")}))
'''


def test_building_a_voyage_embedder_imports_neither_the_sdk_nor_torch() -> None:
    """The reason this client exists, asserted in a fresh interpreter.

    Invariant: constructing `VoyageEmbedder` (which makes its width probe) and the Voyage
    reranker's client imports none of `voyageai`, `torch` or `transformers`. The SDK's own
    `__init__` imports `voyageai.chunking`, and through it `langchain_text_splitters`,
    `transformers` and `torch` wherever those are installed; that was the 10 s and 778 MB per
    MCP server measured on VPS2.

    Red proof, recorded 2026-09-23 against the pre-fix `recall/embeddings.py` and
    `recall/rerank.py` from `22aabd1d` (P1), which built `voyageai.Client`: this test failed at
    `assert loaded["voyageai"] is False`. It needs the `voyage` extra (the embedder refuses
    without it), so it skips on CI's `dev` install.
    """
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("the voyage extra is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", _CONSTRUCT_IN_A_FRESH_PROCESS],
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])

    assert loaded["voyageai"] is False
    assert loaded["torch"] is False
    assert loaded["transformers"] is False
