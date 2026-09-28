"""`recall.dashscope`: the native `text-embedding-v4` client W0 measures, and its registration.

Every test drives the real client against a recorded fake session, so nothing needs a key or the
network. What each test protects, and the mutation of the production code it was watched to fail
on before it was trusted (the red proof), is stated in its docstring.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math
from typing import Any

import pytest

from recall import dashscope
from recall.dashscope import (
    DEFAULT_BASE_URL,
    FALLBACK_TEXT_BYTES,
    MEMORY_RETRIEVAL_INSTRUCTION,
    DashScopeError,
    resolve_base_url,
)
from recall.embedding_registry import registered_profile
from recall.embeddings import embedding_profile_id

PLAIN = "dashscope-text-embedding-v4-1024-v1"
INSTRUCT = "dashscope-text-embedding-v4-1024-memory-instruct-v1"
#: A placeholder, never a real credential: the fake session below accepts anything.
PLACEHOLDER_KEY = "CHANGEME-placeholder-not-a-key"


def _unit(seed: int, dim: int = 1024) -> list[float]:
    raw = [math.sin(seed * 7.0 + i) + 1.5 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _Response:
    def __init__(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    """Answers every POST with one unit vector per text, derived from the text's length."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        reverse: bool = False,
        scale: float = 1.0,
        refuse_bytes_over: int | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.dim = dim
        self.reverse = reverse
        self.refuse: list[tuple[int, str]] = []
        self.scale = scale
        self.refuse_bytes_over = refuse_bytes_over

    def post(self, url: str, *, headers: dict[str, str], data: str, timeout: float) -> _Response:
        body = json.loads(data)
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        if self.refuse:
            status, code = self.refuse.pop(0)
            return _Response(status, {"code": code, "message": "refused"}, {"Retry-After": "0"})
        texts = body["input"]["texts"]
        if self.refuse_bytes_over is not None and any(
            len(t.encode("utf-8")) > self.refuse_bytes_over for t in texts
        ):
            return _Response(400, {"code": "InvalidParameter", "message": "too long"})
        shift = 1 if body["parameters"].get("instruct") else 0
        items = [
            {
                "text_index": i,
                "embedding": [x * self.scale for x in _unit(len(t) + 31 * shift, self.dim)],
            }
            for i, t in enumerate(texts)
        ]
        if self.reverse:
            items.reverse()
        return _Response(200, {"output": {"embeddings": items}, "usage": {"total_tokens": 1}})


def _build(profile_id: str, session: _Session, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(dashscope.requests, "Session", lambda: session)
    return registered_profile(profile_id).build(
        api_key=PLACEHOLDER_KEY, env={"DASHSCOPE_API_KEY": PLACEHOLDER_KEY}
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall.embeddings.time.sleep", lambda _s: None)


def test_query_and_passage_use_separate_encoders_and_the_instruction_rides_only_on_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: passages go as `text_type=document` with no instruction; queries go as
    `text_type=query`, carrying the profile's instruction inside `parameters` only for the
    instruct profile.

    Red proof: mutating `DashScopeEmbedder.embed_query` to pass ``instruct=None`` fails the
    ``instruct`` assertion; mutating `embed_passages` to send ``text_type="query"`` fails the
    passage assertion.
    """
    session = _Session()
    embedder = _build(INSTRUCT, session, monkeypatch)
    embedder.embed_passages(["a window", "another"])
    embedder.embed_query("when did we talk?")
    passage, query = session.calls[1]["body"], session.calls[2]["body"]
    assert passage["parameters"]["text_type"] == "document"
    assert "instruct" not in passage["parameters"]
    assert query["parameters"]["text_type"] == "query"
    assert query["parameters"].get("instruct") == MEMORY_RETRIEVAL_INSTRUCTION
    assert query["parameters"]["dimension"] == 1024
    assert query["model"] == "text-embedding-v4"
    assert session.calls[2]["headers"]["Authorization"] == f"Bearer {PLACEHOLDER_KEY}"
    assert session.calls[2]["url"] == DEFAULT_BASE_URL + dashscope.EMBEDDING_PATH

    plain_session = _Session()
    plain = _build(PLAIN, plain_session, monkeypatch)
    plain.embed_query("when did we talk?")
    assert "instruct" not in plain_session.calls[-1]["body"]["parameters"]


def test_batches_hold_at_most_ten_texts_and_vectors_follow_text_index_not_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: no request exceeds the documented 10 texts, and each vector is placed by its
    `text_index`, so a provider answering out of order cannot swap two windows' vectors.

    Red proof: mutating `_post` to append vectors in response order (ignoring `text_index`) fails
    the equality below, since the fake answers in reverse.
    """
    session = _Session(reverse=True)
    embedder = _build(PLAIN, session, monkeypatch)
    texts = [f"text {'x' * i}" for i in range(23)]
    vectors = embedder.embed_passages(texts)
    sizes = [len(call["body"]["input"]["texts"]) for call in session.calls[1:]]
    assert sizes == [10, 10, 3]
    assert vectors == [_unit(len(t)) for t in texts]


def test_a_refused_batch_with_an_oversized_text_is_resent_once_cut_to_the_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: when the endpoint refuses a batch holding a text over `FALLBACK_TEXT_BYTES`, the
    batch is resent once with those texts cut to the budget, and the cut is counted.

    Red proof: mutating the fallback in `_embed_batch` to resend ``batch`` instead of ``fitted``
    fails with the fake's second 400.
    """
    session = _Session(refuse_bytes_over=FALLBACK_TEXT_BYTES)
    embedder = _build(PLAIN, session, monkeypatch)
    long_text = "é" * FALLBACK_TEXT_BYTES  # 2 bytes each: twice the budget
    vectors = embedder.embed_passages(["short", long_text])
    resent = session.calls[-1]["body"]["input"]["texts"]
    assert resent[0] == "short"
    assert len(resent[1].encode("utf-8")) <= FALLBACK_TEXT_BYTES
    assert embedder.truncated_inputs == 1
    assert len(vectors) == 2


def test_a_refusal_with_no_oversized_text_is_raised_not_resent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: a 400 on ordinary texts (a bad key or parameter) fails at once.

    Red proof: removing the ``not oversized`` condition in `_embed_batch` makes the call resend
    and succeed, so no `DashScopeError` is raised.
    """
    session = _Session()
    embedder = _build(PLAIN, session, monkeypatch)
    session.refuse = [(400, "InvalidParameter")]
    with pytest.raises(DashScopeError) as caught:
        embedder.embed_passages(["short"])
    assert caught.value.http_status == 400


def test_a_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: a 429 carries `http_status`, so the shared retry policy resends it.

    Red proof: calling `_post` directly in `_embed_batch`, without `retry_with_backoff`, makes
    the first 429 raise and fails the call. (Dropping ``http_status`` alone does NOT go red: the
    retry classifier also finds "429" in the error text, so that mutation is harmless.)
    """
    session = _Session()
    embedder = _build(PLAIN, session, monkeypatch)
    session.refuse = [(429, "Throttling.RateQuota")]
    assert len(embedder.embed_passages(["short"])) == 1


def test_construction_refuses_a_wrong_width_or_a_vector_that_is_not_unit_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: the registered 1,024 width and l2 claim are held to the live endpoint.

    Red proof: deleting the norm check in `DashScopeEmbedder.__init__` lets the scaled session
    construct, failing the second ``pytest.raises``.
    """
    with pytest.raises(ValueError):
        _build(PLAIN, _Session(dim=768), monkeypatch)
    with pytest.raises(RuntimeError, match="norm"):
        _build(PLAIN, _Session(scale=2.0), monkeypatch)


def test_only_the_singapore_endpoints_are_accepted() -> None:
    """Invariant: the profile declares the Singapore region, so a base URL for another region or
    host is refused rather than silently producing vectors stored under this profile.

    Red proof: replacing the host condition in `resolve_base_url` with ``True`` accepts the
    Beijing host and fails the first ``pytest.raises``.
    """
    assert resolve_base_url(None) == DEFAULT_BASE_URL
    workspace = "https://ws-abc123.ap-southeast-1.maas.aliyuncs.com/api/v1/"
    assert resolve_base_url(workspace) == workspace.rstrip("/")
    for bad in (
        "https://ws-abc123.cn-beijing.maas.aliyuncs.com/api/v1",
        "https://evil.example.com/api/v1",
        "http://dashscope-intl.aliyuncs.com/api/v1",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ):
        with pytest.raises(ValueError):
            resolve_base_url(bad)


def test_registered_profiles_carry_their_identity_and_differ_only_in_the_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: the two v4 profiles are hosted and carried by the built embedder, and the
    instruct profile's identity names the exact instruction version `recall.dashscope` sends, so
    the instruction is key material: changing its text without a new version cannot reuse vectors.

    Red proof: registering the instruct profile under ``"memory-retrieval-v0"`` fails the
    instruction-version assertion. (Comparing the two fingerprints would guard nothing: they
    differ by profile id alone.)
    """
    from recall.dashscope import MEMORY_RETRIEVAL_INSTRUCTION_VERSION
    from recall.embedding_registry import _DASHSCOPE_SINGAPORE

    assert _DASHSCOPE_SINGAPORE == DEFAULT_BASE_URL
    plain, instruct = registered_profile(PLAIN), registered_profile(INSTRUCT)
    assert plain.hosted and instruct.hosted
    assert instruct.identity().instruction_version == MEMORY_RETRIEVAL_INSTRUCTION_VERSION
    assert instruct.identity().query_mode == "query-instruct"
    assert plain.identity().instruction_version == "none"
    assert embedding_profile_id(_build(INSTRUCT, _Session(), monkeypatch)) == INSTRUCT


def test_document_groups_are_embedded_in_one_pass_and_split_back_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: `embed_document_groups`, which C9's context guard calls, returns one vector per
    text in each group's own order.

    Red proof: mutating the split to ``vectors[start : start + len(group) - 1]`` drops a vector
    and fails the equality.
    """
    embedder = _build(PLAIN, _Session(), monkeypatch)
    groups = [["a", "bb"], ["ccc"], ["dddd", "eeeee", "ffffff"]]
    assert embedder.embed_document_groups(groups) == [
        [_unit(len(t)) for t in group] for group in groups
    ]


def test_the_instruction_check_reports_whether_the_endpoint_honoured_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: `instruction_changes_query_vector` compares the instructed and plain query.

    Red proof: mutating `embed_query_without_instruction` to send the instruction makes both
    vectors equal and fails the assertion.
    """
    assert _build(INSTRUCT, _Session(), monkeypatch).instruction_changes_query_vector() is True


def test_the_w0_variants_differ_from_c9_only_in_their_two_embedding_profiles() -> None:
    """Invariant: `C9_v4` and `C9_v4_instruct` measure the embedder alone.

    Red proof: building `C9_v4` with only ``embedding_profile`` replaced (context left on Voyage)
    fails the expected field set.
    """
    from recall_aml.variants import C9_VARIANT_NAME, variant

    c9 = variant(C9_VARIANT_NAME)
    for name, profile in (("C9_v4", PLAIN), ("C9_v4_instruct", INSTRUCT)):
        arm = variant(name)
        changed = {f.name for f in fields(c9) if getattr(c9, f.name) != getattr(arm, f.name)}
        assert changed == {"name", "embedding_profile", "context_embedding_profile"}
        assert arm.embedding_profile == arm.context_embedding_profile == profile


def test_the_hosted_app_passes_the_dashscope_settings_to_the_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: `_resolve_hosted_embedders` hands DASHSCOPE_* settings to the resolver.

    Red proof: deleting the DashScope loop in `_resolve_hosted_embedders` leaves the key out of
    the environment the resolver sees and fails the assertion.
    """
    import recall_aml.__main__ as main
    from recall_aml.variants import variant

    seen: list[dict[str, str]] = []

    def fake_resolve(profile_id: str, env: dict[str, str]) -> Any:
        seen.append(dict(env))
        return _build(profile_id, _Session(), monkeypatch)

    monkeypatch.setattr(main, "resolve_registered_embedder", fake_resolve)
    monkeypatch.setenv("DASHSCOPE_API_KEY", PLACEHOLDER_KEY)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)

    class _Settings:
        voyage_api_key = PLACEHOLDER_KEY
        embedding_lock_path = None
        embedding_cache_path = None

    main._resolve_hosted_embedders(_Settings(), variant("C9_v4"))  # type: ignore[arg-type]
    assert seen and all(env.get("DASHSCOPE_API_KEY") == PLACEHOLDER_KEY for env in seen)
    assert all(env.get("DASHSCOPE_BASE_URL") == DEFAULT_BASE_URL for env in seen)
