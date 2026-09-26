"""A Voyage request refused for its size is split, and every request Voyage accepts is unchanged.

Audit item 5 (2026-09-24 C9 bug audit): `VoyageEmbedder` cuts a batch by count (128) only, while
Voyage also caps the tokens in one request. Measured 2026-09-26 with the voyage-code-4 tokenizer
on served C9's chunking: pasted CSV in CLBench reaches 240,869 tokens in one 128-text request,
and a refused request is a 400 that `retry_with_backoff` does not retry, so the Add failed every
time AML resent it. The fake provider below refuses any request over a character budget, which
stands in for the token cap.

Red proof, each against `VoyageEmbedder._embed_typed` in `recall/embeddings.py` with this file
unchanged (2026-09-26):

* the split tests, against the pre-fix `_embed_batch` (no split: the refusal propagates): both
  fail at their assertion that the refused batch was split;
* ``test_a_single_refused_text_still_raises``, mutation "drop the ``len(batch) < 2`` guard":
  fails with ``RecursionError`` where ``pytest.raises`` expects the provider's 400, because a
  one-text request is split into itself forever;
* ``test_only_a_400_is_split``, mutation "split on any status": fails, a 401 is resent in halves;
* ``test_an_accepted_request_is_sent_unchanged``, mutation "always split a batch in two before
  sending it": fails, the provider sees two requests of 64 instead of one of 128.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from recall._voyage_http import InvalidRequestError, VoyageError
from recall.embeddings import VoyageEmbedder

BUDGET_CHARS = 1_000


def vector(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)))]


def make_embedder(monkeypatch, *, refuse_status: int = 400, parallel: int = 1):
    requests: list[list[str]] = []

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, texts, *, model, input_type=None):
            texts = list(texts)
            requests.append(texts)
            if sum(len(text) for text in texts) > BUDGET_CHARS:
                error = InvalidRequestError if refuse_status == 400 else VoyageError
                raise error("request over the token cap", http_status=refuse_status)
            return SimpleNamespace(embeddings=[vector(text) for text in texts])

    monkeypatch.setitem(sys.modules, "voyageai", SimpleNamespace(Client=_Client))
    monkeypatch.setattr("recall._voyage_http.Client", _Client)
    embedder = VoyageEmbedder(api_key="secret", model="voyage-code-4", max_parallel_requests=parallel)
    requests.clear()
    return embedder, requests


def texts(count: int, width: int = 20) -> list[str]:
    return [f"t{index:04d}".ljust(width, "x") for index in range(count)]


def test_a_refused_batch_is_split_until_each_request_is_accepted(monkeypatch) -> None:
    embedder, requests = make_embedder(monkeypatch)
    batch = texts(128)  # 2,560 chars: refused at 128 and at 64, accepted at 32

    try:
        result = embedder.embed_passages(batch)
    except InvalidRequestError:
        pytest.fail("the refused batch was not split: the provider's 400 reached the caller")

    assert result == [vector(text) for text in batch]
    accepted = [r for r in requests if sum(map(len, r)) <= BUDGET_CHARS]
    assert [len(r) for r in accepted] == [32, 32, 32, 32]
    assert [text for r in accepted for text in r] == batch


def test_input_order_survives_a_split_among_parallel_requests(monkeypatch) -> None:
    embedder, _requests = make_embedder(monkeypatch, parallel=4)
    # Three count batches: 128 long texts (refused, split), 128 short (accepted), 44 long.
    batch = texts(128) + texts(128, width=5) + texts(44)

    try:
        result = embedder.embed_passages(batch)
    except InvalidRequestError:
        pytest.fail("the refused batch was not split: the provider's 400 reached the caller")

    assert result == [vector(text) for text in batch]


def test_a_single_refused_text_still_raises(monkeypatch) -> None:
    embedder, requests = make_embedder(monkeypatch)

    with pytest.raises(InvalidRequestError):
        embedder.embed_passages(["y" * (BUDGET_CHARS + 1)])
    assert len(requests) == 1


def test_only_a_400_is_split(monkeypatch) -> None:
    embedder, requests = make_embedder(monkeypatch, refuse_status=401)

    with pytest.raises(VoyageError):
        embedder.embed_passages(texts(128))
    assert len(requests) == 1


def test_an_accepted_request_is_sent_unchanged(monkeypatch) -> None:
    embedder, requests = make_embedder(monkeypatch)
    batch = texts(128, width=5)  # 640 chars: accepted whole

    assert embedder.embed_passages(batch) == [vector(text) for text in batch]
    assert requests == [batch]
