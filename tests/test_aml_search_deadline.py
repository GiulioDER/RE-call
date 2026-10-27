"""The Search time budget and the second batch of C9 Full 2 fixes (CCA run 20261020T152413Z).

Owner, 2026-10-20: "let's test the time limit, so we are sure it work and also don't create other
correlated problems". So besides one behaviour proof per fix, this file drives concurrent Searches
against stalled providers and checks what the stall must not touch: the fast Searches beside it,
the Adds running at the same time, and the Searches after it.

Every constant a test shortens is a module attribute read at call time (the budget, the reserve,
the rerank timeout, the image leg's waits), so the tests run in seconds with the production code.

Red proofs, run 2026-10-20 on the remote test host. On 43693e37 this file does not import (the
module is new), which is not behavioural proof, so each test was proved on a mutation of the fix,
failing as shown, then restored and green:

* M1 `DeadlineEmbedder.embed_query` ignoring the budget: the stalled-embedding test fails "DID NOT
  RAISE SearchDeadlineExceeded"; the 503 test fails ``(200, ..., None) == (503, ..., '5')``; the
  concurrency test fails ``assert False`` (stalled Searches answer instead of ending).
* M2 no rerank skip: ``assert [1] == []``.
* M3 image route without a deadline: ``assert 5.008 < 2.5``.
* M4 one multimodal client for queries: ``[180.0, 180.0] == [180.0, 20.0, 180.0]``.
* M5 `LazyMedia` reading every id: ``assert 60 <= 8``.
* M6 `graph_sidecar` from the variant: the graph store is asked for the tenant.
* M7 fixed probe text: the two starts' probe texts are equal.
* M8 an unreadable image re-raised: ``PIL.UnidentifiedImageError`` from the Add.
* M9 no blank-window fallback: ``ValidationError`` for the bounded blank part (the 422).
* M10 image-only query fused with its placeholder text: the text memory is served.
* M11 the guard remembering the placeholder: ``assert True is False``.
* M12 no ``reranked_after_placement``: ``(None, None) == (True, False)``.
* M13 the deadline not mapped in the app: ``(503, service_unavailable, None)`` instead of
  ``search_deadline`` with ``Retry-After: 5``.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from typing import Any

import pytest
from starlette.testclient import TestClient

import recall_aml.retrieval as retrieval_module
import recall_aml.search_deadline as deadline
import recall_aml.service as service_module
from recall.types import Chunk, ScoredChunk
from recall_aml.config import HostedSettings
from recall_aml.models import AddRequest, SearchRequest, TextContentPart
from recall_aml.multimodal import LazyMedia, render_preserved
from recall_aml.search_deadline import DeadlineEmbedder, SearchDeadlineExceeded, search_budget
from tests.test_aml_multimodal import _png_data_url
from tests.test_aml_multimodal_scope import _add_image_memory, _add_text_memory, _search
from tests.test_aml_specialist_fusion import _service

BUDGET = 2.0
RESERVE = 0.5


@pytest.fixture
def short_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2 s Search budget with a 0.5 s reserve, so a stage may wait at most 1.5 s."""
    monkeypatch.setattr(deadline, "SEARCH_BUDGET_SECONDS", BUDGET)
    monkeypatch.setattr(deadline, "RESPONSE_RESERVE_SECONDS", RESERVE)


class _SlowWhen:
    """Delegate to an embedder; sleep ``seconds`` first when the query holds ``marker``."""

    def __init__(self, inner: Any, marker: str, seconds: float) -> None:
        self._inner = inner
        self.marker = marker
        self.seconds = seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed_query(self, text: str) -> list[float]:
        if self.marker in text:
            time.sleep(self.seconds)
        return self._inner.embed_query(text)


def _slow_text_service(seconds: float):  # type: ignore[no-untyped-def]
    """C7 over the in-memory doubles, its query embeddings bounded as `build_app` bounds them."""
    service, repository, code, context, multimodal = _service("C7_routed_specialists")
    for retriever in (service._retriever, *service._specialist_retrievers.values()):
        retriever._embedder = DeadlineEmbedder(_SlowWhen(retriever._embedder, "SLOW", seconds))
    return service, repository


# --- the budget itself ------------------------------------------------------------------------


def test_outside_a_search_a_query_embedding_runs_inline_and_unbounded() -> None:
    """Invariant: an Add, a readiness probe or any call outside a Search keeps today's behaviour."""
    calls: list[str] = []

    class _Inner:
        def embed_query(self, text: str) -> list[float]:
            calls.append(threading.current_thread().name)
            return [1.0]

    assert DeadlineEmbedder(_Inner()).embed_query("q") == [1.0]
    assert calls == [threading.current_thread().name]


def test_a_stalled_query_embedding_ends_at_the_budget(short_budget: None) -> None:
    """Invariant: inside a Search, a query embedding waits at most budget less reserve.

    Defect caught: a stalled Voyage query embedding held the Search for up to three 60 s attempts
    plus Retry-After, past the ~100 s cut.
    """

    class _Stalled:
        def embed_query(self, text: str) -> list[float]:
            time.sleep(5)
            return [1.0]

    started = time.monotonic()
    with search_budget(), pytest.raises(SearchDeadlineExceeded):
        DeadlineEmbedder(_Stalled()).embed_query("q")

    assert time.monotonic() - started < BUDGET - RESERVE + 0.5


def test_a_slow_search_answers_503_with_a_short_retry_after(short_budget: None) -> None:
    """Invariant: a Search whose query embedding runs out of budget answers 503, Retry-After 5,
    before the budget ends, never a hang to the proxy cut.
    """
    from recall_aml.app import create_app

    service, _ = _slow_text_service(5.0)
    _add_text_memory(service, "deadline-user")
    client = TestClient(create_app(HostedSettings("postgresql://unused", "secret", "abc123"), service))

    started = time.monotonic()
    response = client.post(
        "/v1/search",
        headers={"X-Api-Key": "secret"},
        json={"query": "SLOW parking receipt", "user_id": "deadline-user"},
    )

    assert (response.status_code, response.json(), response.headers.get("retry-after")) == (
        503,
        {"error": "search_deadline"},
        "5",
    )
    assert time.monotonic() - started < BUDGET + 0.5


def test_a_rerank_with_too_little_budget_left_serves_the_fused_order(
    short_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: when less than the rerank timeout is left, the reranker is not called and the
    fused order is served (reported as a fallback), so the rerank cannot push a Search past the
    budget.
    """
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    monkeypatch.setattr(retrieval_module, "RERANK_TIMEOUT_SECONDS", 15.0)
    calls: list[int] = []

    class _Reranker:
        def rerank(self, query: str, hits: list[Any]) -> list[Any]:
            calls.append(1)
            return list(reversed(hits))

    service, _, _, _, _ = _service("C7_routed_specialists")
    for retriever in (service._retriever, *service._specialist_retrievers.values()):
        retriever._reranker = _Reranker()
    _add_text_memory(service, "scope-user")

    response = _search(service)

    assert response.data, "precondition: the Search serves items"
    assert calls == []


def test_a_rerank_with_budget_left_still_runs(short_budget: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: with the rerank timeout inside the remaining budget, the reranker runs."""
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    monkeypatch.setattr(retrieval_module, "MIN_RERANK_SECONDS", 0.1)
    calls: list[int] = []

    class _Reranker:
        def rerank(self, query: str, hits: list[Any]) -> list[Any]:
            calls.append(1)
            return hits

    service, _, _, _, _ = _service("C7_routed_specialists")
    for retriever in (service._retriever, *service._specialist_retrievers.values()):
        retriever._reranker = _Reranker()
    _add_text_memory(service, "scope-user")

    _search(service)

    assert calls == [1]


# --- the image legs ---------------------------------------------------------------------------


class _StalledMultimodal:
    """A multimodal embedder whose query embedding stalls (documents still work)."""

    dim = 3
    profile = "test-multimodal"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def embed_documents(self, inputs):  # type: ignore[no-untyped-def]
        return [[0.0, 1.0, 0.0] for _ in inputs]

    def embed_query(self, value):  # type: ignore[no-untyped-def]
        time.sleep(self.seconds)
        return [0.0, 1.0, 0.0]


def test_a_stalled_image_query_serves_the_text_ranking(short_budget: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """U-48. Invariant: when the query carries an image and the image leg stalls, the Search
    answers within the budget with the text ranking instead of failing.

    Defect caught: on the image route the leg had no deadline and no fallback, so a slow or
    failing multimodal provider failed the whole Search (503) or held it to the 180 s timeout.
    """
    monkeypatch.setattr(service_module, "VISUAL_ROUTE_LEG_SECONDS", 1.0)
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    service._multimodal_embedder = _StalledMultimodal(5.0)
    query = [
        {"type": "text", "text": "the parking receipt"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"query")}},
    ]

    async def timed() -> tuple[float, Any]:
        # Timed inside the loop: `asyncio.run` itself then waits for the abandoned worker thread
        # at shutdown, which a long-lived server loop never does.
        started = time.monotonic()
        result = await service.search(
            SearchRequest.model_validate({"query": query, "user_id": "scope-user"})
        )
        return time.monotonic() - started, result

    elapsed, response = asyncio.run(timed())

    assert elapsed < BUDGET + 0.5
    assert response.data, "the text ranking is served"
    assert response.visual_leg is False


def test_the_image_query_client_has_its_own_short_timeout() -> None:
    """U-48/PERF-002. Invariant: a Search's image query uses a client with ``query_timeout``,
    documents keep the long one, so an abandoned query ends near its wait instead of at 180 s.
    """
    import sys
    import types

    made: list[float] = []
    used: list[tuple[str, float]] = []

    class _Client:
        def __init__(self, *, api_key: str, max_retries: int, timeout: float) -> None:
            made.append(timeout)
            self.timeout = timeout

        def multimodal_embed(self, inputs, *, model, input_type, truncation):  # type: ignore[no-untyped-def]
            used.append((input_type, self.timeout))
            return types.SimpleNamespace(embeddings=[[0.0] * 1024 for _ in inputs])

    fake = types.ModuleType("voyageai")
    fake.Client = _Client  # type: ignore[attr-defined]
    saved = sys.modules.get("voyageai")
    sys.modules["voyageai"] = fake
    try:
        from recall_aml.multimodal import VoyageMultimodalEmbedder

        bounded = VoyageMultimodalEmbedder(
            "k", query_timeout=20.0, token_counter=lambda texts: [0] * len(texts)
        )
        VoyageMultimodalEmbedder("k", token_counter=lambda texts: [0] * len(texts))
        bounded.embed_query("a query")
        bounded.embed_documents([{"content": [{"type": "text", "text": "a document"}]}])
    finally:
        if saved is None:
            del sys.modules["voyageai"]
        else:
            sys.modules["voyageai"] = saved

    assert made == [180.0, 20.0, 180.0]
    # CCA V-14: the query is SENT on the short client, documents on the long one. Red proof: the
    # mutation `self._client.multimodal_embed` in `embed_query` sends the query at 180.0.
    assert used == [("query", 20.0), ("document", 180.0)]


# --- correlated problems under concurrency ----------------------------------------------------


def _run_as_served(main: Any) -> Any:
    """Run ``main()`` on the default executor `create_app`'s lifespan installs (`executor_workers`).

    A bare `asyncio.run` gets asyncio's own ``min(32, cpu + 4)`` threads instead: 8 on CI's 4-vCPU
    runner, fewer than this burst's 20 requests, so the fast Searches queued behind the stalled
    ones and started with their budget spent. Master CI failed that way at 1ecb50db5 (run
    40534338969) while hosts with more cores passed.
    """
    from concurrent.futures import ThreadPoolExecutor

    from recall_aml.app import executor_workers

    workers = executor_workers(HostedSettings("postgresql://unused", "secret", "abc123"))

    async def served() -> Any:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recall-aml")
        )
        return await main()

    return asyncio.run(served())


def _outcomes(results: list[tuple[float, Any]]) -> str:
    """Each result as its exception's name or its number of memories, for an assertion message.

    A string, because pytest shortens a list in an assertion message and hides the failing entry.
    """
    return ", ".join(
        type(result).__name__ if isinstance(result, Exception) else f"{len(result.data)} memories"
        for _, result in results
    )


def test_stalled_searches_do_not_slow_the_rest_or_outlive_the_stall(short_budget: None) -> None:
    """Invariant (the owner's "no correlated problems"): with 8 Searches stalled on their query
    embedding and 8 fast ones, plus 4 Adds, all at once:

    * each stalled Search ends with ``SearchDeadlineExceeded`` within the budget;
    * every fast Search and every Add completes, inside the budget (the stalled embeddings run in
      the query pool, so they never take a query-pool slot from a fast Search);
    * once the stall is over, the same users' Searches succeed (nothing is left behind).

    A stalled Search still holds one shared-executor thread while it waits on its embedding, so
    the burst runs on the executor production installs (`_run_as_served`), not on asyncio's
    CPU-sized default.

    Red proof, run 2026-10-27 in CI: with ``QUERY_EMBED_THREADS = 1`` in
    `recall_aml/search_deadline.py`, the fast Searches' embeddings queue behind the stalled ones
    and this test fails at the fast-Search assertion with the fast results reported as
    ``SearchDeadlineExceeded`` (run 40539781611, red in all three suite jobs). Restored, green
    (run 40539763161). On asyncio's default executor the same test failed one of three suite jobs
    (run 40539772152), so on a 4-vCPU runner it is timing-dependent, not deterministic.
    """
    service, _ = _slow_text_service(4.0)
    users = [f"user-{index}" for index in range(8)]
    for user in users:
        _add_text_memory(service, user)

    async def timed(coroutine: Any) -> tuple[float, Any]:
        started = time.monotonic()
        try:
            result = await coroutine
        except Exception as exc:  # noqa: BLE001 - the test classifies every outcome
            result = exc
        return time.monotonic() - started, result

    def add(index: int) -> Any:
        return service.add(
            AddRequest.model_validate(
                {
                    "request_id": f"stress-add-{index}",
                    "user_id": f"adder-{index}",
                    "session_id": "stress",
                    "messages": [{"role": "user", "content": f"The locker code is {index}."}],
                }
            )
        )

    def search(user: str, text: str) -> Any:
        return service.search(SearchRequest.model_validate({"query": text, "user_id": user}))

    async def burst() -> list[tuple[float, Any]]:
        return list(
            await asyncio.gather(
                *(timed(search(user, "SLOW parking receipt")) for user in users),
                *(timed(search(user, "parking receipt")) for user in users),
                *(timed(add(index)) for index in range(4)),
            )
        )

    results = _run_as_served(burst)
    stalled, fast, adds = results[:8], results[8:16], results[16:]

    assert all(isinstance(result, SearchDeadlineExceeded) for _, result in stalled), _outcomes(stalled)
    assert max(elapsed for elapsed, _ in stalled) < BUDGET + 0.5
    assert all(not isinstance(result, Exception) and result.data for _, result in fast), _outcomes(
        fast
    )
    assert max(elapsed for elapsed, _ in fast) < BUDGET
    assert all(not isinstance(result, Exception) for _, result in adds)
    assert max(elapsed for elapsed, _ in adds) < BUDGET

    async def after_stall() -> list[tuple[float, Any]]:
        return list(await asyncio.gather(*(timed(search(user, "parking receipt")) for user in users)))

    after = _run_as_served(after_stall)
    assert all(not isinstance(result, Exception) and result.data for _, result in after), _outcomes(
        after
    )


# --- the media budget (U-23) ------------------------------------------------------------------


def _image_parent(index: int) -> tuple[ScoredChunk, Chunk, Chunk]:
    parent_id = f"raw_{index:03d}"
    media_id = f"media_{index:03d}"
    hit = ScoredChunk(
        Chunk(id=parent_id, source="s", text="t", metadata={"multimodal_parent_id": parent_id}),
        1.0 - index / 1000,
    )
    parent = Chunk(
        id=parent_id,
        source="s",
        text="t",
        metadata={
            "multimodal_manifest": [
                {"type": "text", "text": f"photo {index}"},
                {"type": "image_ref", "media_id": media_id},
            ],
            "multimodal_scalar": False,
        },
    )
    media = Chunk(
        id=media_id,
        source="m",
        text="png",
        metadata={"record_type": "media", "data_url": _png_data_url(str(index).encode())},
    )
    return hit, parent, media


def test_an_image_search_reads_only_the_media_it_can_serve() -> None:
    """U-23. Invariant: with 60 image candidates and top_k 3, the renderer's output is exactly the
    eager one, and only one batch of media rows is read.

    Defect caught: every candidate parent's images (up to about 200, each up to 13 MB of base64)
    were read and decoded before the renderer stopped at top_k and the 30 MiB response budget.
    """
    triples = [_image_parent(index) for index in range(60)]
    hits = [hit for hit, _, _ in triples]
    primary = {parent.id: parent for _, parent, _ in triples}
    media = {row.id: row for _, _, row in triples}
    reads: list[list[str]] = []

    def fetch(ids: list[str]) -> dict[str, Chunk]:
        reads.append(list(ids))
        return {media_id: media[media_id] for media_id in ids if media_id in media}

    ordered = [row.id for _, _, row in triples]
    lazy = render_preserved(
        hits, primary_by_id=primary, media_by_id=LazyMedia(ordered, fetch, batch=8), top_k=3
    )
    eager = render_preserved(hits, primary_by_id=primary, media_by_id=media, top_k=3)

    assert [item.model_dump() for item in lazy] == [item.model_dump() for item in eager]
    assert len(lazy) == 3
    assert sum(len(batch) for batch in reads) <= 8


def test_lazy_media_skips_exactly_what_eager_media_skips() -> None:
    """U-23 control: a missing media row is skipped by the lazy reader as by the eager one, and
    the renderer keeps going down the ranking (rows read on demand, never assumed absent)."""
    triples = [_image_parent(index) for index in range(20)]
    hits = [hit for hit, _, _ in triples]
    primary = {parent.id: parent for _, parent, _ in triples}
    media = {row.id: row for index, (_, _, row) in enumerate(triples) if index % 3}

    lazy = render_preserved(
        hits,
        primary_by_id=primary,
        media_by_id=LazyMedia(
            [row.id for _, _, row in triples],
            lambda ids: {media_id: media[media_id] for media_id in ids if media_id in media},
            batch=2,
        ),
        top_k=10,
    )
    eager = render_preserved(hits, primary_by_id=primary, media_by_id=media, top_k=10)

    assert [item.id for item in lazy] == [item.id for item in eager]
    assert len(eager) == 10


# --- the remaining deferred fixes -------------------------------------------------------------


def test_with_the_compile_off_a_search_skips_the_inert_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-07. Invariant: with ``RECALL_AML_COMPILER=0`` the graph sidecar, which can only pass the
    ranking through, is neither queried nor reported active.

    Defect caught: it ran on every Search (a fingerprint, a dense and a sparse query, and a second
    query embedding on the Context route) and ``/version`` reported it active.
    """
    from tests.test_aml_window_compose import _c9_service

    monkeypatch.setenv("RECALL_AML_COMPILER", "0")
    service = _c9_service()
    asked: list[str] = []
    original = service._repository.graph_store

    def graph_store(tenant: str) -> Any:
        asked.append(tenant)
        return original(tenant)

    monkeypatch.setattr(service._repository, "graph_store", graph_store)

    response = asyncio.run(
        service.search(SearchRequest.model_validate({"query": "fountain pens", "user_id": "u"}))
    )

    assert response is not None
    assert asked == []
    assert service.graph_sidecar is False
    assert service.active_components["graph_sidecar"] is False


def test_with_the_compile_on_a_search_still_consults_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-07 control: the served variant's graph is consulted whenever the compile serves."""
    from tests.test_aml_window_compose import _c9_service

    monkeypatch.delenv("RECALL_AML_COMPILER", raising=False)
    service = _c9_service()

    assert service.graph_sidecar is True
    assert service.active_components["graph_sidecar"] is True


def test_each_start_probes_the_embedders_with_a_fresh_text() -> None:
    """U-18. Invariant: two starts send different probe texts, so a cached vector cannot answer.

    Defect caught: the fixed probe texts were answered from the on-disk query cache on every start
    after the first, so a dead model or entitlement surfaced only at the first Search.
    """
    from recall_aml.readiness import verify_model_readiness
    from recall_aml.variants import variant
    from tests.test_aml_hosted import FakeEmbedder

    seen: list[str] = []

    class _Recording(FakeEmbedder):  # type: ignore[misc]
        def embed_query(self, text: str) -> list[float]:
            seen.append(text)
            return [0.0] * self.dim

    for _ in range(2):
        verify_model_readiness(
            embedder=_Recording(), compiler=None, reranker=None,  # type: ignore[arg-type]
            behavior=variant("A0_raw"),
        )

    assert len(seen) == 2 and seen[0] != seen[1]


def test_one_unreadable_image_no_longer_loses_the_rest_of_the_add() -> None:
    """U-47. Invariant: an Add with text and an image Pillow cannot read stores its text.

    Defect caught: the unreadable image raised for the whole Add, so every message of it was
    lost (first as 32 retried 503s, then as one 422).
    """
    import base64

    service, repository, _, _, _ = _service("C7_routed_specialists")
    corrupt = "data:image/png;base64," + base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"not an image body" * 8
    ).decode("ascii")

    result = asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": "mixed",
                    "user_id": "mixed-user",
                    "session_id": "s",
                    "messages": [
                        {"role": "user", "content": "My locker code is 4417."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "and here is the photo"},
                                {"type": "image_url", "image_url": {"url": corrupt}},
                            ],
                        },
                    ],
                }
            )
        )
    )

    assert result.status == "stored"
    response = _search(service, "mixed-user", "locker code")
    assert any("4417" in str(item.content) for item in response.data)


def test_a_part_list_query_whose_kept_ends_are_blank_is_still_served() -> None:
    """U-21 edge: a 20,000+ character text part blank at both kept ends is bounded on its
    stripped text instead of becoming a blank part (a 422)."""
    text = " " * 12_000 + "where is my locker" + " " * 12_000
    request = SearchRequest.model_validate(
        {"user_id": "u", "query": [{"type": "text", "text": text}]}
    )

    parts = [part for part in request.query if isinstance(part, TextContentPart)]
    assert parts and "where is my locker" in parts[0].text


def test_an_image_only_query_is_ranked_by_its_image_and_not_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-17. Invariant: a query with no text is ranked by the image leg alone when that leg found
    images, and the feedback guard does not record its placeholder as a question.

    Defect caught: the text half of the fusion ranked "[image evidence]", which every image-only
    memory also stores, at equal weight with the image leg; and the guard remembered it.
    """
    monkeypatch.setenv("RECALL_AML_FEEDBACK_GUARD", "1")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(service, "scope-user")
    _add_image_memory(service)
    query = [{"type": "image_url", "image_url": {"url": _png_data_url(b"query")}}]

    response = asyncio.run(
        service.search(SearchRequest.model_validate({"query": query, "user_id": "scope-user"}))
    )

    assert response.visual_leg
    assert all("parking receipt total" not in str(item.content) for item in response.data)
    assert service._asked_questions.asked(service_module.tenant_for("scope-user"), "[image evidence]") is False


def test_version_says_whether_the_reranker_follows_the_atomic_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-03. Invariant: ``/version`` reports that the fixed top five no longer holds when the
    reranker reorders after the atomic placement."""
    from tests.test_aml_window_compose import _c9_service

    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    on = _c9_service().atomic_rescue_profile
    monkeypatch.delenv("RECALL_AML_RERANKER")
    off = _c9_service().atomic_rescue_profile

    assert (on.get("reranked_after_placement"), off.get("reranked_after_placement")) == (True, False)


# --- second CCA batch (run 20261020T212405Z) ---------------------------------------------------


class _FlakyMultimodal:
    """A multimodal embedder whose query embedding raises ``errors`` first, then answers."""

    dim = 3
    profile = "test-multimodal"

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    def embed_documents(self, inputs):  # type: ignore[no-untyped-def]
        return [[0.0, 1.0, 0.0] for _ in inputs]

    def embed_query(self, value):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [0.0, 1.0, 0.0]


def _status(code: int) -> Exception:
    error = RuntimeError(f"HTTP {code}")
    error.http_status = code  # type: ignore[attr-defined]
    return error


IMAGE_ONLY = [{"type": "image_url", "image_url": {"url": _png_data_url(b"query")}}]
TEXT_AND_IMAGE = [{"type": "text", "text": "the parking receipt"}, *IMAGE_ONLY]


def _image_search(service, query):  # type: ignore[no-untyped-def]
    return asyncio.run(
        service.search(SearchRequest.model_validate({"query": query, "user_id": "scope-user"}))
    )


def test_a_transient_image_query_error_is_retried_inside_the_leg() -> None:
    """CCA V-03. Invariant: on the image route one transient 429 is retried, and the Search is
    served with its image leg.

    Defect caught: the query client makes one attempt and the leg's broad catch turned a single
    429 into a permanent answer without images. Red proof: without the retry the leg makes one
    call and the response reports ``visual_leg`` False.
    """
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    flaky = _FlakyMultimodal(_status(429))
    service._multimodal_embedder = flaky

    response = _image_search(service, TEXT_AND_IMAGE)

    assert (flaky.calls, response.visual_leg) == (2, True)


def test_an_image_only_query_whose_image_leg_fails_is_not_served_its_placeholder() -> None:
    """CCA V-02. Invariant: a query with no text whose image leg fails raises (a retryable 503),
    because its "text ranking" is only the "[image evidence]" placeholder's arbitrary order.

    Red proof: the first fail-open caught everything and served that order with a 200.
    """
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    service._multimodal_embedder = _FlakyMultimodal(_status(503), _status(503))

    with pytest.raises(RuntimeError, match="HTTP 503"):
        _image_search(service, IMAGE_ONLY)


def test_a_query_with_text_still_falls_open_when_its_image_leg_fails() -> None:
    """CCA V-02 control: the owner's U-48 fail-open stands for a query that has text."""
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    service._multimodal_embedder = _FlakyMultimodal(_status(503), _status(503))

    response = _image_search(service, TEXT_AND_IMAGE)

    assert response.data and response.visual_leg is False


def test_an_unreadable_query_image_is_answered_422_again() -> None:
    """CCA V-03. Invariant: a query image Pillow cannot read is the permanent 422, as before the
    fail-open; it never reaches a text-only answer.

    Red proof: the first fail-open swallowed the Pillow error and answered 200.
    """
    from recall_aml.app import create_app
    from recall_aml.multimodal import to_voyage_input

    class _Decoding(_FlakyMultimodal):
        """Decodes the query as `VoyageMultimodalEmbedder.embed_query` does (strict fit)."""

        def embed_query(self, value):  # type: ignore[no-untyped-def]
            to_voyage_input(value)
            return super().embed_query(value)

    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    service._multimodal_embedder = _Decoding()
    corrupt = "data:image/png;base64," + base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"not an image body" * 8
    ).decode("ascii")
    client = TestClient(create_app(HostedSettings("postgresql://unused", "secret", "abc123"), service))

    response = client.post(
        "/v1/search",
        headers={"X-Api-Key": "secret"},
        json={
            "user_id": "scope-user",
            "query": [
                {"type": "text", "text": "the parking receipt"},
                {"type": "image_url", "image_url": {"url": corrupt}},
            ],
        },
    )

    assert response.status_code == 422


def test_a_stalled_rerank_ends_at_the_budget_and_serves_the_fused_order(
    short_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CCA V-05. Invariant: a rerank that does not answer is abandoned within the Search budget
    (the client's connect retries can multiply its own timeout), and the fused order is served.

    Red proof: the rerank ran inline with only its client timeout, so this Search waited the full
    5 s stall.
    """
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    monkeypatch.setattr(retrieval_module, "MIN_RERANK_SECONDS", 0.1)

    class _Stalled:
        def rerank(self, query: str, hits: list[Any]) -> list[Any]:
            time.sleep(5)
            return hits

    service, _, _, _, _ = _service("C7_routed_specialists")
    for retriever in (service._retriever, *service._specialist_retrievers.values()):
        retriever._reranker = _Stalled()
    _add_text_memory(service, "scope-user")

    async def timed() -> tuple[float, Any]:
        started = time.monotonic()
        result = await service.search(
            SearchRequest.model_validate({"query": "parking receipt", "user_id": "scope-user"})
        )
        return time.monotonic() - started, result

    elapsed, response = asyncio.run(timed())

    assert elapsed < BUDGET + 0.5
    assert response.data


def test_build_app_bounds_every_retriever_and_the_image_query_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """CCA V-13. Invariant: the served app wraps every retriever's embedder in `DeadlineEmbedder`
    and builds the multimodal embedder with the short query timeout.

    Red proof: removing either wrapper or the ``query_timeout`` argument from `build_app` fails
    the matching assertion (no other test reaches that wiring).
    """
    import recall_aml.__main__ as hosted_main
    from recall_aml.config import VISUAL_QUERY_TIMEOUT_SECONDS
    from tests.test_aml_hosted import FakeEmbedder

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    class _Pool:
        def close(self) -> None:
            pass

    class _Store:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def check_schema(self) -> None:
            pass

    def _multimodal(*_a: Any, **kwargs: Any) -> object:
        seen["multimodal_kwargs"] = kwargs
        return object()

    def _service_ctor(*args: Any, **kwargs: Any) -> None:
        seen["retriever"] = args[2]
        seen["specialists"] = kwargs.get("specialist_retrievers", {})
        raise _Stop

    readiness = {
        "embedder_ready": True,
        "compiler_ready": True,
        "reranker_ready": True,
        "multimodal_ready": True,
        "specialist:voyage-context-4-v1": True,
    }
    monkeypatch.setattr(
        hosted_main,
        "_resolve_hosted_embedders",
        lambda *_: (FakeEmbedder(), {"voyage-context-4-v1": FakeEmbedder()}),
    )
    monkeypatch.setattr(hosted_main, "SharedPool", lambda *_a, **_k: _Pool())
    monkeypatch.setattr(hosted_main, "PgVectorStore", _Store)
    monkeypatch.setattr(hosted_main, "PgHostedRepository", lambda *_a, **_k: object())
    monkeypatch.setattr(hosted_main, "VoyageReranker", lambda **_k: object())
    monkeypatch.setattr(hosted_main, "VoyageMultimodalEmbedder", _multimodal)
    monkeypatch.setattr(hosted_main, "verify_model_readiness", lambda **_k: dict(readiness))
    monkeypatch.setattr(hosted_main, "HostedService", _service_ctor)
    monkeypatch.setenv("RECALL_AML_COMPILER", "0")
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name="C9_routed_specialists_grounded_graph_atomic",
        voyage_api_key="voyage-key",
        embedding_lock_path=tmp_path / "embed.lock",
    )

    with pytest.raises(_Stop):
        hosted_main.build_app(settings)

    assert isinstance(seen["retriever"]._embedder, DeadlineEmbedder)
    assert seen["specialists"] and all(
        isinstance(retriever._embedder, DeadlineEmbedder)
        for retriever in seen["specialists"].values()
    )
    assert seen["multimodal_kwargs"].get("query_timeout") == VISUAL_QUERY_TIMEOUT_SECONDS


def test_a_long_retry_after_ends_the_image_leg_instead_of_outliving_it(short_budget: None) -> None:
    """CCA V-03 (regression diff of the fix). Invariant: when a 429's Retry-After plus one more
    attempt does not fit in the image leg's wait, the leg ends at once (the text ranking is served)
    instead of sleeping in a shared executor thread and resending after the Search has answered.

    Red proof: the first version bounded the backoff by the whole Search budget, so the worker
    slept the 5 s Retry-After while the Search waited out the leg (about 1.5 s here), then resent.
    """
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    rate_limited = _status(429)
    rate_limited.headers = {"retry-after": "5"}  # type: ignore[attr-defined]
    flaky = _FlakyMultimodal(rate_limited)
    service._multimodal_embedder = flaky

    async def timed() -> tuple[float, Any]:
        started = time.monotonic()
        result = await service.search(
            SearchRequest.model_validate({"query": TEXT_AND_IMAGE, "user_id": "scope-user"})
        )
        return time.monotonic() - started, result

    elapsed, response = asyncio.run(timed())

    assert elapsed < 1.0
    assert (flaky.calls, response.visual_leg) == (1, False)


@pytest.mark.parametrize(("retry_after", "calls", "visual_leg"), [("1", 2, True), ("30", 1, False)])
def test_the_image_leg_retries_a_retry_after_only_when_it_fits(
    monkeypatch: pytest.MonkeyPatch, retry_after: str, calls: int, visual_leg: bool
) -> None:
    """CCA V-03 (pass 2). Invariant: inside a 25 s leg, a 429 asking for 1 s is waited out and
    retried, one asking for 30 s ends the leg at once.

    Red proof: `backoff_before` checking only the next attempt (ignoring the backoff's length)
    sleeps the 30 s and holds the Search for the whole 25 s leg (the second case's time bound).
    """
    monkeypatch.setattr(service_module, "VISUAL_QUERY_TIMEOUT_SECONDS", 0.1)
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    rate_limited = _status(429)
    rate_limited.headers = {"retry-after": retry_after}  # type: ignore[attr-defined]
    flaky = _FlakyMultimodal(rate_limited)
    service._multimodal_embedder = flaky

    async def timed() -> tuple[float, Any]:
        started = time.monotonic()
        result = await service.search(
            SearchRequest.model_validate({"query": TEXT_AND_IMAGE, "user_id": "scope-user"})
        )
        return time.monotonic() - started, result

    elapsed, response = asyncio.run(timed())

    assert (flaky.calls, response.visual_leg) == (calls, visual_leg)
    assert elapsed < 3.0
