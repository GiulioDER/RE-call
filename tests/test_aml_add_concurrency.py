"""C9's Add path overlaps independent work and keeps CPU work off the event loop.

Three changes, all performance only, each proved here to leave what an Add stores unchanged:

* the embeddings that do not depend on the compile (the raw windows stored beside the graph
  sidecar, and the atomic views in both scopes) are computed while the compile runs;
* request decoding and chunk building run in worker threads, not on the event loop;
* the default executor is sized from the configured Add and Search concurrency.

The repository here is the production ``PgHostedRepository`` over an in-memory store that
records every upsert, so the persist methods, their embedding calls and their argument order are
the real ones. The code embedder's vectors depend on each text alone, and the Context embedder's
on the whole request, as a contextual provider's do, so a vector computed from the wrong texts,
the wrong grouping or the wrong embedder shows up as a different stored row.

Red proof, 2026-09-26, each by mutating the named production line with this file unchanged,
watching the named assertion fail, then restoring it:

* ``test_c9_add_stores_identical_rows_with_and_without_the_prefetch``: in
  ``HostedService._start_embedding_prefetch`` (``recall_aml/service.py``), computing
  ``prefetch.specialist_view_vectors`` with ``embed_texts(None, texts)`` instead of
  ``embed_texts(context_profile, texts)`` failed ``assert prefetched.writes == inline.writes``
  on the Context view rows' vectors.
* ``test_a_failed_prefetch_leaves_the_add_embedding_inline``: deleting the
  ``except Exception as exc:`` clause that ends ``compute`` in ``_start_embedding_prefetch`` (so
  the provider error escaped the prefetch task) failed the Add with ``TimeoutError: provider
  timeout`` raised from ``_prefetched``, in place of the equal response and rows.
* ``test_c9_add_embeds_independent_rows_while_the_compile_runs``: making
  ``_start_embedding_prefetch`` return None at its top failed
  ``assert compiler.embeddings_seen_during_compile == 3`` (``0 == 3``).
* ``test_c9_add_builds_and_links_chunks_off_the_event_loop``: replacing
  ``await asyncio.to_thread(attach_grounded_relations, normalized_request, chunks)`` in
  ``HostedService._compile_and_persist`` with a direct call failed
  ``assert on_loop == []`` with ``['attach_grounded_relations']``.
* ``test_a_large_body_is_decoded_off_the_event_loop``: replacing the ``asyncio.to_thread``
  branch of ``_parse`` (``recall_aml/app.py``) with ``return _decode_model(body, model)`` failed
  ``assert decoded_on_loop == [False]`` with ``[True]``.
* ``test_the_default_executor_admits_two_threads_per_admitted_request``: deleting the
  ``set_default_executor`` call in ``create_app``'s lifespan failed
  ``assert probe.json() == {"all_met": True}`` with ``{'all_met': False}``: asyncio's default of at
  most 32 threads cannot hold the 36 waiting parties.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import recall_aml.app as app_module
import recall_aml.service as service_module
from recall_aml.app import create_app, executor_workers
from recall_aml.config import HostedSettings
from recall_aml.models import AddRequest, CodingMemoryRecord, Message
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import variant
from tests.test_aml_hosted import make_service

C9 = "C9_routed_specialists_grounded_graph_atomic"
SESSION_TEXT = " ".join(
    f"Step {i}: the worker in src/queue_{i % 7}.py raised TimeoutError, so I moved the retry into "
    f"QueueClient.send and pytest tests/test_queue_{i}.py passed afterwards."
    for i in range(40)
)


class RecordingStore:
    """The slice of ``PgVectorStore`` the Add path writes through, recording every upsert."""

    def __init__(self, writes: list[tuple[Any, ...]], tenant: str = "base") -> None:
        self._writes = writes
        self._tenant = tenant

    def for_tenant(self, tenant: str) -> "RecordingStore":
        return RecordingStore(self._writes, tenant)

    def upsert(self, chunks: list[Any], vectors: list[list[float]]) -> int:
        self._writes.append(
            (
                self._tenant,
                [(chunk.id, chunk.source, chunk.text, chunk.metadata) for chunk in chunks],
                [list(vector) for vector in vectors],
            )
        )
        return len(chunks)


class CodeEmbedder:
    """A non-contextual passage embedder: each vector depends on its own text only."""

    dim = 3
    name = "code"

    def __init__(self, calls: list[tuple[str, tuple[str, ...]]], guard: threading.Lock) -> None:
        self._calls = calls
        self._guard = guard

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        with self._guard:
            self._calls.append((self.name, tuple(texts)))
        return [[float(len(text)), float(sum(map(ord, text)) % 997), 1.0] for text in texts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)


class ContextEmbedder(CodeEmbedder):
    """A contextual embedder: each vector also depends on the whole request it arrived in."""

    name = "context"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        with self._guard:
            self._calls.append((self.name, tuple(texts)))
        group = float(sum(len(text) for text in texts) % 991)
        return [[float(len(text)), float(len(texts)), group] for text in texts]


class RecordingRepository(PgHostedRepository):
    """The production repository, with receipts and locks kept in memory."""

    distributed_locks = False

    def __init__(self, *, prefetch: bool = True) -> None:
        self.writes: list[tuple[Any, ...]] = []
        self.embedding_calls: list[tuple[str, tuple[str, ...]]] = []
        self.receipts: list[str] = []
        guard = threading.Lock()
        behavior = variant(C9)
        super().__init__(
            RecordingStore(self.writes),  # type: ignore[arg-type]
            CodeEmbedder(self.embedding_calls, guard),
            None,
            specialist_embedders={
                behavior.context_embedding_profile: ContextEmbedder(
                    self.embedding_calls, guard
                )
            },
        )
        if not prefetch:
            # What the service saw before the prefetch existed: no way to embed ahead.
            self.embed_texts = None  # type: ignore[assignment,method-assign]

    def get_receipt(self, *args: Any) -> None:
        return None

    def record_receipt(self, tenant: str, request_id: str, fingerprint: str, result: str) -> None:
        self.receipts.append(result)

    def prior_records(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class GroundedCompiler:
    """Returns one record quoting the session verbatim, so the graph links it to raw windows."""

    def __init__(self, repository: RecordingRepository | None = None, wait_for: int = 0) -> None:
        self._repository = repository
        self._wait_for = wait_for
        self.embeddings_seen_during_compile = 0

    def compile_anchored_v3(
        self, messages: list[Message], session_id: str, prior: Any
    ) -> list[CodingMemoryRecord]:
        if self._repository is not None:
            for _ in range(200):
                if len(self._repository.embedding_calls) >= self._wait_for:
                    break
                threading.Event().wait(0.01)
            self.embeddings_seen_during_compile = len(self._repository.embedding_calls)
        content = messages[0].content
        assert isinstance(content, str)
        quote = content[:120]
        return [
            CodingMemoryRecord.model_validate(
                {
                    "kind": "successful repair",
                    "action": quote,
                    "evidence_spans": [
                        {"message_ordinal": 0, "start": 0, "end": len(quote), "quote": quote}
                    ],
                    "evidence_quotes": [quote],
                    "source_session_id": session_id,
                }
            )
        ]


def c9_service(repository: RecordingRepository, compiler: Any) -> HostedService:
    behavior = variant(C9)
    return HostedService(
        repository,
        compiler,
        object(),  # type: ignore[arg-type]
        behavior=behavior,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={behavior.context_embedding_profile: object()},  # type: ignore[dict-item]
    )


def add_request() -> AddRequest:
    return AddRequest(
        request_id="r1",
        user_id="u",
        session_id="s",
        messages=[Message(role="assistant", content=SESSION_TEXT)],
    )


def test_c9_add_stores_identical_rows_with_and_without_the_prefetch() -> None:
    inline = RecordingRepository(prefetch=False)
    prefetched = RecordingRepository(prefetch=True)

    inline_response = asyncio.run(c9_service(inline, GroundedCompiler()).add(add_request()))
    prefetched_response = asyncio.run(
        c9_service(prefetched, GroundedCompiler()).add(add_request())
    )

    tenants = [write[0] for write in inline.writes]
    # The Add reached every namespace C9 writes: raw, graph, Context, and both view scopes.
    assert len(tenants) == 5 and len(set(tenants)) == 5
    assert inline_response.compiled_count == 1
    assert prefetched_response == inline_response
    assert prefetched.receipts == inline.receipts
    assert prefetched.writes == inline.writes
    assert sorted(prefetched.embedding_calls) == sorted(inline.embedding_calls)


def test_a_failed_prefetch_leaves_the_add_embedding_inline() -> None:
    """A provider error while embedding ahead costs the Add nothing but the wasted request."""
    inline = RecordingRepository(prefetch=False)
    failing = RecordingRepository(prefetch=True)

    def refuse(*_args: Any) -> list[list[float]]:
        raise TimeoutError("provider timeout")

    failing.embed_texts = refuse  # type: ignore[method-assign]

    inline_response = asyncio.run(c9_service(inline, GroundedCompiler()).add(add_request()))
    failing_response = asyncio.run(c9_service(failing, GroundedCompiler()).add(add_request()))

    assert failing_response == inline_response
    assert failing.writes == inline.writes


def test_c9_add_embeds_independent_rows_while_the_compile_runs() -> None:
    repository = RecordingRepository(prefetch=True)
    compiler = GroundedCompiler(repository, wait_for=3)

    asyncio.run(c9_service(repository, compiler).add(add_request()))

    # Raw windows (code), views (code) and views (Context), before the compile returned.
    assert compiler.embeddings_seen_during_compile == 3
    assert len(repository.embedding_calls) == 5


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("prefetch", [True, False])
def test_c9_add_builds_and_links_chunks_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, prefetch: bool
) -> None:
    on_loop: list[str] = []
    for name in ("_normalize_messages", "build_chunks", "attach_grounded_relations", "build_view_chunks"):
        original = getattr(service_module, name)

        def recording(*args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
            if _on_loop():
                on_loop.append(__name)
            return __original(*args, **kwargs)

        monkeypatch.setattr(service_module, name, recording)
    repository = RecordingRepository(prefetch=prefetch)

    response = asyncio.run(c9_service(repository, GroundedCompiler()).add(add_request()))

    assert response.compiled_count == 1
    assert on_loop == []


def test_a_large_body_is_decoded_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    decoded_on_loop: list[bool] = []
    original = app_module._decode_json

    def recording(body: bytes) -> Any:
        decoded_on_loop.append(_on_loop())
        return original(body)

    monkeypatch.setattr(app_module, "_decode_json", recording)
    service, _, _ = make_service()
    settings = HostedSettings("postgresql://unused", "secret", "abc123", variant_name="A0_raw")
    client = TestClient(create_app(settings, service))
    large = "ExactError in src/widget.py " * 4_000  # about 112 KB, over the inline limit

    response = client.post(
        "/v1/add",
        headers={"authorization": "Bearer secret"},
        json={
            "request_id": "r1",
            "user_id": "user-a",
            "session_id": "s",
            "messages": [{"role": "user", "content": large}],
        },
    )

    assert response.status_code == 200
    assert decoded_on_loop == [False]
    # Nonbehavioural control: a Search body is small and stays on the loop.
    decoded_on_loop.clear()
    client.post(
        "/v1/search",
        headers={"authorization": "Bearer secret"},
        json={"query": "ExactError", "user_id": "user-a", "top_k": 1},
    )
    assert decoded_on_loop == [True]


def test_the_default_executor_admits_two_threads_per_admitted_request() -> None:
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        add_concurrency=12,
        search_concurrency=6,
        variant_name="A0_raw",
    )
    service, _, _ = make_service()
    app = create_app(settings, service)
    parties = 2 * (settings.add_concurrency + settings.search_concurrency)
    assert parties > 32  # more than asyncio's own default can ever hold

    async def probe(_: Request) -> JSONResponse:
        barrier = threading.Barrier(parties, timeout=3.0)

        def wait() -> bool:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return False
            return True

        met = await asyncio.gather(*(asyncio.to_thread(wait) for _ in range(parties)))
        return JSONResponse({"all_met": all(met)})

    app.add_route("/probe", probe)
    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.json() == {"all_met": True}
    assert executor_workers(settings) == parties + app_module.EXECUTOR_MARGIN_THREADS
