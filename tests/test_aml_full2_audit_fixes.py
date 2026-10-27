"""CCA hunt of the C9 Full 2 build (b712ec33), 2026-10-20: one behaviour proof per fix.

Each test states the invariant it protects and the defect it catches. The red proof for each was
taken against the pre-fix code (the served build) or, where the pre-fix code lacks a symbol the
test needs, against a named mutation of the fix; the run, the failure and the green rerun are
recorded in the commit and in the CCA run ``20261020T152413Z``.

Red proofs, run 2026-10-20 on the remote test host against the served build b712ec33 with only this file added
(16 collected: 11 failed, the 5 controls passed; then 16 passed on the fix):

* ``test_compile_off_startup_needs_no_openrouter_key``: ``RuntimeError: OPENROUTER_API_KEY is
  required`` raised by ``build_app`` (``recall_aml/__main__.py:163`` at b712ec33).
* ``test_compile_off_startup_does_not_probe_the_compiler``: ``assert True is False`` (readiness
  handed the variant's compiler).
* ``test_the_served_reranker_times_out_well_under_the_proxy_cut``: ``assert 60.0 <= 30.0``.
* ``test_a_transient_rerank_refusal_at_startup_is_retried``: ``RuntimeError: HTTP 503`` on the
  first probe call.
* ``test_a_part_list_query_is_bounded_like_a_string_one``: ``assert 40001 <= 20000``.
* ``test_an_image_only_query_is_not_reranked_against_its_placeholder``: ``assert 1 == 0``.
* ``test_image_add_preparation_runs_off_the_event_loop``: ``assert [True] == [False]``.
* ``test_an_image_only_the_visual_leg_found_is_served_as_its_image``: the ``any(_has_image(...))``
  assertion.
* ``test_a_healthy_check_is_shared_for_a_few_seconds``: ``assert 2 == 1``.
* ``test_an_image_pillow_cannot_read_is_answered_422_not_503``: ``assert 503 == 422``.
* ``test_voyage_reranker_builds_its_client_with_the_given_timeout`` fails on b712ec33 with a
  ``TypeError`` (the keyword is new), which is not behavioural proof, so it was proved on a
  mutation instead: ``VoyageReranker._voyage_client`` reverted to ``_voyage_timeout(os.environ)``
  alone fails ``assert [60.0, 60.0] == [15.0, 60.0]``.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from recall_aml.config import HostedSettings
from recall_aml.models import AddRequest, SearchRequest, TextContentPart
from recall_aml.readiness import verify_model_readiness
from recall_aml.variants import variant
from tests.test_aml_multimodal import _png_data_url
from tests.test_aml_multimodal_scope import (
    _add_image_memory,
    _add_text_memory,
    _has_image,
    _search,
)
from tests.test_aml_specialist_fusion import _service

C9 = "C9_routed_specialists_grounded_graph_atomic"


class _Stop(Exception):
    """Ends `build_app` at the stage a test inspects."""


def _c9_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant_name: str = C9, **settings: Any
) -> dict[str, Any]:
    """Run `build_app` up to the readiness probe (C9 by default) and capture what it is handed."""
    import recall_aml.__main__ as hosted_main
    from tests.test_aml_hosted import FakeEmbedder

    seen: dict[str, Any] = {}

    class _Pool:
        def close(self) -> None:
            pass

    class _Store:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def check_schema(self) -> None:
            pass

    def _reranker(**kwargs: Any) -> object:
        seen["reranker_kwargs"] = kwargs
        return object()

    def _readiness(**kwargs: Any) -> dict[str, bool]:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(hosted_main, "_resolve_hosted_embedders", lambda *_: (FakeEmbedder(), {}))
    monkeypatch.setattr(hosted_main, "SharedPool", lambda *_a, **_k: _Pool())
    monkeypatch.setattr(hosted_main, "PgVectorStore", _Store)
    monkeypatch.setattr(hosted_main, "PgHostedRepository", lambda *_a, **_k: object())
    monkeypatch.setattr(hosted_main, "VoyageReranker", _reranker)
    monkeypatch.setattr(hosted_main, "VoyageMultimodalEmbedder", lambda *_a, **_k: object())
    monkeypatch.setattr(hosted_main, "verify_model_readiness", _readiness)
    # The transport only: CI installs `dev` without the `openai` SDK, and these tests read what
    # startup hands the readiness probe, never what the client does.
    monkeypatch.setattr(hosted_main, "build_openrouter_client", lambda *_a, **_k: object())
    config = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name=variant_name,
        voyage_api_key="voyage-key",
        embedding_lock_path=tmp_path / "embed.lock",
        **settings,
    )
    with pytest.raises(_Stop):
        hosted_main.build_app(config)
    return seen


def test_compile_off_startup_needs_no_openrouter_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """U-01 (P1). Invariant: with ``RECALL_AML_COMPILER=0`` C9 starts without an OpenRouter key.

    Defect caught: the switch reached `HostedService` only, so `build_app` still demanded the key
    for a compile that never runs, and a restart without it (or with a refused one) never started.
    Red proof: the served build raises ``OPENROUTER_API_KEY is required`` before readiness.
    """
    monkeypatch.setenv("RECALL_AML_COMPILER", "0")

    seen = _c9_startup(monkeypatch, tmp_path)

    assert "behavior" in seen, "startup must reach the readiness probe"


def test_compile_off_startup_does_not_probe_the_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """U-01 (P1). Invariant: with ``RECALL_AML_COMPILER=0`` readiness makes no compile call.

    Defect caught: readiness got the variant's ``compiler=True`` and made one live OpenRouter
    compile at every start, so an out-of-credit or unreachable OpenRouter kept C9 down.
    Red proof: the served build hands readiness ``behavior.compiler is True``.
    """
    monkeypatch.setenv("RECALL_AML_COMPILER", "0")

    seen = _c9_startup(monkeypatch, tmp_path, openrouter_api_key="openrouter-key")

    assert seen["behavior"].compiler is False
    assert variant(C9).compiler is True, "precondition: the variant itself compiles"


def test_compile_unset_startup_still_probes_the_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """U-01 control: with the switch unset the variant decides, so readiness still probes it."""
    monkeypatch.delenv("RECALL_AML_COMPILER", raising=False)

    seen = _c9_startup(monkeypatch, tmp_path, openrouter_api_key="openrouter-key")

    assert seen["behavior"].compiler is True


def test_the_compiler_switch_cannot_turn_a_compile_on_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """U-01 scope. Invariant: ``RECALL_AML_COMPILER=1`` on a variant without a compiler neither
    requires an OpenRouter key nor probes a compile, because `HostedService` never serves one.

    Red proof (mutation, the switch is new): ``_env_compiler`` returning ``_env_flag(...)`` alone
    starts with a compile probe for C7, failing ``assert True is False``.
    """
    monkeypatch.setenv("RECALL_AML_COMPILER", "1")

    seen = _c9_startup(
        monkeypatch,
        tmp_path,
        variant_name="C7_routed_specialists",
        openrouter_api_key="openrouter-key",
    )

    assert variant("C7_routed_specialists").compiler is False, "precondition"
    assert seen["behavior"].compiler is False


def test_the_served_reranker_times_out_well_under_the_proxy_cut(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """U-19. Invariant: C9's Voyage reranker gives up on a stalled request in at most 30 s.

    Defect caught: each Search waits on the rerank inline with the 60 s embedding default (plus
    connection retries), so a stalled provider took Searches past the ~100 s cut instead of
    falling back to the fused order. Red proof: the served build passes no timeout.
    """
    seen = _c9_startup(monkeypatch, tmp_path, openrouter_api_key="openrouter-key")

    assert 0 < seen["reranker_kwargs"].get("timeout", 60.0) <= 30.0


def test_voyage_reranker_builds_its_client_with_the_given_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-19. Invariant: `VoyageReranker(timeout=...)` reaches the HTTP client; unset keeps the default."""
    import recall.embeddings as embeddings
    from recall.rerank import VoyageReranker

    made: list[float] = []

    class _Client:
        def __init__(self, *, api_key: str, timeout: float) -> None:
            made.append(timeout)

    monkeypatch.setattr(embeddings, "_voyage_client_class", lambda _caller: _Client)
    monkeypatch.delenv("RECALL_VOYAGE_TIMEOUT_SECONDS", raising=False)

    VoyageReranker(api_key="k", timeout=15.0)._voyage_client()
    VoyageReranker(api_key="k")._voyage_client()

    assert made == [15.0, 60.0]


class _FlakyReranker:
    """Fails with the given HTTP status for the first ``failures`` calls, then passes through."""

    def __init__(self, status: int, failures: int = 1) -> None:
        self.status = status
        self.failures = failures
        self.calls = 0

    def rerank(self, query: str, hits: list[Any]) -> list[Any]:
        self.calls += 1
        if self.calls <= self.failures:
            error = RuntimeError(f"HTTP {self.status}")
            error.http_status = self.status  # type: ignore[attr-defined]
            raise error
        return hits


def _probe_reranker(reranker: _FlakyReranker) -> dict[str, bool]:
    from tests.test_aml_hosted import FakeEmbedder

    return verify_model_readiness(
        embedder=FakeEmbedder(),
        compiler=None,
        reranker=reranker,  # type: ignore[arg-type]
        behavior=replace(variant("A0_raw"), reranker=True),
    )


def test_a_transient_rerank_refusal_at_startup_is_retried() -> None:
    """U-02. Invariant: one 503 from Voyage's rerank at startup does not stop C9 starting.

    Defect caught: the probe made one call and any error stopped startup, so a provider blip at a
    restart kept the service down although Search falls back to the fused order. Red proof: the
    served build raises on the first 503 (one call).
    """
    reranker = _FlakyReranker(503)

    status = _probe_reranker(reranker)

    assert (status["reranker_ready"], reranker.calls) == (True, 2)


def test_a_refused_rerank_key_still_stops_startup() -> None:
    """U-02 control: a 401 (refused key) is not transient, so startup still fails on it at once."""
    reranker = _FlakyReranker(401)

    with pytest.raises(RuntimeError, match="401"):
        _probe_reranker(reranker)

    assert reranker.calls == 1


def test_a_part_list_query_is_bounded_like_a_string_one() -> None:
    """U-21. Invariant: a Search question sent as text parts is held to 20,000 characters too.

    Defect caught: only a string query was bounded, and the router's quadratic regex then ran on
    the event loop over all of a part list (21 s at 40,000 characters, measured 2026-10-20). Red
    proof: the served build keeps all 40,000 characters.
    """
    image = {"type": "image_url", "image_url": {"url": _png_data_url(b"q")}}
    request = SearchRequest.model_validate(
        {
            "user_id": "u",
            "query": [
                {"type": "text", "text": "a(" * 10_000},
                image,
                {"type": "text", "text": "b(" * 10_000},
            ],
        }
    )

    texts = [part.text for part in request.query if isinstance(part, TextContentPart)]
    assert len("\n".join(texts)) <= 20_000
    assert [part.type for part in request.query] == ["text", "image_url"]
    assert texts[0].startswith("a(") and texts[0].endswith("b(")


def test_a_short_part_list_query_is_unchanged() -> None:
    """U-21 control: a part list within the bound reaches the service exactly as sent."""
    parts = [
        {"type": "text", "text": "what is in this photo?"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"q")}},
    ]
    request = SearchRequest.model_validate({"user_id": "u", "query": parts})

    assert [part.model_dump() for part in request.query] == parts


class _CountingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, hits: list[Any]) -> list[Any]:
        self.calls += 1
        return hits


def _reranked_service():  # type: ignore[no-untyped-def]
    from recall_aml.retrieval import HostedRetriever
    from recall_aml.service import HostedService
    from tests.test_aml_specialist_fusion import _Embedder, _MultimodalEmbedder, _Repository

    behavior = variant("C7_routed_specialists")
    reranker = _CountingReranker()
    service = HostedService(
        _Repository(behavior.context_embedding_profile),
        None,
        HostedRetriever(_Embedder("code"), reranker),  # type: ignore[arg-type]
        behavior=behavior,
        multimodal_embedder=_MultimodalEmbedder(),  # type: ignore[arg-type]
        specialist_retrievers={
            behavior.context_embedding_profile: HostedRetriever(_Embedder("context"), reranker)  # type: ignore[arg-type]
        },
    )
    return service, reranker


def test_an_image_only_query_is_not_reranked_against_its_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-17. Invariant: with the reranker on, a query with no text keeps the fused order.

    Defect caught: an image-only query reaches the text leg as the literal "[image evidence]",
    and the reranker reordered the text candidates against that string (one provider call of
    noise per Search). Red proof: the served build calls the reranker once.
    """
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    service, reranker = _reranked_service()
    _add_image_memory(service)
    image_only = [{"type": "image_url", "image_url": {"url": _png_data_url(b"query")}}]

    response = asyncio.run(
        service.search(SearchRequest.model_validate({"query": image_only, "user_id": "scope-user"}))
    )

    assert response.data, "precondition: the Search serves items"
    assert reranker.calls == 0


def test_a_query_with_text_is_still_reranked(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-17 control: any text part keeps the reranker on for that Search."""
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    service, reranker = _reranked_service()
    _add_image_memory(service)

    _search(service)

    assert reranker.calls == 1


def test_image_add_preparation_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-22. Invariant: an image Add decodes, hashes and fits its images in a worker thread.

    Defect caught: `prepare_messages` ran on the event loop, stalling every concurrent request
    for 26 ms to 1.35 s per image (measured 2026-10-20). Red proof: the served build calls it
    from the event loop's (main) thread.
    """
    import recall_aml.service as service_module

    threads: list[bool] = []
    original = service_module.prepare_messages

    def recording(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "prepare_messages", recording)
    service, _, _, _, _ = _service("C7_routed_specialists")

    _add_image_memory(service)

    assert threads == [False]


def test_an_image_only_the_visual_leg_found_is_served_as_its_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-39. Invariant: under the dual scope, an image memory the text leg did not return but the
    visual leg did is served with its image.

    Defect caught: the preserved renderer ran only when a TEXT hit carried
    ``multimodal_parent_id``; a visual-only hit carries ``record_type="multimodal_vector"``, so on
    a tenant whose text pool held no image primary the image was served as its text. Red proof:
    the served build serves no item with an image part.
    """
    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", "dual")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(service, "scope-user")
    _add_image_memory(service)
    original = service._retriever.search

    def text_pool_without_images(*args: Any, **kwargs: Any) -> Any:
        # A large tenant whose text candidates hold no image primary, as on the official runs.
        run = original(*args, **kwargs)
        run.hits[:] = [hit for hit in run.hits if "multimodal_parent_id" not in hit.chunk.metadata]
        return run

    monkeypatch.setattr(service._retriever, "search", text_pool_without_images)

    response = _search(service)

    assert response.visual_leg, "precondition: the visual leg ran"
    assert any(_has_image(item) for item in response.data)


def test_a_healthy_check_is_shared_for_a_few_seconds() -> None:
    """U-24. Invariant: concurrent and repeated ``/health`` checks within the cache window cost
    one repository check.

    Defect caught: the public ``/health`` re-read and hashed every migration file and ran the
    schema queries on a pooled connection per call, with no auth and no cache. Red proof: the
    served build checks the repository twice.
    """
    service, repository, _, _, _ = _service("C7_routed_specialists")
    calls: list[int] = []

    def health() -> dict[str, object]:
        calls.append(1)
        return {"status": "ready"}

    repository.health = health

    async def twice() -> list[dict[str, object]]:
        return list(await asyncio.gather(service.health(), service.health()))

    assert asyncio.run(twice()) == [{"status": "ready"}, {"status": "ready"}]
    assert len(calls) == 1


def test_a_failed_health_check_is_never_cached() -> None:
    """U-24 control: an unhealthy answer is re-checked on the next call."""
    service, repository, _, _, _ = _service("C7_routed_specialists")
    calls: list[int] = []

    def health() -> dict[str, object]:
        calls.append(1)
        raise RuntimeError("database unavailable")

    repository.health = health

    for _ in range(2):
        with pytest.raises(RuntimeError, match="unavailable"):
            asyncio.run(service.health())

    assert len(calls) == 2


def test_an_image_pillow_cannot_read_is_answered_422_not_503() -> None:
    """U-47. Invariant: an Add whose admitted image Pillow cannot read gets the permanent 422.

    Defect caught: admission checks only type, size and magic bytes; the Pillow error inside the
    Add fell to the generic handler, so AML got a retryable 503 and resent the same payload up to
    32 times. Red proof: the served build answers 503.
    """
    from recall_aml.app import create_app

    service, _, _, _, _ = _service("C7_routed_specialists")
    client = TestClient(create_app(HostedSettings("postgresql://unused", "secret", "abc123"), service))
    corrupt = b"\x89PNG\r\n\x1a\n" + b"not an image body" * 8
    url = "data:image/png;base64," + base64.b64encode(corrupt).decode("ascii")
    body = AddRequest.model_validate(
        {
            "request_id": "corrupt-image",
            "user_id": "u",
            "session_id": "s",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
        }
    ).model_dump(mode="json")

    response = client.post("/v1/add", headers={"X-Api-Key": "secret"}, json=body)

    assert response.status_code == 422
