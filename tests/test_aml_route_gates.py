"""Route gates: what decides C9's image leg and Code4 tie order (``HostedVariant.route_gates``).

Design: recall-lab ``research/designs/2026-09-28-route-architecture.md``, reviewed the same day.
Served (``route``): a visual word in plain text picks the multimodal route, and only that route
runs the image leg. ``data``: a visual word no longer routes text; the image leg runs when the query
carries an image or the tenant holds image vectors, asked of the store before any multimodal
embedding; the tie order follows the store searched. The retirement of the visual-word route and
the tenant-gated leg are one setting, so image tenants never lose their images.

Red proofs, each run 2026-09-28 against the deliberate mutation named, failing at the assertion
named, then green after restoring the line:

* ``test_route_query_ignores_visual_words_only_when_told``: ``if visual_words and
  _VISUAL_SIGNAL.search(text)`` in ``route_query`` reverted to ``if _VISUAL_SIGNAL.search(text)``;
  fails at ``route_query(..., visual_words=False) == "code"`` with ``multimodal``.
* ``test_a_coding_query_with_a_visual_word_keeps_the_code_path``: ``HostedService.search`` calls
  ``route_query(request.query)`` without ``visual_words``; fails at
  ``response.specialist_route == "code"`` with ``multimodal``.
* ``test_data_gates_give_an_image_tenant_its_images_without_a_visual_word``: the effective scope
  line reverted to ``scope = self.multimodal_scope``; fails at ``response.visual_leg``.
* ``test_data_gates_pay_no_multimodal_embedding_for_a_text_only_tenant``: the pre-check clause
  replaced by ``and True``; fails at ``multimodal.query_inputs == []``.
* ``test_the_first_search_after_an_image_add_sees_the_image``: ``_tenant_has_image_vectors``
  memoised per tenant (a cache filled by the Search before the Add); fails at
  ``after.visual_leg``.
* ``test_data_gates_key_the_tie_order_on_the_store``: the data branch of the tie-order condition
  reverted to ``specialist_route == "code"``; fails at ``stable == [True, False, True]`` with
  ``[True, False, False]``.
* ``test_the_served_gates_are_unchanged``: the served branch keyed on the store
  (``else self._searches_raw_windows(retriever)``); fails at ``stable == [False]`` with ``[True]``.
* ``test_unknown_route_gates_stop_service_startup``: ``self.route_gates`` removed from the
  startup read in ``HostedService.__init__``; fails with DID NOT RAISE.
* ``test_the_router_profile_names_the_data_gates``: the ``+no-visual-words`` branch of
  ``specialist_router_profile`` removed; fails at the profile equality.
* ``test_a_text_only_tenant_gets_identical_output_under_both_gates``: ``visual_route`` also true
  under the data gates (``... or data_gates``); fails at the response dump equality (RRF rescoring,
  score 0.9 against 0.0164). A second mutation, the tie order forced on for every data-gated
  Search, SURVIVES this test: the fake store produces no ties, so the order cannot be seen in its
  output. ``test_data_gates_key_the_tie_order_on_the_store`` observes that condition directly.
* ``test_search_is_answered_for_an_image_query_under_data_gates``: the image-part check in
  ``route_query`` also made conditional on ``visual_words``; fails at
  ``response.specialist_route == "multimodal"`` with ``code``.

``has_multimodal_vectors`` on the PostgreSQL repository is covered by
``test_pg_repository_reports_image_vectors_per_tenant`` (DB-backed).
"""

from __future__ import annotations

import asyncio

import pytest

from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall.types import Chunk
from recall_aml.identity import tenant_for
from recall_aml.models import SearchRequest
from recall_aml.retrieval import HostedRetriever
from recall_aml.specialists import route_query
from recall_aml.storage import PgHostedRepository
from tests.conftest import TEST_DSN, requires_db
from tests.test_aml_multimodal import _png_data_url
from tests.test_aml_multimodal_scope import (
    QUERY,
    _add_image_memory,
    _add_text_memory,
    _has_image,
    _search,
)
from tests.test_aml_specialist_fusion import _service

VISUAL_CODING_QUERY = "Fix the settings screen layout bug in parser.py"
CONTEXT_QUERY = "What did we decide during yesterday's meeting?"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RECALL_AML_ROUTE_GATES", "RECALL_AML_MULTIMODAL_SCOPE"):
        monkeypatch.delenv(name, raising=False)


def _stable_order_spy(monkeypatch) -> list[bool]:
    seen: list[bool] = []
    original = HostedRetriever.search

    def spy(self, *args, **kwargs):
        seen.append(bool(kwargs.get("stable_window_order")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HostedRetriever, "search", spy)
    return seen


def test_route_query_ignores_visual_words_only_when_told() -> None:
    """A visual word routes text to multimodal as served, and to the text routes under data gates."""
    assert route_query("What does this screenshot show?") == "multimodal"
    assert route_query("What does this screenshot show?", visual_words=False) == "code"
    assert route_query(VISUAL_CODING_QUERY, visual_words=False) == "code"
    assert route_query("Which chart did we discuss yesterday?", visual_words=False) == "context"
    image_query = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"q")}},
    ]
    parts = SearchRequest.model_validate({"query": image_query, "user_id": "u"}).query
    assert route_query(parts, visual_words=False) == "multimodal"


def test_data_gates_give_an_image_tenant_its_images_without_a_visual_word(monkeypatch) -> None:
    """Retiring the visual route must not take the image leg away from a tenant with images."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_image_memory(service)

    response = _search(service)

    assert response.specialist_route == "code"
    assert response.visual_leg
    assert multimodal.query_inputs == [QUERY]
    assert _has_image(response.data[0])


def test_data_gates_pay_no_multimodal_embedding_for_a_text_only_tenant(monkeypatch) -> None:
    """The store is asked first; a tenant with no image vectors gets no multimodal call."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    service, repository, _, _, multimodal = _service("C7_routed_specialists")
    _add_text_memory(service, "text-user")

    response = _search(service, "text-user")

    assert multimodal.query_inputs == []
    assert repository.image_probes, "the pre-check must ask the store"
    assert response.visual_leg is False


def test_the_first_search_after_an_image_add_sees_the_image(monkeypatch) -> None:
    """AML searches right after an Add returns: the pre-check must not answer from a stale view."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_text_memory(service, "scope-user")
    before = _search(service)
    _add_image_memory(service)

    after = _search(service)

    assert before.visual_leg is False
    assert after.visual_leg
    assert multimodal.query_inputs == [QUERY]
    assert any(_has_image(item) for item in after.data)


def test_a_coding_query_with_a_visual_word_keeps_the_code_path(monkeypatch) -> None:
    """cca789b BUG-001: "screen" or "UI" in a Coding query must not change its path."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    stable = _stable_order_spy(monkeypatch)
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_text_memory(service, "coding-user")

    response = _search(service, "coding-user", VISUAL_CODING_QUERY)

    assert response.specialist_route == "code"
    assert stable == [True]
    assert multimodal.query_inputs == []
    assert response.visual_leg is False


def test_data_gates_key_the_tie_order_on_the_store(monkeypatch) -> None:
    """Raw window store: Code4 order on every route that searches it; Context4 store: off."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    stable = _stable_order_spy(monkeypatch)
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    image_query = [
        {"type": "text", "text": "the parking receipt"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"receipt")}},
    ]

    responses = [
        _search(service, query=QUERY),
        _search(service, query=CONTEXT_QUERY),
        _search(service, query=image_query),
    ]

    assert [r.specialist_route for r in responses] == ["code", "context", "multimodal"]
    assert stable == [True, False, True]


@pytest.mark.parametrize("gates", [None, "route"])
def test_the_served_gates_are_unchanged(monkeypatch, gates) -> None:
    """Unset or ``route``: a visual word still routes to multimodal, with the served tie order."""
    if gates is not None:
        monkeypatch.setenv("RECALL_AML_ROUTE_GATES", gates)
    stable = _stable_order_spy(monkeypatch)
    service, repository, _, _, multimodal = _service("C7_routed_specialists")
    _add_text_memory(service, "coding-user")

    response = _search(service, "coding-user", VISUAL_CODING_QUERY)

    assert response.specialist_route == "multimodal"
    assert stable == [False]
    assert multimodal.query_inputs == [VISUAL_CODING_QUERY]
    assert repository.image_probes == [], "the served route scope never asks the store"


@pytest.mark.parametrize("query", [QUERY, CONTEXT_QUERY])
def test_a_text_only_tenant_gets_identical_output_under_both_gates(monkeypatch, query) -> None:
    """Byte identity on the code and context routes for every Textual and Coding text tenant."""
    baseline_service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(baseline_service, "text-user")
    baseline = _search(baseline_service, "text-user", query)

    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(service, "text-user")
    response = _search(service, "text-user", query)

    assert response.model_dump_json() == baseline.model_dump_json()
    assert [item.model_dump_json() for item in response.data] == [
        item.model_dump_json() for item in baseline.data
    ]


def test_unknown_route_gates_stop_service_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "tenant")
    with pytest.raises(ValueError, match="unknown route gates"):
        _service("C7_routed_specialists")


def test_the_router_profile_names_the_data_gates(monkeypatch) -> None:
    service, _, _, _, _ = _service("C7_routed_specialists")
    assert service.route_gates == "route"
    assert service.specialist_router_profile == "conservative-specialist-router-v1"

    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    assert service.route_gates == "data"
    assert service.specialist_router_profile == "conservative-specialist-router-v1+no-visual-words"


def test_search_is_answered_for_an_image_query_under_data_gates(monkeypatch) -> None:
    """An image in the query still takes the visual route and returns the image memory."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)
    image_query = [
        {"type": "text", "text": "the parking receipt"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"receipt")}},
    ]

    response = asyncio.run(
        service.search(SearchRequest.model_validate({"query": image_query, "user_id": "scope-user"}))
    )

    assert response.specialist_route == "multimodal"
    assert response.visual_leg
    assert _has_image(response.data[0])


@requires_db
def test_pg_repository_reports_image_vectors_per_tenant(make_store) -> None:
    """The pre-check's real query: true for the tenant that holds a vector, false for another.

    Red proof, 2026-09-28: ``WHERE tenant_id = %s`` in ``PgHostedRepository.has_multimodal_vectors``
    replaced by ``WHERE %s::text IS NOT NULL`` (the whole shared table); fails at the second assertion,
    because the other tenant's vector is visible.
    """
    fixture_store = make_store(3)
    serving_store = PgVectorStore(
        TEST_DSN,
        3,
        table=fixture_store.table,
        tenant="aml_service_readiness",
        shared_pool=SharedPool(TEST_DSN, min_size=1, max_size=4),
        owns_pool=True,
    )
    try:
        repository = PgHostedRepository(serving_store, _ThreeDimensions())  # type: ignore[arg-type]
        holder, other = tenant_for("route-gates-images"), tenant_for("route-gates-text")
        repository.persist_multimodal(
            holder,
            [Chunk(id="mm-1", source="route-gates", text="", metadata={"primary_id": "p-1"})],
            [[1.0, 0.0, 0.0]],
        )

        assert repository.has_multimodal_vectors(holder) is True
        assert repository.has_multimodal_vectors(other) is False
    finally:
        serving_store.close()


class _ThreeDimensions:
    dim = 3

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]
