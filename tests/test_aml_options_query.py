"""E3: ``RECALL_AML_OPTIONS_QUERY=concat`` lets a choice task's options reach the retriever.

Pre-registration: recall-lab ``research/preregistrations/2026-10-04-e3-options-in-query.md``
(``d0148f2``). The switch is default off; under ``concat`` the retrieval legs search with the
question followed by the choices, while the route, the rendering and everything after candidate
retrieval keep the question alone.

Red proofs, each run 2026-10-04 against the deliberate mutation named, failing at the assertion
named, then green after restoring the line:

* ``test_concat_sends_the_choices_to_the_retriever``: ``HostedService.search`` passes
  ``query_text`` instead of ``search_text`` to ``retriever.search``; fails at
  ``searched == [EXPECTED]`` with the bare question.
* ``test_off_and_unset_search_with_the_question_alone``: ``retrieval_text`` ignores ``mode`` (the
  ``if mode == "concat" else []`` clause removed); fails at ``searched == [QUESTION, QUESTION]``.
* ``test_the_route_is_decided_on_the_question_alone``: ``route_query(search_text, ...)`` in place
  of ``route_query(request.query, ...)``; fails at ``response.specialist_route == "context"`` with
  ``code`` (the choice carries a code word).
* ``test_retrieval_text_keeps_the_query_bound``: ``bounded_query(...)`` removed from
  ``retrieval_text``; fails at ``len(text) == MAX_QUERY_CHARS`` with 30,049.
* ``test_unknown_options_query_mode_stops_service_startup``: ``self.options_query`` removed from
  the startup read in ``HostedService.__init__``; fails with DID NOT RAISE.
"""

from __future__ import annotations

import asyncio

import pytest

from recall_aml.models import MAX_QUERY_CHARS, SearchRequest
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import retrieval_text
from tests.test_aml_multimodal_scope import _add_text_memory
from tests.test_aml_specialist_fusion import _service

QUESTION = "What did we decide during yesterday's meeting?"
CHOICES = ["We agreed to fix the build first.", "We agreed to postpone the launch."]
EXPECTED = "\n".join([QUESTION, *CHOICES])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("RECALL_AML_OPTIONS_QUERY", "RECALL_AML_ROUTE_GATES"):
        monkeypatch.delenv(name, raising=False)


def _retriever_spy(monkeypatch) -> list[str]:
    seen: list[str] = []
    original = HostedRetriever.search

    def spy(self, store, query, *args, **kwargs):
        seen.append(query)
        return original(self, store, query, *args, **kwargs)

    monkeypatch.setattr(HostedRetriever, "search", spy)
    return seen


def _search(service, options: list[str] | None):
    payload: dict[str, object] = {"query": QUESTION, "user_id": "options-user"}
    if options is not None:
        payload["options"] = options
    return asyncio.run(service.search(SearchRequest.model_validate(payload)))


def test_retrieval_text_appends_choices_only_under_concat() -> None:
    assert retrieval_text(QUESTION, CHOICES, "off") == QUESTION
    assert retrieval_text(QUESTION, CHOICES, "concat") == EXPECTED
    assert retrieval_text(QUESTION, None, "concat") == QUESTION
    assert retrieval_text(QUESTION, ["", "  "], "concat") == QUESTION


def test_retrieval_text_keeps_the_query_bound() -> None:
    """Choices have no length limit of their own; the combined text keeps the question's."""
    long_choices = ["x" * 15_000, "y" * 15_000]
    text = retrieval_text(QUESTION, long_choices, "concat")
    assert len(text) == MAX_QUERY_CHARS
    assert text.startswith(QUESTION) and text.endswith("y")


def test_concat_sends_the_choices_to_the_retriever(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_OPTIONS_QUERY", "concat")
    searched = _retriever_spy(monkeypatch)
    service, *_ = _service("C7_routed_specialists")
    _add_text_memory(service, "options-user")

    response = _search(service, CHOICES)

    assert searched == [EXPECTED]
    assert response.options_query == "concat"


def test_off_and_unset_search_with_the_question_alone(monkeypatch) -> None:
    searched = _retriever_spy(monkeypatch)
    service, *_ = _service("C7_routed_specialists")
    _add_text_memory(service, "options-user")
    unset = _search(service, CHOICES)
    monkeypatch.setenv("RECALL_AML_OPTIONS_QUERY", "off")
    off = _search(service, CHOICES)

    assert searched == [QUESTION, QUESTION]
    assert unset.options_query == off.options_query == "off"


def test_concat_without_choices_is_the_served_search(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_OPTIONS_QUERY", "concat")
    searched = _retriever_spy(monkeypatch)
    service, *_ = _service("C7_routed_specialists")
    _add_text_memory(service, "options-user")

    response = _search(service, None)

    assert searched == [QUESTION]
    assert response.options_query == "off"


def test_the_route_is_decided_on_the_question_alone(monkeypatch) -> None:
    """A choice that carries a code word must not move a conversational question to Code4."""
    monkeypatch.setenv("RECALL_AML_OPTIONS_QUERY", "concat")
    service, *_ = _service("C7_routed_specialists")
    _add_text_memory(service, "options-user")

    response = _search(service, CHOICES)

    assert response.specialist_route == "context"


def test_unknown_options_query_mode_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_OPTIONS_QUERY", "balanced")
    with pytest.raises(ValueError, match="RECALL_AML_OPTIONS_QUERY"):
        _service("C7_routed_specialists")
