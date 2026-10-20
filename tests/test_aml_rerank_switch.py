"""RK-1: switch C9's Voyage reranker on per Search and choose its model (``recall_aml.rerank_switch``).

Round two, 2026-10-19. Both variables are unset by default, so C9 serves as before. Each test names
the mutation of the production code it was watched to fail on (the red proof), recorded with the
run in the pull request.
"""

from __future__ import annotations

import asyncio

import pytest

from recall_aml.rerank_switch import DEFAULT_RERANK_MODEL, rerank_model
from tests.test_aml_relative_dates_and_adjacency import _search
from tests.test_aml_window_compose import LONG, _VisualStore

QUERY = "when did I talk about fountain pens?"


class _ReversingReranker:
    """A reranker that reverses the fused order and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, hits):  # type: ignore[no-untyped-def]
        self.calls += 1
        return list(reversed(hits))


def _service(reranker: _ReversingReranker):  # type: ignore[no-untyped-def]
    """C9 as served, over the forget tests' in-memory doubles, with the given reranker."""
    from recall_aml.models import AddRequest
    from recall_aml.retrieval import HostedRetriever
    from recall_aml.service import HostedService
    from tests.test_aml_forget import C9, _Compiler, _Embedder, _Repository
    from tests.test_aml_specialist_fusion import _MultimodalEmbedder

    class _Repo(_Repository):
        def multimodal_store(self, tenant):  # type: ignore[no-untyped-def]
            return _VisualStore()

        def media_store(self, tenant):  # type: ignore[no-untyped-def]
            return _VisualStore()

    service = HostedService(
        _Repo(), _Compiler(), HostedRetriever(_Embedder("code"), reranker),  # type: ignore[arg-type]
        behavior=C9, multimodal_embedder=_MultimodalEmbedder(),  # type: ignore[arg-type]
        specialist_retrievers={
            C9.context_embedding_profile: HostedRetriever(_Embedder("context"), reranker)  # type: ignore[arg-type]
        },
        rerank_model="rerank-3",
    )
    asyncio.run(service.add(AddRequest.model_validate({
        "request_id": "rk1-long", "user_id": "rk1-user", "session_id": "rk1-sess",
        "messages": [{"role": role, "content": text, "timestamp": 1_700_000_000_000} for role, text in LONG],
    })))
    return service


def test_c9_never_reranks_unless_asked() -> None:
    """Invariant: the served C9 has the reranker off, so the default Search never calls it.

    Red proof: `HostedService.reranker_enabled` defaulting to True (``_env_flag(RERANKER_ENV,
    True)``) calls the reranker and fails the call-count assertion.
    """
    reranker = _ReversingReranker()
    service = _service(reranker)
    response = _search(service, "rk1-user", QUERY)
    assert response.data, "precondition: windows are served"
    assert (reranker.calls, service.reranker_enabled, service.active_components["reranker"]) == (0, False, False)


def test_the_switch_reranks_every_search_and_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: ``RECALL_AML_RERANKER=1`` makes each Search rerank, and the served order follows
    the reranker; `active_components` reports it.

    Red proof: `HostedService.search` passing ``rerank=self._behavior.reranker`` (the variant's
    False) never calls the reranker and fails the call-count assertion.
    """
    baseline = [item.id for item in _search(_service(_ReversingReranker()), "rk1-user", QUERY).data]
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    reranker = _ReversingReranker()
    service = _service(reranker)
    served = [item.id for item in _search(service, "rk1-user", QUERY).data]
    assert reranker.calls >= 1
    assert served != baseline
    assert service.active_components["reranker"] is True


def test_the_search_reports_the_model_that_reranked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: a reranked Search names the model the service loaded (here rerank-3), which is
    what the per-Search header and the journal carry; an unreranked Search names none.

    Red proof: `HostedService.search` reporting ``RERANK_MODEL.split(":", 1)[1]`` (the config
    default, rerank-2.5) fails the model equality. Found 2026-10-20 on RK-1's stress run, where
    every header said rerank-2.5 while rerank-3 served.
    """
    assert _search(_service(_ReversingReranker()), "rk1-user", QUERY).reranker_model == "none"
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    assert _search(_service(_ReversingReranker()), "rk1-user", QUERY).reranker_model == "rerank-3"


def test_reranked_scores_stay_non_increasing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: a reranker reorders without rescoring, so the returned scores would follow the
    fused order; the retriever hands the fused scores out again in descending order.

    Red proof: `HostedRetriever.search` keeping the reranked hits as they come (``hits =
    reranked``) returns ascending scores under the reversing reranker and fails the ordering.
    """
    monkeypatch.setenv("RECALL_AML_RERANKER", "1")
    scores = [item.score for item in _search(_service(_ReversingReranker()), "rk1-user", QUERY).data]
    assert len(scores) > 1, "precondition: several items"
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1], "precondition: the scores are not all equal"


def test_a_bad_reranker_value_stops_service_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: an unreadable switch is refused at startup, not on every Search.

    Red proof: ``self.reranker_enabled`` removed from the startup read in `HostedService.__init__`
    lets the service start and fails the ``pytest.raises``.
    """
    monkeypatch.setenv("RECALL_AML_RERANKER", "yes")
    with pytest.raises(ValueError, match="RECALL_AML_RERANKER"):
        _service(_ReversingReranker())


def test_the_model_is_one_of_voyages_rerankers() -> None:
    """Invariant: unset keeps the served model; a current Voyage reranker is accepted; anything else
    is refused before the service is built.

    Red proof: `rerank_model` returning ``value`` without the membership check accepts
    ``rerank-9`` and fails the ``pytest.raises``.
    """
    assert rerank_model({}) == DEFAULT_RERANK_MODEL == "rerank-2.5"
    assert rerank_model({"RECALL_AML_RERANK_MODEL": "rerank-3"}) == "rerank-3"
    with pytest.raises(ValueError, match="RECALL_AML_RERANK_MODEL"):
        rerank_model({"RECALL_AML_RERANK_MODEL": "rerank-9"})
