"""The serving path reuses one reranker per configuration, and builds it for the routed profile.

Invariant: `recall_mcp.service._retrieve_trusted`, which every search, evidence and reasoning call
goes through, passes its settings snapshot to the reranker builder as an explicit `env`. The
builder must still serve the per-process instance for that configuration, and must build it for
the profile the request was actually routed to.

Failure mode caught: `recall_mcp.factories._build_reranker` treated any explicit `env` as "build an
ad-hoc instance, never the shared one" and called the builder with the environment only. So each
search re-verified and reloaded the cross-encoder, and under `RECALL_ROUTING_MODE=active` the
routed profile never reached the builder while diagnostics reported it.

Red proof, recorded 2026-09-22 against `origin/master` at `3cc57b81` (the pre-fix
`_build_reranker`, restored into the worktree for the run):

* `test_one_configuration_builds_one_reranker_across_searches` failed with
  `AssertionError: built 2 rerankers for one configuration` (`assert 2 == 1`).
* `test_the_routed_profile_reaches_the_reranker_builder` failed with `assert None == ...`: the
  builder received `profile=None` while the request was served under the routed profile.
* `test_a_changed_configuration_builds_its_own_reranker` guards the opposite defect (a cache that
  over-shares), which the pre-fix code could not exhibit. Its red proof is a mutation of the fixed
  key in `_build_reranker` to `(profile, "")`, dropping `_env_digest(values)`: it then failed with
  `assert 1 == 2`.

All three pass with the fix. The existing tests in `tests/test_retrieval_profiles.py` call
`_build_reranker()` with no environment, which is the cached path serving never took; that is why
they stayed green through the regression.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import recall_mcp.factories as factories
from recall.trust_policy import TrustPolicy
from recall_mcp import service


class _Embedder:
    dim = 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _search(env: dict[str, str], query: str = "q") -> object:
    retrieval = service._retrieve_trusted(
        SimpleNamespace(generation_id="g1"),
        _Embedder(),
        query,
        None,
        3,
        None,
        TrustPolicy.development(),
        env=env,
    )
    return retrieval


@pytest.fixture
def builds(monkeypatch):
    seen: list[dict[str, object]] = []
    sentinel = object()

    def fake_new_reranker(env=None, profile=None):
        seen.append({"env": env, "profile": profile})
        return sentinel

    def fake_trusted_search(_store, _timed, _query, **kwargs):
        return SimpleNamespace(reranker=kwargs["reranker"])

    service._reset_reranker_cache()
    monkeypatch.setattr(factories, "_new_reranker", fake_new_reranker)
    monkeypatch.setattr(service, "trusted_search", fake_trusted_search)
    yield seen, sentinel
    service._reset_reranker_cache()


def test_one_configuration_builds_one_reranker_across_searches(builds) -> None:
    seen, sentinel = builds
    env = {"RECALL_RETRIEVAL_PROFILE": "quality", "RECALL_DECISION_LEDGER": "0"}

    first = _search(env)
    second = _search(dict(env))

    assert len(seen) == 1, f"built {len(seen)} rerankers for one configuration"
    assert first.result.reranker is sentinel
    assert second.result.reranker is sentinel


def test_a_changed_configuration_builds_its_own_reranker(builds) -> None:
    """The cache must not serve a reranker built for a different environment."""
    seen, _sentinel = builds
    base = {"RECALL_RETRIEVAL_PROFILE": "quality", "RECALL_DECISION_LEDGER": "0"}

    _search(base)
    _search({**base, "RECALL_RERANK_BATCH_SIZE": "8"})

    assert len(seen) == 2
    assert seen[1]["env"]["RECALL_RERANK_BATCH_SIZE"] == "8"


def test_the_routed_profile_reaches_the_reranker_builder(builds) -> None:
    seen, _sentinel = builds
    env = {"RECALL_ROUTING_MODE": "active", "RECALL_DECISION_LEDGER": "0"}

    retrieval = _search(env)

    assert retrieval.profile.name != "legacy"  # routing replaced the process profile
    assert seen[0]["profile"] == retrieval.profile
