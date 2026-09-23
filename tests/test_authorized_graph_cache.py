"""A security-filtered reasoning graph is built once per cached projection and caller scope.

With a source security policy configured, `_authorized_graph` ran `decide()` on every node and
refiltered every edge set on every query, and returned a new object each time, so the filtered
graph's `fingerprint` (a full hash, feeding the proposal and planner cache keys) was recomputed
per query too. The filtered view is now cached beside the cached unfiltered projection, keyed on
that projection's identity and on the authorisation scope: the policy digest plus every access
context field `decide()` reads. One caller's view must never be served to another.

Red proofs (2026-09-23, base ``c7f2b9bc``):

* ``tests/test_authorized_graph_cache.py::test_one_caller_scope_reuses_its_filtered_graph``
  failed against the uncached ``recall_mcp.graph_projection._authorized_graph`` with
  ``assert first is second``: every call built a new view.
* ``tests/test_authorized_graph_cache.py::test_two_principals_never_share_a_filtered_graph``,
  against a mutation keying the cache on the policy digest alone, failed
  ``assert _sources(alice) == ["public/a.md"]``: alice was served admin's view, secret source
  included.
"""

from __future__ import annotations

import pytest

import recall_mcp.service as service
from recall.reasoning_graph import build_reasoning_graph
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.types import Chunk

POLICY = SourceSecurityPolicy(
    (
        SourceRule("public", classification="public"),
        SourceRule("secret", classification="confidential", principals=frozenset({"admin"})),
    )
)
ADMIN = AccessContext("admin", "acme", clearance="confidential")
ALICE = AccessContext("alice", "acme", clearance="confidential")


class _Readiness:
    ready = True
    graph_fingerprint = "graph-a"


class _Store:
    tenant = "acme"

    def active_generation_id(self) -> str:
        return "gen-1"

    def graph_readiness(self) -> _Readiness:
        return _Readiness()


def _graph():
    return build_reasoning_graph(
        [
            Chunk("a", "public/a.md", "decision: public note.", {"file": "a.md", "ord": 0}),
            Chunk("b", "secret/b.md", "decision: secret note.", {"file": "b.md", "ord": 0}),
        ],
        tenant_id="acme",
        generation_id="gen-1",
        include_text=True,
    )


@pytest.fixture(autouse=True)
def _cached_projection(monkeypatch):
    service._reset_graph_projection_cache()
    monkeypatch.setattr(service, "project_store_graph", lambda store, **kwargs: _graph())
    yield
    service._reset_graph_projection_cache()


def _cached_graph():
    return service._store_graph(
        _Store(),
        include_text=True,
        policy_fingerprint=service._combined_graph_policy_fingerprint(security_policy=POLICY),
    )


def _view(context: AccessContext):
    return service._authorized_graph(_Store(), _cached_graph(), POLICY, context)


def _sources(graph) -> list[str]:
    return sorted({node.source for node in graph.nodes if node.source})


def test_one_caller_scope_reuses_its_filtered_graph() -> None:
    first = _view(ADMIN)
    second = _view(ADMIN)

    assert _cached_graph() is _cached_graph()  # the unfiltered projection really is cached
    assert first is second
    assert first.fingerprint == second.fingerprint


def test_two_principals_never_share_a_filtered_graph() -> None:
    admin = _view(ADMIN)
    alice = _view(ALICE)

    assert _sources(admin) == ["public/a.md", "secret/b.md"]
    assert _sources(alice) == ["public/a.md"]
    # Each is reused for its own scope, and only for its own.
    assert _view(ALICE) is alice
    assert _view(ADMIN) is admin
    assert _sources(_view(ALICE)) == ["public/a.md"]


def test_a_graph_outside_the_projection_cache_is_filtered_fresh() -> None:
    transient = _graph()

    first = service._authorized_graph(_Store(), transient, POLICY, ALICE)
    second = service._authorized_graph(_Store(), transient, POLICY, ALICE)

    assert first is not second
    assert _sources(first) == _sources(second) == ["public/a.md"]
