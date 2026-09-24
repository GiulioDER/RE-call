"""A cached reasoning graph's own identity checks run once, not on every reasoning query.

`_validate_graph_binding` ended every reasoning query by checking the tenant and generation of
every node, edge and diagnostic of the graph, and `_validate_proposals` rebuilt the set of every
node id. Both depend only on the graph, which is immutable and cached per generation, so the
answers are kept per graph object. A graph that fails is never recorded as checked, so it is
refused on every query.

Red proofs (2026-09-23, base ``c7f2b9bc``):

* ``tests/test_reasoning_graph_validation_once.py::test_a_graph_is_checked_once_however_often_it_is_validated``
  failed against the unchanged ``recall.reasoning._validate_graph_members`` with
  ``assert checks == first_pass`` (the second validation checked every member again).
* ``tests/test_reasoning_graph_validation_once.py::test_a_graph_with_a_foreign_member_is_refused_every_time``,
  against a mutation recording the graph as checked before its members are checked, failed with
  ``DID NOT RAISE`` on the second validation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import recall.reasoning as reasoning
from recall.reasoning import ReasoningValidationError
from recall.reasoning_graph import build_reasoning_graph
from recall.types import Chunk


def _graph():
    return build_reasoning_graph(
        [
            Chunk("a", "a.md", "decision: a.", {"file": "a.md", "ord": 0}),
            Chunk("b", "b.md", "decision: b.", {"file": "b.md", "ord": 0, "supersedes": "a.md"}),
        ],
        tenant_id="acme",
        generation_id="gen-1",
        include_text=True,
    )


def _count_checks(monkeypatch) -> list[int]:
    calls = [0]
    original = reasoning._check_member_identity

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reasoning, "_check_member_identity", counted)
    return calls


def test_a_graph_is_checked_once_however_often_it_is_validated(monkeypatch) -> None:
    calls = _count_checks(monkeypatch)
    graph = _graph()

    reasoning._validate_graph_members(graph)
    first_pass = calls[0]
    reasoning._validate_graph_members(graph)
    reasoning._validate_graph_members(graph)
    checks = calls[0]

    assert first_pass == len(graph.nodes) + len(graph.authored_edges) + len(
        graph.inferred_candidate_edges
    ) + len(graph.diagnostics)
    assert first_pass > 0
    assert checks == first_pass
    # A different graph object, even an equal one, is checked on its own.
    reasoning._validate_graph_members(_graph())
    assert calls[0] == 2 * first_pass


def test_a_graph_with_a_foreign_member_is_refused_every_time() -> None:
    graph = _graph()
    foreign = replace(graph, nodes=(replace(graph.nodes[0], tenant_id="other"), *graph.nodes[1:]))

    for _ in range(2):
        with pytest.raises(ReasoningValidationError, match="tenant_id does not match graph"):
            reasoning._validate_graph_members(foreign)


def test_a_proposal_citing_unknown_evidence_is_refused_on_a_graph_seen_before() -> None:
    graph = _graph()
    reasoning._validate_proposals(graph, [])

    class _Proposal:
        generation_id = graph.generation_id
        pipeline_id = graph.pipeline_fingerprint
        source_evidence_ids = (graph.nodes[0].id, "rg_node_not_in_this_graph")

    with pytest.raises(ReasoningValidationError, match="rg_node_not_in_this_graph"):
        reasoning._validate_proposals(graph, [_Proposal()])
