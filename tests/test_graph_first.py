from __future__ import annotations

from recall.graph_first import build_graph_first_candidates
from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk


def _graph():
    return build_semantic_graph(
        [
            Chunk(
                "c1",
                "release.md",
                "Release decision",
                {
                    "file": "release.md",
                    "recall_graph": {
                        "entities": [
                            {"name": "Release", "kind": "project"},
                            {"name": "Deploy", "kind": "service"},
                        ],
                        "relations": [
                            {
                                "relation": "supports",
                                "subject": "Release",
                                "object": "Deploy",
                            }
                        ],
                    },
                },
            )
        ],
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )


def test_graph_first_candidates_are_deterministic_and_bounded():
    graph = _graph()
    first = build_graph_first_candidates(graph, "release procedure", max_candidates=3)
    second = build_graph_first_candidates(graph, "release procedure", max_candidates=3)

    assert first == second
    assert len(first) == 2
    assert first[0].kind == "entity"
    assert first[1].kind == "relation"
    assert all(len(candidate.query) <= 2_000 for candidate in first)
    assert all(not candidate.query.startswith("-") for candidate in first)


def test_graph_first_uses_exact_aliases_and_no_graph_text():
    graph = _graph()
    assert build_graph_first_candidates(graph, "unrelated operation") == ()
    candidates = build_graph_first_candidates(graph, "release procedure")
    assert all("Release decision" not in candidate.query for candidate in candidates)
    assert candidates[0].entity_ids
    assert candidates[1].relation_ids


def test_graph_first_refuses_invalid_limits():
    graph = _graph()
    for mode in ("invalid", "entity"):
        try:
            build_graph_first_candidates(graph, "release", mode=mode, max_candidates=0)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("invalid graph-first limits must be rejected")
