from __future__ import annotations

from datetime import UTC, date, datetime

from recall.query_entity_resolution import (
    decompose_query,
    extract_query_dates,
    resolve_query_entities,
)
from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk


def test_query_dates_support_equivalent_formats_and_relative_ranges() -> None:
    reference = datetime(2026, 9, 10, 12, tzinfo=UTC)

    dates = extract_query_dates(
        "changes on 2026-08-25, 25 August 2026, and last week",
        reference_time=reference,
    )

    assert [(item.start, item.end, item.precision) for item in dates] == [
        (date(2026, 8, 25), date(2026, 8, 25), "day"),
        (date(2026, 8, 31), date(2026, 9, 6), "relative"),
    ]


def test_query_decomposition_retains_full_query_and_bounded_clauses() -> None:
    assert decompose_query("What changed for A, and why did B change?") == (
        "What changed for A, and why did B change?",
        "What changed for A",
        "why did B change?",
    )


def test_query_resolution_matches_alias_and_date_without_mutating_graph() -> None:
    graph = build_semantic_graph(
        [
            Chunk(
                "target",
                "target.md",
                "target",
                {
                    "file": "target.md",
                    "entity_aliases": {"target.md": ["Target Run", "2026-08-25"]},
                },
            )
        ],
        tenant_id="tenant-a",
        generation_id="generation-a",
    )

    target = next(entity for entity in graph.entities if entity.canonical_name == "target.md")
    alias_result = resolve_query_entities(graph, "What happened to the Target Run?")
    date_result = resolve_query_entities(
        graph,
        "What happened on 25 August 2026?",
        reference_time=datetime(2026, 9, 10, tzinfo=UTC),
    )

    assert alias_result.entity_ids == (target.id,)
    assert {item.match_kind for item in alias_result.entities} == {"exact"}
    assert date_result.entity_ids == (target.id,)
    assert {item.match_kind for item in date_result.entities} == {"date"}
    assert date_result.dates[0].start == date(2026, 8, 25)
    assert graph.entities[0].canonical_name == "target.md"


def test_query_resolution_rejects_ambiguous_entity_labels() -> None:
    graph = build_semantic_graph(
        [
            Chunk("person", "person.md", "", {"person": "Shared"}),
            Chunk("project", "project.md", "", {"project": "Shared"}),
        ],
        tenant_id="tenant-a",
        generation_id="generation-a",
    )

    result = resolve_query_entities(graph, "Shared")

    assert result.entity_ids == ()
