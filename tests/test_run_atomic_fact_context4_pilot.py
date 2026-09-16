from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.run_atomic_fact_context4_pilot import (
    AtomicView,
    Candidate,
    _build_atomic_views,
    equal_rrf,
    rank_atomic,
    summarize,
)
from scripts.run_production_atomic_fact_fresh_audit import build_source_views


def _candidate(source: str, ordinal: int, score: float = 0.0) -> Candidate:
    return Candidate(source, ordinal, f"text for {source} {ordinal}", score)


def _item(source: str, ordinal: int, *, exact: bool, gold: bool) -> dict[str, object]:
    return {
        "source": source,
        "ordinal": ordinal,
        "score": 0.0,
        "exact": exact,
        "gold": gold,
    }


def test_rank_atomic_deduplicates_views_by_best_parent_score() -> None:
    views = [
        AtomicView("a.md", 0, 0, "a0", "parent a"),
        AtomicView("a.md", 0, 1, "a1", "parent a"),
        AtomicView("b.md", 2, 0, "b0", "parent b"),
    ]
    vectors = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]

    result = rank_atomic(views, vectors, [1.0, 0.0], cutoff=2)

    assert [item.identity for item in result] == [("a.md", 0), ("b.md", 2)]
    assert result[0].score == 1.0


def test_equal_rrf_uses_best_rank_then_dense_rank_for_ties() -> None:
    dense = [_candidate("a", 0), _candidate("b", 0), _candidate("c", 0)]
    atomic = [_candidate("c", 0), _candidate("b", 0), _candidate("d", 0)]

    result = equal_rrf(dense, atomic, cutoff=4)

    assert [item.identity for item in result] == [
        ("c", 0),
        ("b", 0),
        ("a", 0),
        ("d", 0),
    ]


def test_summarize_promotes_an_arm_that_clears_every_frozen_gate() -> None:
    rows = []
    for index in range(2):
        rows.append(
            {
                "arms": {
                    "dense": [_item(f"wrong-{index}", 0, exact=False, gold=False)] * 20,
                    "atomic": [_item(f"gold-{index}", 0, exact=True, gold=True)] * 20,
                    "equal_rrf": [_item(f"gold-{index}", 0, exact=True, gold=True)] * 20,
                }
            }
        )

    result = summarize(rows)

    assert result["decision"] == "PROMISING_ATOMIC_FACT_PILOT"
    assert result["chosen_arm"] == "atomic"
    assert result["qualifying_arms"] == ["atomic", "equal_rrf"]


def test_summarize_stops_when_rank_one_gain_is_too_small() -> None:
    row = {
        "arms": {
            "dense": [_item("wrong", 0, exact=False, gold=False)] * 20,
            "atomic": [_item("gold", 0, exact=True, gold=True)] * 20,
            "equal_rrf": [_item("gold", 0, exact=True, gold=True)] * 20,
        }
    }

    result = summarize([row])

    assert result["decision"] == "STOP_ATOMIC_FACT_RETRIEVAL_PILOT"
    assert result["chosen_arm"] is None


def test_atomic_views_preserve_manifest_crlf_chunk_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    path = root / "note.md"
    paragraphs = [
        f"Paragraph {index}: " + ("context words " * 14) + f"unique fact {index}."
        for index in range(8)
    ]
    data = ("\r\n\r\n".join(paragraphs) + "\r\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    source = "test/note.md"
    production_chunks, expected_views = build_source_views(data.decode(), source)
    parents = {(source, ordinal): text for ordinal, text in enumerate(production_chunks)}
    objects = [
        {
            "uri": path.as_uri(),
            "version_id": digest,
            "media_type": "text/markdown",
            "size": len(data),
            "sha256": digest,
            "context_group_id": None,
        }
    ]

    groups, metrics = _build_atomic_views(objects, {"test": root}, parents)

    assert metrics["atomic_views"] == len(expected_views)
    assert [view.parent_ordinal for view in groups[0]] == [
        int(view["parent_ordinal"]) for view in expected_views
    ]


def test_atomic_views_exclude_frozen_index_filenames(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    path = root / "MEMORY.md"
    path.write_text("changed aggregate", encoding="utf-8")
    objects = [
        {
            "uri": path.as_uri(),
            "version_id": "0" * 64,
            "media_type": "text/markdown",
            "size": 1,
            "sha256": "0" * 64,
            "context_group_id": None,
        }
    ]

    groups, metrics = _build_atomic_views(objects, {"test": root}, {})

    assert groups == []
    assert metrics["manifest_objects"] == 0
    assert metrics["excluded_index_sources"] == 1
