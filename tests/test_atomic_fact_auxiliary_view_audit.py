from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_atomic_fact_auxiliary_view import (
    audit_atomic_fact_view,
    build_source_views,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pool(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "memory"
    root.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(1, 251):
        path = root / f"project-{index:03d}.md"
        answer = (
            f"Atomic fact {index} records enough unique words to make this deterministic fixture valid."
        )
        raw = f"# Project {index} title\n\n## Outcome\n\nFACT: {answer}\n"
        path.write_text(raw, encoding="utf-8", newline="\n")
        rows.append(
            {
                "id": f"extractive-positive-{index:03d}",
                "query": f"What fact was recorded for project {index}?",
                "expected_answerability": "answerable",
                "gold_sources": [f"recall/{path.name}"],
                "source_sha256": _sha256_bytes(path.read_bytes()),
                "gold_ordinal": 0,
                "answer_span": answer,
                "answer_span_sha256": _sha256_bytes(answer.encode()),
                "construction": (
                    "extractive_field"
                    if index % 3 == 0
                    else "extractive_heading"
                    if index % 3 == 1
                    else "extractive_fallback"
                ),
            }
        )
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"queries": rows}), encoding="utf-8")
    return pool, {"recall": root}


def test_source_view_is_fact_sized_and_maps_to_unchanged_parent() -> None:
    raw = (
        "---\ntype: project\n---\n\n# Retrieval Project\n\n## Result\n\n"
        "FACT: The atomic auxiliary view keeps the complete recorded fact in one small unit.\n\n"
        "A second paragraph has enough words to become another independent retrieval view.\n"
    )

    chunks, views = build_source_views(raw, "recall/project.md")

    assert len(chunks) == 1
    assert views[0]["parent_ordinal"] == 0
    assert views[0]["rendered"] == (
        "title: Retrieval Project\n"
        "section: Retrieval Project > Result\n"
        "field: FACT\n"
        "content: The atomic auxiliary view keeps the complete recorded fact in one small unit."
    )
    assert views[1]["rendered"].endswith(
        "content: A second paragraph has enough words to become another independent retrieval view."
    )
    assert "source:" not in views[0]["rendered"]


def test_audit_passes_bounded_fact_view_with_complete_gold(tmp_path: Path) -> None:
    pool, roots = _write_pool(tmp_path)

    result = audit_atomic_fact_view(pool, roots, enforce_frozen_hash=False)

    assert result["decision"] == "GO_BUILD_ATOMIC_FACT_AUXILIARY_SHADOW"
    assert result["failed_gates"] == []
    assert result["metrics"]["gold"]["unique_view"] == 250
    assert result["metrics"]["gold"]["parent_ordinal_match"] == 250
    assert result["metrics"]["gold"]["complete_content"] == 250
    assert result["metrics"]["coverage"]["oversized_views"] == 0


def test_audit_stops_when_gold_maps_to_the_wrong_parent(tmp_path: Path) -> None:
    pool, roots = _write_pool(tmp_path)
    payload = json.loads(pool.read_text(encoding="utf-8"))
    payload["queries"][0]["gold_ordinal"] = 1
    pool.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_atomic_fact_view(pool, roots, enforce_frozen_hash=False)

    assert result["decision"] == "STOP_ATOMIC_FACT_AUXILIARY_SHADOW"
    assert result["gates"]["gold_parent_matches"] is False
    assert "gold_parent_matches" in result["failed_gates"]
