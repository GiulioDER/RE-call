from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_production_atomic_fact_fresh_audit import (
    INPUT_HASHES,
    _render,
    audit_fresh_pool,
    build_source_views,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_render_degrades_structure_without_truncating_fact_content() -> None:
    """Red proof mutates `_render` to replace content with its first 200 characters.

    The assertion on the complete content then fails, proving this test observes the production
    invariant rather than only the length ceiling.
    """
    content = "Complete fact " + "content " * 37
    rung, rendered = _render(
        title="T" * 256,
        section="S" * 512,
        field="FACT",
        content=content,
    )

    assert rung == "drop-section"
    assert len(rendered) <= 800
    assert rendered.endswith(f"content: {content}")
    assert content in rendered


def _write_pool(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "memory"
    root.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(1, 251):
        path = root / f"fixture-{index:03d}.md"
        answer = f"Recorded fact {index} contains enough distinct words for reliable audit coverage."
        mode = index % 3
        if mode == 0:
            raw = f"# Fixture title {index}\n\nFACT: {answer}\n"
            construction = "extractive_field"
        elif mode == 1:
            raw = f"# Fixture title {index}\n\n## Outcome\n\n{answer}\n"
            construction = "extractive_heading"
        else:
            raw = f"{answer}\n"
            construction = "extractive_fallback"
        path.write_text(raw, encoding="utf-8", newline="\n")
        _, views = build_source_views(raw, f"recall/{path.name}")
        gold = views[0]
        rows.append(
            {
                "id": f"atomic-fact-fresh-{index:03d}",
                "query": f"Question {index}?",
                "expected_answerability": "answerable",
                "gold_sources": [f"recall/{path.name}"],
                "source_sha256": _sha(path.read_bytes()),
                "gold_ordinal": gold["parent_ordinal"],
                "answer_span": gold["content"],
                "answer_span_sha256": _sha(str(gold["content"]).encode()),
                "construction": construction,
            }
        )
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "input_hashes": dict(sorted(INPUT_HASHES.items())),
                "excluded_sources": 400,
                "eligible_sources": 500,
                "queries": rows,
            }
        ),
        encoding="utf-8",
    )
    return pool, {"recall": root}


def test_fresh_audit_passes_complete_parent_mapped_gold(tmp_path: Path) -> None:
    """The aggregate gate must observe complete content and its exact parent."""
    pool, roots = _write_pool(tmp_path)

    result = audit_fresh_pool(pool, roots)

    assert result["decision"] == "GO_BUILD_ATOMIC_FACT_CONTEXT4_SHADOW"
    assert result["failed_gates"] == []
    assert result["metrics"]["gold"]["unique_view"] == 250
    assert result["metrics"]["gold"]["parent_ordinal_match"] == 250
    assert result["metrics"]["coverage"]["oversized_views"] == 0


def test_field_eligibility_applies_after_label_separation() -> None:
    answer = "A" * 250 + " complete field fact with eight distinct trailing words here now."
    raw = f"# Long field fixture\n\nFACT: {answer}\n"

    _, views = build_source_views(raw, "recall/field.md")

    assert len(answer) <= 320
    assert len(f"FACT: {answer}") > len(answer)
    assert views[0]["content"] == answer
    assert views[0]["field"] == "FACT"
