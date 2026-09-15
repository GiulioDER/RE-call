from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_guarded_spare_slot_extractive_pool import _build_rows, build_pool


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _source(path: Path, title: str, field: str, value: str) -> None:
    path.write_text(
        f"# {title}\n\n{field}: {value}\n",
        encoding="utf-8",
        newline="\n",
    )


def _inputs(tmp_path: Path) -> tuple[list[tuple[str, Path]], Path, Path]:
    root = tmp_path / "memory"
    root.mkdir()
    _source(
        root / "kept.md",
        "A sufficiently descriptive kept memo",
        "FACT",
        "The canonical answer contains enough distinct words to qualify as an exact frozen span.",
    )
    _source(
        root / "old.md",
        "A sufficiently descriptive old memo",
        "FACT",
        "This old pool answer also contains enough distinct words but must never be selected again.",
    )
    _source(
        root / "traced.md",
        "A sufficiently descriptive traced memo",
        "FACT",
        "This traced answer contains enough distinct words but must remain development data only.",
    )
    trace = tmp_path / "trace.json"
    _write_json(trace, {"rows": [{"trace": {"pool": [{"source": "test/traced.md"}]}}]})
    old_pool = tmp_path / "old-pool.json"
    _write_json(
        old_pool,
        {"queries": [{"expected_answerability": "answerable", "gold_sources": ["test/old.md"]}]},
    )
    return [("test", root)], trace, old_pool


def test_pool_excludes_trace_and_prior_pool_sources(tmp_path: Path) -> None:
    """Red proof for ``_candidates``: omitting old-pool exclusion selected ``old.md``."""
    roots, trace, old_pool = _inputs(tmp_path)

    payload = build_pool(
        roots,
        trace,
        old_pool,
        count=1,
        seed="test-seed",
        negative_prefix="ZXQXACT",
    )

    positives = [
        item for item in payload["queries"] if item["expected_answerability"] == "answerable"
    ]
    assert [item["gold_sources"] for item in positives] == [["test/kept.md"]]
    assert positives[0]["gold_ordinal"] == 0
    assert positives[0]["answer_span"].startswith("The canonical answer")
    assert positives[0]["answer_span_sha256"] == hashlib.sha256(
        positives[0]["answer_span"].encode("utf-8")
    ).hexdigest()
    negatives = [
        item for item in payload["queries"] if item["expected_answerability"] == "unanswerable"
    ]
    assert len(negatives) == 1
    assert "ZXQXACT-0001" in negatives[0]["query"]
    assert negatives[0]["answer_span"] == "NOT_FOUND"


def test_pool_refuses_an_existing_negative_prefix(tmp_path: Path) -> None:
    roots, trace, old_pool = _inputs(tmp_path)
    with (roots[0][1] / "kept.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\nZXQXACT already exists here.\n")

    with pytest.raises(ValueError, match="negative prefix"):
        build_pool(
            roots,
            trace,
            old_pool,
            count=1,
            seed="test-seed",
            negative_prefix="ZXQXACT",
        )


def test_pool_refuses_when_too_few_eligible_sources_remain(tmp_path: Path) -> None:
    roots, trace, old_pool = _inputs(tmp_path)
    with pytest.raises(ValueError, match="INSUFFICIENT_POOL"):
        build_pool(
            roots,
            trace,
            old_pool,
            count=2,
            seed="test-seed",
            negative_prefix="ZXQXACT",
        )


def test_pool_continues_past_duplicate_questions() -> None:
    """Red proof for ``_build_rows``: slicing first stopped before a later unique item."""
    common = {
        "source_sha256": "a" * 64,
        "source_ordinal": 0,
        "answer_span": "A distinct answer span with enough words for this deterministic fixture.",
        "answer_span_sha256": "b" * 64,
        "construction": "extractive_field",
    }
    candidates = [
        {**common, "source": "one.md", "question": "What fact was recorded for shared?"},
        {**common, "source": "two.md", "question": "What fact was recorded for shared?"},
        {**common, "source": "three.md", "question": "What fact was recorded for unique?"},
    ]

    rows = _build_rows(candidates, 2, "test-seed", "ZXQXACT")

    assert len(rows) == 4
    assert {
        tuple(item["gold_sources"])
        for item in rows
        if item["expected_answerability"] == "answerable"
    } == {("one.md",), ("three.md",)}
