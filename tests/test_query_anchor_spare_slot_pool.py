from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_query_anchor_spare_slot_pool import build_pool


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pool_excludes_every_consumed_extractive_source(tmp_path: Path) -> None:
    """A consumed source cannot leak into the new untouched holdout.

    Red proof targets ``build_pool``. The deliberate baseline replaced consumed source loading with
    an empty set, so the function returned a pool instead of raising ``INSUFFICIENT_POOL``.
    """
    root = tmp_path / "memory"
    root.mkdir()
    source = root / "used.md"
    source.write_text(
        "# A sufficiently descriptive used memory title\n\n"
        "## Result\n\n"
        "This recorded result contains enough exact words to form a deterministic answer span.\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.json"
    old_pool = tmp_path / "old.json"
    consumed = tmp_path / "consumed.json"
    _write_json(trace, {"rows": []})
    _write_json(old_pool, {"queries": []})
    _write_json(
        consumed,
        {"queries": [{"gold_sources": ["recall/used.md"]}]},
    )

    with pytest.raises(ValueError, match="INSUFFICIENT_POOL"):
        build_pool(
            [("recall", root)],
            trace,
            old_pool,
            consumed,
            count=1,
            seed="query-anchor-spare-slot-v1",
        )


def test_pool_builds_alphabetic_absent_controls_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    source = root / "fresh.md"
    source.write_text(
        "# A sufficiently descriptive fresh memory title\n\n"
        "## Result\n\n"
        "This fresh result contains enough exact words to form a deterministic answer span.\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.json"
    old_pool = tmp_path / "old.json"
    consumed = tmp_path / "consumed.json"
    for path, payload in (
        (trace, {"rows": []}),
        (old_pool, {"queries": []}),
        (consumed, {"queries": []}),
    ):
        _write_json(path, payload)

    first = build_pool(
        [("recall", root)], trace, old_pool, consumed, count=1, seed="seed"
    )
    second = build_pool(
        [("recall", root)], trace, old_pool, consumed, count=1, seed="seed"
    )
    control = next(
        row for row in first["queries"] if row["expected_answerability"] == "unanswerable"
    )
    nonce = control["query"].split()[3]

    assert first == second
    assert nonce.isalpha()
    assert 10 <= len(nonce) <= 14
    assert nonce.casefold() not in source.read_text(encoding="utf-8").casefold()
