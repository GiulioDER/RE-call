"""Shared storage regressions from the integrated release's graph audit."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.types import Chunk


@pytest.mark.parametrize(
    ("relation", "seed_ord", "ordinals", "expected"),
    [
        ("source", 0, {"neighbor": -(2**63)}, ["neighbor"]),
        ("ordinal", 0, {"neighbor": -(2**63)}, []),
        ("source", 2**63, {"neighbor": 2**63 + 1}, ["neighbor"]),
        ("ordinal", 2**63, {"neighbor": 2**63 + 1}, ["neighbor"]),
        ("source", 0, {"a-far": 2**63 + 1, "z-near": 2**63}, ["z-near"]),
    ],
)
def test_related_ordering_keeps_exact_python_integers(
    tmp_path: Path, relation: str, seed_ord: int, ordinals: dict[str, int], expected: list[str]
) -> None:
    """Valid Python ordinals must preserve exact distance ordering at the public store entry."""
    chunks = [Chunk("seed", "notes.md", "seed", {"ord": seed_ord})]
    chunks.extend(Chunk(cid, "notes.md", cid, {"ord": ordinal}) for cid, ordinal in ordinals.items())
    with LiteStore(tmp_path / "memory.db", dim=2) as store:
        store.upsert(chunks, [[1.0, 0.0]] * len(chunks))
        try:
            result = store.related_chunks("seed", relation, 1)
        except (sqlite3.Error, OverflowError) as exc:
            actual = [type(exc).__name__]
        else:
            assert result is not None
            actual = [chunk.id for chunk in result[1]]
        assert actual == expected, "related ranking lost exact Python integer semantics"


def test_related_reads_one_revision_across_two_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source replacement after the seed read must not mix a newer neighbor with that seed."""
    path = tmp_path / "memory.db"
    old = [Chunk("seed", "notes.md", "old seed", {"ord": 0}), Chunk("peer", "notes.md", "old peer", {"ord": 1})]
    new = [Chunk("seed", "notes.md", "new seed", {"ord": 0}), Chunk("peer", "notes.md", "new peer", {"ord": 1})]
    with LiteStore(path, dim=2) as reader, LiteStore(path, dim=2) as writer:
        reader.upsert(old, [[1.0, 0.0]] * 2)
        lookup = reader.chunk_by_id
        changed = False

        def replace_after_seed(chunk_id: str) -> Chunk | None:
            nonlocal changed
            chunk = lookup(chunk_id)
            if chunk_id == "seed" and not changed:
                writer.upsert(new, [[1.0, 0.0]] * 2)
                changed = True
            return chunk

        monkeypatch.setattr(reader, "chunk_by_id", replace_after_seed)
        result = reader.related_chunks("seed", "source", 1)
        assert changed and writer.chunk_by_id("peer").text == "new peer", "negative control: replacement did not commit"
        assert result is not None
        assert (result[0].text, [chunk.text for chunk in result[1]]) == ("old seed", ["old peer"]), (
            "related retrieval mixed two source revisions"
        )
