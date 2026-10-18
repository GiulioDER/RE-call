"""PR894 public regressions for FIX001, FIX002 and FIX003.

Baseline 6aa7ff5bdb4230ad620849eb1211e119859b04e4, measured 2026-10-13.
All cases fail at their named assertions before repair and pass unchanged afterward.
FIX001 targets LiteStore._hit metadata detachment, including nested values and
subsequent public scoped admission. FIX002 targets query_sparse's joined matrix,
FTS and scoring view. FIX003 targets delete_sources_across's empty-table guard.
The native-red receipt records exact nodes and intended failures. Data is synthetic.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.lite import store as lite_module
from recall.scope import Scope
from recall.types import Chunk, ScoredChunk

VECTOR = [1.0] + [0.0] * 7
OTHER = [0.0, 1.0] + [0.0] * 6


def _chunk(text: str = "alpha", **metadata: Any) -> Chunk:
    return Chunk(id="a", source="a.md", text=text, metadata=metadata)


def _query(store: LiteStore, leg: str, scope: Scope | None = None) -> list[ScoredChunk]:
    if leg == "dense":
        return store.query_dense(VECTOR, 1, scope=scope)
    return store.query_sparse("alpha", 1, vec=VECTOR, scope=scope)


@pytest.mark.parametrize("leg", ["dense", "sparse"])
@pytest.mark.parametrize("field", ["file", "type", "security_policy_digest", "numeric_values"])
def test_outbound_metadata_preserves_persisted_rows(tmp_path: Path, leg: str, field: str) -> None:
    """FIX001: direct and nested edits cannot change the next public result."""
    metadata = {"file": "private/a.md", "type": "stored", "security_policy_digest": "old", "numeric_values": ["5"]}
    with closing(LiteStore(tmp_path / "m.db", dim=8)) as store:
        store.upsert([_chunk(**metadata)], [VECTOR])
        hit = _query(store, leg)[0]
        if field == "numeric_values":
            hit.chunk.metadata[field].clear()
        else:
            hit.chunk.metadata[field] = "edited"
        with closing(LiteStore(store.path, dim=8)) as fresh:
            actual = _query(store, leg)[0].chunk.metadata
            assert actual == _query(fresh, leg)[0].chunk.metadata == metadata, "returned metadata changed persisted retrieval"


@pytest.mark.parametrize("leg", ["dense", "sparse"])
@pytest.mark.parametrize("field", ["file", "security_policy_digest", "type"])
def test_returned_hit_cannot_change_scoped_admission(tmp_path: Path, leg: str, field: str) -> None:
    """FIX001: changing presentation metadata cannot authorize a persisted row."""
    scopes = {
        "file": Scope(source_prefixes=("public",)),
        "security_policy_digest": Scope(security_policy_digest="current"),
        "type": Scope(facet="edited"),
    }
    values = {"file": "public/a.md", "security_policy_digest": "current", "type": "edited"}
    with closing(LiteStore(tmp_path / "m.db", dim=8)) as store:
        store.upsert([_chunk(file="private/a.md", security_policy_digest="old", type="stored")], [VECTOR])
        assert _query(store, leg, scopes[field]) == []
        _query(store, leg)[0].chunk.metadata[field] = values[field]
        assert _query(store, leg, scopes[field]) == [], "returned metadata rewrote scoped admission"


@pytest.mark.parametrize("keyword", ["alpha", "beta"])
@pytest.mark.parametrize("scored", [False, True])
def test_sparse_keyword_membership_uses_one_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keyword: str, scored: bool,
) -> None:
    """FIX002: an existing-ID update cannot join later FTS matches to earlier text."""
    with closing(LiteStore(tmp_path / "m.db", dim=8)) as store, closing(LiteStore(store.path, dim=8)) as writer:
        store.upsert([_chunk()], [VECTOR])
        original = store._load
        changed = False

        def read() -> Any:
            nonlocal changed
            matrix = original()
            if not changed:
                changed = True
                writer.upsert([_chunk("beta")], [OTHER])
            return matrix

        monkeypatch.setattr(store, "_load", read)
        hits = store.query_sparse(keyword, 1, vec=VECTOR if scored else None)
        expected = ["alpha"] if keyword == "alpha" else []
        assert [hit.chunk.text for hit in hits] == expected, "sparse query mixed keyword and row revisions"
        if scored and hits:
            assert hits[0].score == pytest.approx(1.0)
        assert [hit.chunk.text for hit in store.query_sparse("beta", 1)] == ["beta"]


def test_sparse_score_uses_the_selected_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX002: a second writer before scoring cannot replace the selected vector."""
    with closing(LiteStore(tmp_path / "m.db", dim=8)) as store, closing(LiteStore(store.path, dim=8)) as writer:
        store.upsert([_chunk()], [VECTOR])
        original = lite_module._matches

        def match(scope: Scope, row: Any) -> bool:
            writer.upsert([_chunk()], [OTHER])
            return original(scope, row)

        monkeypatch.setattr(lite_module, "_matches", match)
        hits = store.query_sparse("alpha", 1, vec=VECTOR)
        assert hits[0].score == pytest.approx(1.0), "sparse score used a later embedding revision"
        monkeypatch.setattr(lite_module, "_matches", original)
        assert store.cosines_for(["a"], VECTOR) == {"a": pytest.approx(0.0)}


def test_empty_table_selection_is_refused_without_deletion(tmp_path: Path) -> None:
    """FIX003 P1: an empty destructive target must not expand to the persisted table."""
    with closing(LiteStore(tmp_path / "m.db", dim=8)) as store, closing(LiteStore(store.path, dim=8)) as observer:
        store.upsert([_chunk()], [VECTOR])
        assert [hit.chunk.id for hit in observer.query_dense(VECTOR, 1)] == ["a"]
        error = None
        try:
            store.delete_sources_across([], ["a.md"])
        except ValueError as exc:
            error = exc
        assert error is not None, "empty table selection was not refused"
        assert store.count() == observer.count() == 1
        assert [hit.chunk.id for hit in observer.query_dense(VECTOR, 1)] == ["a"]
        assert store.delete_sources_across([], []) == 0
