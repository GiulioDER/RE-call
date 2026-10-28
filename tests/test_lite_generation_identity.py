"""A lite search and a lite current-state projection name the lite generation, not "None".

`LiteStore.snapshot` pins a SQLite read view and yields None. `_searched_generation` in
`recall_mcp/retrieval.py` and `project_current_state` in `recall/current_state.py` both entered any
store's snapshot and used `str(generation_id)`, so on a lite server `recall_search` and
`recall_evidence` reported `index_generation: "None"` and a current-state projection was bound to
generation "None", while the lite calibration binding names `LiteStore.lite_generation_id()`.

Red proof, recorded 2026-10-28 against master `81215e0a7` (before the fix), run DB-less on the
test host:

- `tests/test_lite_generation_identity.py::test_a_lite_search_names_the_lite_generation` fails at
  `assert retrieval.result["index_generation"] == store.lite_generation_id()`
  (`'None' == 'lite-...'`), targeting `_searched_generation`'s `yield str(generation_id)`;
- `tests/test_lite_generation_identity.py::test_a_lite_current_state_names_the_lite_generation`
  fails at `assert projection.generation_id == store.lite_generation_id()` (`'None' == 'lite-...'`),
  targeting `project_current_state`'s `str(generation_id)` inside the snapshot.

Both pass once a snapshot that yields None reads `store.generation_id` inside the pinned view.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from recall.current_state import project_current_state
from recall.lite.store import LiteStore
from recall.types import Chunk
from recall_mcp.retrieval import _retrieve_trusted

NOW = datetime(2026, 10, 28, tzinfo=UTC)


def _store(tmp_path: Path) -> LiteStore:
    store = LiteStore(tmp_path / "memory.db", dim=4, tenant="default")
    store.upsert([Chunk("a", "a.md", "the rate limit is ten", {"file": "a.md"})], [[1.0, 0.0, 0.0, 0.0]])
    return store


def _search(store, embedder, query, *, index_generation, **_kwargs):
    with store.snapshot():
        return {"index_generation": index_generation}


def test_a_lite_search_names_the_lite_generation(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        retrieval = _retrieve_trusted(
            store,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            "what is the rate limit",
            None,
            5,
            None,
            None,
            env={},
            reranker_builder=lambda *_args, **_kwargs: None,
            admission_factory=lambda _profile: nullcontext(),
            trusted_search_fn=_search,
        )
        assert retrieval.result["index_generation"] == store.lite_generation_id()


def test_a_lite_current_state_names_the_lite_generation(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        projection = project_current_state(store, as_of=NOW)
        assert projection.generation_id == store.lite_generation_id()
