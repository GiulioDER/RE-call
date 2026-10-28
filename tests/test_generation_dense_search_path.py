"""A generation-scoped dense query is exact by default, and HNSW only when an operator asks.

Invariant: with `RECALL_GENERATION_DENSE_SEARCH` unset, `GenerationStore.query_dense` runs under
`_EXACT_SCAN_GUARDS` and never issues the HNSW tuning statement.

Failure mode caught: the leg depending on the planner again. Before this, the exact plan was served
only because a pooled connection's prepared statement fell back to a cached generic plan; measured
2026-10-01 on a copy of the production table, the custom HNSW plan returned 0.875 of the memory
tenant's true top 20 (0.918 at the maximum `ef_search`), and a planner statistic was enough to
switch to it silently. These tests pin the path by the statements sent, which no plan cache,
statistic or GUC can change.

Red proof (2026-10-01, Linux), node
`tests/test_generation_dense_search_path.py::test_the_default_is_exact_and_never_tunes_hnsw`:
- mutating the default in `recall.generation_store._generation_dense_search` from `"exact"` to
  `"hnsw"` fails the guard assertion, the statements being the HNSW tuning and the search;
- deleting the `exact` branch at the top of `GenerationStore._query_dense` fails the same way.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import pytest

from recall.generation_store import GENERATION_DENSE_SEARCH_ENV, GenerationStore
from recall.store import _EXACT_SCAN_GUARDS, _HNSW_FILTERED_TUNING_SQL

_ROW = ("chunk-1", "a.md", "text", {}, None, None, 0.9)


class _RecordingStore(GenerationStore):
    """A generation store whose connection records every statement and returns one row.

    One row, so the HNSW path never reaches `_dense_exact_fallback`, which runs the exact guards
    itself and would make an HNSW query look exact.
    """

    def __init__(self) -> None:  # no database: nothing here connects
        self._tenant = "acme"
        self._pinned_generation = ContextVar("pinned_generation", default="gen-1")
        self._fixed_generation = None
        self.statements: list[str] = []

    def _with_retry(self, op):  # type: ignore[no-untyped-def]
        return op(self)

    @contextmanager
    def transaction(self):  # type: ignore[no-untyped-def]
        yield

    def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
        self.statements.append(str(sql))
        return self

    def fetchall(self):  # type: ignore[no-untyped-def]
        return [_ROW]


def _search(monkeypatch, mode: str | None) -> list[str]:
    if mode is None:
        monkeypatch.delenv(GENERATION_DENSE_SEARCH_ENV, raising=False)
    else:
        monkeypatch.setenv(GENERATION_DENSE_SEARCH_ENV, mode)
    store = _RecordingStore()
    hits = store.query_dense([0.1, 0.2, 0.3], 5)
    assert [hit.chunk.id for hit in hits] == ["chunk-1"]
    return store.statements


def test_the_default_is_exact_and_never_tunes_hnsw(monkeypatch) -> None:
    statements = _search(monkeypatch, None)

    assert all(guard in statements for guard in _EXACT_SCAN_GUARDS), (
        f"the default generation dense query must run under _EXACT_SCAN_GUARDS; it sent {statements}"
    )
    assert _HNSW_FILTERED_TUNING_SQL not in statements, (
        "the default path issued the HNSW tuning, so it is planning for the index"
    )


def test_hnsw_stays_reachable_when_an_operator_asks(monkeypatch) -> None:
    statements = _search(monkeypatch, "hnsw")

    assert _HNSW_FILTERED_TUNING_SQL in statements
    assert not any(guard in statements for guard in _EXACT_SCAN_GUARDS)


def test_an_unknown_path_is_refused_by_name(monkeypatch) -> None:
    monkeypatch.setenv(GENERATION_DENSE_SEARCH_ENV, "approximate")
    with pytest.raises(ValueError, match="RECALL_GENERATION_DENSE_SEARCH"):
        _RecordingStore().query_dense([0.1, 0.2, 0.3], 5)


@pytest.mark.parametrize(("raw", "expected"), [(" HNSW ", "hnsw"), ("Exact", "exact"), ("", "exact")])
def test_case_spaces_and_an_empty_value_are_normalised(monkeypatch, raw: str, expected: str) -> None:
    """CCA of #847, ENV-001: the key reads like the other mode keys (case and spaces ignored, empty
    means the default), so a cosmetic difference is not a failure on every search.

    Red proof, 2026-10-28: with the parser before this fix (an exact, case-sensitive match), each
    case raised ValueError instead of taking its path.
    """
    statements = _search(monkeypatch, raw)
    if expected == "hnsw":
        assert _HNSW_FILTERED_TUNING_SQL in statements
    else:
        assert all(guard in statements for guard in _EXACT_SCAN_GUARDS)


def test_mcp_startup_refuses_an_unknown_path() -> None:
    """CCA of #847, ENV-001: a bad value refuses at startup, naming the variable. It used to pass
    startup and the readiness handshake, then fail every search, which federation's default
    `omit` records only as a failed leg.

    Red proof, 2026-10-28: with the `parse_generation_dense_search` call removed from
    `recall_mcp.settings._validate_runtime_options`, `Settings.from_env` returned normally
    (DID NOT RAISE).
    """
    from recall_mcp.settings import Settings

    with pytest.raises(ValueError, match="RECALL_GENERATION_DENSE_SEARCH"):
        Settings.from_env({"RECALL_GENERATION_DENSE_SEARCH": "approximate"})
    Settings.from_env({"RECALL_GENERATION_DENSE_SEARCH": " HNSW "})
