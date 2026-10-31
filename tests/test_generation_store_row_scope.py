"""Every chunk-table read `GenerationStore` inherits is confined to its pinned generation.

`recall_chunks_v1` holds every generation a tenant has built, so a read filtered by tenant alone
mixes them: an inventory lists folders only a retired generation holds, a centroid averages two
generations of one folder, and `top_cosine` reads the parent's vectors during carry-forward and
reports that the corpus has not moved. These reads used to be overridden one by one in
`GenerationStore`, each restating the base SQL with a generation predicate. They are now the base
statements, confined through `PgVectorStore._row_scope`, so the confinement is INHERITED, and an
inherited guarantee is exactly the one a new base method can forget without anything failing.

Two checks, no database. Each inherited read is run against a recording connection, and every
statement it sends to `recall_chunks_v1` must carry the generation predicate with the pinned
generation bound. And the set of base methods that touch the chunk table and are not overridden is
derived from the source, so a new one fails here, by name, until it is either checked or exempted
with a reason.

Red proof, recorded per the repository rule (each mutation restored afterwards):

* `test_inherited_reads_bind_the_pinned_generation` failed for all 13 cases with "sent a
  recall_chunks_v1 statement without the generation predicate" when
  `GenerationStore._row_scope`/`_row_scope_args` were mutated to return no predicate, and failed
  for `count` alone when `{scope}` was deleted from `PgVectorStore.count`'s statement.
* `test_every_inherited_chunk_table_read_is_checked_or_exempt` failed naming `probe_rows` when a
  method `probe_rows` reading `self._table` was added to `PgVectorStore`, and (after S7d split the
  store into mixins) failed the same way when `probe_mixin_rows` was added to
  `recall.store_supersession.SupersessionMixin` instead.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from recall import generation_store as generation_store_mod
from recall.generation_store import GenerationStore
from recall.store import PgVectorStore

GENERATION = "gen_pinned_for_row_scope"
DIM = 4
VEC = [0.1] * DIM


class _Cursor:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self._log = log
        self.itersize = 0

    def execute(self, sql: object, params: object = None, **_: object) -> "_Cursor":
        self._log.append((re.sub(r"\s+", " ", str(sql)).strip(), params))
        return self

    def fetchall(self) -> list[Any]:
        return []

    def fetchone(self) -> None:
        return None

    def fetchmany(self, *_: object) -> list[Any]:
        return []

    def __iter__(self):
        return iter(())

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _Connection:
    closed = False
    broken = False

    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self._log = log

    def execute(self, sql: object, params: object = None, **_: object) -> _Cursor:
        return _Cursor(self._log).execute(sql, params)

    def cursor(self, *_: object, **__: object) -> _Cursor:
        return _Cursor(self._log)

    @contextmanager
    def transaction(self, *_: object, **__: object):
        yield self

    def close(self) -> None:
        pass


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> tuple[GenerationStore, list[tuple[str, Any]]]:
    log: list[tuple[str, Any]] = []
    monkeypatch.setattr(PgVectorStore, "_connect", lambda self: _Connection(log))
    store = GenerationStore("postgresql://u@localhost/db", DIM, tenant="t1")
    monkeypatch.setattr(store, "_generation_id", lambda: GENERATION)
    return store, log


#: Each inherited chunk-table read, called the way a caller calls it. A generator is drained,
#: because its statement is sent only when it is first advanced.
CALLS: dict[str, Callable[[GenerationStore], object]] = {
    "_cosines_for": lambda s: s._cosines_for(["c1"], VEC),
    "_iter_chunks_with_times": lambda s: list(s._iter_chunks_with_times(10, include_text=False)),
    "_query_dense_exact": lambda s: s._query_dense_exact(VEC, 5, "a.md"),
    "chunk_metadata_by_ids": lambda s: s.chunk_metadata_by_ids(["c1"]),
    "chunks_by_ids": lambda s: s.chunks_by_ids(["c1"]),
    "count": lambda s: s.count(),
    "iter_chunks": lambda s: list(s.iter_chunks(10)),
    "scope_centroids": lambda s: s.scope_centroids("folder"),
    "scope_inventory": lambda s: s.scope_inventory("folder"),
    "scope_undeclared_count": lambda s: s.scope_undeclared_count("folder"),
    "top_cosine": lambda s: s.top_cosine(VEC),
}

#: `_query_dense` is overridden, but only to choose a path; both paths are base statements.
EXTRA_CALLS: dict[str, Callable[[GenerationStore], object]] = {
    "_query_dense[hnsw]": lambda s: s._query_dense(VEC, 5, "a.md"),
    "iter_chunks_with_times": lambda s: list(s.iter_chunks_with_times(10)),
}

#: Inherited methods that touch the chunk table and are NOT confined, each with why.
EXEMPT: dict[str, str] = {
    "_dense_exact_fallback": "runs the statement its caller built; checked through _query_dense",
    "_upsert_in": "a write; GenerationStore refuses every write before reaching it",
    "analyze": "ANALYZE is table maintenance, not a read of rows",
    "analyze_if_stale": "ANALYZE is table maintenance, not a read of rows",
    "sparse_row_count": "reads recall_sparse_v1, keyed by chunk table, not the chunk rows",
    "table": "returns the table name; sends no statement",
    # ⚠️ Pre-existing and NOT decided: both read every generation of the tenant. The only caller
    # of the first is recall_aml, which passes a plain PgVectorStore; the second reports
    # table-level readiness. Listed so the gap is visible, not because it is known to be right.
    "explicit_superseded_chunk_ids": "pre-existing, tenant-wide on GenerationStore; undecided",
    "readiness_facts": "pre-existing, tenant-wide on GenerationStore; undecided",
}


def _inherited_chunk_table_methods() -> set[str]:
    def methods(module: object, name: str) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))  # type: ignore[arg-type]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        # Under mutation testing the file is mutmut's rewrite: each method keeps its name and its
        # real body, and its mutants sit beside it as `xǁClassǁname__mutmut_N`. Those are not
        # methods of the class's design, and counting them failed the job's clean stats run.
        return {
            n.name: n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and "__mutmut_" not in n.name
        }

    # Every class PgVectorStore is assembled from, not only its own body: a chunk-table read moved
    # into a mixin (S7d) is inherited by GenerationStore all the same.
    base: dict[str, ast.FunctionDef] = {}
    for klass in reversed(PgVectorStore.__mro__):
        if klass.__module__.startswith("recall"):
            base.update(methods(sys.modules[klass.__module__], klass.__name__))
    overridden = set(methods(generation_store_mod, "GenerationStore"))
    touching = {name for name, fn in base.items() if "self._table" in ast.unparse(fn)}
    return touching - overridden


def test_every_inherited_chunk_table_read_is_checked_or_exempt() -> None:
    inherited = _inherited_chunk_table_methods()
    assert inherited, "found no inherited chunk-table methods; the source walk is vacuous"
    unaccounted = inherited - set(CALLS) - set(EXEMPT)
    assert not unaccounted, (
        f"GenerationStore inherits chunk-table reads nothing checks: {sorted(unaccounted)}. "
        "Confine each through _row_scope and add it to CALLS, or exempt it with a reason."
    )
    stale = (set(CALLS) | set(EXEMPT)) - inherited
    assert not stale, f"listed but no longer inherited chunk-table methods: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted(CALLS) + sorted(EXTRA_CALLS))
def test_inherited_reads_bind_the_pinned_generation(
    name: str, recorded: tuple[GenerationStore, list[tuple[str, Any]]], monkeypatch
) -> None:
    store, log = recorded
    monkeypatch.setenv(generation_store_mod.GENERATION_DENSE_SEARCH_ENV, "hnsw")
    (CALLS | EXTRA_CALLS)[name](store)

    reads = [(sql, params) for sql, params in log if "recall_chunks_v1" in sql]
    assert reads, f"{name} sent no recall_chunks_v1 statement; the case checks nothing"
    for sql, params in reads:
        assert re.search(r"\bgeneration_id = %(\(generation\))?s", sql), (
            f"{name} sent a recall_chunks_v1 statement without the generation predicate: {sql}"
        )
        bound = params.values() if isinstance(params, dict) else params
        assert GENERATION in list(bound), f"{name} did not bind the pinned generation: {params}"
