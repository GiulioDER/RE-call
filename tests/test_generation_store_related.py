from recall.generation_store import GenerationStore

from datetime import UTC, datetime


class _Result:
    def __init__(self, *, one=None, many=()):
        self._one = one
        self._many = list(many)

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _Connection:
    def __init__(self, calls):
        self.calls = calls

    def execute(self, sql, params):
        self.calls.append((sql, params))
        if "AND chunk_id = %s" in sql:
            return _Result(
                one=("seed", "file:///uploads/a.md", "seed", {"file": "a.md", "ord": 0})
            )
        return _Result(
            many=[
                (
                    "neighbor",
                    "file:///uploads/a.md",
                    "neighbor",
                    {"file": "a.md", "ord": 1},
                )
            ]
        )


def test_generation_related_chunks_uses_generation_scoped_columns(monkeypatch) -> None:
    store = object.__new__(GenerationStore)
    store._tenant = "tenant-a"
    calls = []
    connection = _Connection(calls)
    monkeypatch.setattr(store, "_generation_id", lambda: "generation-1")
    monkeypatch.setattr(store, "_with_retry", lambda operation: operation(connection))

    seed, related = store.related_chunks("seed", "source", 1)

    assert seed.id == "seed"
    assert [chunk.id for chunk in related] == ["neighbor"]
    assert len(calls) == 2
    assert all("SELECT id" not in sql for sql, _params in calls)
    assert all("chunk_id" in sql and "generation_id" in sql for sql, _params in calls)
    assert all("tenant-a" not in sql for sql, _params in calls)
    assert calls[0][1] == ("tenant-a", "generation-1", "seed")


def test_scored_chunk_by_id_fetches_exactly_one_generation_bound_parent(monkeypatch) -> None:
    """Active rescue cannot fetch a same-id parent from another generation.

    Red proof receipt ``atomic-active-parent-load-01`` targets
    ``GenerationStore.scored_chunk_by_id``. Removing the generation predicate or parameter makes
    the SQL and parameter assertions fail.
    """

    indexed_at = datetime(2026, 9, 16, tzinfo=UTC)
    first_indexed_at = datetime(2026, 8, 1, tzinfo=UTC)
    calls = []

    class Connection:
        def execute(self, sql, params):
            calls.append((sql, params))
            return _Result(
                one=(
                    "rescued",
                    "file:///uploads/rescued.md",
                    "rescued text",
                    {"file": "rescued.md", "ord": 2},
                    indexed_at,
                    first_indexed_at,
                )
            )

    store = object.__new__(GenerationStore)
    store._tenant = "tenant-a"
    monkeypatch.setattr(store, "_generation_id", lambda: "generation-1")
    monkeypatch.setattr(store, "_with_retry", lambda operation: operation(Connection()))

    hit = store.scored_chunk_by_id("rescued", 0.75)

    assert hit is not None
    assert hit.chunk.id == "rescued"
    assert hit.score == 0.75
    assert hit.indexed_at == indexed_at
    assert hit.first_indexed_at == first_indexed_at
    assert "tenant_id = %s AND generation_id = %s" in calls[0][0]
    assert calls[0][1] == ("tenant-a", "generation-1", "rescued")


def test_both_postgres_ordinal_windows_follow_the_resolvers_constant(monkeypatch) -> None:
    """Invariant: the `ordinal` read of `PgVectorStore` and `GenerationStore` bounds its LIMIT by
    `recall.types.ORDINAL_WINDOW`, the window the resolver (`recall.related`) applies after it.

    Failure mode: a copied literal `<= 2` kept the PostgreSQL window when the constant moved, so a
    wider window returned fewer neighbours there than on Lite, and a narrower one spent LIMIT slots
    on rows the resolver drops (CCA CODE-001, run 20261115T155439Z).

    Red proof: this node against `recall/store.py` and `recall/generation_store.py` as of
    98ae7efba (both `... abs(...) <= 2`) fails its `<= 5` assertion; green with the fix.
    """
    from recall import types
    from recall.store import PgVectorStore

    # raising=False: against the pre-fix tree, which has no such name, this fails in the window
    # assertion below rather than in setup.
    monkeypatch.setattr(types, "ORDINAL_WINDOW", 5, raising=False)

    generation = object.__new__(GenerationStore)
    generation._tenant = "tenant-a"
    generation_calls = []
    monkeypatch.setattr(generation, "_generation_id", lambda: "generation-1")
    monkeypatch.setattr(generation, "_with_retry", lambda operation: operation(_Connection(generation_calls)))
    generation.related_chunks("seed", "ordinal", 3)

    class _LegacyConnection:
        def __init__(self, calls):
            self.calls = calls

        def execute(self, sql, params):
            self.calls.append((sql, params))
            if "AND id = %s" in sql:
                return _Result(one=("seed", "file:///uploads/a.md", "seed", {"file": "a.md", "ord": 0}))
            return _Result(many=[])

    legacy = object.__new__(PgVectorStore)
    legacy._tenant = "tenant-a"
    legacy._table = "chunks"
    legacy_calls = []
    monkeypatch.setattr(legacy, "_with_retry", lambda operation: operation(_LegacyConnection(legacy_calls)))
    legacy.related_chunks("seed", "ordinal", 3)

    for name, calls in (("GenerationStore", generation_calls), ("PgVectorStore", legacy_calls)):
        assert len(calls) == 2, f"{name} answered the ordinal relation without its bounded read"
        assert "<= 5" in calls[1][0], f"{name}'s ordinal window is not recall.types.ORDINAL_WINDOW"
