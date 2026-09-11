from recall.generation_store import GenerationStore


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
