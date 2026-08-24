from datetime import UTC, datetime

import pytest

from recall.current_state import project_current_state
from recall.types import Chunk


NOW = datetime(2026, 8, 24, tzinfo=UTC)


class Store:
    tenant = "tenant-a"
    generation_id = "gen-1"

    def __init__(self, chunks, candidates=()):
        self._chunks = chunks
        self._candidates = dict(candidates)

    def iter_chunks(self, batch_size=1000):
        del batch_size
        return iter(self._chunks)

    def supersession_all(self):
        return {}, frozenset(), self._candidates


def test_current_state_is_deterministic_and_generation_bound() -> None:
    store = Store(
        [
            Chunk("a", "a.md", "old", {"file": "a.md"}),
            Chunk("b", "b.md", "new", {"file": "b.md"}),
        ],
        {"a.md": [("b.md", NOW)]},
    )
    first = project_current_state(store, as_of=NOW)
    second = project_current_state(store, as_of=NOW)
    assert first == second
    assert first.generation_id == "gen-1"
    states = {record.source: record.state for record in first.records}
    assert states == {"a.md": "superseded", "b.md": "current"}


def test_current_state_fails_closed_on_multiple_live_successors() -> None:
    store = Store(
        [Chunk("a", "a.md", "old", {"file": "a.md"})],
        {"a.md": [("b.md", NOW), ("c.md", NOW)]},
    )
    record = project_current_state(store, as_of=NOW).records[0]
    assert record.state == "ambiguous"
    assert "multiple_live_successors" in record.diagnostics


def test_current_state_reports_validity_window_states() -> None:
    store = Store(
        [
            Chunk("future", "future.md", "future", {"file": "future.md", "valid_from": "2027-01-01"}),
            Chunk("past", "past.md", "past", {"file": "past.md", "valid_until": "2026-01-01"}),
        ]
    )
    states = {record.source: record.state for record in project_current_state(store, as_of=NOW).records}
    assert states == {"future.md": "not_yet_valid", "past.md": "expired"}


def test_current_state_fails_closed_on_malformed_supersession_metadata() -> None:
    store = Store([Chunk("bad", "bad.md", "bad", {"file": "bad.md", "supersedes": 7})])
    record = project_current_state(store, as_of=NOW).records[0]
    assert record.state == "invalid"
    assert "malformed_supersession_metadata" in record.diagnostics


def test_current_state_serving_bound_fails_closed_before_assembling_more_records() -> None:
    store = Store(
        [
            Chunk("a", "a.md", "a", {"file": "a.md"}),
            Chunk("b", "b.md", "b", {"file": "b.md"}),
        ]
    )
    with pytest.raises(ValueError, match="exceeds max_records"):
        project_current_state(store, as_of=NOW, max_records=1)


def test_current_state_rejects_an_unbounded_serving_limit() -> None:
    with pytest.raises(ValueError, match="<= 1000"):
        project_current_state(Store([]), as_of=NOW, max_records=1001)
