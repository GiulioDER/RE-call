"""LW-1: each retrieved session's last window follows the top ranks, behind a default-off flag.

Red proofs, each run with ``PYTHONDONTWRITEBYTECODE=1`` against a deliberate mutation of the named
production line, each failing at the assertion named here, then restored and run green:

- ``test_last_window_is_the_highest_segment_of_a_one_add_session``: ``last_window`` taking ``min``
  instead of ``max`` picks segment 0, failing the ``== "s1-seg5"`` assertion.
- ``test_a_session_split_over_several_adds_has_no_last_window``: the several-Adds check removed
  (``if not raws:`` alone) returns the first Add's segment 5, failing the ``is None`` assertion.
- ``test_last_windows_follow_the_top_block_in_first_appearance_order``: iterating
  ``sorted(sources)`` instead of first appearance in ``with_last_windows`` puts session A's window
  first, failing the ranked id list assertion.
- ``test_a_last_window_already_ranked_is_neither_added_nor_repeated``: dropping the
  ``last.id not in seen`` check adds the head's own window again, failing the ``added == 0``
  assertion.
- ``test_c9_serves_no_last_windows_unless_the_flag_is_set``: ``last_window_append`` defaulting to
  ``True`` on ``HostedVariant`` fails the ``is False`` assertion.
- ``test_flag_on_places_the_session_last_window_after_the_top_ten``: skipping
  ``run.hits[:] = extended`` in ``HostedService.search`` leaves the ranking unchanged, failing the
  ``ids_on[10] == "s1-w11"`` assertion.
- ``test_a_failed_lookup_serves_the_ranking_without_last_windows``: narrowing the LW-1
  ``except Exception`` in ``HostedService.search`` to ``except ValueError`` lets the store's
  ``RuntimeError`` escape Search, so the test fails at its ``_search(failing, ...)`` call instead
  of receiving the unchanged ranking.
"""

from __future__ import annotations

import asyncio

import pytest

from recall.types import Chunk, ScoredChunk
from recall_aml.identity import tenant_for
from recall_aml.last_window import last_window, with_last_windows
from recall_aml.models import SearchRequest
from recall_aml.variants import variant
from tests.test_aml_specialist_fusion import _Store, _service


def _raw(chunk_id: str, session: str, segment: int, event_time: str = "2026-09-01T10:00:00Z") -> Chunk:
    return Chunk(
        id=chunk_id,
        source=f"aml://session/{session}",
        text=f"{session} window {segment}",
        metadata={
            "kind": "raw",
            "record_type": "raw",
            "source_session_id": session,
            "segment": segment,
            "event_time": event_time,
        },
    )


def test_last_window_is_the_highest_segment_of_a_one_add_session() -> None:
    """The final window of a session stored by one Add, whatever order the store returns it in."""
    windows = [_raw(f"s1-seg{i}", "s1", i) for i in range(6)]
    compiled = Chunk(
        id="s1-compiled",
        source="aml://session/s1",
        text="decision: ship it",
        metadata={"kind": "architectural decision", "segment": 9},
    )
    flagged = _raw("s1-bool", "s1", 0)
    flagged.metadata["segment"] = True

    chosen = last_window([windows[3], compiled, flagged, windows[5], *windows[:3], windows[4]])

    assert chosen is not None
    assert chosen.id == "s1-seg5"
    assert last_window([compiled]) is None


def test_a_session_split_over_several_adds_has_no_last_window() -> None:
    """Two Adds of one session both start at segment 0 and share a timestamp; neither end is chosen."""
    first = [_raw(f"s1-add1-seg{i}", "s1", i) for i in range(6)]
    second = [_raw(f"s1-add2-seg{i}", "s1", i) for i in range(2)]

    assert last_window([*first, *second]) is None
    assert last_window(first) is not None


def test_last_windows_follow_the_top_block_in_first_appearance_order() -> None:
    """Session B appears first in the top ten, so its last window comes before A's; the tail keeps order."""
    corpus = {
        "aml://session/A": [_raw(f"A{i}", "A", i) for i in range(5)],
        "aml://session/B": [_raw(f"B{i}", "B", i) for i in range(5)],
    }
    ranked = ["B0", "A0", "B1", "A1", "B2", "A2", "B3", "A3", "A4-x", "B-x", "T1", "A4", "T2"]
    by_id = {chunk.id: chunk for chunks in corpus.values() for chunk in chunks}
    by_id["A4-x"] = _raw("A4-x", "A", 0)
    by_id["A4-x"].metadata["kind"] = "atomic"
    by_id["B-x"] = _raw("B-x", "B", 0)
    by_id["T1"] = _raw("T1", "T", 0)
    by_id["T2"] = _raw("T2", "T", 1)
    hits = [ScoredChunk(by_id[chunk_id], 1.0 - rank / 100) for rank, chunk_id in enumerate(ranked)]

    extended, added = with_last_windows(hits, lambda source: corpus[source])

    assert added == 2
    assert [hit.chunk.id for hit in extended] == [
        *ranked[:10],
        "B4",
        "A4",
        "T1",
        "T2",
    ]
    assert extended[10].score == pytest.approx(hits[9].score)


def test_a_last_window_already_ranked_is_neither_added_nor_repeated() -> None:
    """Nothing is added for a session whose last window already sits in the top block."""
    corpus = {"aml://session/A": [_raw(f"A{i}", "A", i) for i in range(3)]}
    hits = [ScoredChunk(chunk, 0.9 - index / 10) for index, chunk in enumerate(reversed(corpus["aml://session/A"]))]

    extended, added = with_last_windows(hits, lambda source: corpus[source])

    assert added == 0
    assert [hit.chunk.id for hit in extended] == ["A2", "A1", "A0"]


def test_c9_serves_no_last_windows_unless_the_flag_is_set() -> None:
    """The official variant must not change its Search until the owner turns LW-1 on."""
    assert variant("C9_routed_specialists_grounded_graph_atomic").last_window_append is False


class _SourceStore(_Store):
    def chunks_for_source(self, source):
        return [chunk for chunk in self.repository.chunks[self.tenant].values() if chunk.source == source]


class _FailingSourceStore(_Store):
    def chunks_for_source(self, source):
        raise RuntimeError("database went away")


def _seeded(store_class, user: str):
    service, repository, _, _, _ = _service()
    tenant = tenant_for(user)
    for segment in range(12):
        chunk = _raw(f"s1-w{segment:02d}", "s1", segment)
        repository.chunks[tenant][chunk.id] = Chunk(
            id=chunk.id, source=chunk.source, text="fix parser pytest " * 3, metadata=chunk.metadata
        )
    repository.tenant_store = lambda name: store_class(repository, name)
    return service


def _search(service, user: str):
    return asyncio.run(
        service.search(SearchRequest.model_validate({"query": "Fix parser.py and run pytest", "user_id": user}))
    )


def test_flag_on_places_the_session_last_window_after_the_top_ten(monkeypatch) -> None:
    """With the flag on, Search returns the unchanged top ten, then the session's last window."""
    service = _seeded(_SourceStore, "lw-user")
    monkeypatch.setenv("RECALL_AML_LAST_WINDOW", "0")
    off = _search(service, "lw-user")
    ids_off = [item.id for item in off.data]
    assert off.specialist_route == "code"
    assert off.last_windows_added == 0
    assert "s1-w11" not in ids_off[:10]

    monkeypatch.setenv("RECALL_AML_LAST_WINDOW", "1")
    on = _search(service, "lw-user")
    ids_on = [item.id for item in on.data]

    assert ids_on[:10] == ids_off[:10]
    assert ids_on[10] == "s1-w11"
    assert on.last_windows_added == 1
    assert len(ids_on) == len(set(ids_on))


def test_a_failed_lookup_serves_the_ranking_without_last_windows(monkeypatch) -> None:
    """A store error inside LW-1 must cost the extra windows, never the Search."""
    working = _seeded(_SourceStore, "lw-fail")
    monkeypatch.setenv("RECALL_AML_LAST_WINDOW", "0")
    expected = [item.id for item in _search(working, "lw-fail").data]

    failing = _seeded(_FailingSourceStore, "lw-fail")
    monkeypatch.setenv("RECALL_AML_LAST_WINDOW", "1")
    response = _search(failing, "lw-fail")

    assert [item.id for item in response.data] == expected
    assert response.last_windows_added == 0
