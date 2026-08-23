"""The two helpers that decide whether a BEAM probe is measuring anything at all.

Neither needs Postgres — `require_indexed` touches only `store.count()` and `rebuild_dates` only
`store.iter_chunks()` — so both are pinned here rather than left to a live-DB run.

`rebuild_dates` is the subtler of the two. `dedup_probe`'s headline mechanism is "keep the NEWEST
of each near-duplicate cluster", and on its documented invocation the date map was blanked, so the
comparison `if di and di > dj` could never fire: the probe measured keep-the-highest-RANKED and
published it as a newest-wins curve. The map now has two routes — built during `ingest`, or
recovered from an already-indexed store — and the test below is a differential oracle over both,
because "they agree by construction" is a claim, not a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.beam.systems import (
    _filename,
    _iso_date,
    _turn_document,
    rebuild_dates,
    require_indexed,
)
from recall.index import chunk_text

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


@dataclass
class _Chunk:
    source: str
    text: str


class _FakeStore:
    """Only the two methods the helpers actually use."""

    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks

    def count(self) -> int:
        return len(self._chunks)

    def iter_chunks(self) -> list[_Chunk]:
        return list(self._chunks)


def _indexed_turn(turn: dict[str, str], index: int = 0) -> list[_Chunk]:
    """A turn as it lands in the store: chunked exactly as the ingest path chunks it."""
    name = _filename(index)
    return [_Chunk(source=f"/tmp/beam-xyz/{name}", text=piece)
            for piece in chunk_text(_turn_document(turn, index))]


# --- require_indexed ----------------------------------------------------------------------


def test_require_indexed_refuses_an_empty_tenant() -> None:
    with pytest.raises(SystemExit, match="ZERO chunks"):
        require_indexed(_FakeStore([]), tenant="beam-1m-0", table="t", what="threshold_probe")


def test_require_indexed_names_the_tenant_and_table_it_refused() -> None:
    with pytest.raises(SystemExit) as exc:
        require_indexed(_FakeStore([]), tenant="beam-1m-7", table="bench_beam_cal",
                        what="lexical_probe")
    message = str(exc.value)
    assert "beam-1m-7" in message and "bench_beam_cal" in message and "lexical_probe" in message


def test_require_indexed_returns_the_count_when_the_tenant_is_populated() -> None:
    store = _FakeStore([_Chunk("a.md", "x"), _Chunk("b.md", "y")])
    assert require_indexed(store, tenant="t", table="tbl", what="probe") == 2


# --- rebuild_dates ------------------------------------------------------------------------


def test_rebuild_dates_recovers_exactly_what_the_ingest_path_recorded() -> None:
    """Differential oracle: the recovered map must EQUAL the one ingest builds.

    `BeamRecallSystem.ingest` records `{_filename(i): _iso_date(turn["date"])}`. Anything else
    here is a divergence between the two routes, which is precisely the class of defect that made
    the collapse inert in the first place.
    """
    turn = {"role": "user", "content": "The TTL is 20 minutes now.", "date": "March-01-2024"}
    expected = {_filename(0): _iso_date(turn["date"])}
    assert rebuild_dates(_FakeStore(_indexed_turn(turn))) == expected
    assert expected[_filename(0)] == "2024-03-01", "guard the oracle itself against a silent ''"


def test_rebuild_dates_survives_a_document_split_across_many_chunks() -> None:
    # Only the FIRST chunk carries the `# Speaker — date` header. A later chunk that happens to
    # start with `#` must not overwrite the recovered date with "".
    turn = {"role": "assistant", "content": "\n\n".join(
        [f"# section {i} — not a date\n\n{'filler ' * 60}" for i in range(8)]
    ), "date": "April-15-2024"}
    chunks = _indexed_turn(turn)
    assert len(chunks) > 1, "this test is only meaningful on a multi-chunk document"
    assert rebuild_dates(_FakeStore(chunks)) == {_filename(0): "2024-04-15"}


def test_rebuild_dates_returns_nothing_when_no_chunk_carries_a_header() -> None:
    # The input that must make dedup_probe refuse rather than publish a rank-wins curve.
    assert rebuild_dates(_FakeStore([_Chunk("turn_000000.md", "just prose, no header")])) == {}


def test_rebuild_dates_keeps_an_unparseable_date_empty_rather_than_guessing() -> None:
    turn = {"role": "user", "content": "hello", "date": "not-a-real-date-at-all"}
    assert rebuild_dates(_FakeStore(_indexed_turn(turn))) == {_filename(0): ""}
