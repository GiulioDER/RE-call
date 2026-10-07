"""`trusted_search` end to end on a `LiteStore`: the same verdicts the Postgres store gets.

The first three cases are `tests/test_trusted_search.py`'s, run on SQLite instead of pgvector with
the same memos and assertions. No database server.

Invariants and the failure each one catches:
- T1 a memo another memo declares it supersedes is demoted and names its successor.
- T2 a memo past its `valid_until` is `expired`, and a search with nothing else abstains.
- T3 provenance (file, ordinal, indexed time, source) reaches the hit.
- T4 a supersession written by another process is seen by a store already serving, without a
  restart (the cache is keyed on the corpus version in the file, not on this instance's writes).
- T5 asked as of an instant before the successor was written, the older memo is still `ok`:
  edges are dated by the claiming chunk's first write.
- T6 a delete whose version bump fails deletes nothing: the delete and the bump are one
  transaction, so no cache keyed on the version can keep a deleted successor's edge.

Node IDs, all in `tests/test_lite_trusted_search.py`:
T1 `test_superseded_memory_loses_to_successor`, T2 `test_expired_only_memory_abstains`,
T3 `test_provenance_populated_end_to_end`,
T4 `test_a_supersession_written_elsewhere_is_seen_without_a_restart`,
T5 `test_as_of_before_the_successor_the_older_memo_is_current`,
T6 `test_a_delete_that_cannot_move_the_corpus_version_deletes_nothing`.

Red proof, 2026-10-07, each mutation alone against `recall/lite/store.py`, failing in the named
assertion (JUnit XML), then restored byte for byte and green:
- N1 (T1) `LiteStore.supersession_all` resolving no rows: the stale memo ranked first
  (`rate_v1.md`).
- N2 (T2) `valid_until` dropped from the metadata `LiteStore._insert` writes: the search did not
  abstain.
- N3 (T3) `indexed_at` not returned by `LiteStore._hit`: `provenance.indexed_at` was None.
- N4 (T4) the cache in `LiteStore.supersession_all` trusting this instance's writes: "the serving
  store kept a stale supersession map".
- N5 (T5) edges left undated in `LiteStore.supersession_all`: "an edge written later demoted the
  memo as of an earlier instant".
- N6 (T6) baseline, not a mutation: `LiteStore.delete_sources` at the PR head, which committed the
  DELETE and the version bump as two autocommit statements: "the delete committed without moving
  the corpus version, so a serving store keeps the deleted successor's edge" (`assert []`), on
  VPS2.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.lite import LiteStore
from tests.conftest import dev_search
from tests.test_trusted_search import EXPIRED, V1, V2

DIM = 64


def _store(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "memory.db", dim=DIM)


def _index(root: Path, store: LiteStore, files: dict[str, str]) -> None:
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=DIM), env={}).index_path(root)


def test_superseded_memory_loses_to_successor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _index(tmp_path / "m", store, {"rate_v1.md": V1, "rate_v2.md": V2})
    res = dev_search(store, HashingEmbedder(dim=DIM), "API rate limit requests per second", k=5)
    files = [h.provenance.file for h in res.hits]
    assert "rate_v1.md" in files and "rate_v2.md" in files
    assert res.hits[0].provenance.file == "rate_v2.md"
    assert res.hits[0].verdict == "ok"
    stale = next(h for h in res.hits if h.provenance.file == "rate_v1.md")
    assert stale.verdict == "superseded", "the superseded memo was not demoted"
    assert stale.validity.superseded_by == "rate_v2.md"
    assert res.abstained is False


def test_expired_only_memory_abstains(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _index(tmp_path / "m", store, {"freeze.md": EXPIRED})
    res = dev_search(store, HashingEmbedder(dim=DIM), "deploy freeze winter release", k=5)
    assert res.abstained is True
    assert any(h.verdict == "expired" for h in res.hits)
    assert res.reason != ""


def test_provenance_populated_end_to_end(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _index(tmp_path / "m", store, {"rate_v1.md": V1})
    res = dev_search(store, HashingEmbedder(dim=DIM), "API rate limit requests per second", k=3)
    h = res.hits[0]
    assert h.provenance.file == "rate_v1.md"
    assert h.provenance.ord == 0
    assert h.provenance.indexed_at is not None
    assert h.provenance.source.endswith("rate_v1.md")


def test_a_supersession_written_elsewhere_is_seen_without_a_restart(tmp_path: Path) -> None:
    serving = _store(tmp_path)
    root = tmp_path / "m"
    _index(root, serving, {"rate_v1.md": V1})
    before = dev_search(serving, HashingEmbedder(dim=DIM), "API rate limit requests per second", k=5)
    assert before.hits[0].verdict == "ok"
    writer = _store(tmp_path)  # a second connection, as a separate `recall index` run would be
    _index(root, writer, {"rate_v2.md": V2})
    after = dev_search(serving, HashingEmbedder(dim=DIM), "API rate limit requests per second", k=5)
    stale = next(h for h in after.hits if h.provenance.file == "rate_v1.md")
    assert stale.verdict == "superseded", "the serving store kept a stale supersession map"


def test_as_of_before_the_successor_the_older_memo_is_current(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "m"
    _index(root, store, {"rate_v1.md": V1})
    time.sleep(0.05)
    between = datetime.now(UTC)
    time.sleep(0.05)
    _index(root, store, {"rate_v2.md": V2})
    res = dev_search(store, HashingEmbedder(dim=DIM), "API rate limit requests per second", k=5, known_as_of=between)
    older = next(h for h in res.hits if h.provenance.file == "rate_v1.md")
    assert older.verdict == "ok", "an edge written later demoted the memo as of an earlier instant"


def test_a_delete_that_cannot_move_the_corpus_version_deletes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _store(tmp_path)
    _index(tmp_path / "m", writer, {"rate_v1.md": V1, "rate_v2.md": V2})
    successor = next(c.source for c in writer.iter_chunks() if c.metadata.get("file") == "rate_v2.md")
    serving = _store(tmp_path)
    assert serving.supersession()[0] == {"rate_v1.md": "rate_v2.md"}

    def locked() -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(writer, "_bump", locked)
    with pytest.raises(sqlite3.OperationalError):
        writer.delete_sources([successor])
    assert _store(tmp_path).chunks_for_source(successor), (
        "the delete committed without moving the corpus version, so a serving store keeps the deleted successor's edge"
    )
    assert serving.supersession()[0] == {"rate_v1.md": "rate_v2.md"}
