"""scripts/check_hnsw_recall.py measures the HNSW path, whatever the served default is.

Invariant: each measurement runs `GenerationStore.query_dense` with
`RECALL_GENERATION_DENSE_SEARCH=hnsw`, records that path in its result, and leaves the variable as
it found it. Failure mode caught (CCA of #847, DOC-001): since #847 the served default is the
exact scan, so the script, which #847's own comment sends operators to before they opt into HNSW,
compared exact search with itself and reported full recall for an index it never walked.

No database: the store is a stand-in that records which path the switch selects at query time.

Red proof, 2026-10-28: against the script before this fix, ``test_the_measurement_runs_on_the_hnsw_path``
failed at ``assert seen == ["hnsw"]`` (the switch read ``exact``).
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from recall import generation_store
from recall.generation_store import GENERATION_DENSE_SEARCH_ENV
from scripts import check_hnsw_recall


class _PathRecordingStore:
    _tenant = "acme"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def query_dense(self, vector: list[float], k: int) -> list[object]:
        self.seen.append(generation_store._generation_dense_search())  # noqa: SLF001
        return []


def _run(monkeypatch: pytest.MonkeyPatch, before: str | None) -> tuple[list[str], dict[str, object]]:
    if before is None:
        monkeypatch.delenv(GENERATION_DENSE_SEARCH_ENV, raising=False)
    else:
        monkeypatch.setenv(GENERATION_DENSE_SEARCH_ENV, before)
    store = _PathRecordingStore()
    matrix = np.eye(3, dtype=np.float32)
    args = argparse.Namespace(k=[1], queries=1, seed=0)
    row = check_hnsw_recall._measure(store, "gen-1", ["a", "b", "c"], matrix, args)  # type: ignore[arg-type]  # noqa: SLF001
    return store.seen, row


@pytest.mark.parametrize("before", [None, "exact"])
def test_the_measurement_runs_on_the_hnsw_path(monkeypatch: pytest.MonkeyPatch, before: str | None) -> None:
    import os

    seen, row = _run(monkeypatch, before)

    assert seen == ["hnsw"]
    assert row["path"] == "hnsw"
    assert os.environ.get(GENERATION_DENSE_SEARCH_ENV) == before
