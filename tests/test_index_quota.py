"""The indexing spend quota's integration point.

Counting requests prices a 20 MB index and a 200-byte one identically, so the budget that
actually bounds cost is measured in bytes. `index_memory` hands those bytes to `on_measured`
after its per-request caps pass and before anything is embedded; the server debits the tenant's
byte bucket there.

Two properties matter and neither is visible from the limiter's own tests:

1. the number handed over is the real candidate size, not an estimate, and
2. refusing costs nothing — a rejected request must not have embedded or written a single chunk.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.store import PgVectorStore
from recall_mcp.limits import Rate, RateLimited, RateLimiter
from recall_mcp.service import index_memory

from tests.conftest import TEST_DSN, requires_db


class _E:
    dim = 4
    name = "e"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def store():
    name = "q_" + uuid.uuid4().hex[:8]
    s = PgVectorStore(TEST_DSN, dim=4, table=name)
    s.ensure_schema()
    try:
        yield s
    finally:
        s.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {name}")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    d = tmp_path / "memory"
    d.mkdir()
    for i in range(4):
        (d / f"memo{i}.md").write_text("x" * 1000, encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    return d


@requires_db
def test_on_measured_receives_the_real_candidate_size(store, corpus):
    """Measured, not estimated — the number that bills must be the number that runs."""
    seen: list[tuple[int, int]] = []
    index_memory(store, _E(), str(corpus), on_measured=lambda f, b: seen.append((f, b)))

    assert len(seen) == 1, "the hook must fire exactly once per request"
    files, total_bytes = seen[0]
    assert files == 4
    assert total_bytes == 4000
    assert store.count() > 0


@requires_db
def test_a_refusal_from_the_hook_costs_nothing(store, corpus):
    """Pre-flight is the whole point: a cap that trips after the spend is not a cap."""
    embedder = _E()

    def refuse(_files: int, _bytes: int) -> None:
        raise RateLimited("over quota", retry_after_seconds=60.0)

    with pytest.raises(RateLimited):
        index_memory(store, embedder, str(corpus), on_measured=refuse)

    assert embedder.calls == 0, "embedded despite being refused — the spend already happened"
    assert store.count() == 0


@requires_db
def test_the_quota_stops_a_loop_that_stays_under_the_per_request_cap(store, corpus):
    """The gap this closes.

    Every one of these requests is individually legal — well under RECALL_INDEX_MAX_BYTES. Only
    the aggregate is abusive, which is exactly the case request-counting misses and the reason
    the budget is denominated in bytes.
    """
    limiter = RateLimiter({"index_bytes": Rate(10_000.0, 0.001)})  # ~2.5 requests' worth
    embedder = _E()

    def debit(_files: int, total_bytes: int) -> None:
        limiter.check("acme", "index_bytes", float(total_bytes))

    accepted = 0
    for _ in range(10):
        try:
            index_memory(store, embedder, str(corpus), on_measured=debit)
            accepted += 1
        except RateLimited:
            break

    assert accepted == 2, f"quota admitted {accepted} requests of 4000 bytes against 10000"


@requires_db
def test_the_billed_set_is_the_set_that_gets_indexed(tmp_path, store, monkeypatch):
    """One walk, not two — the bill and the work must describe the same files.

    `index_memory` measured the tree, billed that number, and then let `index_path` walk again
    to decide what to index. Anything appearing under the root between those two walks was
    embedded without having been counted, so it escaped both the per-request byte cap and the
    tenant's hourly quota. The window is a full directory walk wide, and a corpus synced into
    the index root on a timer — the deployment this is built for — lands inside it routinely.

    Simulated by making a file appear during the measurement, which is the same interleaving.
    """
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("x" * 1000, encoding="utf-8")

    billed: list[tuple[int, int]] = []

    def measure(n_files: int, n_bytes: int) -> None:
        billed.append((n_files, n_bytes))
        # A writer lands new files after the measurement but before the indexing walk.
        for i in range(5):
            (corpus / f"late{i}.md").write_text("y" * 1000, encoding="utf-8")

    result = index_memory(store, _E(), str(corpus), on_measured=measure)

    assert billed == [(1, 1000)], billed
    assert result.files == 1, (
        f"billed 1 file but indexed {result.files} — the walk that bills is not the walk that runs"
    )
