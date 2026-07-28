"""A benchmark run must refuse to index on top of an existing one.

This exists because it happened. Two copies of a run script wrote into the same table, so every
tenant held its corpus TWICE: 11,764 rows against a correct 5,882. Nothing errored. The candidate
pool is a fixed size, so duplicates halve the number of DISTINCT documents it can hold, and every
depth of the published curve came in about 0.05 low — a plausible, internally consistent, entirely
wrong result that was reported three times with confidence intervals before anyone counted rows.

The lesson is not "be careful with launchers". It is that the run had no post-condition: it
verified the PROCESS ("is it still going?") and never the ARTIFACT ("does the table hold what it
must?"). One `count()` would have caught it in a second.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.embeddings import HashingEmbedder
from recall.eval.locomo import run_conversation
from recall.store import PgVectorStore
from tests.conftest import TEST_DSN, requires_db

_CONV = {
    "speaker_a": "Caroline",
    "speaker_b": "Mel",
    "session_1_date_time": "1 Jan 2023",
    "session_1": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "I adopted a rescue greyhound."},
        {"speaker": "Mel", "dia_id": "D1:2", "text": "What did you name her?"},
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "Pepper, after the spice."},
    ],
}
_QA = [{"question": "What is the dog called?", "category": 1, "evidence": ["D1:3"]}]


def _fresh(tenant: str, table: str) -> PgVectorStore:
    """A store on an empty table.

    Tests share a database, and the guard under test fires on ANY pre-existing rows — including
    rows a previous run of this same test left behind. Dropping first is what makes the assertions
    about the guard rather than about test order.
    """
    store = PgVectorStore(TEST_DSN, dim=64, tenant=tenant, table=table)
    store.ensure_schema()
    store.drop_table()
    store.ensure_schema()
    return store


def _run(store: PgVectorStore, corpus_dir: Path, **kw):
    return run_conversation(
        _CONV, _QA, store=store, embedder=HashingEmbedder(dim=64),
        k=3, corpus_dir=corpus_dir, ks=[1, 3], candidate_k=10, **kw,
    )


@requires_db
def test_a_clean_run_is_not_blocked(tmp_path: Path) -> None:
    """The guard must not fire on the ordinary path, or it just gets disabled."""
    with _fresh("t-clean", "lab_dbl_clean") as store:
        _run(store, tmp_path / "c1")
        assert store.count() > 0


@requires_db
def test_indexing_over_an_existing_tenant_is_refused(tmp_path: Path) -> None:
    """The actual defect: a second run into a populated tenant silently doubled the corpus."""
    with _fresh("t-dbl", "lab_dbl_guard") as store:
        _run(store, tmp_path / "c1")
        first = store.count()

        with pytest.raises(RuntimeError) as exc:
            _run(store, tmp_path / "c2")

        assert store.count() == first, "the refused run must not have written anything"
    message = str(exc.value)
    assert "already" in message.lower()
    assert "t-dbl" in message, "the error must name the tenant so the cause is obvious"


@requires_db
def test_the_refusal_can_be_overridden_deliberately(tmp_path: Path) -> None:
    """A resume is legitimate; doing it BY ACCIDENT is what must be impossible."""
    with _fresh("t-ovr", "lab_dbl_ovr") as store:
        _run(store, tmp_path / "c1")
        first = store.count()
        _run(store, tmp_path / "c2", allow_existing=True)
        assert store.count() > first, "the override must actually permit the write"
