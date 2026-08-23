"""CCA second-pass config/dedup fixes (CODE-002/003/004, EVAL-003). Pure-unit, no database."""
from __future__ import annotations

import inspect
import os

import pytest

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def test_locomo_reuses_labelled_make_embedder() -> None:  # CODE-002
    from recall.eval import labelled, locomo

    assert locomo._make_embedder is labelled._make_embedder


def test_default_candidate_k_is_one_source_of_truth() -> None:  # CODE-004
    from recall import retriever
    from recall.eval import labelled, locomo

    assert retriever.DEFAULT_CANDIDATE_K == 20
    assert locomo.DEFAULT_CANDIDATE_K == retriever.DEFAULT_CANDIDATE_K
    # both harness entry points bind their candidate_k default to the shared constant
    assert (
        inspect.signature(labelled.evaluate).parameters["candidate_k"].default
        == retriever.DEFAULT_CANDIDATE_K
    )
    assert (
        inspect.signature(retriever.HybridRetriever.__init__).parameters["candidate_k"].default
        == retriever.DEFAULT_CANDIDATE_K
    )


def test_perq_dsn_reads_recall_dsn_like_siblings() -> None:  # CODE-003
    from recall.eval import longmemeval_perq

    expected = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")
    assert longmemeval_perq.DEFAULT_DSN == expected


def test_write_conversation_corpus_skips_non_numeric_session(tmp_path) -> None:  # EVAL-003
    from recall.eval.locomo import write_conversation_corpus

    conv = {
        "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "hello"}],
        "session_1_date_time": "2026-01-01",
        # a non-numeric `session_*` key used to crash the sort with ValueError; it must be skipped
        "session_summary": "not a dialog list",
    }
    n = write_conversation_corpus(conv, tmp_path)
    assert n == 1
    assert (tmp_path / "D1_1.md").exists()
