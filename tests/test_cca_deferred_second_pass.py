"""CCA second-pass deferred fixes: EVAL-002 (trusted_search candidate_k) and PERF-003 (reuse the
already-retrieved result for the abstain decision instead of a second retrieval)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from recall.retriever import DEFAULT_CANDIDATE_K
from recall.types import RetrievalResult, StalenessReport

from .conftest import TEST_DSN, requires_db


def test_trusted_search_forwards_candidate_k(monkeypatch) -> None:  # EVAL-002 (unit, no DB)
    from recall import trust
    from recall.trust_policy import TrustPolicy

    # Development mode: this unit test fakes the retriever and never has a calibration,
    # so strict would refuse before reaching the candidate_k plumbing under test.
    _DEV = TrustPolicy.development()

    captured: dict[str, int] = {}

    class _FakeRetriever:
        def __init__(self, store, embedder, **kw) -> None:
            captured["candidate_k"] = kw.get("candidate_k")

        def search(self, query, k, source=None) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                hits=[],
                gap_warning=False,
                staleness=StalenessReport(
                    stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=1)
                ),
            )

    monkeypatch.setattr(trust, "HybridRetriever", _FakeRetriever)

    trust.trusted_search(object(), object(), "q", policy=_DEV)
    assert captured["candidate_k"] == DEFAULT_CANDIDATE_K  # unset -> library default (unchanged)
    trust.trusted_search(object(), object(), "q", candidate_k=7, policy=_DEV)
    assert captured["candidate_k"] == 7  # threaded through, not silently reverted to the default


@requires_db
def test_reused_result_abstain_equals_trusted_search(tmp_path) -> None:  # PERF-003 (correctness)
    # The reuse only changes throughput, never the number: evaluate() on the result a hybrid arm
    # already retrieved must give the SAME abstained as a fresh trusted_search at the same pool.
    from recall.calibration import from_samples
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from recall.retriever import HybridRetriever
    from recall.store import PgVectorStore
    from recall.trust import evaluate
    from tests.conftest import dev_search

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("the deploy target is staging", encoding="utf-8")
    (corpus / "b.md").write_text("the database backup runs nightly", encoding="utf-8")
    (corpus / "c.md").write_text("the oncall rotation is weekly", encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    table = "t_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus)
        cal = from_samples(emb.name, [0.9, 0.85], [0.2, 0.15])
        retr = HybridRetriever(store, emb)  # default candidate_k, as the false-abstain arm uses

        for query in ["deploy target staging", "database backup nightly", "utterly unrelated xyzzy"]:
            res = retr.search(query, k=5)
            sup, unres = store.supersession() if res.hits else ({}, frozenset())
            reused = evaluate(res, sup, cal, datetime.now(timezone.utc), unres).abstained
            fresh = dev_search(store, emb, query, k=5, calibration=cal).abstained
            assert reused == fresh, f"reuse diverged from trusted_search on {query!r}"
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_labelled_evaluate_reuses_hybrid_results(tmp_path) -> None:  # PERF-003 wiring, end-to-end
    from recall.embeddings import HashingEmbedder
    from recall.eval.labelled import evaluate

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("the deploy target is staging", encoding="utf-8")
    (corpus / "b.md").write_text("the database backup runs nightly", encoding="utf-8")
    (corpus / "c.md").write_text("the oncall rotation is weekly", encoding="utf-8")
    (corpus / "d.md").write_text("the incident review is on friday", encoding="utf-8")

    questions = [
        {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
        {"id": "q2", "query": "database backup", "relevant_files": ["b.md"], "answerable": True},
        {"id": "q3", "query": "oncall rotation", "relevant_files": ["c.md"], "answerable": True},
        {"id": "q4", "query": "incident review", "relevant_files": ["d.md"], "answerable": True},
        {"id": "q5", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
        {"id": "q6", "query": "unrelated nonsense xyzzy", "relevant_files": [], "answerable": False},
    ]
    rep = evaluate(TEST_DSN, corpus, questions, HashingEmbedder(dim=64), k=3)
    # held = questions[1::2] = q2, q4, q6 -> 2 answerable held (reused) + 1 unanswerable held
    assert rep["false_abstain"]["n"] == 2
    assert rep["abstention_accuracy"]["n"] == 1
    assert "hybrid" in rep["arms"]


@requires_db
def test_perq_evaluate_reuses_hybrid_result(tmp_path) -> None:  # PERF-003 wiring, end-to-end
    from recall.embeddings import HashingEmbedder
    from recall.eval.longmemeval_perq import evaluate as perq_evaluate
    from recall.index import Indexer
    from recall.store import PgVectorStore

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "s1.md").write_text("the deploy target is staging", encoding="utf-8")
    (corpus / "s2.md").write_text("the database backup runs nightly", encoding="utf-8")
    (corpus / "s3.md").write_text("the oncall rotation is weekly", encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    master = "m_" + uuid.uuid4().hex[:8]
    mstore = PgVectorStore(TEST_DSN, dim=emb.dim, table=master)
    try:
        mstore.ensure_schema()
        Indexer(mstore, emb).index_path(corpus)
        questions = [
            {"id": "q1", "query": "deploy target", "haystack_files": ["s1.md", "s2.md"],
             "relevant_files": ["s1.md"], "answerable": True, "question_type": "single"},
            {"id": "q2", "query": "database backup", "haystack_files": ["s2.md", "s3.md"],
             "relevant_files": ["s2.md"], "answerable": True, "question_type": "single"},
            {"id": "q3", "query": "nonsense xyzzy", "haystack_files": ["s1.md"],
             "relevant_files": [], "answerable": False, "question_type": "single"},
            {"id": "q4", "query": "unrelated plugh", "haystack_files": ["s3.md"],
             "relevant_files": [], "answerable": False, "question_type": "single"},
        ]
        rep = perq_evaluate(TEST_DSN, master, questions, emb, k=3)
        # held = questions[1::2] = q2, q4 -> 1 answerable held (reused) + 1 unanswerable held
        assert rep["false_abstain"]["n"] == 1
        assert rep["abstention_accuracy"]["n"] == 1
    finally:
        mstore.drop_table()
        mstore.close()
