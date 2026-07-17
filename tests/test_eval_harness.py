from recall.embeddings import HashingEmbedder
from recall.eval.harness import run_ablations

from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_run_ablations_hashing():
    # dense + hybrid only (skip rerank to avoid the model download in the test)
    results = run_ablations(TEST_DSN, [HashingEmbedder(dim=64)], fusions=["dense", "hybrid"])
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.p_at_5 <= 1.0
        assert 0.0 <= r.ndcg_at_10 <= 1.0
        assert r.fcr_no_guard == 1.0
        assert 0.0 <= r.fcr_with_guard <= 1.0
    # sanity: retrieval actually finds some relevant docs
    assert any(r.mrr > 0 for r in results)


@requires_db
def test_run_trust_eval_baseline_trusts_stale_and_trust_layer_does_not(tmp_path):
    import json

    from recall.embeddings import HashingEmbedder
    from recall.eval.harness import run_trust_eval

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "limit_v1.md").write_text(
        "the api request limit is one hundred requests per second per key", encoding="utf-8"
    )
    (corpus / "limit_v2.md").write_text(
        "---\nsupersedes: limit_v1.md\n---\nthrottling revision: twenty requests per second "
        "enforced at the gateway",
        encoding="utf-8",
    )
    (corpus / "freeze.md").write_text(
        "---\nvalid_until: 2020-01-01\n---\na deploy freeze is in effect for the winter release",
        encoding="utf-8",
    )
    (corpus / "filler.md").write_text("unrelated observability notes about tracing", encoding="utf-8")
    (corpus / "metrics_doc.md").write_text(
        "dashboards display latency percentiles for every deployed service", encoding="utf-8"
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                # two answerable + one unanswerable query give the in-run calibration a
                # sane threshold below the trust queries' stale-hit cosines
                {"id": "q1", "query": "notes about tracing observability", "answerable": True,
                 "relevant_ids": ["filler.md:0"]},
                {"id": "q2", "query": "latency percentiles dashboards", "answerable": True,
                 "relevant_ids": ["metrics_doc.md:0"]},
                {"id": "u1", "query": "zebra migration patterns in the serengeti",
                 "answerable": False, "relevant_ids": []},
                {"id": "t1", "query": "api request limit one hundred requests per second per key",
                 "trust": True, "expect": "successor", "stale_ids": ["limit_v1.md:0"],
                 "successor_ids": ["limit_v2.md:0"]},
                {"id": "t2", "query": "a deploy freeze is in effect", "trust": True,
                 "expect": "abstain", "stale_ids": ["freeze.md:0"], "successor_ids": []},
            ]
        ),
        encoding="utf-8",
    )
    results = run_trust_eval(
        TEST_DSN, [HashingEmbedder(dim=64)], corpus_dir=corpus, queries_path=queries
    )
    assert len(results) == 1
    r = results[0]
    # the stale memory is worded closest to the query: plain search returns it as the answer
    assert r.str_baseline > 0.0
    # the trust layer never presents a stale memory as trustworthy
    assert r.str_trust == 0.0
    assert r.successor_acc == 1.0
    assert r.abstain_acc == 1.0
    # trust evaluation must not damage ordinary answerable retrieval
    assert r.mrr_answerable_trust == r.mrr_answerable_baseline
