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
