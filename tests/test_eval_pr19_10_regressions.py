import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from recall.eval import synthetic, scale, harness
from recall.embeddings import HashingEmbedder


@pytest.mark.parametrize("options", [
    {"n_answerable": -1}, {"n_unanswerable": -1}, {"n_successor": -1},
    {"n_abstain": -1}, {"n_filler_chunks": -1},
    {"filler_chunks_per_file": 0}, {"filler_chunks_per_file": -1},
    {"n_unanswerable": 1000000},
])
def test_invalid_generation_options_refuse_before_output_cleanup(tmp_path, monkeypatch, options):
    """BUG-001, baseline e5f7892b8 reaches destructive preparation before rejecting invalid counts.

    The preparation sentinel fails the intended assertion without letting the nonadvancing
    filler loop run. The fixed boundary must raise ValueError before reaching it.
    """
    def forbid_cleanup(root):
        pytest.fail("invalid options reached destructive output preparation")
    monkeypatch.setattr(synthetic, "_prepare_root", forbid_cleanup)
    with pytest.raises(ValueError):
        synthetic.generate(tmp_path / "corpus", **options)


def test_fact_classes_have_distinct_subjects_even_when_random_words_repeat(tmp_path, monkeypatch):
    """BUG-002, baseline e5f7892b8 resets suffixes and duplicates subjects across all classes."""
    monkeypatch.setattr(synthetic, "_ADJECTIVES", ["quartz"])
    monkeypatch.setattr(synthetic, "_NOUNS", ["ledger"])
    corpus = synthetic.generate(tmp_path / "corpus", n_answerable=2, n_unanswerable=0,
                               n_successor=2, n_abstain=2, n_filler_chunks=0)
    subjects = [q["subject"] for q in corpus.queries]
    assert len(subjects) == len(set(subjects)), "fact classes share a subject and ground truth"


def test_source_filtered_recall_runs_when_unfiltered_search_misses(monkeypatch):
    """BUG-003, baseline e5f7892b8 never executes the filtered arm and reports its rate as0."""
    calls = []
    source = "/corpus/facts/gold.md"
    gold = SimpleNamespace(chunk=SimpleNamespace(source=source, metadata={"file":"facts/gold.md", "ord":0}))
    class Store:
        def sources_for_identifiers(self, identifiers):
            assert identifiers == ["facts/gold.md"]
            return {"facts/gold.md": [source]}
    class Retriever:
        def __init__(self, store, embedder):
            pass
        def search(self, query, k, source=None):
            calls.append(source)
            return SimpleNamespace(hits=[gold] if source else [])
    monkeypatch.setattr(scale, "HybridRetriever", Retriever)
    plain, filtered, _ = scale.measure_retrieval(Store(), HashingEmbedder(dim=8),
        [{"query":"gold", "answerable":True, "relevant_ids":["facts/gold.md:0"]}])
    assert filtered["rate"] == 1.0, "filtered recall inherited the unfiltered miss"
    assert plain["rate"] == 0.0
    assert calls == [None, source]


@pytest.mark.parametrize("answerable_count,gap_count,missing", [(1,2,"false_abstain"),(2,1,"gap_fcr")])
def test_singleton_classes_do_not_publish_in_sample_heldout_rates(
    tmp_path, monkeypatch, answerable_count, gap_count, missing
):
    """DOC-001, baseline e5f7892b8 publishes singleton in-sample rates as held-out results."""
    import json
    queries = [{"query":f"answer{i}", "answerable":True, "relevant_ids":["a.md:0"]}
               for i in range(answerable_count)]
    queries += [{"query":f"gap{i}", "answerable":False, "relevant_ids":[]} for i in range(gap_count)]
    qp = tmp_path / "queries.json"
    qp.write_text(json.dumps(queries))
    np = tmp_path / "near.json"
    np.write_text(json.dumps([{"query":"near"}]))
    monkeypatch.setattr(harness, "measure_top_cosines",
                        lambda *args: ([0.8]*answerable_count,[0.2]*gap_count))
    monkeypatch.setattr(harness, "research_search",
                        lambda *args, **kwargs: SimpleNamespace(abstained=False, hits=[]))
    result = harness.run_nearmiss_eval("unused", [HashingEmbedder(dim=8)],
        queries_path=qp, nearmiss_path=np, store_factory=lambda emb:nullcontext(object()))[0]
    assert math.isnan(getattr(result, missing)), "singleton in-sample score published as held-out"
    if answerable_count == 1:
        assert math.isnan(result.mrr_answerable)
    other = result.gap_fcr if missing == "false_abstain" else result.false_abstain
    assert math.isfinite(other), "the class with a valid held-out fold lost coverage"
    assert "NaN" not in harness.nearmiss_results_to_markdown([result])
