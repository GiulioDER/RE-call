from benchmarks.mtrag import run
from benchmarks.mtrag.run import ndcg_at, query_text, recall_at, score_predictions


def test_query_modes() -> None:
    task = {
        "task_id": "q",
        "input": [
            {"speaker": "user", "text": "first"},
            {"speaker": "agent", "text": "reply"},
            {"speaker": "user", "text": "second"},
            {"speaker": "user", "text": "third"},
            {"speaker": "user", "text": "fourth"},
        ],
    }
    assert query_text(task, "last") == "fourth"
    assert query_text(task, "recent3") == "second\nthird\nfourth"


def test_binary_metrics() -> None:
    relevant = {"a", "b"}
    assert recall_at(["a", "x", "b"], relevant, 1) == 0.5
    assert recall_at(["a", "x", "b"], relevant, 3) == 1.0
    assert ndcg_at(["a", "b"], relevant, 2) == 1.0
    assert 0.0 < ndcg_at(["x", "a", "b"], relevant, 3) < 1.0


def test_score_predictions_overall_is_pooled_not_domain_macro() -> None:
    """`overall` weights domains by judged-query count, and this fixture proves which one it is.

    The domains are deliberately UNBALANCED: three judged queries in clapnq (all hits) against
    one in cloud (a miss). Pooled gives 3/4 = 0.75; an unweighted mean of the two non-empty
    domain figures gives (1.0 + 0.0)/2 = 0.5. A balanced fixture cannot tell those apart, which
    is how the previous version of this test passed under either definition while its name
    asserted one of them.
    """
    predictions = [
        {"task_id": "q1", "contexts": [{"document_id": "a"}]},
        {"task_id": "q2", "contexts": [{"document_id": "b"}]},
        {"task_id": "q3", "contexts": [{"document_id": "c"}]},
        {"task_id": "q4", "contexts": [{"document_id": "x"}]},
    ]
    qrels = {
        "clapnq": {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}},
        "cloud": {"q4": {"d"}},
        "fiqa": {},
        "govt": {},
    }
    scores = score_predictions(predictions, qrels)

    assert scores["overall"]["count"] == 4
    assert scores["overall"]["nDCG@5"] == 0.75
    assert scores["overall"]["Recall@5"] == 0.75

    # The per-domain figures stay unweighted within a domain.
    assert scores["domains"]["clapnq"]["nDCG@5"] == 1.0
    assert scores["domains"]["cloud"]["nDCG@5"] == 0.0
    assert scores["domains"]["fiqa"]["nDCG@5"] is None

    # Pin the choice itself: the pooled figure must NOT equal the domain macro. Without this,
    # switching the aggregation to a macro average would still satisfy every assertion above
    # that happens to hold under both.
    populated = [
        scores["domains"][d]["nDCG@5"]
        for d in ("clapnq", "cloud", "fiqa", "govt")
        if scores["domains"][d]["nDCG@5"] is not None
    ]
    domain_macro = sum(populated) / len(populated)
    assert domain_macro == 0.5
    assert scores["overall"]["nDCG@5"] != domain_macro


def test_the_default_split_is_dev_not_the_sealed_test_set() -> None:
    """The held-out set must be chosen deliberately, never arrived at by default.

    MTRAG-UN is what the official leaderboard scored and what the archived 2026-08-04 Task A
    baseline already used. A new arm comparison that lands on it by default silently spends the
    held-out set, and no error is raised when that happens — which is why this is asserted rather
    than left to a docstring.
    """
    args = run.parse_args(["--mtrag-root", ".", "--output-dir", "."])

    assert args.split == "dev"


def test_every_learned_sparse_arm_can_actually_reach_depth_100() -> None:
    """Recall@100 is only a depth measurement if the fused pool can hold 100 candidates.

    Each leg contributes at most `candidate_k`, so an arm at candidate_k=20 tops out at a pool of
    40 and its Recall@100 would silently be a statement about the POOL. These arms are declared
    at candidate_k=100 precisely so the number means what it says; if anyone lowers it, this
    fails instead of the metric quietly changing meaning.
    """
    undersized = {arm.name: arm.pool_bound() for arm in run.SPARSE_ARMS if arm.pool_bound() < 100}

    assert undersized == {}


def test_the_sparse_arms_vary_only_the_sparse_backend() -> None:
    """The comparison is only clean if nothing else moves between control and primary.

    `hybrid_lexical` and `hybrid_splade` must differ in exactly one field. If a future edit also
    changed candidate_k or the reranker, the measured delta would no longer be attributable to
    the learned sparse leg, and the result would look identical.
    """
    control = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_lexical")
    primary = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade")
    differing = {
        field for field in ("query_mode", "candidate_k", "use_dense", "use_sparse", "rerank",
                            "sparse_backend")
        if getattr(control, field) != getattr(primary, field)
    }

    assert differing == {"sparse_backend"}


def test_speaker_prefix_is_stripped_from_dev_queries() -> None:
    """MTRAG-human ships every turn prefixed with '|user|: '. It is not part of the question.

    Leaving it in feeds the literal token into both the embedder and the sparse encoder on EVERY
    dev query, which depresses the whole run and makes the numbers incomparable with the
    established baseline. Nothing errors; the scores are just quietly worse.

    Implementation matches benchmarks/mtrag/probe/fix_retrieval_2x2.py on bench/mtrag-arm-r, so
    normalisation is IDENTICAL to the measured baseline rather than merely similar.
    """
    assert run.strip_speaker("|user|: Do the Arizona Cardinals play outside the US?") == (
        "Do the Arizona Cardinals play outside the US?"
    )


def test_speaker_stripping_keeps_colons_inside_the_question() -> None:
    """Only the leading speaker tag goes. A colon in the question body is content."""
    assert run.strip_speaker("|user|: What is X: a thing?") == "What is X: a thing?"


def test_speaker_stripping_handles_the_multi_turn_concatenation() -> None:
    """The `_questions` file concatenates turns, one per line, each with its own prefix."""
    raw = "|user|: where do the arizona cardinals play\n|user|: Do they play outside the US?"

    assert run.strip_speaker(raw) == (
        "where do the arizona cardinals play\nDo they play outside the US?"
    )


def test_dev_queries_reach_the_retriever_without_their_speaker_prefix() -> None:
    """The end the bug actually lives at: what `query_text` hands the retriever.

    Testing `strip_speaker` alone would pass while the loader never called it, which is exactly
    the shape of the original defect.
    """
    task = {"task_id": "q1", "_domain": "clapnq", "_text": "|user|: How many teams are in the NFL?"}

    assert run.query_text(task, "last") == "How many teams are in the NFL?"
