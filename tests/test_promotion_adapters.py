"""Corpus adapters, and the runner that does not know which corpus it is running."""
from __future__ import annotations

import json

import pytest

from recall.eval.promotion.adapters import (
    ADAPTERS,
    CorpusUnavailable,
    LabelledAdapter,
    LadderAdapter,
    LocomoAdapter,
    LongMemEvalAdapter,
    MtragAdapter,
    PepsAdapter,
    label_kind,
)
from recall.eval.promotion.run import ArmConfig, SearchOutcome, score_arm


def test_every_named_corpus_is_reachable_from_the_table() -> None:
    assert set(ADAPTERS) == {
        "labelled", "peps", "locomo", "ladder", "longmemeval", "mtrag"
    }


# --------------------------------------------------------------------------------------------
# The two adapters whose source data ships in the repository.
# --------------------------------------------------------------------------------------------


def test_the_labelled_corpus_freezes_both_classes() -> None:
    frozen = LabelledAdapter().freeze()
    assert any(question.answerable for question in frozen)
    assert any(not question.answerable for question in frozen)
    assert {question.corpus for question in frozen} == {"labelled"}


def test_the_labelled_corpus_queries_verify_against_their_own_frozen_hashes() -> None:
    adapter = LabelledAdapter()
    queries = adapter.queries()
    for question in adapter.freeze():
        question.verify(queries[question.question_id])


def test_the_peps_question_set_is_wired(tmp_path) -> None:
    """It sat in the tree referenced by no code until this module."""
    frozen = PepsAdapter().freeze()
    assert len(frozen) > 10
    assert any(question.answerable for question in frozen)
    assert any(not question.answerable for question in frozen)


def test_every_adapter_declares_the_id_space_its_labels_live_in() -> None:
    """A wrong one scores the whole corpus a miss and reports a working run — which is what the
    first end-to-end run of this harness did."""
    assert {name: label_kind(cls()) for name, cls in ADAPTERS.items()} == {
        "labelled": "file_ord",
        "peps": "source",
        "locomo": "locomo_dia",
        "ladder": "ladder_doc",
        "longmemeval": "source",
        "mtrag": "chunk_id",
    }


def test_an_adapter_that_declares_no_id_space_is_refused() -> None:
    """No inherited default: a new adapter must not be able to get this silently wrong."""

    class Undeclared(LabelledAdapter):
        label_kind = ""

    with pytest.raises(TypeError, match="declares no label_kind"):
        label_kind(Undeclared())


def test_freezing_is_deterministic() -> None:
    assert LabelledAdapter().freeze() == LabelledAdapter().freeze()


# --------------------------------------------------------------------------------------------
# The adapters whose data is external: availability is stated, never silent.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter",
    [
        LocomoAdapter(),
        LadderAdapter(),
        LongMemEvalAdapter(),
        MtragAdapter(),
    ],
    ids=["locomo", "ladder", "longmemeval", "mtrag"],
)
def test_an_adapter_without_its_data_names_the_missing_file(tmp_path, adapter) -> None:
    """A corpus that quietly contributes zero questions shrinks the evidence base while the run
    still prints success."""
    if adapter.available():
        pytest.skip(f"{adapter.name} source data is present in this checkout")
    assert "source data not present" in (adapter.unavailable_reason() or "")
    with pytest.raises(CorpusUnavailable):
        adapter.freeze()


def test_locomo_freezes_category_five_as_unanswerable(tmp_path) -> None:
    """LOCOMO's own adversarial class, which has no supporting evidence by construction."""
    path = tmp_path / "locomo10.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "qa": [
                        {"question": "when did they meet", "evidence": ["D1:2"], "category": 1},
                        {"question": "what is her cat called", "evidence": [], "category": 5},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    frozen = LocomoAdapter(path).freeze()
    assert [question.question_id for question in frozen] == ["conv-1/0", "conv-1/1"]
    # BARE turn ids: a LOCOMO run indexes one conversation at a time, and that is what
    # `recall/eval/locomo.py::_retrieved_dia_ids` recovers from the indexed filename.
    assert frozen[0].expected_relevance_labels == ("D1:2",)
    assert frozen[1].expected_relevance_labels == ()


def test_longmemeval_freezes_the_abstention_class_as_unanswerable(tmp_path) -> None:
    path = tmp_path / "longmemeval_s.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "what did i order",
                    "question_type": "single-session-user",
                    "answer_session_ids": ["s7"],
                },
                {
                    "question_id": "q2_abs",
                    "question": "what did i never say",
                    "question_type": "single-session-user_abs",
                    "answer_session_ids": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    frozen = LongMemEvalAdapter(path).freeze()
    assert frozen[0].expected_relevance_labels == ("s7.md",)
    assert frozen[1].expected_relevance_labels == ()


def test_the_ladder_adapter_reads_through_the_ladders_own_verified_reader(tmp_path) -> None:
    """The ladder's instances are already frozen with their own digest. This re-expresses them;
    it does not re-derive them, and an edited ladder manifest is refused before a question
    crosses over."""
    from benchmarks.ladder.manifest import RING_MAX, RING_ORIGINAL, Instance, write_manifest

    path = tmp_path / "manifest.jsonl"
    instances = [
        Instance(
            instance_id="i-orig", corpus="locomo", source_question_id="s1",
            question="when did they meet", label="answerable", ring=RING_ORIGINAL,
            excised_doc_ids=(), gold_doc_ids=("c1/D1:2",), pair_id="p1",
        ),
        Instance(
            instance_id="i-max", corpus="locomo", source_question_id="s1",
            question="when did they meet", label="unanswerable", ring=RING_MAX,
            excised_doc_ids=("c1/D1:2",), gold_doc_ids=("c1/D1:2",), pair_id="p1",
        ),
    ]
    write_manifest(path, instances, ring_widths=[1], corpus_hashes={"locomo": "x"},
                   manifest_version="1.0")
    frozen = {q.question_id: q for q in LadderAdapter(path).freeze()}
    assert frozen["i-orig"].expected_relevance_labels == ("c1/D1:2",)
    # Its gold document was excised, so it is unanswerable by construction.
    assert frozen["i-max"].expected_relevance_labels == ()

    tampered = path.read_text(encoding="utf-8").replace("when did they meet", "when did they eat")
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="does not match header digest"):
        LadderAdapter(path).freeze()


def test_mtrag_splits_its_domains_into_separate_corpora() -> None:
    """This repository's macro average is the UNWEIGHTED mean over corpora, and MT-RAG's domains
    differ in judged-query count by nearly a factor of two — the AUD-1 finding."""
    from recall.eval.promotion.adapters import SourceQuestion

    adapter = MtragAdapter()
    assert adapter.corpus_of(SourceQuestion("clapnq/t1", "q", ())) == "mtrag:clapnq"
    assert adapter.corpus_of(SourceQuestion("govt/t9", "q", ())) == "mtrag:govt"


# --------------------------------------------------------------------------------------------
# The runner. No database, no corpus — only frozen questions and a callable.
# --------------------------------------------------------------------------------------------

ARM = ArmConfig(
    label="baseline",
    embedding_profile_id="bge-small-symmetric-v1",
    retrieval_profile="fast",
    generation="g1",
    candidate_pool=20,
    embedding_fingerprint="f" * 64,
)


def _search(_query: str) -> SearchOutcome:
    return SearchOutcome(
        retrieved_chunk_ids=("a.md:0",),
        dense_cosine=0.8,
        confidence=0.7,
        trust_verdict="ok",
        reranking_status="not_configured",
        stage_timings={"dense": 2.0},
    )


def test_the_runner_scores_a_manifest_and_resumes_without_re_paying(tmp_path) -> None:
    frozen = LabelledAdapter().freeze()
    queries = LabelledAdapter().queries()
    ledger = tmp_path / "arm.jsonl"

    first = score_arm(frozen, queries, _search, ARM, ledger)
    assert first.scored_now == len(frozen) and first.resumed == 0

    calls: list[str] = []

    def _counting(query: str) -> SearchOutcome:
        calls.append(query)
        return _search(query)

    second = score_arm(frozen, queries, _counting, ARM, ledger)
    assert calls == []
    assert second.scored_now == 0 and second.resumed == len(frozen)
    # A resumed run and a fresh run must produce the same gate input.
    assert second.records == first.records


def test_a_resumed_run_returns_every_record_not_only_this_invocations(tmp_path) -> None:
    """Otherwise the gate runs on whatever fraction survived the last crash, and looks identical
    from the outside."""
    frozen = LabelledAdapter().freeze()
    queries = LabelledAdapter().queries()
    ledger = tmp_path / "arm.jsonl"
    score_arm(frozen[:3], queries, _search, ARM, ledger)
    result = score_arm(frozen, queries, _search, ARM, ledger)
    assert result.scored_now == len(frozen) - 3
    assert len(result.records) == len(frozen)


def test_a_drifted_query_fails_before_the_search_is_paid_for(tmp_path) -> None:
    frozen = LabelledAdapter().freeze()
    queries = dict(LabelledAdapter().queries())
    queries[frozen[0].question_id] = "a completely different question"
    searched: list[str] = []

    def _recording(query: str) -> SearchOutcome:
        searched.append(query)
        return _search(query)

    with pytest.raises(ValueError, match="changed after the manifest was frozen"):
        score_arm(frozen, queries, _recording, ARM, tmp_path / "arm.jsonl")
    assert "a completely different question" not in searched


def test_a_frozen_question_with_no_query_text_is_refused(tmp_path) -> None:
    frozen = LabelledAdapter().freeze()
    queries = dict(LabelledAdapter().queries())
    del queries[frozen[0].question_id]
    with pytest.raises(KeyError, match="supplied no query text"):
        score_arm(frozen, queries, _search, ARM, tmp_path / "arm.jsonl")


def test_a_search_that_reports_its_own_total_is_refused(tmp_path) -> None:
    """`total` is the harness's wall clock; a search supplying one could under-report it."""

    def _liar(_query: str) -> SearchOutcome:
        return SearchOutcome(
            retrieved_chunk_ids=(),
            dense_cosine=float("nan"),
            confidence=float("nan"),
            trust_verdict="abstained",
            reranking_status="not_configured",
            stage_timings={"total": 0.0001},
        )

    frozen = LabelledAdapter().freeze()
    with pytest.raises(ValueError, match="reported its own 'total'"):
        score_arm(
            frozen, LabelledAdapter().queries(), _liar, ARM, tmp_path / "arm.jsonl"
        )


def test_the_runner_records_the_wall_clock_it_actually_spent(tmp_path) -> None:
    frozen = LabelledAdapter().freeze()[:1]
    result = score_arm(
        frozen, LabelledAdapter().queries(), _search, ARM, tmp_path / "arm.jsonl"
    )
    record = result.records[0]
    assert record.stage_timings["dense"] == 2.0
    assert record.total_ms >= 0.0
