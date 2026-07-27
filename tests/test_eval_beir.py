"""Turning BEIR datasets into corpora the `labelled` harness can score.

Everything here is plumbing, and plumbing is where ground truth gets silently corrupted: a
filename collision makes two documents into one and leaves the qrels pointing at whichever was
written last, with no error and a plausible-looking score at the end. These tests exist for that
class of failure, not for the file I/O.
"""
from __future__ import annotations

import json


from recall.eval.beir import (
    build_questions,
    materialize,
    read_beir,
    safe_filename,
    select_documents,
)


def test_safe_filename_does_not_collide_for_ids_differing_only_in_illegal_characters():
    """The silent corruption this module exists to avoid.

    BEIR document ids contain slashes, colons and dots. Sanitising them by replacing illegal
    characters maps `doc/1` and `doc_1` onto the same name — the second write clobbers the first,
    the qrels still reference both, and one question is scored against the wrong document forever
    after. Nothing raises.
    """
    assert safe_filename("doc/1") != safe_filename("doc_1")
    assert safe_filename("MED-10:2") != safe_filename("MED-10_2")


def test_safe_filename_is_a_legal_single_path_segment():
    for raw in ("doc/1", "a\\b", "c:d", "..", "with space", "PLAIN-123"):
        name = safe_filename(raw)
        assert "/" not in name and "\\" not in name
        assert name not in ("", ".", "..")
        assert name.endswith(".txt")


def test_safe_filename_is_deterministic():
    assert safe_filename("MED-2043") == safe_filename("MED-2043")


def test_select_documents_keeps_every_gold_document():
    """A dropped gold document turns a scorable question into an unanswerable one and quietly
    lowers the measured score for reasons that have nothing to do with the embedder."""
    qrels = {"q1": {"d3": 1}, "q2": {"d7": 2}}
    selected = select_documents(qrels, [f"d{i}" for i in range(10)], max_docs=4, seed=0)
    assert "d3" in selected
    assert "d7" in selected
    assert len(selected) == 4


def test_select_documents_never_truncates_gold_below_the_budget():
    # More gold documents than the budget allows. Silently dropping some would corrupt the qrels;
    # the budget yields instead, and the caller can see the corpus came back larger than asked.
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(6)}
    selected = select_documents(qrels, [f"d{i}" for i in range(10)], max_docs=3, seed=0)
    assert {f"d{i}" for i in range(6)} <= set(selected)


def test_select_documents_is_deterministic_for_a_given_seed():
    qrels = {"q1": {"d0": 1}}
    ids = [f"d{i}" for i in range(50)]
    assert select_documents(qrels, ids, max_docs=10, seed=5) == select_documents(qrels, ids, max_docs=10, seed=5)
    assert select_documents(qrels, ids, max_docs=10, seed=5) != select_documents(qrels, ids, max_docs=10, seed=6)


def test_build_questions_ignores_relevance_zero_judgements():
    """BEIR qrels carry a graded relevance and 0 means NOT relevant.

    Treating every judged pair as relevant inflates hit@5 for both embedders and compresses the
    gap toward zero — the study's entire response variable, quietly shrunk.
    """
    questions = build_questions(
        queries={"q1": "what is the dose"},
        qrels={"q1": {"d1": 0, "d2": 1}},
        selected={"d1", "d2"},
    )
    assert questions[0]["relevant_files"] == [safe_filename("d2")]


def test_build_questions_drops_a_question_whose_gold_is_not_in_the_corpus():
    # Scoring a question whose answer was never indexed measures nothing but the subsample.
    questions = build_questions(
        queries={"q1": "answerable", "q2": "orphaned"},
        qrels={"q1": {"d1": 1}, "q2": {"d99": 1}},
        selected={"d1"},
    )
    assert [q["id"] for q in questions] == ["q1"]


def test_build_questions_emits_the_shape_the_labelled_harness_consumes():
    questions = build_questions(
        queries={"q1": "what is the dose"},
        qrels={"q1": {"d2": 1}},
        selected={"d2"},
    )
    assert questions == [{
        "id": "q1",
        "query": "what is the dose",
        "answerable": True,
        "relevant_files": [safe_filename("d2")],
    }]
    json.dumps(questions)  # must survive the round trip to disk


def test_build_questions_sorts_relevant_files_for_reproducibility():
    # Set iteration order is not stable across runs; an unsorted list makes two identical runs
    # produce different question files and breaks byte-level reproducibility of the artifact.
    questions = build_questions(
        queries={"q1": "q"},
        qrels={"q1": {"d3": 1, "d1": 1, "d2": 1}},
        selected={"d1", "d2", "d3"},
    )
    assert questions[0]["relevant_files"] == sorted(questions[0]["relevant_files"])


def _write_beir(root, docs, queries, qrels_rows, *, header=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": i, "title": t, "text": b}) for i, (t, b) in docs.items()),
        encoding="utf-8")
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": i, "text": q}) for i, q in queries.items()), encoding="utf-8")
    (root / "qrels").mkdir(exist_ok=True)
    lines = (["query-id\tcorpus-id\tscore"] if header else []) + [
        f"{q}\t{d}\t{s}" for q, d, s in qrels_rows]
    (root / "qrels" / "test.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_materialize_writes_every_file_its_questions_reference(tmp_path):
    """The invariant the harness rests on: a `relevant_files` entry that is not on disk scores as
    a miss for every embedder, so a broken mapping looks exactly like a hard corpus."""
    src = tmp_path / "nfcorpus"
    _write_beir(
        src,
        docs={"MED-1": ("Dosage", "take one"), "MED/2": ("Timing", "at night"),
              "MED_2": ("Decoy", "unrelated"), "MED-9": ("Filler", "noise")},
        queries={"q1": "what dose", "q2": "when to take"},
        qrels_rows=[("q1", "MED-1", 1), ("q2", "MED/2", 2), ("q2", "MED_2", 0)],
    )
    out = tmp_path / "out"
    manifest = materialize(src, out, max_docs=4, seed=0)

    questions = json.loads((out / "questions.json").read_text(encoding="utf-8"))
    referenced = {f for q in questions for f in q["relevant_files"]}
    on_disk = {p.name for p in out.glob("*.txt")}
    assert referenced <= on_disk
    assert manifest["questions"] == 2
    assert manifest["documents_written"] == 4

    # The relevance-0 decoy must not have become gold for q2, and the two colliding ids
    # ('MED/2' vs 'MED_2') must have landed in different files.
    q2 = next(q for q in questions if q["id"] == "q2")
    assert q2["relevant_files"] == [safe_filename("MED/2")]
    assert safe_filename("MED/2") in on_disk and safe_filename("MED_2") in on_disk


def test_read_beir_parses_qrels_whether_or_not_a_header_row_is_present(tmp_path):
    # Mirrors differ. Reading the header as data yields int('score') -> ValueError, and skipping a
    # real data row silently loses one question's ground truth.
    for header in (True, False):
        src = tmp_path / f"ds-{header}"
        _write_beir(src, docs={"d1": ("T", "body")}, queries={"q1": "ask"},
                    qrels_rows=[("q1", "d1", 1)], header=header)
        _docs, _queries, qrels = read_beir(src)
        assert qrels == {"q1": {"d1": 1}}


def test_read_beir_keeps_the_title_with_the_body(tmp_path):
    # Titles carry real retrieval signal and are informative on some corpora and empty on others.
    # Dropping them would vary difficulty BY CORPUS, which is the axis the study correlates on.
    src = tmp_path / "titled"
    _write_beir(src, docs={"d1": ("Insulin resistance", "the body stops responding")},
                queries={"q1": "ask"}, qrels_rows=[("q1", "d1", 1)])
    docs, _q, _r = read_beir(src)
    assert "Insulin resistance" in docs["d1"]
    assert "the body stops responding" in docs["d1"]
