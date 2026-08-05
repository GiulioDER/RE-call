"""The frozen input manifest: it must refuse an edit, not repair one."""
from __future__ import annotations

import json

import pytest

from recall.eval.promotion.manifest import (
    FrozenQuestion,
    manifest_digest,
    question_input_hash,
    read_manifest,
    write_manifest,
)


def _question(qid: str = "q01", corpus: str = "labelled", query: str = "why postgres") -> FrozenQuestion:
    return FrozenQuestion(
        question_id=qid,
        corpus=corpus,
        input_hash=question_input_hash(
            question_id=qid, corpus=corpus, query=query, expected_relevance_labels=("a.md:0",)
        ),
        expected_relevance_labels=("a.md:0",),
    )


def test_the_digest_does_not_depend_on_the_order_questions_were_emitted() -> None:
    a, b = _question("q01"), _question("q02")
    assert manifest_digest([a, b], corpus_hashes={"c": "x"}) == manifest_digest(
        [b, a], corpus_hashes={"c": "x"}
    )


def test_the_digest_covers_the_provenance_and_not_only_the_bodies(tmp_path) -> None:
    """The header is the field a tamperer edits to claim a provenance the manifest lacks."""
    questions = [_question()]
    one = manifest_digest(questions, corpus_hashes={"labelled:queries.json": "aaa"})
    two = manifest_digest(questions, corpus_hashes={"labelled:queries.json": "bbb"})
    assert one != two


def test_an_edited_label_is_refused_rather_than_read(tmp_path) -> None:
    """The failure this file exists for: a label corrected after seeing a candidate's results."""
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_question()], corpus_hashes={"labelled:queries.json": "aaa"})
    lines = path.read_text(encoding="utf-8").splitlines()
    body = json.loads(lines[1])
    body["expected_relevance_labels"] = ["b.md:0"]
    path.write_text(
        lines[0] + "\n" + json.dumps(body, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match header digest"):
        read_manifest(path)


def test_a_round_trip_returns_the_same_questions(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    questions = [_question("q01"), _question("q02")]
    digest = write_manifest(path, questions, corpus_hashes={"labelled:queries.json": "aaa"})
    loaded, header = read_manifest(path)
    assert header["digest"] == digest
    assert sorted(loaded, key=lambda q: q.question_id) == sorted(
        questions, key=lambda q: q.question_id
    )


def test_verify_refuses_a_query_that_drifted_under_the_frozen_manifest() -> None:
    """The manifest digest cannot catch this: it proves the FILE is intact, not the CORPUS."""
    question = _question(query="why postgres")
    question.verify("why postgres")
    with pytest.raises(ValueError, match="changed after the manifest was frozen"):
        question.verify("why postgres over a document store")


def test_input_hash_cannot_be_collided_by_moving_a_boundary() -> None:
    """NUL-terminated fields: without them ("ab","c") and ("a","bc") hash alike."""
    left = question_input_hash(
        question_id="ab", corpus="c", query="q", expected_relevance_labels=()
    )
    right = question_input_hash(
        question_id="a", corpus="bc", query="q", expected_relevance_labels=()
    )
    assert left != right


def test_an_empty_manifest_is_refused(tmp_path) -> None:
    """A gate over no questions passes every check it is asked."""
    with pytest.raises(ValueError, match="empty manifest"):
        write_manifest(tmp_path / "m.jsonl", [], corpus_hashes={})


def test_a_duplicate_question_is_refused_at_freeze_time(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate frozen question"):
        write_manifest(
            tmp_path / "m.jsonl", [_question("q01"), _question("q01")], corpus_hashes={}
        )


def test_the_same_id_in_two_corpora_is_not_a_duplicate(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    write_manifest(
        path,
        [_question("q01", corpus="labelled"), _question("q01", corpus="peps")],
        corpus_hashes={},
    )
    loaded, _ = read_manifest(path)
    assert {question.corpus for question in loaded} == {"labelled", "peps"}


def test_the_bytes_do_not_depend_on_the_operating_system(tmp_path) -> None:
    """The digest is over "\\n"-joined lines; text-mode translation would write CRLF on Windows."""
    path = tmp_path / "m.jsonl"
    write_manifest(path, [_question()], corpus_hashes={"c": "x"})
    assert b"\r\n" not in path.read_bytes()
