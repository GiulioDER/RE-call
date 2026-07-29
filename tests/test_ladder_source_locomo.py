"""LOCOMO -> documents + questions, with the id-collision trap the existing runner already hit.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

`dia_id` is unique only WITHIN a conversation (recall/eval/locomo.py:415). Loading ten
conversations into one id space without namespacing would silently make "D1:3" from conversation
0 and conversation 7 the same document — and the excision would remove the wrong turn while every
count still looked right.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ladder.sources.locomo import load_locomo

SAMPLE = [
    {
        "sample_id": "conv-0",
        "conversation": {
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Caroline", "text": "I went to the support group."},
                {"dia_id": "D1:2", "speaker": "Melanie", "text": "How was it?"},
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {"question": "An adversarial one", "adversarial_answer": "no", "category": 5},
        ],
    },
    {
        "sample_id": "conv-1",
        "conversation": {
            "session_1_date_time": "1 Jan 2023",
            "session_1": [{"dia_id": "D1:1", "speaker": "Ann", "text": "Different conversation."}],
        },
        "qa": [{"question": "Who spoke?", "answer": "Ann", "evidence": ["D1:1"], "category": 1}],
    },
]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return path


def test_doc_ids_are_namespaced_by_conversation(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    ids = [doc_id for doc_id, _ in corpus.documents]
    assert "conv-0/D1:1" in ids
    assert "conv-1/D1:1" in ids
    assert len(ids) == len(set(ids))


def test_gold_ids_are_namespaced_to_match_the_documents(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    q = next(q for q in corpus.questions if q.question_id == "conv-0/qa0")
    assert q.gold_doc_ids == ("conv-0/D1:1",)


def test_adversarial_category_five_questions_are_dropped(tmp_path: Path):
    """They are unanswerable by ANNOTATION, not by excision — a different construction."""
    corpus = load_locomo(_write(tmp_path))
    assert all("adversarial" not in q.question.lower() for q in corpus.questions)
    assert len(corpus.questions) == 2


def test_cluster_members_group_turns_by_conversation(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    assert corpus.cluster_members["conv-0"] == ("conv-0/D1:1", "conv-0/D1:2")
    assert corpus.cluster_members["conv-1"] == ("conv-1/D1:1",)


def test_document_text_carries_speaker_and_date(tmp_path: Path):
    corpus = load_locomo(_write(tmp_path))
    text = dict(corpus.documents)["conv-0/D1:1"]
    assert "Caroline" in text and "7 May 2023" in text and "support group" in text


def test_content_hash_changes_when_a_turn_changes(tmp_path: Path):
    before = load_locomo(_write(tmp_path)).content_hash
    altered = json.loads(json.dumps(SAMPLE))
    altered[0]["conversation"]["session_1"][0]["text"] = "changed"
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert load_locomo(path).content_hash != before


def test_evidence_naming_a_turn_the_conversation_does_not_have_is_dropped_and_counted(tmp_path: Path):
    """rings.build_rings refuses such gold, so it must never reach the builder — and the drop is
    counted, because a corpus that quietly discards questions still yields a clean-looking curve."""
    altered = json.loads(json.dumps(SAMPLE))
    altered[0]["qa"][0]["evidence"] = ["D9:9"]
    path = tmp_path / "ghost.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    corpus = load_locomo(path)
    assert all(q.question_id != "conv-0/qa0" for q in corpus.questions)
    assert corpus.dropped["evidence_not_in_conversation"] == 1


def test_questions_without_evidence_are_dropped_not_silently_ungolded(tmp_path: Path):
    altered = json.loads(json.dumps(SAMPLE))
    altered[1]["qa"][0].pop("evidence")
    path = tmp_path / "noevi.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    corpus = load_locomo(path)
    assert all(q.gold_doc_ids for q in corpus.questions)
    assert len(corpus.questions) == 1
