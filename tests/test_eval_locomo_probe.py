from __future__ import annotations

from recall.embeddings import HashingEmbedder
from recall.eval import locomo
from recall.retriever import LegProbe
from tests.conftest import requires_db

#: Minimal LOCOMO-shaped conversation. Real payloads carry `speaker_a`/`speaker_b`,
#: one `session_N_date_time` string per session, and `session_N` lists of
#: {speaker, dia_id, text} — see locomo10.json.
CONVERSATION = {
    "speaker_a": "Ann",
    "speaker_b": "Bob",
    "session_1_date_time": "1:00 pm on 8 May, 2023",
    "session_1": [
        {"speaker": "Ann", "dia_id": "D1:1", "text": "I adopted a tabby cat named Mochi."},
        {"speaker": "Bob", "dia_id": "D1:2", "text": "I started a pottery class on Tuesdays."},
        {"speaker": "Ann", "dia_id": "D1:3", "text": "My flight to Lisbon leaves on 12 June."},
    ],
}

QA = [
    {"question": "What is the name of Ann's cat?", "answer": "Mochi",
     "evidence": ["D1:1"], "category": 2},
    {"question": "When does Ann fly to Lisbon?", "answer": "12 June",
     "evidence": ["D1:3"], "category": 2},
]


@requires_db
def test_run_conversation_forwards_the_probe_and_fires_once_per_question(tmp_path, make_store):
    emb = HashingEmbedder(dim=64)
    store = make_store(64)
    seen: list[LegProbe] = []

    res = locomo.run_conversation(
        CONVERSATION,
        QA,
        store=store,
        embedder=emb,
        k=5,
        corpus_dir=tmp_path / "corpus",
        ks=[1, 5],
        probe=seen.append,
    )

    answerable = [q for q in res["questions"] if "evidence" in q]
    assert len(answerable) == 2
    # One probe per probed search, in question order. Task 5's CLI pairs these two lists with
    # zip(strict=True), so this ordering IS the contract that makes its records trustworthy.
    assert len(seen) == 2
    assert [p.query for p in seen] == [q["question"] for q in answerable]
    assert all(p.dense for p in seen)


@requires_db
def test_run_conversation_without_a_probe_still_scores(tmp_path, make_store):
    emb = HashingEmbedder(dim=64)
    store = make_store(64)

    res = locomo.run_conversation(
        CONVERSATION, QA, store=store, embedder=emb, k=5,
        corpus_dir=tmp_path / "corpus", ks=[1, 5],
    )

    assert len([q for q in res["questions"] if "evidence" in q]) == 2
