"""The F1 storyline replay changes only what it claims to: gated questions, on top, within top_k.

Red proof, 2026-09-25, each by a deliberate mutation of the named line, then restored green:

- ``test_a_chunk_closes_at_twenty_messages``: ``len(current) >= CHUNK_MESSAGES`` mutated to
  ``>``; failed on ``assert [len(c["messages"]) for c in chunks] == [20, 5]`` (got [21, 4]).
- ``test_a_chunk_closes_before_the_message_that_passes_2000_words``: ``words + size >
  CHUNK_WORDS`` mutated to ``words > CHUNK_WORDS``; failed on the chunk sizes (got
  [[900, 900, 300], [2500]]).
- ``test_chunks_never_cross_a_session``: the per-batch reset moved outside the batch loop; failed
  on the chunk shapes (got [(0, 3), (1, 3)]).
- ``test_an_ungated_question_keeps_the_stored_items_exactly``: the ``not gated(question)``
  clause of ``arm_items`` removed; failed on the identity assertion (a storyline at index 0).
- ``test_every_arm_stays_within_top_k``: ``[:TOP_K]`` removed from ``arm_items``; failed with
  101 items.
- ``test_the_gate_fires_on_summary_intent_only``: ``overview`` removed from ``GATE``; failed on
  the overview question. The restored module then passed all 8.
"""

from __future__ import annotations

from benchmarks.beam.f1_storyline_replay import (
    TOP_K,
    aml_chunks,
    arm_items,
    digest_items,
    gated,
)


def _conversation(*batches: list[int]) -> dict:
    """Batches of messages, each message given by its word count."""
    return {"conversation": 0, "batches": [
        {"date": f"March-{i + 1}-2024",
         "messages": [{"role": "user", "content": " ".join(["w"] * words)} for words in batch]}
        for i, batch in enumerate(batches)]}


def test_a_chunk_closes_at_twenty_messages() -> None:
    chunks = aml_chunks(_conversation([1] * 25))
    assert [len(c["messages"]) for c in chunks] == [20, 5]


def test_a_chunk_closes_before_the_message_that_passes_2000_words() -> None:
    chunks = aml_chunks(_conversation([900, 900, 300, 2500]))
    assert [[len(m["content"].split()) for m in c["messages"]] for c in chunks] == [
        [900, 900], [300], [2500]]


def test_chunks_never_cross_a_session() -> None:
    chunks = aml_chunks(_conversation([1, 1], [1]))
    assert [(c["batch"], len(c["messages"])) for c in chunks] == [(0, 2), (1, 1)]
    assert chunks[1]["date"] == "March-2-2024"


def test_the_gate_fires_on_summary_intent_only() -> None:
    assert gated("Can you provide a detailed summary of the whole process?")
    assert gated("Give me an overview of how my project went")
    assert gated("请总结一下我们的讨论")
    assert not gated("What database did I choose for the MVP?")
    assert not gated("When did Caroline go to the LGBTQ support group?")


RECORDS = [{"conversation": 0, "chunk": i, "date": "March-1-2024", "digest": f"digest {i}",
            "storyline": f"story after {i}"} for i in range(3)]
ITEMS = [{"id": f"raw-{i}", "content": f"item {i}"} for i in range(100)]


def test_an_ungated_question_keeps_the_stored_items_exactly() -> None:
    for arm in ("r0", "story", "digests", "story_digests"):
        assert arm_items(arm, ITEMS, RECORDS, "What database did I choose?") == ITEMS


def test_a_gated_question_gets_the_latest_storyline_on_top() -> None:
    items = arm_items("story_digests", ITEMS, RECORDS, "Summarize my project")
    assert items[0]["content"].endswith("story after 2")
    assert [i["id"] for i in items[1:4]] == ["digest-0-0", "digest-0-1", "digest-0-2"]
    assert items[4:] == ITEMS[:96]


def test_every_arm_stays_within_top_k() -> None:
    for arm in ("r0", "story", "digests", "story_digests", "story_all"):
        assert len(arm_items(arm, ITEMS, RECORDS, "Summarize everything")) == TOP_K


def test_digests_past_the_cap_keep_chronological_order() -> None:
    records = [{"conversation": 0, "chunk": i, "date": "d", "digest": f"topic{i % 3} note {i}",
                "storyline": "s"} for i in range(40)]
    chosen = digest_items(records, "summary of topic1")
    assert len(chosen) == 24
    chunks = [int(item["id"].rsplit("-", 1)[1]) for item in chosen]
    assert chunks == sorted(chunks)
    assert {c for c in range(40) if c % 3 == 1} <= set(chunks)
