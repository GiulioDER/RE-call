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

Amendment 1, same day:

- ``test_a_storyline_past_the_bound_needs_compression``: ``>`` mutated to ``>=`` in
  ``needs_compression``; failed on ``assert not needs_compression(... 700 words)``.
- ``test_an_overlong_storyline_is_replaced_by_its_compression``: the line assigning the
  compressed storyline removed from ``builder_call``; failed on ``'long long ...' ==
  'short story'``. The restored module then passed all 10.

Amendment 3, same day:

- ``test_a_failed_compression_keeps_the_new_storyline``: the fail-forward branch mutated to keep
  the previous storyline (``result["storyline"] = storyline``); failed on
  ``assert result["storyline"].startswith("new")``. The restored module then passed all 11.

Amendment 4, same day:

- ``test_the_replicate_arm_answers_a_gated_question_from_the_stored_items``: ``"r0b"`` removed
  from the first condition of ``arm_items`` ALONE stayed green (that condition is redundant: the
  arm adds nothing on top either way), so it is not the proof. The proof: ``"r0b"`` also added to
  the storyline arms' tuple; failed on the equality (a storyline at index 0). The restored module
  then passed all 12.
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
    for arm in ("r0", "r0b", "story", "digests", "story_digests"):
        assert arm_items(arm, ITEMS, RECORDS, "What database did I choose?") == ITEMS


def test_the_replicate_arm_answers_a_gated_question_from_the_stored_items() -> None:
    assert arm_items("r0b", ITEMS, RECORDS, "Summarize my project") == ITEMS


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


def test_a_storyline_past_the_bound_needs_compression() -> None:
    from benchmarks.beam.f1_storyline_replay import COMPRESS_ABOVE_WORDS, needs_compression

    assert not needs_compression(" ".join(["w"] * COMPRESS_ABOVE_WORDS))
    assert needs_compression(" ".join(["w"] * (COMPRESS_ABOVE_WORDS + 1)))


def test_an_overlong_storyline_is_replaced_by_its_compression(monkeypatch) -> None:
    import benchmarks.beam.f1_storyline_replay as replay

    calls = []

    def fake(spend, system, user, max_tokens, fields):
        calls.append(system)
        if system == replay.BUILDER_SYSTEM:
            return {"digest": "d", "storyline": " ".join(["long"] * 900)}, {"usage": {}}
        return {"storyline": "short story"}, {"usage": {}}

    monkeypatch.setattr(replay, "_json_call", fake)
    chunk = {"date": "d", "messages": [{"role": "user", "content": "hi"}]}
    result, meta = replay.builder_call(None, "", chunk)
    assert calls == [replay.BUILDER_SYSTEM, replay.COMPRESS_SYSTEM]
    assert result["storyline"] == "short story"
    assert meta["compress"]["words_before"] == 900


def test_a_failed_compression_keeps_the_new_storyline(monkeypatch) -> None:
    import benchmarks.beam.f1_storyline_replay as replay

    def fake(spend, system, user, max_tokens, fields):
        if system == replay.BUILDER_SYSTEM:
            return {"digest": "d", "storyline": " ".join(["new"] * 900)}, {"usage": {}}
        raise RuntimeError("builder call failed with status 200: truncated")

    monkeypatch.setattr(replay, "_json_call", fake)
    chunk = {"date": "d", "messages": [{"role": "user", "content": "hi"}]}
    result, meta = replay.builder_call(None, "old", chunk)
    assert result["storyline"].startswith("new")
    assert "compress_failed" in meta
