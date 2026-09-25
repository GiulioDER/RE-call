"""Contract proofs for the live C9 BEAM probe and its prep step.

Red proof receipts, 2026-09-24: each named test was run against the stated one-line mutation of
``benchmarks/beam/aml_c9_probe.py`` or ``aml_c9_prep.py``, failed on its assertion, and passed
again once reverted.

* ``test_chronological_orders_by_batch_then_position``: sorting on ``rank`` alone in
  ``chronological`` leaves the returned order.
* ``test_an_unlocatable_item_sorts_after_its_batch``: ``position if position >= 0 else -1``
  puts the compiled record first.
* ``test_event_ordering_matches_the_aml_formula``: dropping ``* f1`` from
  ``event_ordering_score`` scores an incomplete ordering 1.0.
* ``test_the_judge_rejects_an_off_scale_or_partial_answer``: removing the ``any(v not in
  (0.0, 0.5, 1.0) ...)`` clause in ``_scores`` accepts a 0.7 (``DID NOT RAISE ValueError``).
  Removing the index clause instead only moves the failure to a ``KeyError``, which ``_judged``
  already retries, so it is not used as proof.
* ``test_a_top_arm_answers_from_the_first_n_returned_items``: ``items[: TOP_ARMS[arm]]``
  replaced by ``items[-TOP_ARMS[arm]:]`` answers from the tail.
* ``test_a_quote_counts_only_when_it_is_verbatim_in_the_context``: ``if quote and quote in
  haystack`` replaced by ``if quote`` credits an invented passage.
* ``test_the_summary_block_is_this_conversation_in_session_order``: ``key=lambda s: s["batch"]``
  replaced by ``key=lambda s: -s["batch"]`` puts the last session first.
* ``test_prep_dates_every_message_in_batch_order``: dropping ``+ ordinal * 60_000`` gives
  every message of a batch the same timestamp.
"""

from __future__ import annotations

import pytest

import benchmarks.beam.aml_c9_probe as probe
from benchmarks.beam.aml_c9_prep import batch_millis, conversation_record


USER = "beam-probe-x-conv-0"


def _conversation() -> dict:
    return {
        "conversation": 0,
        "batches": [
            {"messages": [{"content": "alpha beta gamma delta epsilon zeta eta theta"}]},
            {"messages": [{"content": "one two three four five six seven eight nine ten"}]},
        ],
    }


def _item(content: str, batch: int) -> dict:
    return {"content": content, "session_id": f"{USER}/batch-{batch:02d}"}


def test_chronological_orders_by_batch_then_position() -> None:
    items = [_item("six seven eight", 1), _item("gamma delta", 0), _item("two three", 1),
             _item("alpha beta", 0)]
    ordered = probe.chronological(items, _conversation(), USER)
    assert [i["content"] for i in ordered] == ["alpha beta", "gamma delta", "two three",
                                               "six seven eight"]


def test_an_unlocatable_item_sorts_after_its_batch() -> None:
    items = [_item("a compiled summary not in the text", 0), _item("gamma delta", 0),
             _item("two three", 1)]
    ordered = probe.chronological(items, _conversation(), USER)
    assert [i["content"] for i in ordered] == ["gamma delta", "a compiled summary not in the text",
                                               "two three"]


def test_event_ordering_matches_the_aml_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    def exact(_spend, messages, _max_tokens, json_mode=False):
        first, second = messages[1]["content"].split("\n\nSecond snippet: ")
        return "YES" if first.removeprefix("First snippet: ") == second else "NO"

    monkeypatch.setattr(probe, "complete", exact)
    spend = probe.Spend(1.0)
    reference = ["a", "b", "c", "d"]
    assert probe.event_ordering_score(spend, reference, ["a", "b", "c", "d"]) == 1.0
    # Two of four events, in order: precision 1, recall 0.5, f1 2/3; unmatched ranks tie.
    partial = probe.event_ordering_score(spend, reference, ["a", "b"])
    assert 0.0 < partial < 2 / 3 + 1e-9
    assert probe.event_ordering_score(spend, reference, ["d", "c", "b", "a"]) == 0.0


def test_the_judge_rejects_an_off_scale_or_partial_answer() -> None:
    assert probe._scores('{"scores": [{"index": 0, "score": 1.0}, {"index": 1, "score": 0.5}]}', 2) == [1.0, 0.5]
    with pytest.raises(ValueError):
        probe._scores('{"scores": [{"index": 0, "score": 1.0}]}', 2)
    with pytest.raises(ValueError):
        probe._scores('{"scores": [{"index": 0, "score": 0.7}, {"index": 1, "score": 1.0}]}', 2)


def test_prep_dates_every_message_in_batch_order() -> None:
    row = {
        "chat": [[{"role": "user", "content": "first", "time_anchor": "March-15-2024"},
                  {"role": "assistant", "content": "second"}]],
        "probing_questions": "{}",
    }
    record = conversation_record(row, "100K", 0)
    stamps = [m["timestamp"] for m in record["batches"][0]["messages"]]
    assert stamps[0] == batch_millis("March-15-2024")
    assert stamps[1] > stamps[0]
    assert batch_millis("not a date") is None


def test_a_top_arm_answers_from_the_first_n_returned_items() -> None:
    items = [_item(f"item {i}", 0) for i in range(100)]
    assert probe.arm_items("top10", items, _conversation(), USER) == items[:10]
    assert probe.arm_items("top40", items, _conversation(), USER) == items[:40]
    assert probe.arm_items("returned", items, _conversation(), USER) == items


def test_a_quote_counts_only_when_it_is_verbatim_in_the_context() -> None:
    context = "The user moved to Lisbon in March.\n\nShe  started at the aquarium."
    scores, invented = probe.verified_quotes(
        context,
        {0: "moved to lisbon in march", 1: "she started at the aquarium", 2: "she moved to Porto"},
        4,
    )
    assert scores == [1.0, 1.0, 0.0, 0.0]
    assert invented == 1


def test_the_summary_block_is_this_conversation_in_session_order() -> None:
    summaries = [
        {"conversation": 0, "batch": 1, "summary": "second"},
        {"conversation": 1, "batch": 0, "summary": "other user"},
        {"conversation": 0, "batch": 0, "summary": "first"},
    ]
    block = probe.summary_block(0, summaries)
    assert block == "Session 1 summary: first\n\nSession 2 summary: second"
