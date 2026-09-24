"""Measurement helpers for the C9 LoCoMo route comparison.

Pre-registration: ``docs/preregistrations/2026-09-23-aml-c9-locomo-route-comparison.md``. Each test
names the mutation of ``scripts/aml_locomo_route_compare.py`` that was watched to fail it.
"""

from __future__ import annotations

from scripts.aml_locomo_route_compare import (
    build_corpus,
    evidence_ids,
    paired_bootstrap,
    score,
    session_timestamp_ms,
    turn_present,
)


def _sample() -> dict:
    return {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Ana",
            "speaker_b": "Ben",
            "session_2_date_time": "10:37 am on 27 June, 2023",
            "session_2": [{"speaker": "Ben", "dia_id": "D2:1", "text": "Later."}],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a cat."},
                {
                    "speaker": "Ben",
                    "dia_id": "D1:2",
                    "text": "Nice!",
                    "blip_caption": "a photo of a cat",
                },
            ],
        },
        "qa": [
            {"question": "What did Ana adopt?", "evidence": ["D1:1 D1:2"], "category": 1},
            {"question": "Adversarial?", "evidence": ["D1:1"], "category": 5},
            {"question": "Unresolvable?", "evidence": ["D9:9", "D"], "category": 2},
        ],
    }


def test_corpus_resolves_gold_turns_to_the_ingested_text_and_session() -> None:
    """Gold evidence must name exactly the text and session that were added.

    Red proof (mutation): dropping the ``blip_caption`` suffix in ``turn_content`` fails the
    ``gold_turns`` equality; keeping category 5 fails the question count.
    """
    adds, questions = build_corpus([_sample()], "run", None)

    assert [add["session_id"] for add in adds] == ["conv-x:session_1", "conv-x:session_2"]
    assert adds[0]["messages"][0] == {
        "role": "user",
        "content": "Ana: I adopted a cat.",
        "timestamp": session_timestamp_ms("1:56 pm on 8 May, 2023"),
    }
    assert len(questions) == 1
    assert questions[0]["gold_turns"] == [
        "Ana: I adopted a cat.",
        "Ben: Nice! [shared image: a photo of a cat]",
    ]
    assert questions[0]["gold_sessions"] == ["conv-x:session_1"]


def test_packed_evidence_ids_are_split_and_malformed_ones_dropped() -> None:
    """Red proof (mutation): not splitting on whitespace loses ``D4:4`` (only ``["D9:1"]``)."""
    assert evidence_ids(["D9:1 D4:4", "D9:1", "D", "session 3"]) == ["D9:1", "D4:4"]


def test_session_timestamp_is_utc_milliseconds() -> None:
    """Red proof (mutation): dropping ``* 1_000`` in ``session_timestamp_ms`` fails the equality."""
    assert session_timestamp_ms("1:56 pm on 8 May, 2023") == 1_683_554_160_000


def test_turn_presence_needs_the_turn_or_an_edge_of_a_long_one() -> None:
    """A split long turn counts by its first or last twelve words; a short turn needs all of it.

    Red proof (mutation): replacing the edge fallback with ``return False`` fails the split-turn
    assertion; lowering ``EDGE_WORDS`` to 2 makes the short-turn negative pass as a hit.
    """
    long_turn = "Ana: " + " ".join(f"w{i}" for i in range(40))
    window_with_head = "prefix " + " ".join(long_turn.split()[:15])

    assert turn_present("Ana: I adopted a cat.", "x Ana:  I adopted\na cat. y")
    assert turn_present(long_turn, window_with_head)
    assert not turn_present("Ana: I adopted a cat.", "Ana: I adopted")


def test_score_cuts_to_depth_before_matching() -> None:
    """Gold at rank 11 must miss at 10 and hit at 20, at turn and session level.

    Red proof (mutation): scoring ``items`` instead of ``items[:depth]`` makes ``turn_hit@10`` 1.
    """
    filler = [{"content": "noise", "session_id": "other", "kind": "raw"}] * 10
    gold = {"content": "Ana: I adopted a cat.", "session_id": "s1", "kind": "raw"}
    row = score(filler + [gold], ["Ana: I adopted a cat."], {"s1"})

    assert (row["turn_hit@10"], row["turn_hit@20"]) == (0, 1)
    assert (row["session_hit@10"], row["session_hit@20"]) == (0, 1)


def test_paired_bootstrap_signs_the_difference_as_treatment_minus_control() -> None:
    """Red proof (mutation): computing ``c - t`` reports -50 points and swaps rescues."""
    result = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1])

    assert result["delta_points"] == 50.0
    assert (result["rescues"], result["regressions"]) == (2, 0)
    assert result["ci95_low_points"] <= 50.0 <= result["ci95_high_points"]
