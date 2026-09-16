from __future__ import annotations

import json

from scripts.build_atomic_fact_blind_source_holdout import (
    USER_TEMPLATE,
    _input_sources,
    construct_pool,
    validate_question,
    writer_user_prompt,
)


def test_input_sources_reads_all_frozen_schema_shapes() -> None:
    """Read pool, trace, and answerable production gold sources.

    Red proof: before the first green run, the deliberate mutation removed the answerable list
    branch in `_input_sources`. This node failed at the intended `recall/live.md` assertion because
    the mutated function returned an empty set.
    """

    pool = {"queries": [{"gold_sources": ["recall/pool.md"]}]}
    trace = {"rows": [{"trace": {"pool": [{"source": "recall/trace.md"}]}}]}
    production = [
        {"answerable": True, "relevant_files": ["recall/live.md"]},
        {"answerable": False, "relevant_files": ["recall/control.md"]},
    ]
    assert _input_sources(pool) == {"recall/pool.md"}
    assert _input_sources(trace) == {"recall/trace.md"}
    assert _input_sources(production) == {"recall/live.md"}


def test_writer_prompt_contains_only_the_fact_payload() -> None:
    """Keep source structure out of the question writer input.

    Red proof: before the first green run, the deliberate mutation appended a source label in
    `writer_user_prompt`. This node failed at the intended exact prompt equality.
    """

    content = "The calibration cutoff changes only after certification."
    assert writer_user_prompt(content) == USER_TEMPLATE.format(content=content)


def test_preflight_stops_before_model_when_population_is_too_small(monkeypatch) -> None:
    """Do not spend model calls when the frozen target is mathematically impossible.

    Red proof: before the first green run, the deliberate mutation bypassed the candidate-count
    branch in `construct_pool`. This node failed at the intended fake writer assertion because the
    mutated implementation attempted a model call.
    """

    import scripts.build_atomic_fact_blind_source_holdout as module

    class NoCallWriter:
        def complete(self, system: str, user: str) -> str:
            raise AssertionError("writer must not be called after an insufficient-population stop")

    monkeypatch.setattr(module, "load_exclusions", lambda paths: (set(), {}))
    monkeypatch.setattr(
        module,
        "candidate_rows",
        lambda roots, excluded: (
            [
                {
                    "source": f"recall/{index}.md",
                    "view": {"content": f"Unique fact number {index} has enough words for testing."},
                }
                for index in range(45)
            ],
            {},
            86,
        ),
    )
    pool, audit = construct_pool([], [], NoCallWriter())
    assert pool["attempted_sources"] == 0
    assert pool["eligible_source_candidates"] == 45
    assert audit["decision"] == "STOP_BLIND_HOLDOUT_CONSTRUCTION"


def test_question_validation_rejects_five_token_answer_copy() -> None:
    """Reject a question that copies any five contiguous answer tokens.

    Red proof: before the first green run, the deliberate mutation changed the overlap window in
    `validate_question` from five tokens to six. This node failed at the intended assertion because
    the copied five-token phrase was accepted.
    """

    answer = (
        "Calibration thresholds stay fixed during routine updates until a certified evaluation "
        "approves change."
    )
    response = json.dumps(
        {"question": "Why do thresholds stay fixed during routine maintenance?"}
    )
    assert validate_question(response, answer) == (None, "answer_ngram")


def test_question_validation_accepts_natural_noncopying_question() -> None:
    """Control: a natural, nonstructural question remains eligible.

    This is a nonbehavioral positive control for the two rejection tests. It proves their fixture
    is not rejected merely because every well-formed question fails validation.
    """

    answer = (
        "Calibration thresholds stay fixed during routine updates until a certified evaluation "
        "approves change."
    )
    response = json.dumps(
        {"question": "When is it acceptable to change the calibration cutoff in production?"}
    )
    assert validate_question(response, answer) == (
        "When is it acceptable to change the calibration cutoff in production?",
        None,
    )
