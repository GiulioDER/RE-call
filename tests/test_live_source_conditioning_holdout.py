from __future__ import annotations

from scripts.run_live_source_conditioning_holdout import _decision, _target_analysis


def _score(*, source_hit: bool, complete: bool, facts: int) -> dict[str, object]:
    return {
        "source_hit": source_hit,
        "complete": complete,
        "covered_facts": [f"fact-{index}" for index in range(facts)],
    }


def test_target_population_is_selected_only_by_control_failure_shape() -> None:
    """The target requires a control source hit and fact miss.

    Red proof on 2026-09-13: I plausibly mutated ``_target_analysis`` to omit the
    ``not baseline complete`` predicate. This assertion failed because ``complete`` entered
    ``target_query_ids``. The production symbol under proof is ``_target_analysis``.
    """
    rows = [
        {
            "query": {"id": "rescued"},
            "label": {"facts": [{}]},
            "scores": {
                "baseline": _score(source_hit=True, complete=False, facts=0),
                "candidate": _score(source_hit=True, complete=True, facts=1),
            },
        },
        {
            "query": {"id": "complete"},
            "label": {"facts": [{}]},
            "scores": {
                "baseline": _score(source_hit=True, complete=True, facts=1),
                "candidate": _score(source_hit=True, complete=True, facts=1),
            },
        },
        {
            "query": {"id": "missing-source"},
            "label": {"facts": [{}]},
            "scores": {
                "baseline": _score(source_hit=False, complete=False, facts=0),
                "candidate": _score(source_hit=False, complete=False, facts=0),
            },
        },
    ]

    observed = _target_analysis(rows)

    assert observed["target_query_ids"] == ["rescued"]
    assert observed["target_fact_rescues"] == 1
    assert observed["complete_gain_ids"] == ["rescued"]
    assert observed["fact_gain_ids"] == ["rescued"]


def test_decision_requires_live_integrity_and_registered_quality_gates() -> None:
    """ADVANCE requires the full apparatus plus every quality and safety gate.

    Red proof on 2026-09-13: I plausibly mutated ``_decision`` to ignore selection liveness.
    The intended ``REPAIR`` assertion failed with ``ADVANCE``. The production symbol under proof
    is ``_decision`` and the mutation removed the minimum selection difference condition.
    """
    summary = {
        "arms": {
            "baseline": {
                "complete_queries": 9,
                "covered_facts": 9,
                "unanswerable_answers": 1,
                "context_precision": 0.60,
            },
            "candidate": {
                "complete_queries": 10,
                "covered_facts": 10,
                "unanswerable_answers": 1,
                "context_precision": 0.56,
            },
        }
    }
    target = {
        "target_queries": 4,
        "target_fact_rescues": 1,
        "complete_loss_ids": [],
        "fact_loss_ids": [],
    }
    integrity = {
        "requests": 36,
        "candidate_hash_parity": 36,
        "baseline_hash_parity": 36,
        "shadow_ok": 36,
        "timing_receipts": 36,
        "selection_differences": 2,
    }

    assert _decision(summary, target, integrity) == "ADVANCE"
    assert _decision(summary, {**target, "target_queries": 3}, integrity) == "INSUFFICIENT"
    assert _decision(summary, target, {**integrity, "selection_differences": 1}) == "REPAIR"
    unsafe = {
        **summary,
        "arms": {
            **summary["arms"],
            "candidate": {**summary["arms"]["candidate"], "unanswerable_answers": 2},
        },
    }
    assert _decision(unsafe, target, integrity) == "CLOSE"
