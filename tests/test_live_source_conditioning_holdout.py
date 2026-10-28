from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

import scripts.run_live_source_conditioning_holdout as holdout

from scripts.run_live_source_conditioning_holdout import (
    _challenge_config,
    _decision,
    _target_analysis,
)


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


def test_buried_challenge_selects_its_frozen_inputs_and_prior_sources() -> None:
    """The buried run cannot silently reuse the saturated first holdout.

    Red proof on 2026-09-13: I plausibly mutated ``_challenge_config`` so the buried query digest
    reused the independent holdout digest. The intended digest assertion failed. The production
    symbol under proof is ``_challenge_config``.
    """
    config = _challenge_config("buried")

    assert config["query_sha256"] == (
        "9eefe96c8bb07cb71b2bde71f359c222f6966e488c85097249e5536b6b074794"
    )
    assert config["fact_sha256"] == (
        "9e824f03caef370490f9be53b16451bc8fa29ee8d616aff7ff05351f6d0b86c5"
    )
    assert len(config["prior_query_sets"]) == 2


@pytest.mark.parametrize("challenge", ["independent", "buried"])
def test_archived_prior_inputs_reach_validation_without_bypassing_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, challenge: str
) -> None:
    """DAT-001: external archived prior inputs retain count, digest and overlap gates.

    Red proof: mutate main's prior_paths to config["prior_query_sets"], ignoring
    the parsed override. Both nodes fail the intended validated-input assertion
    with FileNotFoundError. Restoring the override makes the same nodes green.
    A second mutation using the later pseudonymized prior digest fails the literal
    registered-identity assertion. Original frozen sources must share the labels'
    source namespace. The model loader stops before serving can be constructed.
    """
    config = holdout._challenge_config(challenge)
    count = 1 if challenge == "independent" else 2
    assert config["prior_query_sha256"] == [
        "06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f",
        "cf91bb848a518991a49863c79ae3014a4aa9c4ba3f717877c2019266173b5940",
    ][:count]
    queries = [{"id": f"q{i}", "answerable": i < 18} for i in range(36)]
    labels = [{"query_id": f"q{i}", "source": f"holdout-{i}"} for i in range(18)]
    query_path, label_path, artifact_path = [tmp_path / name for name in ["query", "labels", "model"]]
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    artifact_path.write_bytes(b"synthetic artifact")
    paths = [tmp_path / f"prior-{i}" for i in range(count)]
    for i, path in enumerate(paths):
        path.write_text(json.dumps([{"relevant_files": [f"prior-source-{i}"]}]), encoding="utf-8")
    config = {
        **config,
        "query_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "fact_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "prior_query_sets": [str(tmp_path / f"missing-prior-{i}") for i in range(count)],
        "prior_query_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
    }
    monkeypatch.setattr(holdout, "_challenge_config", lambda _: config)
    monkeypatch.setattr(
        holdout, "EXPECTED_ARTIFACT_SHA256", hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )

    def stop_before_serving(_path: Path) -> None:
        raise RuntimeError("validated external prior inputs")

    monkeypatch.setattr(holdout, "load_source_conditioning_artifact", stop_before_serving)
    base_args = [
        "holdout", "--challenge", challenge, "--query-set", str(query_path),
        "--fact-labels", str(label_path), "--artifact", str(artifact_path),
        "--output", str(tmp_path / "output"), "--generation-id", holdout.EXPECTED_GENERATION_ID,
    ]
    prior_args = [arg for path in paths for arg in ["--prior-query-set", str(path)]]
    monkeypatch.setattr(sys, "argv", base_args + prior_args)
    observed = None
    try:
        holdout.main()
    except (OSError, ValueError, RuntimeError) as error:
        observed = error
    assert isinstance(observed, RuntimeError) and str(observed) == "validated external prior inputs", (
        f"external prior inputs did not reach validation: {observed!r}"
    )

    # Omitting a prior or supplying an extra one must not weaken the coverage gate.
    wrong_count = paths[:-1] if count > 1 else paths + paths
    monkeypatch.setattr(
        sys, "argv", base_args + [arg for path in wrong_count for arg in ["--prior-query-set", str(path)]]
    )
    with pytest.raises(ValueError, match="every registered prior query set"):
        holdout.main()
    monkeypatch.setattr(sys, "argv", base_args + prior_args)
    original = paths[0].read_bytes()
    paths[0].write_bytes(b"[]")
    with pytest.raises(RuntimeError, match="prior query set differs from the registered digest"):
        holdout.main()
    paths[0].write_bytes(original)

    # Registered external priors still pass through the real contamination check.
    paths[0].write_text(json.dumps([{"relevant_files": ["holdout-0"]}]), encoding="utf-8")
    config["prior_query_sha256"][0] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="holdout source overlap with prior gold"):
        holdout.main()
