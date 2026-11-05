"""Every registered gate refuses the inputs it cannot decide on (M4).

Invariant: each gate below, given an entry of `tests/undecidable.py` for an input it takes, raises
or returns its negative answer. The failure mode is CCA's `fail-open` class: a NaN that compares
False with a threshold, a clamp that turns NaN into 1.0, a truthy string `"false"`, a misspelled
key that falls back to the permissive default.

Gates cannot be discovered by name, so the registry is explicit: a new gate is added here by hand,
and that is this mechanism's known weakness. Each gate's negative control is a decidable input that
passes, so a gate that refuses everything cannot satisfy its row.

Red proof (2026-11-04). The first two are master's code, live fail-open defects the corpus found
on its first run:

- `recall.trust_gate._verdict` without its `math.isfinite(threshold)` test:
  `test_trust_verdict_refuses_an_undecidable_threshold[nan]` failed, the verdict `ok`.
- `recall.calibration.Calibration` without its `__post_init__`:
  `test_calibration_refuses_an_undecidable_threshold[nan]` failed (no ValueError), and with the
  check restored but `confidence` back on `math.isnan`,
  `test_confidence_is_never_high_for_an_undecidable_cosine[inf]` failed at 1.0.
- `recall._env.strict_bool` returning False for an unknown value:
  `test_strict_bool_refuses_an_undecidable_flag[misspelled-false]` failed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.undecidable import FLAGS, MISSPELLED_KEYS, MODE_SPELLINGS, NUMBERS, STRING_FALSES

NOW = datetime(2026, 11, 4, tzinfo=UTC)


def _hit(score: float):
    from recall.types import Chunk, ScoredChunk

    return ScoredChunk(chunk=Chunk(id="a", source="/r/a.md", text="t", metadata={"file": "a.md"}), score=score)


# -- the trust gate -------------------------------------------------------------------------------
def test_trust_verdict_passes_a_decidable_hit() -> None:
    from recall.trust_gate import _verdict

    assert _verdict(_hit(0.9), {}, 0.5, NOW)[0] == "ok"


@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_trust_verdict_refuses_an_undecidable_threshold(name: str) -> None:
    from recall.trust_gate import _verdict

    assert _verdict(_hit(0.9), {}, NUMBERS[name], NOW)[0] == "low_confidence"


@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_trust_verdict_refuses_an_undecidable_score(name: str) -> None:
    from recall.trust_gate import _verdict

    assert _verdict(_hit(NUMBERS[name]), {}, 0.5, NOW)[0] == "low_confidence"


# -- calibration ----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_calibration_refuses_an_undecidable_threshold(name: str) -> None:
    from recall.calibration import Calibration

    with pytest.raises(ValueError, match="threshold must be finite"):
        Calibration(embedder="e", threshold=NUMBERS[name])


@pytest.mark.parametrize("scale", [*NUMBERS.values(), 0.0, -0.1], ids=[*NUMBERS, "zero", "negative"])
def test_calibration_refuses_an_undecidable_scale(scale: float) -> None:
    from recall.calibration import Calibration

    with pytest.raises(ValueError, match="scale must be finite and positive"):
        Calibration(embedder="e", threshold=0.5, scale=scale)


@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_confidence_is_never_high_for_an_undecidable_cosine(name: str) -> None:
    from recall.calibration import Calibration

    calibration = Calibration(embedder="e", threshold=0.5)
    assert calibration.confidence(0.9) > 0.5  # negative control: a decidable cosine above the bar
    assert calibration.confidence(NUMBERS[name]) < 0.5


@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_from_samples_refuses_an_undecidable_sample(name: str) -> None:
    from recall.calibration import from_samples

    with pytest.raises(ValueError):
        from_samples("e", [0.9, NUMBERS[name], 0.8], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        from_samples("e", [0.9, 0.8, 0.85], [0.1, NUMBERS[name], 0.3])


# -- configuration parsers ------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FLAGS))
def test_strict_bool_refuses_an_undecidable_flag(name: str) -> None:
    from recall._env import strict_bool

    assert strict_bool("true", name="X") is True and strict_bool("false", name="X") is False
    with pytest.raises(ValueError):
        strict_bool(FLAGS[name], name="X")


@pytest.mark.parametrize("name", sorted(MODE_SPELLINGS))
def test_trust_policy_never_relaxes_on_a_misspelled_mode(name: str) -> None:
    from recall.trust_policy import TrustPolicy

    try:
        policy = TrustPolicy(mode=MODE_SPELLINGS[name])  # type: ignore[arg-type]
    except ValueError:
        return
    assert policy.strict


@pytest.mark.parametrize("name", sorted(STRING_FALSES))
def test_write_time_hook_is_off_for_a_string_false(name: str) -> None:
    from recall_hooks.write_time import settings

    assert settings({"write_time": {"enabled": False}})["enabled"] is False
    assert settings({"write_time": {"enabled": STRING_FALSES[name]}})["enabled"] is False


@pytest.mark.parametrize("where", sorted(MISSPELLED_KEYS))
def test_source_policy_refuses_a_misspelled_key(where: str, tmp_path: Path) -> None:
    from recall.security_policy import load_source_policy

    rule: dict[str, object] = {
        "prefix": "notes/", "classification": "restricted", "principals": ["a"], "purposes": ["b"],
        "redactions": [],
    }
    payload: dict[str, object] = {"default_deny": True, "rules": [rule]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(path)}) is not None
    if where == "policy":
        payload[MISSPELLED_KEYS[where]] = payload.pop("default_deny")
    else:
        rule[MISSPELLED_KEYS[where]] = rule.pop("classification")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(path)})


# -- measurement and spend gates ------------------------------------------------------------------
PROMOTION = {
    **{key: True for key in (
        "contract_passed", "isolation_passed", "persistence_passed", "idempotency_passed",
        "release_frozen", "external_https_ready", "stable_30_days_committed", "paired_identity_passed",
    )},
    "net_successful_cells": 8, "task_count": 34, "paired_cells": 102, "seeds": [0, 1, 2],
    "invalid_cells": 0, "new_control_failures": 0, "add_concurrency": 16, "add_errors": 0,
    "search_concurrency": 16, "search_errors": 0, "add_p95_seconds": 1.0,
    "search_p95_seconds": 1.0, "provider_spend_usd": 1.0,
}


@pytest.mark.parametrize("field", ["add_p95_seconds", "search_p95_seconds", "provider_spend_usd"])
@pytest.mark.parametrize("name", sorted(NUMBERS))
def test_promotion_gate_refuses_an_undecidable_measurement(field: str, name: str) -> None:
    from scripts.aml_promotion_gate import decide

    assert decide(PROMOTION)[0] is True
    assert decide({**PROMOTION, field: NUMBERS[name]})[0] is False
